"""
End-to-end hardware-pipeline test (no physical hardware required).

Drives the MuJoCo Cat-v0 / CatNoTail-v0 sim as the "physical robot" and runs the
ACTUAL hardware inference pipeline from controller.py (student_obs, policy_action,
joint_targets: projected-gravity obs, previous action + step, ONNX inference, action
low-pass, joint-range mapping). Verifies each stage matches the simulation/training
pipeline, then confirms closed-loop righting.

The sim's front_body world-frame quaternion stands in for the front IMU (i.e. we
assume the on-robot IMU alignment yields a z-up world frame); joint encoders are
read from qpos. Everything downstream of that is the real controller code.

Runs from any directory (needs pyserial, since it imports controller.py):
    python hardware/e2e_test.py                    # both variants
    python hardware/e2e_test.py --variant notail
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "hardware"))

import gymnasium as gym
import cat_env  # noqa: F401  (registers Cat-v0 / CatNoTail-v0)
import distillation as D
import controller as C  # the real hardware controller
from variants import VARIANTS, student_path, onnx_path

# Floor for the closed-loop check. This is a coarse pipeline-breakage tripwire, not a
# performance target: N=100 random release attitudes is noisy. The released students
# score about 50-60/100 (tail) and 20-35/100 (no tail). A zero-action policy scores about
# 4-13/100, because some releases start close to upright, so the floors sit above that
# baseline: they catch a broken pipeline as well as a collapse to passivity.
RIGHTING_FLOOR = {"tail": 30, "notail": 15}

results = []
def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    results.append(ok)


# ---- shared helpers: emulate the on-robot sensors ----
def emulate_sensors(u):
    """Front IMU quaternion (world frame) + 4 joint encoder angles, from sim state."""
    front_quat = u.data.xquat[u._body_idx["front_body"]].copy()
    qi = u._joint_qpos_idx
    joints = np.array([u.data.qpos[qi["rot1"]], u.data.qpos[qi["pitch"]],
                       u.data.qpos[qi["rot2"]], u.data.qpos[qi["tail"]]])
    return front_quat, joints

def sim_student_frame(obs):
    """Ground-truth student frame from the sim obs (front_proj_grav[0:3] + joints[12:16])."""
    return np.concatenate([obs[0:3], obs[12:16]])

def tilt_deg(u, body):
    up = R.from_quat(u.data.xquat[u._body_idx[body]], scalar_first=True).apply([0, 0, 1])
    return np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0)))


def run(variant):
    """Run all checks for one variant. Returns True if every check passed."""
    n_before = len(results)
    cfg = VARIANTS[variant]
    ONNX_FILE = onnx_path(variant)
    PTH_FILE = student_path(variant)

    print(f"\n=== variant={variant}  env={cfg['env_id']}  onnx={os.path.basename(ONNX_FILE)} ===")
    print(f"obs={C.STUDENT_OBS_DIM}-dim  filter_alpha={C.FILTER_ALPHA}  "
          f"ranges(roll/pitch/tail)={C.ROLL_RANGE}/{C.PITCH_RANGE}/{C.TAIL_RANGE}")

    sess = ort.InferenceSession(ONNX_FILE)
    inp = sess.get_inputs()[0].name
    onnx_dim = sess.get_inputs()[0].shape[-1]

    student = D.StudentPolicy(D.STUDENT_OBS_DIM, 3)
    try:
        student.load_state_dict(torch.load(PTH_FILE, map_location="cpu"))
    except RuntimeError:
        # Almost always a stale artifact rather than a real bug: the observation
        # layout changed and this .pth predates it. Say so instead of dumping a
        # size-mismatch traceback.
        check("student checkpoint matches current obs layout", False,
              f"{os.path.basename(PTH_FILE)} was built for a different observation "
              f"width (expected {D.STUDENT_OBS_DIM}). Re-run distillation.py and "
              f"onnx_conversion.py for --variant {variant}.")
        return False
    student.eval()

    # ---- TEST 0: dimensions line up across the whole chain ----
    check(f"dims consistent (controller/distill/onnx = {D.STUDENT_OBS_DIM})",
          C.STUDENT_OBS_DIM == D.STUDENT_OBS_DIM == onnx_dim,
          f"ctrl={C.STUDENT_OBS_DIM} distill={D.STUDENT_OBS_DIM} onnx={onnx_dim}")

    # ---- TEST 1: ONNX == PyTorch student ----
    X = np.random.randn(1000, D.STUDENT_OBS_DIM).astype(np.float32)
    with torch.no_grad():
        yt = student(torch.from_numpy(X)).numpy()
    yo = sess.run(None, {inp: X})[0]
    d1 = float(np.max(np.abs(yt - yo)))
    check("ONNX == PyTorch student", d1 < 1e-4, f"max|diff|={d1:.2e}")

    env = gym.make(cfg["env_id"])
    u = env.unwrapped

    # ---- TEST 2: controller obs-build == sim student obs (over a rollout) ----
    # Both sides are fed the same prev_action (tail zeroed for notail, as deployed).
    # distillation.collect_data fed back the unzeroed tail command instead; that
    # train/deploy difference is outside what this test compares.
    obs, _ = env.reset()
    prev_c = np.zeros(C.ACT_DIM)
    prev_s = np.zeros(D.ACT_DIM)
    step_i = 0
    maxdiff2 = 0.0
    for _ in range(300):
        fq, jt = emulate_sensors(u)
        ctrl_obs = C.student_obs(fq, jt, prev_c, step_i, variant)
        sf = D.zero_tail(sim_student_frame(obs), variant)
        sim_obs = D.build_student_obs(sf, prev_s, obs[D.STEP_IDX]).astype(np.float32)
        maxdiff2 = max(maxdiff2, float(np.max(np.abs(ctrl_obs[0] - sim_obs))))
        raw = C.policy_action(sess, inp, ctrl_obs, variant)
        prev_c = raw.copy(); prev_s = raw.copy()
        step_i += 1
        obs, _, term, trunc, _ = env.step(raw)
        if term or trunc:
            obs, _ = env.reset()
            prev_c = np.zeros(C.ACT_DIM); prev_s = np.zeros(D.ACT_DIM); step_i = 0
    check("controller obs == sim student obs", maxdiff2 < 1e-6, f"max|diff|={maxdiff2:.2e}")

    # ---- TEST 3: controller filter+map == sim internal executed target ----
    # Disable the sim's domain-randomization action delay so the comparison is 1:1.
    obs, _ = env.reset()
    u.action_delay = 0
    u.action_buffer = []
    rng = np.random.default_rng(0)
    c_filt = np.zeros(3)
    maxdiff3 = 0.0
    n_cmp = 0
    for _ in range(300):
        raw = rng.uniform(-1, 1, 3).astype(np.float32)
        # Controller path: per-channel low-pass, then the joint-range map.
        c_targets, c_filt = C.joint_targets(raw, c_filt)
        obs, _, term, trunc, _ = env.step(raw)
        if not (term or trunc):
            sim_exec = np.asarray(u.ctrls[-1][:3])  # sim's mapped [rot1,pitch,tail] target
            maxdiff3 = max(maxdiff3, float(np.max(np.abs(np.array(c_targets[:3]) - sim_exec))))
            n_cmp += 1
        else:
            obs, _ = env.reset()
            u.action_delay = 0
            u.action_buffer = []
            c_filt = np.zeros(3)          # cat_env resets its filter per episode
    check("controller filter+map == sim executed target", maxdiff3 < 1e-6,
          f"max|diff|={maxdiff3:.2e} over {n_cmp} steps")

    # ---- TEST 4: closed-loop righting through the full hardware pipeline ----
    # Build obs from emulated sensors -> ONNX -> feed to env (sim map+PD == controller+Teensy).
    N = 100
    ok = 0
    ftilt, rtilt = [], []
    infer_times = []
    for _ in range(N):
        obs, _ = env.reset()
        prev_c = np.zeros(C.ACT_DIM)
        step_i = 0
        done = False
        while not done:
            fq, jt = emulate_sensors(u)
            x = C.student_obs(fq, jt, prev_c, step_i, variant)
            t0 = time.perf_counter()
            raw = C.policy_action(sess, inp, x, variant)
            infer_times.append((time.perf_counter() - t0) * 1e3)
            prev_c = raw.copy()
            step_i += 1
            obs, _, term, trunc, _ = env.step(raw)
            done = term or trunc
        fa, ra = tilt_deg(u, "front_body"), tilt_deg(u, "rear_body")
        ftilt.append(fa); rtilt.append(ra)
        ok += (fa < 30 and ra < 30)
    env.close()
    pct = ok
    floor = RIGHTING_FLOOR[variant]
    check(f"closed-loop righting >= {floor}% ({variant} tripwire)", pct >= floor,
          f"{pct}/100 both-upright  |  final tilt med f/r "
          f"{np.median(ftilt):.0f}/{np.median(rtilt):.0f} deg")
    print(f"       ONNX inference: mean {np.mean(infer_times):.3f} ms, "
          f"p99 {np.percentile(infer_times,99):.3f} ms  (budget @50Hz = 20 ms)")
    return all(results[n_before:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=list(VARIANTS), default=None,
                        help="ablation condition to test (default: both)")
    args = parser.parse_args()
    for v in ([args.variant] if args.variant else list(VARIANTS)):
        run(v)
    print("\n=== SUMMARY ===")
    print(f"{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)
