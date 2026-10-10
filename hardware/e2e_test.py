"""Check the robot's inference pipeline end to end on the MuJoCo sim (no hardware needed).

    python -m hardware.e2e_test                    # both variants
    python -m hardware.e2e_test --variant notail

The Cat-v0 / CatNoTail-v0 sim stands in for the robot and drives the controller's own
code (student_obs, policy_action, joint_targets). The checks: the observation width
agrees across the chain, the ONNX export matches the PyTorch student, the controller
observation matches the sim student observation, the controller filter and joint
mapping match the target the sim executes, and the closed loop rights the robot.
The sim's front_body world quaternion stands in for the aligned front IMU and the
joint encoders are read from qpos. Needs pyserial, since it imports the controller.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R

import gymnasium as gym
import train.cat_env  # noqa: F401  (registers Cat-v0 and CatNoTail-v0)
import train.distillation as D
import hardware.controller as C
from model.variants import VARIANTS, student_path, onnx_path

# Upright drops of 100 needed to pass, between zero action (4-13) and the students (20-60)
RIGHTING_FLOOR = {"tail": 30, "notail": 15}

# Outcome of every check run so far
results = []


def check(name, ok, detail=""):
    """Print a check's PASS/FAIL line and record its outcome."""
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    results.append(ok)


def emulate_sensors(sim):
    """Read the robot's sensors from the sim state.

    Returns (front IMU quaternion [w, x, y, z] in the world frame,
    joint angles [rot1, pitch, rot2, tail] in rad).
    """
    front_quat = sim.data.xquat[sim._body_idx["front_body"]].copy()
    qpos_idx = sim._joint_qpos_idx
    joints = np.array([sim.data.qpos[qpos_idx["rot1"]], sim.data.qpos[qpos_idx["pitch"]],
                       sim.data.qpos[qpos_idx["rot2"]], sim.data.qpos[qpos_idx["tail"]]])
    return front_quat, joints


def sim_student_frame(obs):
    """Return the sim's student frame: front projected gravity obs[0:3] + joints obs[12:16]."""
    return np.concatenate([obs[0:3], obs[12:16]])


def tilt_deg(sim, body):
    """Return the angle between a body's z axis and world up, in deg."""
    up = R.from_quat(sim.data.xquat[sim._body_idx[body]], scalar_first=True).apply([0, 0, 1])
    return np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0)))


def load_student(variant):
    """Load the PyTorch student; record a FAIL and return None if the checkpoint is stale."""
    pth_file = student_path(variant)
    student = D.StudentPolicy(D.STUDENT_OBS_DIM, 3)
    try:
        student.load_state_dict(torch.load(pth_file, map_location="cpu"))
    except RuntimeError:
        # A size mismatch: the .pth predates the current observation layout
        check("student checkpoint matches current obs layout", False,
              f"{os.path.basename(pth_file)} was built for a different observation "
              f"width (expected {D.STUDENT_OBS_DIM}). Re-run train.distillation and "
              f"hardware.onnx_conversion for --variant {variant}.")
        return None
    student.eval()
    return student


def check_onnx_matches_torch(sess, input_name, student):
    """Check that the ONNX export and the PyTorch student agree on random inputs."""
    x = np.random.randn(1000, D.STUDENT_OBS_DIM).astype(np.float32)
    with torch.no_grad():
        y_torch = student(torch.from_numpy(x)).numpy()
    y_onnx = sess.run(None, {input_name: x})[0]
    max_diff = float(np.max(np.abs(y_torch - y_onnx)))
    check("ONNX == PyTorch student", max_diff < 1e-4, f"max|diff|={max_diff:.2e}")


def check_obs_matches_sim(env, sess, input_name, variant):
    """Check that the controller observation equals the sim student observation.

    Both sides get the deployed prev_action, with the tail zeroed for no-tail; DAgger in
    train/distillation.py fed back the unzeroed tail instead, which is not compared here.
    """
    # Fly the sim with the controller for 300 steps, comparing the observations at each one
    sim = env.unwrapped
    obs, _ = env.reset()
    prev_ctrl = np.zeros(C.ACT_DIM)
    prev_sim = np.zeros(D.ACT_DIM)
    step_i = 0
    max_diff = 0.0
    for _ in range(300):
        # Build both observations from the same sim state
        front_quat, joints = emulate_sensors(sim)
        ctrl_obs = C.student_obs(front_quat, joints, prev_ctrl, step_i, variant)
        frame = D.zero_tail(sim_student_frame(obs), variant)
        sim_obs = D.build_student_obs(frame, prev_sim, obs[D.STEP_IDX]).astype(np.float32)
        max_diff = max(max_diff, float(np.max(np.abs(ctrl_obs[0] - sim_obs))))

        # Step with the controller's action, starting a new episode when one ends
        raw = C.policy_action(sess, input_name, ctrl_obs, variant)
        prev_ctrl = raw.copy()
        prev_sim = raw.copy()
        step_i += 1
        obs, _, term, trunc, _ = env.step(raw)
        if term or trunc:
            obs, _ = env.reset()
            prev_ctrl = np.zeros(C.ACT_DIM)
            prev_sim = np.zeros(D.ACT_DIM)
            step_i = 0

    # Report the largest difference
    check("controller obs == sim student obs", max_diff < 1e-6, f"max|diff|={max_diff:.2e}")


def check_targets_match_sim(env):
    """Check that the controller's filter and joint mapping equal the sim's executed target."""
    # Turn off the sim's random action delay so each step compares one to one
    sim = env.unwrapped
    env.reset()
    sim.action_delay = 0
    sim.action_buffer = []

    # Feed random actions to both and compare the [rot1, pitch, tail] targets
    rng = np.random.default_rng(0)
    filt = np.zeros(3)
    max_diff = 0.0
    n_compared = 0
    for _ in range(300):
        raw = rng.uniform(-1, 1, 3).astype(np.float32)
        targets, filt = C.joint_targets(raw, filt)
        _, _, term, trunc, _ = env.step(raw)
        if not (term or trunc):
            sim_targets = np.asarray(sim.ctrls[-1][:3])
            max_diff = max(max_diff, float(np.max(np.abs(np.array(targets[:3]) - sim_targets))))
            n_compared += 1
        else:
            # New episode: the sim resets its filter, so the controller's starts over too
            env.reset()
            sim.action_delay = 0
            sim.action_buffer = []
            filt = np.zeros(3)
    check("controller filter+map == sim executed target", max_diff < 1e-6,
          f"max|diff|={max_diff:.2e} over {n_compared} steps")


def check_closed_loop(env, sess, input_name, variant):
    """Check that the full pipeline rights the robot in enough of 100 random drops.

    Emulated sensors -> controller observation -> ONNX -> sim, whose joint mapping and
    PD stand in for the controller's mapping and the Teensy PD. Closes env after the
    last drop.
    """
    # Fly 100 random drops through the controller code
    sim = env.unwrapped
    n_drops = 100
    n_upright = 0
    front_tilts, rear_tilts = [], []
    infer_ms = []
    for _ in range(n_drops):
        # Fly one drop, timing each inference
        env.reset()
        prev_action = np.zeros(C.ACT_DIM)
        step_i = 0
        done = False
        while not done:
            front_quat, joints = emulate_sensors(sim)
            obs = C.student_obs(front_quat, joints, prev_action, step_i, variant)
            t0 = time.perf_counter()
            raw = C.policy_action(sess, input_name, obs, variant)
            infer_ms.append((time.perf_counter() - t0) * 1e3)
            prev_action = raw.copy()
            step_i += 1
            _, _, term, trunc, _ = env.step(raw)
            done = term or trunc

        # It counts as righted if both bodies end within 30 deg of upright
        front_tilt, rear_tilt = tilt_deg(sim, "front_body"), tilt_deg(sim, "rear_body")
        front_tilts.append(front_tilt)
        rear_tilts.append(rear_tilt)
        n_upright += (front_tilt < 30 and rear_tilt < 30)

    # Close the sim
    env.close()

    # Report the success count against the floor, and the inference latency
    floor = RIGHTING_FLOOR[variant]
    check(f"closed-loop righting >= {floor}% ({variant} tripwire)", n_upright >= floor,
          f"{n_upright}/{n_drops} both-upright  |  final tilt med f/r "
          f"{np.median(front_tilts):.0f}/{np.median(rear_tilts):.0f} deg")
    print(f"       ONNX inference: mean {np.mean(infer_ms):.3f} ms, "
          f"p99 {np.percentile(infer_ms, 99):.3f} ms  (budget @50Hz = 20 ms)")


def run(variant):
    """Run every check for one variant."""
    # Print the variant and the controller constants under test
    cfg = VARIANTS[variant]
    onnx_file = onnx_path(variant)
    print(f"\n=== variant={variant}  env={cfg['env_id']}  onnx={os.path.basename(onnx_file)} ===")
    print(f"obs={C.STUDENT_OBS_DIM}-dim  filter_alpha={C.FILTER_ALPHA}  "
          f"ranges(roll/pitch/tail)={C.ROLL_RANGE}/{C.PITCH_RANGE}/{C.TAIL_RANGE}")

    # Load the exported student and the PyTorch student it came from
    sess = ort.InferenceSession(onnx_file)
    input_name = sess.get_inputs()[0].name
    onnx_dim = sess.get_inputs()[0].shape[-1]
    student = load_student(variant)
    if student is None:
        return

    # Check the network: observation width, then ONNX against PyTorch
    check(f"dims consistent (controller/distill/onnx = {D.STUDENT_OBS_DIM})",
          C.STUDENT_OBS_DIM == D.STUDENT_OBS_DIM == onnx_dim,
          f"ctrl={C.STUDENT_OBS_DIM} distill={D.STUDENT_OBS_DIM} onnx={onnx_dim}")
    check_onnx_matches_torch(sess, input_name, student)

    # Check the controller code against the sim
    env = gym.make(cfg["env_id"])
    check_obs_matches_sim(env, sess, input_name, variant)
    check_targets_match_sim(env)
    check_closed_loop(env, sess, input_name, variant)


def main():
    """Run the checks for one or both variants; exit with 1 if any check failed."""
    # Parse the command line
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=list(VARIANTS), default=None,
                        help="ablation condition to test (default: both)")
    args = parser.parse_args()

    # Run the checks
    for variant in ([args.variant] if args.variant else list(VARIANTS)):
        run(variant)

    # Summarize
    print("\n=== SUMMARY ===")
    print(f"{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
