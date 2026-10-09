"""Build docs/assets/data/page-data.js, the data behind the project page.

The page loads this file with a plain <script> tag (window.PAGE_DATA = {...}) so it
also works when opened from file://. Sections:

  trials    every hardware drop: identity, whole-cat score, flown policy, and the
            per-sample front quaternion, joint angles and mean body tilt
  tilt      Fig. 4 curves: mean +/- SD of the mean body tilt per condition, first
            33 control steps, every trial in the condition folder
  table2    Table II: raw whole-cat score per arm plus the adjusted tail effect
  sim       one nominal-physics rollout of each deployed student per release
  training  teacher training curves from the TensorBoard logs
  meta      robot facts and the model/cat.xml geometry needed to draw the robot

Run from the repository root:

    python docs/assets/data/build_page_data.py

Table II's adjusted columns (ANCOVA with HC3 errors, Holm, blocked permutation) are
read from the whole-cat analysis output, model_results.csv and sweep_results.csv,
which must already exist in --stats. Everything else is computed here from the
drop logs, the policies, model/cat.xml and the TensorBoard event files.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R, Slerp

HERE = Path(__file__).resolve().parent
G = 9.81

SCORE_DISTANCE_M = 2.5    # whole-cat score is taken at this much ballistic fall
TILT_STEPS = 33           # Fig. 4 window, in control steps
TRAINING_POINTS = 150     # training curves are block-averaged down to this many points

FILENAME_RE = re.compile(r"^(?P<morph>wt|nt)_roll_(?P<roll>\d{3})deg"
                         r"(?:_pitch_(?P<pitch>[+-]\d{3})deg)?_"
                         r"(?P<date>\d{4})_rep(?P<rep>\d{2})\.csv$")
MORPH = {"wt": "tail", "nt": "notail"}
FOLDER_MORPH = {"with_tail": "tail", "no_tail": "notail"}

# Release conditions, in the paper's order: roll sweep, then the pitch sweep at roll 180.
CONDITIONS = [(45, 0), (90, 0), (180, 0), (180, 15), (180, 30), (180, 45)]
ROLL_SWEEP = {(45, 0), (90, 0), (180, 0)}
PITCH_SWEEP = {(180, 0), (180, 15), (180, 30), (180, 45)}

# Logged columns, in model joint order [rot1, pitch, rot2, tail].
JOINT_COLUMNS = ["F_M1", "F_M2", "B_M2", "B_M1"]
QUAT_COLUMNS = ["F_Q0", "F_Q1", "F_Q2", "F_Q3"]


def r4(values):
    """Round to 4 decimals for the JSON payload (and turn -0.0 into 0.0)."""
    if values is None:
        return None
    arr = np.round(np.asarray(values, dtype=float), 4) + 0.0
    return arr.tolist()


def rear_rotation(front, joints):
    """Rear-body world rotation: R_front * Rx(rot1) * Ry(pitch) * Rx(rot2)."""
    joints = np.asarray(joints, dtype=float)
    return (front * R.from_euler("x", joints[..., 0]) * R.from_euler("y", joints[..., 1])
            * R.from_euler("x", joints[..., 2]))


def body_tilt_deg(rotation):
    """|roll| + |pitch| of the extrinsic xyz Euler decomposition, in degrees (Fig. 4)."""
    roll_pitch = rotation.as_euler("xyz")[..., :2]
    return np.degrees(np.abs(roll_pitch).sum(axis=-1))


def mean_tilt_deg(front, joints):
    """Per-sample mean of the front and reconstructed rear body tilt."""
    return 0.5 * (body_tilt_deg(front) + body_tilt_deg(rear_rotation(front, joints)))


def whole_cat_score(frame, distance=SCORE_DISTANCE_M):
    """(score, S_upright, S_align) at `distance` m of ballistic fall, or None.

    Same definition as the whole-cat analysis: both body rotations are SLERPed to
    t0 + sqrt(2 d / g); S_upright uses the normalized sum of the two body up-vectors,
    S_align the angle of the front-to-rear relative rotation. A drop whose impact
    (accelerometer peak) comes before that instant is excluded and returns None.
    """
    time = frame["Time"].to_numpy(float)
    accel = np.maximum(frame["F_ACC"].to_numpy(float), frame["B_ACC"].to_numpy(float))
    impact = int(np.nanargmax(accel))
    target = time[0] + math.sqrt(2.0 * distance / G)
    if impact < 2 or time[impact] < target:
        return None
    quat = frame[QUAT_COLUMNS].to_numpy(float)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    front = R.from_quat(quat, scalar_first=True)
    rear = rear_rotation(front, frame[JOINT_COLUMNS].to_numpy(float))
    span = slice(0, impact + 1)
    f = Slerp(time[span], front[span])([target])[0]
    r = Slerp(time[span], rear[span])([target])[0]
    up_sum = f.apply([0.0, 0.0, 1.0]) + r.apply([0.0, 0.0, 1.0])
    norm = np.linalg.norm(up_sum)
    tilt = math.pi if norm < 1e-8 else math.acos(float(np.clip(up_sum[2] / norm, -1.0, 1.0)))
    s_up = float(np.clip(1.0 - tilt / math.pi, 0.0, 1.0))
    s_align = float(np.clip(1.0 - (f.inv() * r).magnitude() / math.pi, 0.0, 1.0))
    return s_up * s_align, s_up, s_align


def find_trials(data_dir):
    """[(path, meta)] for every drop log under data_dir, in a stable order."""
    out = []
    for path in sorted(Path(data_dir).rglob("*.csv")):
        m = FILENAME_RE.fullmatch(path.name)
        if m is None:
            continue
        folder = next(p for p in path.relative_to(data_dir).parts if p in FOLDER_MORPH)
        meta = {
            "id": path.stem,
            "morph": FOLDER_MORPH[folder],
            "roll": int(m["roll"]),
            "pitch": int(m["pitch"] or 0),
            "date": m["date"],
            "rep": int(m["rep"]),
        }
        if MORPH[m["morph"]] != meta["morph"]:
            raise ValueError(f"{path}: file prefix disagrees with its folder")
        out.append((path, meta))
    if not out:
        raise FileNotFoundError(f"no drop logs under {data_dir}")
    return out


def build_trials(trials):
    records = []
    for path, meta in trials:
        frame = pd.read_csv(path)
        quat = frame[QUAT_COLUMNS].to_numpy(float)
        joints = frame[JOINT_COLUMNS].to_numpy(float)
        tilt = mean_tilt_deg(R.from_quat(quat, scalar_first=True), joints)
        score = whole_cat_score(frame)
        # The no-tail controller pins the tail command to exactly 0, so a nonzero
        # Cmd_B1 means the with-tail policy flew (agrees with an ONNX replay of
        # every log).
        policy = "tail" if np.any(frame["Cmd_B1"].to_numpy(float) != 0.0) else "notail"
        time = frame["Time"].to_numpy(float)
        # Blanket impact = the accelerometer peak, as in the analysis' exclusion rule.
        acc = np.maximum(frame["F_ACC"].to_numpy(float), frame["B_ACC"].to_numpy(float))
        impact_t = float(time[int(np.nanargmax(acc))] - time[0])
        records.append({
            **meta,
            "included": score is not None,
            "score": None if score is None else r4(score[0]),
            "s_up": None if score is None else r4(score[1]),
            "s_align": None if score is None else r4(score[2]),
            "policy": policy,
            "impact_t": round(impact_t, 4),
            "t": r4(time - time[0]),
            "q": r4(quat),
            "j": r4(joints),
            "tilt": r4(tilt),
        })
    return records


def build_tilt(trials, steps=TILT_STEPS):
    """Fig. 4 curves, grouped by morphology and condition folder.

    Runs are stacked by control step (no resampling), cut to the shortest run in the
    group; x is the mean of the runs' own timestamps; the band is the population SD.
    """
    groups = {}
    for path, meta in trials:
        groups.setdefault((meta["morph"], meta["roll"], meta["pitch"]), []).append(path)
    curves = []
    for morph in ("tail", "notail"):
        for roll, pitch in CONDITIONS:
            paths = groups.get((morph, roll, pitch))
            if not paths:
                continue
            runs = []
            for path in paths:
                frame = pd.read_csv(path).iloc[:steps]
                time = frame["Time"].to_numpy(float)
                quat = frame[QUAT_COLUMNS].to_numpy(float)
                tilt = mean_tilt_deg(R.from_quat(quat, scalar_first=True),
                                     frame[JOINT_COLUMNS].to_numpy(float))
                runs.append((time - time[0], tilt))
            k = min(len(t) for t, _ in runs)
            tilts = np.array([tilt[:k] for _, tilt in runs])
            curves.append({
                "morph": morph, "roll": roll, "pitch": pitch, "n": len(runs),
                "t": r4(np.mean([t[:k] for t, _ in runs], axis=0)),
                "mean": r4(tilts.mean(axis=0)),
                "sd": r4(tilts.std(axis=0)),
            })
    return curves


def build_table2(records, stats_dir):
    """Table II rows: raw mean +/- sample SD per arm, then the adjusted analysis."""
    stats_dir = Path(stats_dir)
    try:
        model = pd.read_csv(stats_dir / "model_results.csv")
        sweeps = pd.read_csv(stats_dir / "sweep_results.csv")
    except FileNotFoundError as err:
        raise SystemExit(f"{err.filename} not found: run the whole-cat analysis with "
                         f"its output in {stats_dir} first (or pass --stats)") from err
    adjusted = {
        "all": model.loc[model["outcome"] == "posture_score"].iloc[0],
        "roll": sweeps.loc[sweeps["sweep"] == "no_pitch_roll_sweep"].iloc[0],
        "pitch": sweeps.loc[sweeps["sweep"] == "pitch_sweep_at_roll_180"].iloc[0],
    }
    members = {"all": None, "roll": ROLL_SWEEP, "pitch": PITCH_SWEEP}
    labels = {"all": "All trials", "roll": "Roll sweep", "pitch": "Pitch sweep"}
    rows = []
    for key in ("all", "roll", "pitch"):
        arms = {}
        for morph in ("notail", "tail"):
            scores = [t["score"] for t in records
                      if t["included"] and t["morph"] == morph
                      and (members[key] is None or (t["roll"], t["pitch"]) in members[key])]
            arms[morph] = {"mean": r4(np.mean(scores)), "sd": r4(np.std(scores, ddof=1)),
                           "n": len(scores)}
        a = adjusted[key]
        if "n" in a and int(a["n"]) != arms["notail"]["n"] + arms["tail"]["n"]:
            raise ValueError(f"{key}: analysis n={int(a['n'])} does not match the trials here")
        rows.append({
            "analysis": key, "label": labels[key],
            "notail": arms["notail"], "tail": arms["tail"],
            "effect": r4(a["estimate"]),
            "ci": r4([a["ci_low"], a["ci_high"]]),
            "p": r4(a["p_hc3"]),
            # The overall comparison is the primary analysis and carries no Holm value.
            "p_holm": None if key == "all" else r4(a["p_holm_three_requested"]),
            "p_perm": r4(a["p_permutation"]),
        })
    return rows


def build_sim(root):
    """One deterministic rollout of each deployed student per release condition.

    Nominal physics: the env's per-reset domain-randomization draw is overwritten
    with the nominal model, with no action delay. Zero initial angular velocity and
    joint angles, clean (noise-free) observations, and the student observation and
    no-tail masking of evaluate.py. Sample k is the state at t = 0.02 k, the one
    the k-th action is computed from; the last sample is the end of the episode.
    """
    sys.path.insert(0, str(root))
    import gymnasium as gym
    import mujoco
    import torch
    import cat_env  # noqa: F401  (registers Cat-v0 / CatNoTail-v0)
    from distillation import (StudentPolicy, build_student_obs, zero_tail,
                              STUDENT_OBS_DIM, STEP_IDX, FRONT_GRAV_SLICE,
                              JOINT_ANGLE_SLICE, ACT_DIM)
    from variants import VARIANTS, student_path

    model_file = {"tail": "model/cat.xml", "notail": "model/cat_notail.xml"}
    rollouts = []
    for variant in ("tail", "notail"):
        policy = StudentPolicy(STUDENT_OBS_DIM, ACT_DIM)
        policy.load_state_dict(torch.load(student_path(variant), map_location="cpu"))
        policy.eval()
        env = gym.make(VARIANTS[variant]["env_id"],
                       model_path=str(Path(root) / model_file[variant]))
        u = env.unwrapped
        front_id = u._body_idx["front_body"]
        rear_id = u._body_idx["rear_body"]
        for roll, pitch in CONDITIONS:
            np.random.seed(0)
            env.reset(seed=0)
            # Undo the domain-randomization draw reset_model() just made.
            m = u.model
            m.body_mass[:] = u.nominal_mass
            m.body_ipos[:] = u.nominal_ipos
            m.body_inertia[:] = u.nominal_inertia
            m.dof_damping[:] = u.nominal_damping
            m.dof_armature[:] = u.nominal_armature
            m.dof_frictionloss[:] = u.nominal_frictionloss
            m.actuator_ctrlrange[:] = u.nominal_ctrlrange
            for pd, (kp, kd) in zip(u.pd, u.pd_nominal):
                pd.kp, pd.kd = kp, kd
            u.action_delay = 0
            u.action_buffer = []
            u.action_filt = np.zeros(ACT_DIM)
            mujoco.mj_setConst(m, u.data)
            # Release attitude as in cat_env/evaluate.py: extrinsic xyz Euler, yaw 0.
            qpos = u.init_qpos.copy()
            x, y, z, w = R.from_euler("xyz", [roll, pitch, 0.0], degrees=True).as_quat()
            qpos[3:7] = [w, x, y, z]
            qpos[7:] = 0.0
            u.set_state(qpos, np.zeros_like(u.init_qvel))
            for pd, name in zip(u.pd, u.joint_names):
                pd.reset(u.data.qpos[u._joint_qpos_idx[name]])
            obs = u._get_obs()

            prev_action = np.zeros(ACT_DIM)
            quats, rear_quats, joints = [], [], []

            def record():
                # xquat is only refreshed at the start of mj_step; recompute it so the
                # body orientations and qpos describe the same instant.
                mujoco.mj_kinematics(u.model, u.data)
                quats.append(u.data.xquat[front_id].copy())
                rear_quats.append(u.data.xquat[rear_id].copy())
                joints.append(u.data.qpos[7:11].copy())

            record()
            done = False
            while not done:
                frame = np.concatenate([obs[FRONT_GRAV_SLICE], obs[JOINT_ANGLE_SLICE]])
                frame = zero_tail(frame, variant)
                x_np = build_student_obs(frame, prev_action, obs[STEP_IDX])
                with torch.no_grad():
                    action = policy(torch.FloatTensor(x_np).unsqueeze(0)).squeeze(0).numpy()
                if variant == "notail":
                    action[2] = 0.0
                prev_action = np.asarray(action, dtype=float).copy()
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                record()

            quats, joints = np.array(quats), np.array(joints)
            tilt = 0.5 * (body_tilt_deg(R.from_quat(quats, scalar_first=True))
                          + body_tilt_deg(R.from_quat(np.array(rear_quats), scalar_first=True)))
            rollouts.append({
                "morph": variant, "roll": roll, "pitch": pitch,
                "t": r4(0.02 * np.arange(len(quats))),
                "q": r4(quats), "j": r4(joints), "tilt": r4(tilt),
            })
        env.close()
    return rollouts


def build_training(log_dir, points=TRAINING_POINTS):
    """Teacher curves, block-averaged: each point is the mean of consecutive logged
    values, placed at the mean timestep of its block."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    curves = []
    for morph in ("tail", "notail"):
        acc = EventAccumulator(str(Path(log_dir) / f"teacher_{morph}"),
                               size_guidance={"scalars": 0})
        acc.Reload()
        series = {}
        for tag in ("rewards/up_mean", "rewards/r_pos"):
            if tag not in acc.Tags()["scalars"]:
                continue
            events = acc.Scalars(tag)
            steps = np.array([e.step for e in events], dtype=float)
            values = np.array([e.value for e in events], dtype=float)
            blocks = np.array_split(np.arange(len(events)), min(points, len(events)))
            series["step"] = [int(round(steps[b].mean())) for b in blocks]
            series[tag.split("/")[1]] = r4([values[b].mean() for b in blocks])
        curves.append({"morph": morph, **series})
    return curves


def build_meta(root):
    """Robot facts (paper) and drawing geometry (model/cat.xml, all joints at zero)."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(Path(root) / "model" / "cat.xml"))

    def geoms(name, kind):
        body_id = m.body(name).id
        return [m.geom(i) for i in range(m.ngeom)
                if m.geom_bodyid[i] == body_id and m.geom_type[i] == kind]

    box, sphere, cylinder = (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_SPHERE,
                             mujoco.mjtGeom.mjGEOM_CYLINDER)
    front_origin = np.zeros(3)
    spine = front_origin + m.body("spine_1").pos
    rear = spine + m.body("spine_2").pos + m.body("rear_body").pos
    tail = rear + m.body("tail").pos
    tip = tail + m.body("tail").ipos          # 150 g point mass at the end of the rod
    mass = {name: float(m.body(name).mass[0])
            for name in ("front_body", "spine_1", "spine_2", "rear_body", "tail")}

    def joint_info(name, origin):
        j = m.joint(name)
        axis = "xyz"[int(np.argmax(np.abs(j.axis)))]
        return {"name": name, "axis": axis, "origin": r4(origin), "range": r4(j.range)}

    return {
        "robot": {
            "mass_kg": {"tail": 1.86, "notail": 1.71},
            "length_cm": {"tail": 58.7, "notail": 44.7},
            "policy_hz": 50, "pd_hz": 1000, "control_steps": 50, "episode_s": 1.0,
            "freefall_trigger_ms2": 3.5,
        },
        "evaluation": {
            "score_distance_m": SCORE_DISTANCE_M,
            "score_time_s": r4(math.sqrt(2.0 * SCORE_DISTANCE_M / G)),
            "tilt_steps": TILT_STEPS,
            "g": G,
        },
        "model": {
            "source": "model/cat.xml",
            "units": "m, kg; positions in the front-body frame with every joint at 0",
            "front_body": {"center": r4(front_origin),
                           "half_size": r4(geoms("front_body", box)[0].size),
                           "mass": r4(mass["front_body"])},
            "spine": {"center": r4(spine),
                      "mass": r4(mass["spine_1"] + mass["spine_2"])},
            "rear_body": {"center": r4(rear),
                          "half_size": r4(geoms("rear_body", box)[0].size),
                          "mass": r4(mass["rear_body"])},
            "tail": {"pivot": r4(tail), "tip": r4(tip),
                     "rod_radius": r4(geoms("tail", cylinder)[0].size[0]),
                     "tip_radius": r4(geoms("tail", sphere)[0].size[0]),
                     "tip_mass": r4(mass["tail"])},
            "paws": {
                "radius": r4(geoms("front_body", sphere)[0].size[0]),
                "front": [r4(front_origin + g.pos) for g in geoms("front_body", sphere)],
                "rear": [r4(rear + g.pos) for g in geoms("rear_body", sphere)],
            },
        },
        "joints": [
            joint_info("rot1", spine),
            joint_info("pitch", spine + m.body("spine_2").pos),
            joint_info("rot2", rear),
            joint_info("tail", tail),
        ],
        "kinematics": [
            "q is the front-body world quaternion (w, x, y, z), body to world",
            "j is [rot1, pitch, rot2, tail] in rad",
            "R_rear = R_front * Rx(rot1) * Ry(pitch) * Rx(rot2)",
            "R_tail = R_rear * Ry(tail)",
            "front-body points x (front_body, paws.front, spine.center) sit at "
            "p_front + R_front * x",
            "p_rear = p_front + R_front * spine.center "
            "+ R_front * Rx(rot1) * Ry(pitch) * (rear_body.center - spine.center)",
            "rear-body points x (paws.rear, tail.pivot) sit at "
            "p_rear + R_rear * (x - rear_body.center)",
            "tail tip = tail pivot + R_tail * (tail.tip - tail.pivot)",
            "tilt = mean over front and rear of |roll| + |pitch| (extrinsic xyz Euler), deg",
        ],
    }


def main():
    root = HERE.parents[2]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=root,
                        help="repository root: code, policies and model are loaded "
                             "from here (default: %(default)s)")
    parser.add_argument("--data", type=Path, default=None,
                        help="drop logs (default: ROOT/data/hardware_trials)")
    parser.add_argument("--stats", type=Path, default=None,
                        help="folder holding the whole-cat analysis output "
                             "model_results.csv and sweep_results.csv (default: ROOT/results)")
    parser.add_argument("--logs", type=Path, default=None,
                        help="TensorBoard logs with teacher_tail/ and teacher_notail/ "
                             "(default: ROOT/results/training_logs)")
    parser.add_argument("--out", type=Path, default=HERE / "page-data.js",
                        help="output script (default: %(default)s)")
    args = parser.parse_args()
    args.root = args.root.resolve()
    data = args.data or args.root / "data" / "hardware_trials"
    stats = args.stats or args.root / "results"
    logs = args.logs or args.root / "results" / "training_logs"

    trials = find_trials(data)
    records = build_trials(trials)
    payload = {
        "meta": build_meta(args.root),
        "trials": records,
        "tilt": build_tilt(trials),
        "table2": build_table2(records, stats),
        "sim": build_sim(args.root),
        "training": build_training(logs),
    }
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    args.out.write_text("// Generated by docs/assets/data/build_page_data.py -- do not edit.\n"
                        f"window.PAGE_DATA = {text};\n")
    print(f"{len(records)} trials, {len(payload['tilt'])} tilt curves, "
          f"{len(payload['sim'])} sim rollouts -> {args.out} "
          f"({args.out.stat().st_size / 1024:.0f} KB)")
    for s in payload["sim"]:
        print(f"  sim {s['morph']:6s} roll {s['roll']:3d} pitch {s['pitch']:2d}: "
              f"tilt {s['tilt'][0]:6.1f} -> {s['tilt'][-1]:5.1f} deg")


if __name__ == "__main__":
    main()
