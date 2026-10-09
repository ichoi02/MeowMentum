"""Analysis helpers for the drop-test logs in raw/.

Column semantics (written by hardware/controller.py::main, same order as its header):
  F_Q0..F_Q3  front IMU quaternion (w,x,y,z), ALREADY world-aligned -- controller.py
              applies align_imu_quaternions before logging, so do NOT re-align here.
  F_M1, F_M2  front encoders: rot1 (spine roll, rad), pitch (spine pitch, rad)
  B_M1, B_M2  back encoders:  tail (rad), rot2 (rear roll, rad)
  Cmd_F1/F2, Cmd_B1/B2   PD targets for rot1, pitch, tail, rot2 (rot2 target = -rot1)
  B_Q0..B_Q3  rear IMU quaternion -- identically (1,0,0,0) in every file in raw/, i.e.
              the rear IMU never reported. That is why the rear orientation has to be
              reconstructed from the front IMU + encoders (compute_rear_body_dynamics).
"""
import numpy as np
from scipy.spatial.transform import Rotation as R


def compute_tilt_angle(quat_wxyz, degrees=True):
    """Tilt as |roll| + |pitch| of the body in the world frame. 0 deg = upright.

    Roll and pitch come from the extrinsic "xyz" Euler decomposition -- the same
    convention the drop orientations are built with (R.from_euler("xyz",
    [roll, pitch, 0]) in cat_env.py and test.py), so a roll_180deg_pitch_+015deg
    run starts at 195 deg. Yaw is the final rotation about world z and drops out,
    so the measure does not depend on heading.

    This is NOT the reward's uprightness (cat_env.py::_get_reward uses the angle
    between body +z and world +z). Pitch is confined to [-90, 90] deg and roll to
    [-180, 180] deg, so the sum spans [0, 270] deg, and roll is ill-defined at
    pitch = +-90 deg (scipy warns on gimbal lock).

    Args:
        quat_wxyz: (4,) or (N,4) world-aligned body quaternion, scalar-first.
        degrees: return degrees (default) or radians.

    Returns:
        float, or (N,) array, in [0, 270] deg.
    """
    q = np.asarray(quat_wxyz, dtype=float)
    roll_pitch = R.from_quat(q, scalar_first=True).as_euler("xyz")[..., :2]
    tilt = np.abs(roll_pitch).sum(axis=-1)
    return np.degrees(tilt) if degrees else tilt


def compute_rear_body_dynamics(front_quat_wxyz, rot1, pitch, rot2):
    """Rear-body world orientation from the front IMU + the three spine encoders.

    Walks the model/cat.xml joint chain, front body -> rear body:
        R_rear = R_front @ Rx(rot1) @ Ry(pitch) @ Rx(rot2)
    Matches mj_kinematics to ~1e-4 deg (hardware/reconstruct_viz.py::rear_world_quat).

    The tail joint is a child of the rear body, so it does not enter this product and
    is not an argument. Encoder angles are joint-side radians (the Teensy divides out
    the 9.68:1 roll / 34.014:1 pitch gearing before it reports).

    Args:
        front_quat_wxyz: (4,) or (N,4) world-aligned front quaternion (F_Q0..F_Q3).
        rot1:  spine roll,  F_M1 (rad)
        pitch: spine pitch, F_M2 (rad)
        rot2:  rear roll,   B_M2 (rad)

    Returns:
        (4,) or (N,4) rear-body world quaternion, scalar-first, ready for
        compute_tilt_angle.
    """
    r_front = R.from_quat(np.asarray(front_quat_wxyz, dtype=float), scalar_first=True)
    r_rear = (r_front
              * R.from_euler("x", np.asarray(rot1, dtype=float))
              * R.from_euler("y", np.asarray(pitch, dtype=float))
              * R.from_euler("x", np.asarray(rot2, dtype=float)))
    return r_rear.as_quat(scalar_first=True)


# ---------------------------------------------------------------------------
# Dataset assembly: raw/**/*.csv -> per-run time series of the plotted metrics
# ---------------------------------------------------------------------------
import json
import re
from pathlib import Path

import pandas as pd
from scipy.signal import savgol_filter

G = 9.81
UPRIGHT_THRESH_DEG = 45.0   # "righted" gate for the time-to-right metric

# x-axis order: pure roll, then roll 180 with increasing pitch.
CONDITIONS = [
    ("roll_045deg", "45"),
    ("roll_090deg", "90"),
    ("roll_180deg", "180"),
    ("roll_180deg_pitch_+015deg", "180·p15"),
    ("roll_180deg_pitch_+030deg", "180·p30"),
    ("roll_180deg_pitch_+045deg", "180·p45"),
]
COND_LABEL = dict(CONDITIONS)

_RUN_RE = re.compile(r"^(nt|wt)_(?P<cond>.+)_(?P<date>\d{4})_rep(?P<rep>\d+)\.csv$")


def fall_time(height_m):
    """Free-fall time for a drop height (s). Sets the airtime budget a run gets."""
    return float(np.sqrt(2.0 * float(height_m) / G))


def angular_speed(quat_wxyz, t, smooth=True):
    """Angular speed magnitude (deg/s) from a quaternion series.

    Central difference on the body-frame increment: |rotvec(R_k^-1 R_k+1)| / dt,
    which is frame-independent, so front and rear are directly comparable.

    Smoothed because the inputs are quantized: the roll encoders step 0.0135 rad
    (48 ticks/rev x 9.68 gearing), which alone injects ~39 deg/s of noise into the
    reconstructed rear rate at 50 Hz. savgol(7, 2) keeps the ~200 ms maneuver
    features while removing that. Peaks are lightly flattened -- read levels, not
    exact maxima.
    """
    r = R.from_quat(np.asarray(quat_wxyz, dtype=float), scalar_first=True)
    t = np.asarray(t, dtype=float)
    dt = np.diff(t)
    step = np.degrees((r[:-1].inv() * r[1:]).magnitude()) / dt
    # Land the (N-1) differences back on the N sample times.
    w = np.concatenate([step[:1], 0.5 * (step[:-1] + step[1:]), step[-1:]])
    if smooth and len(w) >= 7:
        w = savgol_filter(w, 7, 2)
    return np.clip(w, 0.0, None)


def run_series(df):
    """One telemetry CSV -> the four metric time series, clocked from release.

    t[0] is free-fall detection (controller.py arms on acc_mag < 3.5), so t is
    time since release, up to one 50 Hz tick (<=20 ms) of detection latency.
    """
    t = df["Time"].to_numpy(dtype=float)
    t = t - t[0]
    front_q = df[["F_Q0", "F_Q1", "F_Q2", "F_Q3"]].to_numpy(dtype=float)
    rear_q = compute_rear_body_dynamics(front_q, df["F_M1"], df["F_M2"], df["B_M2"])
    return {
        "t": t,
        "tilt_front": compute_tilt_angle(front_q),
        "tilt_rear": compute_tilt_angle(rear_q),
        "avel_front": angular_speed(front_q, t),
        "avel_rear": angular_speed(rear_q, t),
    }


def load_runs(root="."):
    """Walk raw/ and return one record per CSV: identity + metric series."""
    runs = []
    for path in sorted(Path(root).rglob("*.csv")):
        m = _RUN_RE.match(path.name)
        if not m or m.group("cond") not in COND_LABEL:
            continue
        s = run_series(pd.read_csv(path))
        runs.append({
            "variant": path.name[:2],          # nt | wt
            "cond": m.group("cond"),
            "date": m.group("date"),
            "rep": int(m.group("rep")),
            "file": str(path),
            **s,
        })
    return runs


def time_to_right(series, thresh=UPRIGHT_THRESH_DEG):
    """First time (s) both bodies are under `thresh` deg of tilt; NaN if never.

    Uncensored -- the dashboard applies the drop-height deadline itself, so a run
    that rights after touchdown is still recorded here and marked censored there.
    """
    ok = (series["tilt_front"] < thresh) & (series["tilt_rear"] < thresh)
    idx = np.flatnonzero(ok)
    return float(series["t"][idx[0]]) if len(idx) else float("nan")


def export_json(out_path, root="."):
    """Emit the compact payload the HTML dashboard embeds (series stay raw so the
    height slider can re-evaluate every metric client-side without a round trip)."""
    runs = load_runs(root)
    payload = {
        "conditions": [{"key": k, "label": v} for k, v in CONDITIONS],
        "threshold_deg": UPRIGHT_THRESH_DEG,
        "g": G,
        "runs": [{
            "variant": r["variant"], "cond": r["cond"],
            "date": r["date"], "rep": r["rep"],
            "t": [round(float(x), 4) for x in r["t"]],
            "tf": [round(float(x), 2) for x in r["tilt_front"]],
            "tr": [round(float(x), 2) for x in r["tilt_rear"]],
            "wf": [round(float(x), 1) for x in r["avel_front"]],
            "wr": [round(float(x), 1) for x in r["avel_rear"]],
            "t_right": (None if np.isnan(time_to_right(r)) else round(time_to_right(r), 4)),
        } for r in runs],
    }
    Path(out_path).write_text(json.dumps(payload, separators=(",", ":")))
    return payload


if __name__ == "__main__":
    p = export_json("dashboard_data.json")
    print(f"{len(p['runs'])} runs -> dashboard_data.json")
