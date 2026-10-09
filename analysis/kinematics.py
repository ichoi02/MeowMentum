"""Shared kinematics and trial loading for the hardware drop logs.

Each trial in data/hardware_trials/ is one CSV written by hardware/controller.py:
one row per 50 Hz control step for 1.0 s of wall clock after the free-fall trigger
(49 or 50 rows). Column semantics (data/README.md has the full dictionary):
  Time        s since the controller started (not since the drop); the first row
              is the first control step after free fall was detected
  F_Q0..F_Q3  front IMU quaternion (w,x,y,z), ALREADY world-aligned -- controller.py
              applies align_imu_quaternions before logging, so do NOT re-align here
  F_M1, F_M2  front encoders: rot1 (spine roll, rad), pitch (spine pitch, rad)
  F_ACC       front accelerometer magnitude (m/s^2, gravity included)
  B_M1, B_M2  rear encoders: tail (rad), rot2 (rear roll, rad)
  Cmd_F1/F2, Cmd_B1/B2   joint targets for rot1, pitch, tail, rot2 (rad; rot2 = -rot1)
  B_Q0..B_Q3, B_ACC      rear IMU placeholders: (1,0,0,0) and 0 in every file. The
              rear IMU never reported, so the rear-body orientation is reconstructed
              from the front IMU and the spine encoders (compute_rear_body_quat).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

G = 9.81

QUAT_COLUMNS = ["F_Q0", "F_Q1", "F_Q2", "F_Q3"]

# Canonical trial names: <nt|wt>_roll_<RRR>deg[_pitch_<+PPP>deg]_<MMDD>_rep<NN>.csv
TRIAL_NAME_RE = re.compile(
    r"^(?P<morph>wt|nt)_roll_(?P<roll>\d{3})deg"
    r"(?:_pitch_(?P<pitch>[+-]\d{3})deg)?_"
    r"(?P<date>\d{4})_rep(?P<rep>\d{2})\.csv$"
)

# Release conditions in the order the paper lists them: the roll sweep at 0 deg
# pitch, then the pitch sweep at 180 deg roll (whose 0 deg point is r180).
ROLL_SWEEP = ("r45", "r90", "r180")
PITCH_SWEEP = ("r180", "r180_p15", "r180_p30", "r180_p45")
CONDITION_ORDER = ("r45", "r90", "r180", "r180_p15", "r180_p30", "r180_p45")
MORPHOLOGIES = ("with_tail", "no_tail")

# Non-trial CSVs that live alongside the trials.
NON_TRIAL_FILES = {"MANIFEST.csv"}


def parse_trial_name(path):
    """Canonical trial filename -> morphology, release condition, date label, rep."""
    match = TRIAL_NAME_RE.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"Unrecognized canonical filename: {path}")
    data = match.groupdict()
    roll = int(data["roll"])
    pitch = int(data["pitch"] or 0)
    condition = f"r{roll}" if data["pitch"] is None else f"r{roll}_p{pitch}"
    return {
        "morphology": "with_tail" if data["morph"] == "wt" else "no_tail",
        "tail": int(data["morph"] == "wt"),
        "roll_deg": roll,
        "pitch_deg": pitch,
        "condition": condition,
        "date": data["date"],
        "rep": int(data["rep"]),
    }


def find_trials(root):
    """Every trial CSV under `root`, sorted by path.

    The sort order is part of the analysis: the blocked permutation test draws its
    permutations in this order, so its p-values depend on it.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")
    paths = sorted(p for p in root.rglob("*.csv") if p.name not in NON_TRIAL_FILES)
    if not paths:
        raise FileNotFoundError(f"no trial CSVs found under {root}")
    for path in paths:
        parse_trial_name(path)  # fail loudly on a misnamed file
    return paths


def compute_tilt_angle(quat_wxyz, degrees=True):
    """Tilt as |roll| + |pitch| of the body in the world frame. 0 deg = upright.

    Roll and pitch come from the extrinsic "xyz" Euler decomposition -- the same
    convention the simulated drop orientations are built with (R.from_euler("xyz",
    [roll, pitch, 0]) in cat_env/cat_env.py), so a roll 180 deg, pitch 15 deg
    release starts at 195 deg. Yaw is the final rotation about world z and drops
    out, so the measure does not depend on heading.

    This is NOT the reward's uprightness (cat_env.py::_get_reward uses the angle
    between body +z and world +z), nor the whole-cat score's. Pitch is confined to
    [-90, 90] deg and roll to [-180, 180] deg, so the sum spans [0, 270] deg, and
    roll is ill-defined at pitch = +-90 deg (scipy warns on gimbal lock).

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


def rear_body_rotation(front, rot1, pitch, rot2):
    """Rear-body world rotation from the front rotation and the three spine joints.

    Walks the model/cat.xml joint chain, front body -> rear body (paper Eq. 2):
        R_rear = R_front @ Rx(rot1) @ Ry(pitch) @ Rx(rot2)

    The tail joint is a child of the rear body, so it does not enter this product.
    Encoder angles are joint-side radians (the Teensy divides out the 9.68:1 roll /
    34.014:1 pitch gearing before it reports).

    Args:
        front: scipy Rotation (single or stack), front body in the world frame.
        rot1:  spine roll,  F_M1 (rad)
        pitch: spine pitch, F_M2 (rad)
        rot2:  rear roll,   B_M2 (rad)
    """
    return (front
            * R.from_euler("x", np.asarray(rot1, dtype=float))
            * R.from_euler("y", np.asarray(pitch, dtype=float))
            * R.from_euler("x", np.asarray(rot2, dtype=float)))


def compute_rear_body_quat(front_quat_wxyz, rot1, pitch, rot2):
    """rear_body_rotation on scalar-first quaternions (F_Q0..F_Q3 in, rear quat out)."""
    front = R.from_quat(np.asarray(front_quat_wxyz, dtype=float), scalar_first=True)
    return rear_body_rotation(front, rot1, pitch, rot2).as_quat(scalar_first=True)


def fall_time(height_m):
    """Free-fall time (s) to fall `height_m` from rest."""
    return float(np.sqrt(2.0 * float(height_m) / G))


def fall_distance(t):
    """Free-fall distance (m) at time t (s) after release."""
    return 0.5 * G * np.asarray(t, dtype=float) ** 2


def load_drop(path, steps=None):
    """One trial CSV -> the first `steps` control steps (all if None), from release.

    t[0] is free-fall detection (controller.py arms on |acc| < 3.5 m/s^2), so t is
    time since release, up to one 50 Hz tick of detection latency. Rows are kept as
    logged: nothing is resampled, so row k of every trial is the k-th policy step.

    Returns a dict with t (s), the front, rear and mean |roll|+|pitch| tilt (deg),
    and the logged rows themselves under "frame".
    """
    frame = pd.read_csv(path).iloc[:steps]
    t = frame["Time"].to_numpy(dtype=float)
    front_q = frame[QUAT_COLUMNS].to_numpy(dtype=float)
    rear_q = compute_rear_body_quat(front_q, frame["F_M1"], frame["F_M2"], frame["B_M2"])
    tilt_front = compute_tilt_angle(front_q)
    tilt_rear = compute_tilt_angle(rear_q)
    return {
        "t": t - t[0],
        "tilt_front": tilt_front,
        "tilt_rear": tilt_rear,
        # Fig. 4's quantity: the two bodies averaged per step, before any averaging
        # across trials.
        "tilt_mean": 0.5 * (tilt_front + tilt_rear),
        "frame": frame,
    }
