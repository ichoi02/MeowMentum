"""Load the hardware drop logs and compute body tilt, rear-body orientation and fall timing.

Each trial in results/hardware_trials/ is one CSV written by hardware/controller.py: one row
per 50 Hz control step, starting at the first step after free fall was detected (49 or 50
rows). The columns:
  Time                 s since the controller started, not since the drop
  F_Q0..F_Q3           front IMU quaternion [w, x, y, z], already world-aligned
  F_M1, F_M2           front encoders: spine roll rot1 and spine pitch (rad)
  B_M1, B_M2           rear encoders: tail and rear roll rot2 (rad)
  F_ACC                front accelerometer magnitude (m/s^2, gravity included)
  Cmd_F1, Cmd_F2       joint targets for rot1 and pitch (rad)
  Cmd_B1, Cmd_B2       joint targets for tail and rot2 = -rot1 (rad)
  B_Q0..B_Q3, B_ACC    rear IMU placeholders, (1, 0, 0, 0) and 0 in every file
The rear IMU never reported, so the rear-body orientation is reconstructed from the front
IMU and the spine encoders.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# Gravitational acceleration, in m/s^2
G = 9.81

# Front IMU quaternion columns, scalar-first [w, x, y, z]
QUAT_COLUMNS = ["F_Q0", "F_Q1", "F_Q2", "F_Q3"]

# Canonical trial file name: <nt|wt>_roll_<RRR>deg[_pitch_<+PPP>deg]_<MMDD>_rep<NN>.csv
TRIAL_NAME_RE = re.compile(
    r"^(?P<morph>wt|nt)_roll_(?P<roll>\d{3})deg"
    r"(?:_pitch_(?P<pitch>[+-]\d{3})deg)?_"
    r"(?P<date>\d{4})_rep(?P<rep>\d{2})\.csv$"
)

# Release conditions in the paper's order: the roll sweep at 0 deg pitch, then the pitch
# sweep at 180 deg roll (whose 0 deg point is r180)
ROLL_SWEEP = ("r45", "r90", "r180")
PITCH_SWEEP = ("r180", "r180_p15", "r180_p30", "r180_p45")
CONDITION_ORDER = ("r45", "r90", "r180", "r180_p15", "r180_p30", "r180_p45")
MORPHOLOGIES = ("with_tail", "no_tail")

# CSVs stored alongside the trials that are not trials
NON_TRIAL_FILES = {"MANIFEST.csv"}


def parse_trial_name(path):
    """Parse a canonical trial file name into morphology, release condition, date and rep.

    Raises:
        ValueError: if the file name is not canonical.
    """
    # Match the file name against the canonical pattern
    match = TRIAL_NAME_RE.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"Unrecognized canonical filename: {path}")
    fields = match.groupdict()

    # Name the release condition, for example "r180" or "r180_p15"
    roll = int(fields["roll"])
    pitch = int(fields["pitch"] or 0)
    condition = f"r{roll}" if fields["pitch"] is None else f"r{roll}_p{pitch}"

    return {
        "morphology": "with_tail" if fields["morph"] == "wt" else "no_tail",
        "tail": int(fields["morph"] == "wt"),
        "roll_deg": roll,
        "pitch_deg": pitch,
        "condition": condition,
        "date": fields["date"],
        "rep": int(fields["rep"]),
    }


def find_trials(root):
    """Return every trial CSV under `root`, sorted by path.

    The blocked permutation test draws its permutations in this order, so its p-values
    depend on it.
    """
    # Check that the folder exists
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")

    # Collect the trial CSVs, skipping the manifest
    paths = sorted(p for p in root.rglob("*.csv") if p.name not in NON_TRIAL_FILES)
    if not paths:
        raise FileNotFoundError(f"no trial CSVs found under {root}")

    # Reject any file whose name is not canonical
    for path in paths:
        parse_trial_name(path)

    return paths


def compute_tilt_angle(quat_wxyz):
    """Return the tilt |roll| + |pitch| of a body in the world frame, in deg (0 = upright).

    Roll and pitch come from the extrinsic "xyz" Euler decomposition, the convention the
    simulated drop orientations are built with, so a 180 deg roll, 15 deg pitch release
    starts at 195 deg. Yaw drops out. The tilt spans [0, 270] deg, and roll is ill-defined
    at pitch = +-90 deg.

    Args:
        quat_wxyz: (4,) or (N, 4) world-aligned body quaternion, scalar-first.

    Returns:
        A scalar, or an (N,) array.
    """
    q = np.asarray(quat_wxyz, dtype=float)
    roll_pitch = R.from_quat(q, scalar_first=True).as_euler("xyz")[..., :2]
    return np.degrees(np.abs(roll_pitch).sum(axis=-1))


def rear_body_rotation(front, rot1, pitch, rot2):
    """Return the rear-body world rotation from the front rotation and the spine joints.

    Walks the model/cat.xml joint chain from the front to the rear body (paper Eq. 2):
    R_rear = R_front * Rx(rot1) * Ry(pitch) * Rx(rot2). The tail joint is a child of the
    rear body and does not enter.

    Args:
        front: scipy Rotation (single or stack) of the front body in the world frame.
        rot1: spine roll, F_M1 (rad, joint side of the gearing).
        pitch: spine pitch, F_M2 (rad, joint side of the gearing).
        rot2: rear roll, B_M2 (rad, joint side of the gearing).
    """
    return (front
            * R.from_euler("x", np.asarray(rot1, dtype=float))
            * R.from_euler("y", np.asarray(pitch, dtype=float))
            * R.from_euler("x", np.asarray(rot2, dtype=float)))


def compute_rear_body_quat(front_quat_wxyz, rot1, pitch, rot2):
    """Return the rear-body quaternion [w, x, y, z] from the front quaternion and spine joints."""
    front = R.from_quat(np.asarray(front_quat_wxyz, dtype=float), scalar_first=True)
    return rear_body_rotation(front, rot1, pitch, rot2).as_quat(scalar_first=True)


def fall_time(height_m):
    """Return the time (s) to fall `height_m` metres from rest."""
    return float(np.sqrt(2.0 * float(height_m) / G))


def fall_distance(t):
    """Return the free-fall distance (m) at time `t` (s) after release."""
    return 0.5 * G * np.asarray(t, dtype=float) ** 2


def load_drop(path, steps=None):
    """Load the first `steps` control steps of one trial (all if None) and their body tilts.

    Rows are kept as logged, with nothing resampled, so row k of every trial is the k-th
    policy step. Time is measured from the first row, the free-fall detection, which lags
    release by at most one 50 Hz tick.

    Returns:
        A dict with "t" (s since release), "tilt_front", "tilt_rear" and "tilt_mean" (deg;
        the mean of the two bodies per step is Fig. 4's quantity), and the logged rows
        under "frame".
    """
    # Read the logged rows
    frame = pd.read_csv(path).iloc[:steps]
    t = frame["Time"].to_numpy(dtype=float)

    # Compute the tilt of the front body and of the reconstructed rear body
    front_q = frame[QUAT_COLUMNS].to_numpy(dtype=float)
    rear_q = compute_rear_body_quat(front_q, frame["F_M1"], frame["F_M2"], frame["B_M2"])
    tilt_front = compute_tilt_angle(front_q)
    tilt_rear = compute_tilt_angle(rear_q)

    return {
        "t": t - t[0],
        "tilt_front": tilt_front,
        "tilt_rear": tilt_rear,
        "tilt_mean": 0.5 * (tilt_front + tilt_rear),
        "frame": frame,
    }
