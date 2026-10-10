"""Whole-cat score per trial and the tail/no-tail statistics behind Table II.

Each trial is scored at a fixed ballistic fall distance (2.5 m by default). The
whole-cat score is the product of two 0-1 scores: the uprightness of the spherical
midpoint of the front and rear up-vectors, and the alignment of the full front/rear
relative attitude. It ignores common yaw and reaches 1 only when the cat is both
upright and aligned.

Table II is printed. With --output, these files are also written to that folder:
  trial_metrics.csv         one row per included trial: score, components, covariates
  excluded_trials.csv       trials whose impact came before the evaluation point
  model_results.csv         all-trials ANCOVA for the score and each diagnostic
  sweep_results.csv         roll-sweep and pitch-sweep ANCOVAs
  pitch_level_analysis.csv  pitch-leveling ANCOVA (third member of the Holm family)
  table2.csv, table2.md     paper Table II
  REPORT.md                 the tables above with the metric and model definitions

Usage, from the repository root:
    python -m results.plot_scripts.whole_cat_stats
    python -m results.plot_scripts.whole_cat_stats --distance 2.8 --output whole_cat_2p8m
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, signal, stats
from scipy.spatial.transform import Rotation, Slerp

from results.plot_scripts.kinematics import (
    CONDITION_ORDER, G, PITCH_SWEEP, ROLL_SWEEP, find_trials, parse_trial_name,
    rear_body_rotation,
)

# Command-line defaults
DEFAULT_INPUT = Path("results/hardware_trials")
DEFAULT_DISTANCE_M = 2.5
DEFAULT_PERMUTATIONS = 5000
DEFAULT_SEED = 20260819

# Log columns every trial must have; rows missing any of them are dropped
REQUIRED_COLUMNS = {
    "Time", "F_Q0", "F_Q1", "F_Q2", "F_Q3", "F_M1", "F_M2", "B_M1", "B_M2",
    "F_ACC", "B_ACC", "Cmd_F1", "Cmd_F2", "Cmd_B1", "Cmd_B2"
}

# Outcomes of the all-trials ANCOVA and their report labels
OUTCOMES = {
    "posture_score": "coupled whole-cat pose score (0-1)",
    "whole_body_upright_score": "whole-body centerline uprightness (0-1)",
    "spine_alignment_score": "front/rear attitude alignment (0-1)",
    "front_upright_score": "front-body uprightness score (0-1)",
    "back_upright_score": "rear-body uprightness score (0-1)",
    "whole_body_tilt_deg": "whole-body centerline tilt (deg)",
    "spine_misalignment_deg": "front/rear relative attitude (deg)",
    "orientation_error_deg": "front-body 3D orientation error (deg)",
    "angular_speed_deg_s": "3D angular speed (deg/s)",
    "abs_roll_deg": "absolute roll error (deg)",
    "abs_pitch_deg": "absolute pitch error (deg)",
    "abs_roll_rate_deg_s": "absolute roll rate (deg/s)",
    "abs_pitch_rate_deg_s": "absolute pitch rate (deg/s)",
}

# ANCOVA design: initial-state covariates, and the column of the tail indicator
COVARIATES = ("initial_posture_score", "initial_angular_speed_deg_s")
TAIL_COLUMN = 1

# Release-condition sweeps analysed on their own
SWEEPS = {
    "no_pitch_roll_sweep": {
        "label": "No pitch; roll 45/90/180 deg",
        "conditions": ROLL_SWEEP,
    },
    "pitch_sweep_at_roll_180": {
        "label": "180-deg roll; pitch 0/15/30/45 deg",
        "conditions": PITCH_SWEEP,
    },
}

# Table II rows as (row label, sweep key or None for all trials), and the notes under it
TABLE2_ROWS = (
    ("All trials", None),
    ("Roll sweep", "no_pitch_roll_sweep"),
    ("Pitch sweep", "pitch_sweep_at_roll_180"),
)
TABLE2_NOTES = [
    "Raw scores: mean ± sample SD (ddof=1), number of trials in parentheses. The "
    "adjusted tail effect (with tail minus no tail) is the OLS coefficient after "
    "controlling for release condition, collection date, initial whole-cat score and "
    "initial angular speed; CI and p use HC3 standard errors with a t reference.",
    "Holm-adjusted p: Holm over three tests -- the roll sweep, the pitch sweep and the "
    "pitch-leveling test in pitch_level_analysis.csv (`p_holm_three_requested` in "
    "sweep_results.csv). Holm over the two sweeps alone is `p_holm_two_sweeps`.",
    "Blocked permutation p: tail labels shuffled within condition x date blocks; only "
    "blocks containing both morphologies are used (n_permutation trials).",
]


@dataclass
class OLSResult:
    """Coefficients, HC3 covariance and residual statistics of an OLS fit."""

    beta: np.ndarray
    covariance_hc3: np.ndarray
    df_resid: int
    sigma: float
    xtx_inverse: np.ndarray


def angular_distance(angle_deg: np.ndarray | float) -> np.ndarray | float:
    """Return the absolute angle (deg) after wrapping it to [-180, 180)."""
    return np.abs((np.asarray(angle_deg) + 180.0) % 360.0 - 180.0)


def body_up(rotation: Rotation) -> np.ndarray:
    """Return the body +z axis in world coordinates."""
    return np.asarray(rotation.apply([0.0, 0.0, 1.0]), dtype=float)


def tilt_score(rotation: Rotation) -> float:
    """Return the yaw-invariant uprightness, linear in tilt: 1 upright, 0 inverted."""
    tilt = math.acos(float(np.clip(body_up(rotation)[2], -1.0, 1.0)))
    return float(np.clip(1.0 - tilt / math.pi, 0.0, 1.0))


def whole_body_roll_pitch(
    front: Rotation, rear: Rotation, center_up: np.ndarray
) -> tuple[float, float]:
    """Return the fused roll and pitch (rad) of the whole-cat frame with up-axis `center_up`.

    The forward axis is the mean front/rear +x made perpendicular to `center_up`,
    falling back to the front +x and then to a world axis when it vanishes. Fused
    pitch = asin(-R31) and roll = asin(R32) depend only on tilt, not on world yaw.
    """
    # Project the mean longitudinal axis perpendicular to the up-axis
    forward = front.apply([1.0, 0.0, 0.0]) + rear.apply([1.0, 0.0, 0.0])
    forward -= np.dot(forward, center_up) * center_up

    # Fall back to the front axis, then to a world axis, when the projection vanishes
    if np.linalg.norm(forward) < 1e-8:
        forward = front.apply([1.0, 0.0, 0.0])
        forward -= np.dot(forward, center_up) * center_up
    if np.linalg.norm(forward) < 1e-8:
        reference = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(reference, center_up)) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        forward = reference - np.dot(reference, center_up) * center_up

    # Complete the orthonormal frame and read roll and pitch from its bottom row
    center_forward = forward / np.linalg.norm(forward)
    center_side = np.cross(center_up, center_forward)
    center_rotation = np.column_stack((center_forward, center_side, center_up))
    pitch = math.asin(float(np.clip(-center_rotation[2, 0], -1.0, 1.0)))
    roll = math.asin(float(np.clip(center_rotation[2, 1], -1.0, 1.0)))
    return roll, pitch


def coupled_pose_metrics(front: Rotation, rear: Rotation) -> dict[str, float]:
    """Return the whole-cat score, its two components and the whole-body angles.

    The whole-body up-axis is the normalized sum of the front and rear up-vectors,
    their geodesic midpoint on S2. Spine alignment uses the full relative rotation,
    so it also sees twist about the up-axis.
    """
    # Tilt, roll and pitch of the whole-body centerline
    up_sum = body_up(front) + body_up(rear)
    if np.linalg.norm(up_sum) < 1e-8:
        # Antipodal up-vectors have no midpoint: count the whole body as inverted
        body_tilt = math.pi
        center_roll = math.nan
        center_pitch = math.nan
    else:
        center_up = up_sum / np.linalg.norm(up_sum)
        body_tilt = math.acos(float(np.clip(center_up[2], -1.0, 1.0)))
        center_roll, center_pitch = whole_body_roll_pitch(front, rear, center_up)

    # Angle of the rear attitude relative to the front
    relative_angle = float((front.inv() * rear).magnitude())

    # Score both components linearly from 1 (upright, aligned) to 0 and multiply them
    upright = float(np.clip(1.0 - body_tilt / math.pi, 0.0, 1.0))
    aligned = float(np.clip(1.0 - relative_angle / math.pi, 0.0, 1.0))
    return {
        "posture_score": upright * aligned,
        "whole_body_upright_score": upright,
        "spine_alignment_score": aligned,
        "whole_body_tilt_deg": math.degrees(body_tilt),
        "whole_body_roll_deg": math.degrees(center_roll),
        "whole_body_pitch_deg": math.degrees(center_pitch),
        "spine_misalignment_deg": math.degrees(relative_angle),
    }


def low_pass(values: np.ndarray, time_mid: np.ndarray, cutoff_hz: float = 15.0) -> np.ndarray:
    """Apply a zero-phase 4th-order Butterworth low-pass along axis 0.

    `values` is returned unchanged when it has fewer than 16 samples or when its
    sample rate is too low for the cutoff.
    """
    # Skip filtering when there are too few samples
    if len(values) < 16:
        return values

    # Skip filtering when the sample rate is too low for the cutoff
    sample_rate_hz = 1.0 / np.median(np.diff(time_mid))
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 2.2 * cutoff_hz:
        return values

    # Filter forward and backward
    sos = signal.butter(4, cutoff_hz, btype="low", fs=sample_rate_hz, output="sos")
    return signal.sosfiltfilt(sos, values, axis=0)


def display_path(path: Path) -> str:
    """Return the path relative to the working directory, the repository root."""
    return Path(os.path.relpath(path)).as_posix()


def load_trial_log(path: Path) -> pd.DataFrame:
    """Read a trial CSV, check its columns and drop rows with a missing value."""
    # Check that every required column is present
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    # Drop incomplete rows and check the remaining time stamps
    frame = frame.dropna(subset=sorted(REQUIRED_COLUMNS)).reset_index(drop=True)
    time = frame["Time"].to_numpy(float)
    if len(time) < 10 or np.any(np.diff(time) <= 0):
        raise ValueError(f"{path}: insufficient or non-monotonic time samples")
    return frame


def front_rotations(frame: pd.DataFrame, path: Path) -> Rotation:
    """Return the front-body world rotations from the F_Q0..F_Q3 quaternions (w, x, y, z)."""
    q_scalar_first = frame[["F_Q0", "F_Q1", "F_Q2", "F_Q3"]].to_numpy(float)
    q_norm = np.linalg.norm(q_scalar_first, axis=1)
    if np.any(q_norm < 0.5):
        raise ValueError(f"{path}: invalid near-zero quaternion")
    q_scalar_first = q_scalar_first / q_norm[:, None]
    return Rotation.from_quat(q_scalar_first[:, [1, 2, 3, 0]])


def front_angular_velocity(
    time: np.ndarray, rotations: Rotation
) -> tuple[np.ndarray, np.ndarray]:
    """Return the 15 Hz low-passed angular velocity (deg/s) between consecutive samples.

    Returns:
        The (N-1, 3) angular velocity and the (N-1,) midpoint times it belongs to.
    """
    delta = rotations[:-1].inv() * rotations[1:]
    angular_velocity = np.degrees(delta.as_rotvec()) / np.diff(time)[:, None]
    midpoint_time = (time[:-1] + time[1:]) / 2.0
    return low_pass(angular_velocity, midpoint_time), midpoint_time


def extract_trial(path: Path, target_distance_m: float) -> dict[str, object]:
    """Return one trial's metadata, scores, diagnostics and initial-state covariates.

    The trial is scored at t0 + sqrt(2 d / g), the instant it has fallen
    `target_distance_m`. A drop whose impact comes earlier raises a RuntimeError
    starting with "short_drop", which main records as an exclusion.
    """
    # Read the log
    meta = parse_trial_name(path)
    frame = load_trial_log(path)
    time = frame["Time"].to_numpy(float)

    # Locate the impact at the acceleration peak (B_ACC is 0 in every log, so the front IMU)
    accel = np.maximum(frame["F_ACC"].to_numpy(float), frame["B_ACC"].to_numpy(float))
    impact_idx = int(np.nanargmax(accel))
    impact_time = time[impact_idx]
    duration = impact_time - time[0]
    observed_fall_distance = 0.5 * G * duration**2

    # Exclude the drop when the impact comes before the evaluation instant
    target_time = time[0] + math.sqrt(2.0 * target_distance_m / G)
    if impact_idx < 2 or impact_time < target_time:
        raise RuntimeError(
            f"short_drop: estimated fall distance {observed_fall_distance:.3f} m "
            f"is below target {target_distance_m:.3f} m"
        )

    # Front attitude from the IMU, rear attitude from the spine encoders (paper Eq. 2)
    rotations = front_rotations(frame, path)
    rear_rotations = rear_body_rotation(rotations, frame["F_M1"], frame["F_M2"], frame["B_M2"])

    # Keep the samples up to and including the impact
    fall_time = time[: impact_idx + 1]
    fall_rotations = rotations[: impact_idx + 1]
    fall_rear_rotations = rear_rotations[: impact_idx + 1]

    # Attitudes at the start and, interpolated, at the evaluation instant
    target_rotation = Slerp(fall_time, fall_rotations)([target_time])[0]
    target_rear_rotation = Slerp(fall_time, fall_rear_rotations)([target_time])[0]
    initial_rotation = rotations[0]
    initial_rear_rotation = rear_rotations[0]
    target_euler = target_rotation.as_euler("xyz", degrees=True)
    initial_euler = initial_rotation.as_euler("xyz", degrees=True)

    # Front-body angular velocity at the evaluation instant and at the start (median of 5)
    angular_velocity, midpoint_time = front_angular_velocity(fall_time, fall_rotations)
    target_velocity = np.array(
        [np.interp(target_time, midpoint_time, angular_velocity[:, axis]) for axis in range(3)]
    )
    initial_velocity = np.median(angular_velocity[:5], axis=0)

    # Whole-cat scores at the evaluation instant and at the start
    pose = coupled_pose_metrics(target_rotation, target_rear_rotation)
    initial_pose = coupled_pose_metrics(initial_rotation, initial_rear_rotation)
    return {
        **meta,
        "source_file": display_path(path),
        "n_samples": len(frame),
        "impact_time_s": duration,
        "estimated_fall_distance_m": observed_fall_distance,
        **pose,
        "abs_whole_body_pitch_deg": abs(pose["whole_body_pitch_deg"]),
        "front_upright_score": tilt_score(target_rotation),
        "back_upright_score": tilt_score(target_rear_rotation),
        "orientation_error_deg": np.degrees(target_rotation.magnitude()),
        "angular_speed_deg_s": np.linalg.norm(target_velocity),
        "abs_roll_deg": float(angular_distance(target_euler[0])),
        "abs_pitch_deg": float(angular_distance(target_euler[1])),
        "abs_roll_rate_deg_s": abs(target_velocity[0]),
        "abs_pitch_rate_deg_s": abs(target_velocity[1]),
        "initial_orientation_error_deg": np.degrees(initial_rotation.magnitude()),
        "initial_posture_score": initial_pose["posture_score"],
        "initial_whole_body_upright_score": initial_pose["whole_body_upright_score"],
        "initial_spine_alignment_score": initial_pose["spine_alignment_score"],
        "initial_whole_body_pitch_deg": initial_pose["whole_body_pitch_deg"],
        "initial_abs_whole_body_pitch_deg": abs(initial_pose["whole_body_pitch_deg"]),
        "initial_front_upright_score": tilt_score(initial_rotation),
        "initial_back_upright_score": tilt_score(initial_rear_rotation),
        "initial_angular_speed_deg_s": np.linalg.norm(initial_velocity),
        "initial_roll_deg": initial_euler[0],
        "initial_pitch_deg": initial_euler[1],
        "initial_roll_rate_deg_s": initial_velocity[0],
        "initial_pitch_rate_deg_s": initial_velocity[1],
    }


def build_design(
    frame: pd.DataFrame,
    include_date: bool = True,
    covariates: tuple[str, ...] = COVARIATES,
) -> np.ndarray:
    """Return the ANCOVA design matrix for the trials in `frame`.

    Columns: intercept, tail indicator (TAIL_COLUMN), a dummy for each release
    condition and each collection date after the first, and the z-scored covariates
    (all zero when a covariate is constant).
    """
    # Intercept and tail indicator
    columns = [np.ones(len(frame)), frame["tail"].to_numpy(float)]

    # Release-condition dummies, the first condition present in CONDITION_ORDER as baseline
    present = set(frame["condition"])
    conditions = [condition for condition in CONDITION_ORDER if condition in present]
    for condition in conditions[1:]:
        columns.append((frame["condition"] == condition).to_numpy(float))

    # Collection-date dummies, the earliest date as baseline
    if include_date:
        dates = frame["date"].astype(str)
        for date in sorted(dates.unique())[1:]:
            columns.append((dates == date).to_numpy(float))

    # Z-scored covariates
    for covariate in covariates:
        values = frame[covariate].to_numpy(float)
        scale = values.std(ddof=1)
        columns.append((values - values.mean()) / scale if scale > 0 else np.zeros(len(values)))
    return np.column_stack(columns)


def fit_ols(y: np.ndarray, x: np.ndarray) -> OLSResult:
    """Fit y on the design x by OLS with HC3 heteroskedasticity-consistent covariance."""
    # Some BLAS builds raise spurious floating-point flags in matmul on these small matrices
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        # Least-squares coefficients and residual degrees of freedom
        xtx_inverse = np.linalg.pinv(x.T @ x)
        beta = xtx_inverse @ x.T @ y
        residuals = y - x @ beta
        df_resid = len(y) - np.linalg.matrix_rank(x)

        # HC3 sandwich covariance
        leverage = np.einsum("ij,jk,ik->i", x, xtx_inverse, x)
        adjusted_squared = (residuals / np.maximum(1.0 - leverage, 1e-8)) ** 2
        meat = x.T @ (x * adjusted_squared[:, None])
        covariance_hc3 = xtx_inverse @ meat @ xtx_inverse

    # Reject non-finite coefficients and compute the residual SD
    if not np.all(np.isfinite(beta)):
        raise FloatingPointError("non-finite OLS coefficients")
    sigma = math.sqrt(float(residuals @ residuals) / df_resid)
    return OLSResult(beta, covariance_hc3, df_resid, sigma, xtx_inverse)


def coefficient_row(result: OLSResult, index: int = TAIL_COLUMN) -> dict[str, float]:
    """Return the estimate, HC3 SE, 95% CI and two-sided p (t reference) of one coefficient."""
    estimate = result.beta[index]
    se = math.sqrt(max(result.covariance_hc3[index, index], 0.0))
    critical = stats.t.ppf(0.975, result.df_resid)
    statistic = estimate / se
    return {
        "estimate": estimate,
        "se_hc3": se,
        "ci_low": estimate - critical * se,
        "ci_high": estimate + critical * se,
        "t_hc3": statistic,
        "p_hc3": 2.0 * stats.t.sf(abs(statistic), result.df_resid),
        "residual_sd": result.sigma,
        "df_resid": result.df_resid,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return the Holm step-down adjusted p-values, in the input order."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running_max, 1.0)
    return adjusted.tolist()


def overlap_permutation(
    frame: pd.DataFrame,
    outcome: str,
    n_permutations: int,
    seed: int,
    covariates: tuple[str, ...] = COVARIATES,
) -> tuple[float, float, int]:
    """Run the blocked permutation test of the tail coefficient.

    Tail labels are shuffled within condition x date blocks. Only blocks that
    contain both morphologies can be permuted, so the test, including its observed
    estimate, uses those trials alone.

    Returns:
        The observed tail coefficient, the permutation p-value and the number of
        trials used.
    """
    # Keep the trials in blocks that contain both morphologies
    strata = frame.groupby(["condition", "date"])["tail"].agg(["min", "max"])
    overlap = set(strata[(strata["min"] == 0) & (strata["max"] == 1)].index)
    mask = [(condition, date) in overlap
            for condition, date in zip(frame["condition"], frame["date"])]
    subset = frame.loc[mask].copy().reset_index(drop=True)

    # Observed tail coefficient
    y = subset[outcome].to_numpy(float)
    observed = fit_ols(y, build_design(subset, covariates=covariates)).beta[TAIL_COLUMN]

    # Null distribution: refit with the tail labels shuffled within each block
    rng = np.random.default_rng(seed)
    blocks = [group.index.to_numpy() for _, group in subset.groupby(["condition", "date"])]
    original_tail = subset["tail"].to_numpy(int)
    permuted_frame = subset.copy()
    null = np.empty(n_permutations)
    for iteration in range(n_permutations):
        permuted = original_tail.copy()
        for indices in blocks:
            permuted[indices] = rng.permutation(permuted[indices])
        permuted_frame["tail"] = permuted
        x_permuted = build_design(permuted_frame, covariates=covariates)
        null[iteration] = fit_ols(y, x_permuted).beta[TAIL_COLUMN]

    # Two-sided p-value, counting the observed labelling as one permutation
    p_value = (1.0 + np.sum(np.abs(null) >= abs(observed))) / (n_permutations + 1.0)
    return observed, float(p_value), len(subset)


def regression_power(
    effect: float, sigma: float, coefficient_variance_unit: float, df: int, alpha: float
) -> float:
    """Return the power of the two-sided t test of one coefficient against a true `effect`.

    The coefficient's standard error is sigma * sqrt(coefficient_variance_unit), where
    `coefficient_variance_unit` is its diagonal entry of (X'X)^-1.
    """
    ncp = effect / (sigma * math.sqrt(coefficient_variance_unit))
    critical = stats.t.ppf(1.0 - alpha / 2.0, df)
    return float(stats.nct.cdf(-critical, df, ncp) + stats.nct.sf(critical, df, ncp))


def regression_mde(
    sigma: float, coefficient_variance_unit: float, df: int, alpha: float, target: float = 0.8
) -> float:
    """Return the smallest effect the two-sided t test detects with `target` power."""
    def power_shortfall(effect: float) -> float:
        """Return the power at `effect` minus the target power."""
        return regression_power(effect, sigma, coefficient_variance_unit, df, alpha) - target

    # Widen the bracket until the power reaches the target
    upper = sigma
    while power_shortfall(upper) < 0:
        upper *= 1.5
        if upper > 10.0 * sigma:
            raise RuntimeError("Could not bracket the regression MDE")

    # Solve for the effect with exactly the target power
    return float(optimize.brentq(power_shortfall, 1e-9, upper))


def tail_contrast(
    frame: pd.DataFrame,
    outcome: str,
    n_permutations: int,
    seed: int,
    covariates: tuple[str, ...] = COVARIATES,
) -> tuple[OLSResult, dict[str, object]]:
    """Fit the ANCOVA of `outcome` and return it with the tail-effect statistics.

    The statistics are the HC3 coefficient row, the tail effect refit without the
    date dummies, and the blocked permutation test.
    """
    # ANCOVA with all terms
    y = frame[outcome].to_numpy(float)
    model = fit_ols(y, build_design(frame, covariates=covariates))
    adjusted = coefficient_row(model)

    # Same model without the date dummies
    x_no_date = build_design(frame, include_date=False, covariates=covariates)
    no_date = coefficient_row(fit_ols(y, x_no_date))

    # Blocked permutation test
    overlap_estimate, p_permutation, n_overlap = overlap_permutation(
        frame, outcome, n_permutations, seed, covariates=covariates
    )
    return model, {
        **adjusted,
        "estimate_no_date": no_date["estimate"],
        "p_no_date": no_date["p_hc3"],
        "overlap_estimate": overlap_estimate,
        "p_permutation": p_permutation,
        "n_overlap": n_overlap,
    }


def analyze_outcomes(frame: pd.DataFrame, n_permutations: int, seed: int) -> pd.DataFrame:
    """Return the all-trials ANCOVA of the tail effect, one row per outcome in OUTCOMES."""
    # Tail effect on each outcome
    rows: list[dict[str, object]] = []
    for outcome, label in OUTCOMES.items():
        _, contrast = tail_contrast(frame, outcome, n_permutations, seed)
        rows.append({"outcome": outcome, "label": label, **contrast})

    # Holm adjustment over the outcomes
    results = pd.DataFrame(rows)
    results["p_holm_all_outcomes"] = holm_adjust(results["p_hc3"].astype(float).tolist())
    return results


def analyze_predefined_sweeps(
    frame: pd.DataFrame,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    """Return the pooled tail contrast of the whole-cat score in each sweep of SWEEPS."""
    # Tail effect and its 80%-power MDE per sweep; the pitch sweep is the first row
    rows: list[dict[str, object]] = []
    for sweep in ("pitch_sweep_at_roll_180", "no_pitch_roll_sweep"):
        definition = SWEEPS[sweep]
        in_sweep = frame["condition"].isin(definition["conditions"])
        subset = frame.loc[in_sweep].copy().reset_index(drop=True)
        model, contrast = tail_contrast(subset, "posture_score", n_permutations, seed)
        mde = regression_mde(
            model.sigma, float(model.xtx_inverse[TAIL_COLUMN, TAIL_COLUMN]), model.df_resid,
            alpha=0.05,
        )
        rows.append(
            {
                "sweep": sweep,
                "label": definition["label"],
                "conditions": ",".join(definition["conditions"]),
                "n": len(subset),
                **contrast,
                "mde_80": mde,
            }
        )

    # Holm adjustment over the two sweeps
    result = pd.DataFrame(rows)
    result["p_holm_two_sweeps"] = holm_adjust(result["p_hc3"].tolist())
    result["p_permutation_holm_two_sweeps"] = holm_adjust(result["p_permutation"].tolist())
    return result


def analyze_pitch_leveling(
    frame: pd.DataFrame,
    n_permutations: int,
    seed: int,
) -> dict[str, object]:
    """Return the tail contrast of the final absolute whole-cat pitch in the pitch sweep."""
    subset = frame.loc[frame["condition"].isin(PITCH_SWEEP)].copy().reset_index(drop=True)
    covariates = ("initial_abs_whole_body_pitch_deg", "initial_angular_speed_deg_s")
    _, contrast = tail_contrast(
        subset, "abs_whole_body_pitch_deg", n_permutations, seed, covariates=covariates
    )
    return {
        "analysis": "pitch_leveling_at_roll_180",
        "label": "Final absolute whole-cat pitch; roll 180, pitch 0/15/30/45 deg",
        "conditions": ",".join(PITCH_SWEEP),
        "n": len(subset),
        **contrast,
    }


def build_table2(
    frame: pd.DataFrame, results: pd.DataFrame, sweep_results: pd.DataFrame
) -> pd.DataFrame:
    """Return paper Table II: raw whole-cat score per arm plus the adjusted tail effect.

    Raw SDs are sample SDs (ddof=1). The Holm column is p_holm_three_requested: the
    family is {roll sweep, pitch sweep, pitch leveling}, not the two sweeps alone.
    The all-trials row is the primary analysis and has no Holm value.
    """
    # Build one row per Table II analysis
    rows = []
    for label, sweep in TABLE2_ROWS:
        # Trials and fitted model behind this row
        if sweep is None:
            conditions = CONDITION_ORDER
            subset = frame
            model = results.loc[results["outcome"] == "posture_score"].iloc[0]
            p_holm = math.nan
        else:
            conditions = SWEEPS[sweep]["conditions"]
            subset = frame.loc[frame["condition"].isin(conditions)]
            model = sweep_results.loc[sweep_results["sweep"] == sweep].iloc[0]
            p_holm = float(model["p_holm_three_requested"])
        row = {"analysis": label, "conditions": ",".join(conditions)}

        # Raw score per morphology
        for morphology in ("no_tail", "with_tail"):
            scores = subset.loc[subset["morphology"] == morphology, "posture_score"]
            row[f"n_{morphology}"] = len(scores)
            row[f"mean_{morphology}"] = scores.mean()
            row[f"sd_{morphology}"] = scores.std(ddof=1)

        # Adjusted tail effect
        row.update({
            "tail_effect": float(model["estimate"]),
            "ci_low": float(model["ci_low"]),
            "ci_high": float(model["ci_high"]),
            "p_hc3": float(model["p_hc3"]),
            "p_holm": p_holm,
            "p_permutation": float(model["p_permutation"]),
            "n_permutation": int(model["n_overlap"]),
            "df_resid": int(model["df_resid"]),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def format_table2(table: pd.DataFrame) -> list[str]:
    """Return Table II as Markdown lines, at the precision the paper prints."""
    lines = [
        "| Analysis | No tail mean ± SD (n) | With tail mean ± SD (n) | Adjusted tail effect "
        "| 95% HC3 CI | HC3 p | Holm-adjusted p | Blocked permutation p |",
        "|---|---|---|---:|---|---:|---:|---:|",
    ]
    for row in table.itertuples():
        holm = "—" if math.isnan(row.p_holm) else f"{row.p_holm:.4f}"
        lines.append(
            f"| {row.analysis} "
            f"| {row.mean_no_tail:.3f} ± {row.sd_no_tail:.3f} ({row.n_no_tail}) "
            f"| {row.mean_with_tail:.3f} ± {row.sd_with_tail:.3f} ({row.n_with_tail}) "
            f"| {row.tail_effect:+.3f} | [{row.ci_low:.3f}, {row.ci_high:.3f}] "
            f"| {row.p_hc3:.4f} | {holm} | {row.p_permutation:.4f} |"
        )
    return lines


def write_table2(table: pd.DataFrame, output_dir: Path) -> None:
    """Write Table II to table2.csv and, with its notes, to table2.md."""
    table.to_csv(output_dir / "table2.csv", index=False)
    lines = ["# Table II: whole-cat self-righting scores", "", *format_table2(table), ""]
    lines += [f"- {note}" for note in TABLE2_NOTES]
    (output_dir / "table2.md").write_text("\n".join(lines) + "\n")


def write_report(
    frame: pd.DataFrame,
    excluded: list[dict[str, str]],
    results: pd.DataFrame,
    sweep_results: pd.DataFrame,
    pitch_level: dict[str, object],
    table2: pd.DataFrame,
    output_dir: Path,
    target_distance_m: float,
    n_permutations: int,
    seed: int,
) -> None:
    """Write REPORT.md: every table, the metric definition and the limitations."""
    # Header with the trial counts and the analysis settings
    target_time = math.sqrt(2.0 * target_distance_m / G)
    lines = [
        "# Whole-cat score analysis",
        "",
        f"Generated by `results/plot_scripts/whole_cat_stats.py`: {len(frame)} trials included, "
        f"{len(excluded)} excluded (impact before {target_distance_m:g} m of fall). Each trial "
        f"is scored at t0 + {target_time:.3f} s, the instant of {target_distance_m:g} m of "
        f"ballistic fall after free-fall detection. Permutation tests: {n_permutations} "
        f"permutations, seed {seed}.",
        "",
    ]

    # Table II and its notes
    lines += [
        "## Table II",
        "",
        *format_table2(table2),
        "",
        *[f"- {note}" for note in TABLE2_NOTES],
        "",
    ]

    # All-trials ANCOVA of every outcome
    lines += [
        "## All trials: score, components and diagnostics",
        "",
        "Adjusted tail effect (with tail minus no tail) from the all-trials ANCOVA. "
        "Positive score effects and negative angle/rate effects favor the tail.",
        "",
        "| Outcome | Effect | 95% HC3 CI | HC3 p | Holm p (13 outcomes) | Blocked permutation p |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            f"| {row.label} | {row.estimate:+.3f} | {row.ci_low:+.3f} to {row.ci_high:+.3f} "
            f"| {row.p_hc3:.4f} | {row.p_holm_all_outcomes:.4f} | {row.p_permutation:.4f} |"
            for row in results.itertuples()
        ],
        "",
    ]

    # Sweep ANCOVAs
    lines += [
        "## Sweeps",
        "",
        "| Sweep | n | Effect | 95% HC3 CI | HC3 p | Holm p (3 tests) | Holm p (2 sweeps) "
        "| Permutation p (n) | 80% MDE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {row.label} | {row.n} | {row.estimate:+.3f} | {row.ci_low:+.3f} to "
            f"{row.ci_high:+.3f} | {row.p_hc3:.4f} | {row.p_holm_three_requested:.4f} "
            f"| {row.p_holm_two_sweeps:.4f} | {row.p_permutation:.4f} ({row.n_overlap}) "
            f"| {row.mde_80:.3f} |"
            for row in sweep_results.itertuples()
        ],
        "",
    ]

    # Pitch-leveling ANCOVA
    lines += [
        "## Pitch leveling within the pitch sweep",
        "",
        "Final absolute whole-cat fused pitch (0 deg is level, so a negative effect favors "
        "the tail), adjusted for condition, date, initial absolute whole-cat pitch and initial "
        "angular speed. This is the third member of the Holm family used in Table II.",
        "",
        "| Outcome | n | Tail effect | 95% HC3 CI | HC3 p | Holm p (3 tests) "
        "| Permutation p (n) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Absolute final whole-cat pitch | {pitch_level['n']} "
        f"| {pitch_level['estimate']:+.2f} deg "
        f"| {pitch_level['ci_low']:+.2f} to {pitch_level['ci_high']:+.2f} deg "
        f"| {pitch_level['p_hc3']:.4f} | {pitch_level['p_holm_three_requested']:.4f} "
        f"| {pitch_level['p_permutation']:.4f} ({pitch_level['n_overlap']}) |",
        "",
    ]

    # Metric definition and limitations
    lines += [
        "## Metric",
        "",
        "```text",
        "center_up       = normalize(front_up + rear_up)",
        "U               = 1 - whole_body_tilt / pi",
        "A               = 1 - relative_front_rear_angle / pi",
        "whole_cat_score = U * A",
        "```",
        "",
        "The score ranges from 0 to 1. It ignores common yaw, penalizes relative spine twist, "
        "and reaches 1 only when the whole-body centerline is upright and the front/rear frames "
        "are aligned. The rear body is reconstructed from the front IMU and the spine encoders.",
        "",
        "## Limitations",
        "",
        "- Tail/no-tail assignment was not randomized or interleaved within collection session; "
        "date and morphology are partly confounded.",
        "- The metric was developed after viewing these data. Treat the p-values as exploratory.",
        "- Per-trial provenance (which policy flew each drop, file date labels) is in the "
        "'Data notes' of the top-level README.md and in results/hardware_trials/MANIFEST.csv. "
        "Date is a model term and defines the permutation blocks.",
    ]

    # Excluded trials, if any
    if excluded:
        lines += ["", "## Excluded trials", ""]
        lines += [f"- `{row['source_file']}`: {row['reason']}" for row in excluded]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def score_trials(
    input_dir: Path, distance_m: float = DEFAULT_DISTANCE_M
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Score every trial under `input_dir` at `distance_m` of fall.

    Returns:
        The per-trial metrics, and the drops set aside because they were too short to
        reach the evaluation distance.
    """
    # Score every trial, setting aside drops too short to reach the evaluation distance
    records: list[dict[str, object]] = []
    excluded: list[dict[str, str]] = []
    for path in find_trials(input_dir):
        try:
            records.append(extract_trial(path, distance_m))
        except RuntimeError as error:
            excluded.append({"source_file": display_path(path), "reason": str(error)})
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("No trials passed preprocessing")

    # Reject repeated trials
    duplicate_key = frame.duplicated(["morphology", "condition", "date", "rep"], keep=False)
    if duplicate_key.any():
        raise ValueError(
            f"Duplicate canonical trials:\n{frame.loc[duplicate_key].to_string(index=False)}"
        )
    return frame, excluded


def analyze(frame: pd.DataFrame, n_permutations: int, seed: int) -> dict[str, object]:
    """Run the all-trials, sweep and pitch-leveling analyses and build Table II."""
    # All-trials ANCOVA of every outcome
    results = analyze_outcomes(frame, n_permutations, seed)

    # Sweep and pitch-leveling analyses, Holm-adjusted together as one family of three
    sweep_results = analyze_predefined_sweeps(frame, n_permutations, seed)
    pitch_level = analyze_pitch_leveling(frame, n_permutations, seed)
    family_adjusted = holm_adjust(
        sweep_results["p_hc3"].astype(float).tolist() + [float(pitch_level["p_hc3"])]
    )
    sweep_results["p_holm_three_requested"] = family_adjusted[:2]
    pitch_level["p_holm_three_requested"] = family_adjusted[2]

    # Table II
    table2 = build_table2(frame, results, sweep_results)
    return {"results": results, "sweep_results": sweep_results, "pitch_level": pitch_level,
            "table2": table2}


def write_outputs(
    output_dir: Path,
    frame: pd.DataFrame,
    excluded: list[dict[str, str]],
    analysis: dict[str, object],
    distance_m: float,
    n_permutations: int,
    seed: int,
) -> None:
    """Write the per-trial metrics, the analysis tables, Table II and the report."""
    # Per-trial metrics and the exclusions
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "trial_metrics.csv", index=False)
    pd.DataFrame(excluded, columns=("source_file", "reason")).to_csv(
        output_dir / "excluded_trials.csv", index=False)

    # Analysis tables
    analysis["results"].to_csv(output_dir / "model_results.csv", index=False)
    analysis["sweep_results"].to_csv(output_dir / "sweep_results.csv", index=False)
    pd.DataFrame([analysis["pitch_level"]]).to_csv(
        output_dir / "pitch_level_analysis.csv", index=False)

    # Table II and the report
    write_table2(analysis["table2"], output_dir)
    write_report(frame, excluded, analysis["results"], analysis["sweep_results"],
                 analysis["pitch_level"], analysis["table2"], output_dir, distance_m,
                 n_permutations, seed)


def main(argv: list[str] | None = None) -> dict[str, object]:
    """Score every trial, run the tail/no-tail analyses and print Table II.

    Returns:
        The trial metrics ("frame"), the excluded trials, the all-trials, sweep and
        pitch-leveling results, and Table II.
    """
    # Parse the command line
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="folder of trial CSVs, searched recursively "
                             f"(default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=Path, default=None,
                        help="folder to also write the CSV/Markdown outputs to "
                             "(default: print Table II only)")
    parser.add_argument("--distance", type=float, default=DEFAULT_DISTANCE_M,
                        help="fall distance (m) at which each trial is scored "
                             "(default: %(default)s)")
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS,
                        help="blocked permutations per test (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="permutation RNG seed (default: %(default)s)")
    args = parser.parse_args(argv)

    # Score the trials and run the analyses
    frame, excluded = score_trials(args.input, args.distance)
    analysis = analyze(frame, args.permutations, args.seed)

    # Write the tables when an output folder is given
    if args.output is not None:
        write_outputs(args.output, frame, excluded, analysis, args.distance,
                      args.permutations, args.seed)

    # Print Table II and a summary
    print("\n".join(format_table2(analysis["table2"])))
    print(f"\nIncluded {len(frame)}; excluded {len(excluded)} "
          f"short {'drop' if len(excluded) == 1 else 'drops'}")
    if args.output is not None:
        print(f"Results written to {args.output}")
    return {"frame": frame, "excluded": excluded, **analysis}


if __name__ == "__main__":
    main()
