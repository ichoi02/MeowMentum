#!/usr/bin/env python3
"""Whole-cat score per trial and the tail/no-tail statistics behind Table II.

The primary endpoint is a coupled whole-cat pose score at a fixed 2.5 m ballistic
fall distance. It separates (1) tilt of the spherical midpoint of the front and
rear up-vectors from (2) the full relative attitude between the two bodies, then
multiplies their 0--1 scores. This is yaw-invariant for common whole-body yaw,
uses both body attitudes, and requires both righting and spine alignment for 1.

Outputs (default results/):
  trial_metrics.csv         one row per included trial: score, components, covariates
  excluded_trials.csv       trials whose impact came before the evaluation point
  model_results.csv         all-trials ANCOVA for the score and each diagnostic
  sweep_results.csv         roll-sweep and pitch-sweep ANCOVAs
  pitch_level_analysis.csv  pitch-leveling ANCOVA (third member of the Holm family)
  table2.csv, table2.md     paper Table II
  REPORT.md                 the tables above with the metric and model definitions

Usage:
    python analysis/whole_cat_stats.py
    python analysis/whole_cat_stats.py --distance 2.8 --output /tmp/whole_cat_2p8m
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, signal, stats
from scipy.spatial.transform import Rotation, Slerp

from kinematics import (
    CONDITION_ORDER, G, PITCH_SWEEP, ROLL_SWEEP, find_trials, parse_trial_name,
    rear_body_rotation,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "hardware_trials"
DEFAULT_OUTPUT = ROOT / "results"
DEFAULT_DISTANCE_M = 2.5
DEFAULT_PERMUTATIONS = 5000
DEFAULT_SEED = 20260819

REQUIRED_COLUMNS = {
    "Time", "F_Q0", "F_Q1", "F_Q2", "F_Q3", "F_M1", "F_M2", "B_M1", "B_M2",
    "F_ACC", "B_ACC", "Cmd_F1", "Cmd_F2", "Cmd_B1", "Cmd_B2"
}
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
COVARIATES = ("initial_posture_score", "initial_angular_speed_deg_s")
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
# Table II rows: (row label, sweep key or None for all trials).
TABLE2_ROWS = (
    ("All trials", None),
    ("Roll sweep", "no_pitch_roll_sweep"),
    ("Pitch sweep", "pitch_sweep_at_roll_180"),
)


@dataclass
class OLSResult:
    beta: np.ndarray
    covariance_hc3: np.ndarray
    residuals: np.ndarray
    fitted: np.ndarray
    df_resid: int
    sigma: float
    xtx_inverse: np.ndarray
    names: list[str]


def angular_distance(angle_deg: np.ndarray | float) -> np.ndarray | float:
    return np.abs((np.asarray(angle_deg) + 180.0) % 360.0 - 180.0)


def body_up(rotation: Rotation) -> np.ndarray:
    """Local +z expressed in world coordinates."""
    return np.asarray(rotation.apply([0.0, 0.0, 1.0]), dtype=float)


def tilt_score(rotation: Rotation) -> float:
    """Yaw-invariant uprightness, linear in tilt: 1 upright, 0 inverted."""
    tilt = math.acos(float(np.clip(body_up(rotation)[2], -1.0, 1.0)))
    return float(np.clip(1.0 - tilt / math.pi, 0.0, 1.0))


def coupled_pose_metrics(front: Rotation, rear: Rotation) -> dict[str, float]:
    """Whole-body tilt and spine alignment from coupled front/rear attitudes.

    The normalized sum is the equal-weight geodesic midpoint of the two up-vectors
    on S2 whenever they are not antipodal. The relative SO(3) rotation also sees
    twist about the up-axis, which an up-vector-only metric would miss.
    """
    up_front = body_up(front)
    up_rear = body_up(rear)
    up_sum = up_front + up_rear
    if np.linalg.norm(up_sum) < 1e-8:
        # There is no unique spherical midpoint for antipodal up-vectors; treating
        # the common-body attitude as failed is the conservative endpoint choice.
        body_tilt = math.pi
        center_roll = math.nan
        center_pitch = math.nan
    else:
        center_up = up_sum / np.linalg.norm(up_sum)
        body_tilt = math.acos(float(np.clip(center_up[2], -1.0, 1.0)))

        # Complete the midpoint up-vector into a whole-cat body frame. The mean
        # front/rear longitudinal (+x) direction is projected perpendicular to
        # center_up, then re-orthogonalized. This preserves the up-axis midpoint
        # while providing body-fixed axes for yaw-invariant fused roll and pitch
        # diagnostics. Degenerate forward vectors use either segment.
        forward = front.apply([1.0, 0.0, 0.0]) + rear.apply([1.0, 0.0, 0.0])
        forward -= np.dot(forward, center_up) * center_up
        if np.linalg.norm(forward) < 1e-8:
            forward = front.apply([1.0, 0.0, 0.0])
            forward -= np.dot(forward, center_up) * center_up
        if np.linalg.norm(forward) < 1e-8:
            reference = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(reference, center_up)) > 0.9:
                reference = np.array([0.0, 1.0, 0.0])
            forward = reference - np.dot(reference, center_up) * center_up
        center_forward = forward / np.linalg.norm(forward)
        center_side = np.cross(center_up, center_forward)
        center_rotation = np.column_stack((center_forward, center_side, center_up))
        # Fused pitch = asin(-R31), fused roll = asin(R32). Unlike Euler
        # roll/pitch, these depend only on tilt and are invariant to world yaw.
        center_pitch = math.asin(float(np.clip(-center_rotation[2, 0], -1.0, 1.0)))
        center_roll = math.asin(float(np.clip(center_rotation[2, 1], -1.0, 1.0)))

    relative_angle = float((front.inv() * rear).magnitude())
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
    if len(values) < 16:
        return values
    fs = 1.0 / np.median(np.diff(time_mid))
    if not np.isfinite(fs) or fs <= 2.2 * cutoff_hz:
        return values
    sos = signal.butter(4, cutoff_hz, btype="low", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, values, axis=0)


def display_path(path: Path) -> str:
    """Repo-relative path when the file is inside the repo, so outputs do not depend
    on the working directory."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def extract_trial(path: Path, target_distance_m: float) -> dict[str, object]:
    meta = parse_trial_name(path)
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame = frame.dropna(subset=sorted(REQUIRED_COLUMNS)).reset_index(drop=True)
    time = frame["Time"].to_numpy(float)
    if len(time) < 10 or np.any(np.diff(time) <= 0):
        raise ValueError(f"{path}: insufficient or non-monotonic time samples")

    # Impact = the acceleration peak. B_ACC is 0 in every log (no rear IMU), so this
    # is the front accelerometer peak. A trial is excluded when the impact comes
    # before the evaluation instant t0 + sqrt(2 d / g), i.e. the drop was too short.
    accel = np.maximum(frame["F_ACC"].to_numpy(float), frame["B_ACC"].to_numpy(float))
    impact_idx = int(np.nanargmax(accel))
    impact_time = time[impact_idx]
    duration = impact_time - time[0]
    observed_fall_distance = 0.5 * G * duration**2
    target_time = time[0] + math.sqrt(2.0 * target_distance_m / G)
    if impact_idx < 2 or impact_time < target_time:
        raise RuntimeError(
            f"short_drop: estimated fall distance {observed_fall_distance:.3f} m "
            f"is below target {target_distance_m:.3f} m"
        )

    q_scalar_first = frame[["F_Q0", "F_Q1", "F_Q2", "F_Q3"]].to_numpy(float)
    q_norm = np.linalg.norm(q_scalar_first, axis=1)
    if np.any(q_norm < 0.5):
        raise ValueError(f"{path}: invalid near-zero quaternion")
    q_scalar_first = q_scalar_first / q_norm[:, None]
    rotations = Rotation.from_quat(q_scalar_first[:, [1, 2, 3, 0]])
    # The rear IMU never reported, so the rear body comes from the front IMU and
    # the spine encoders (paper Eq. 2).
    rear_rotations = rear_body_rotation(
        rotations, frame["F_M1"], frame["F_M2"], frame["B_M2"]
    )

    # Slerp gives the orientation at the exact physical cutoff, rather than the
    # nearest telemetry frame (which can differ by ~10 ms across trials).
    target_rotation = Slerp(time[: impact_idx + 1], rotations[: impact_idx + 1])([target_time])[0]
    target_rear_rotation = Slerp(
        time[: impact_idx + 1], rear_rotations[: impact_idx + 1]
    )([target_time])[0]
    initial_rotation = rotations[0]
    initial_rear_rotation = rear_rotations[0]
    target_euler = target_rotation.as_euler("xyz", degrees=True)
    initial_euler = initial_rotation.as_euler("xyz", degrees=True)

    # Front-body angular velocity from consecutive quaternions, 15 Hz low-passed.
    # The initial value (a covariate) is the median of the first five rates.
    dt = np.diff(time[: impact_idx + 1])
    delta = rotations[:impact_idx].inv() * rotations[1 : impact_idx + 1]
    angular_velocity = np.degrees(delta.as_rotvec()) / dt[:, None]
    midpoint_time = (time[:impact_idx] + time[1 : impact_idx + 1]) / 2.0
    angular_velocity = low_pass(angular_velocity, midpoint_time)
    target_velocity = np.array(
        [np.interp(target_time, midpoint_time, angular_velocity[:, axis]) for axis in range(3)]
    )
    initial_velocity = np.median(angular_velocity[: min(5, len(angular_velocity))], axis=0)

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
) -> tuple[np.ndarray, list[str]]:
    """Intercept, tail indicator, condition and date dummies, z-scored covariates."""
    columns = [np.ones(len(frame)), frame["tail"].to_numpy(float)]
    names = ["intercept", "with_tail"]
    conditions = [condition for condition in CONDITION_ORDER if condition in set(frame["condition"])]
    for condition in conditions[1:]:
        indicator = (frame["condition"] == condition).to_numpy(float)
        columns.append(indicator)
        names.append(f"condition[{condition}]")
    if include_date:
        dates = sorted(frame["date"].astype(str).unique())
        for date in dates[1:]:
            columns.append((frame["date"].astype(str) == date).to_numpy(float))
            names.append(f"date[{date}]")
    for covariate in covariates:
        values = frame[covariate].to_numpy(float)
        scale = values.std(ddof=1)
        columns.append((values - values.mean()) / scale if scale > 0 else np.zeros(len(values)))
        names.append(f"z({covariate})")
    return np.column_stack(columns), names


def fit_ols(y: np.ndarray, x: np.ndarray, names: list[str]) -> OLSResult:
    """OLS with HC3 heteroskedasticity-consistent covariance."""
    # Some BLAS builds (seen with numpy 2.2 on Apple silicon) raise spurious
    # divide/overflow/invalid flags inside matmul on these small, finite matrices;
    # the products are correct, so the flags are silenced and finiteness checked.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        xtx_inverse = np.linalg.pinv(x.T @ x)
        beta = xtx_inverse @ x.T @ y
        fitted = x @ beta
        residuals = y - fitted
        rank = np.linalg.matrix_rank(x)
        df_resid = len(y) - rank
        leverage = np.einsum("ij,jk,ik->i", x, xtx_inverse, x)
        adjusted_squared = (residuals / np.maximum(1.0 - leverage, 1e-8)) ** 2
        meat = x.T @ (x * adjusted_squared[:, None])
        covariance_hc3 = xtx_inverse @ meat @ xtx_inverse
    if not np.all(np.isfinite(beta)):
        raise FloatingPointError("non-finite OLS coefficients")
    sigma = math.sqrt(float(residuals @ residuals) / df_resid)
    return OLSResult(beta, covariance_hc3, residuals, fitted, df_resid, sigma, xtx_inverse, names)


def coefficient_row(result: OLSResult, index: int = 1) -> dict[str, float]:
    """Estimate, HC3 SE, 95% CI and two-sided p (t reference) for one coefficient."""
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
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(p_values) - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def overlap_permutation(
    frame: pd.DataFrame,
    outcome: str,
    n_permutations: int,
    seed: int,
    covariates: tuple[str, ...] = COVARIATES,
) -> tuple[float, float, int]:
    """Blocked permutation test of the tail coefficient.

    Tail labels are shuffled within condition x date blocks. Only blocks that
    contain both morphologies can be permuted, so the test (including its observed
    estimate) uses those trials alone; n_overlap reports how many.
    """
    strata = frame.groupby(["condition", "date"])["tail"].agg(["min", "max"])
    overlap = set(strata[(strata["min"] == 0) & (strata["max"] == 1)].index)
    mask = [(condition, date) in overlap for condition, date in zip(frame["condition"], frame["date"])]
    subset = frame.loc[mask].copy().reset_index(drop=True)
    x, names = build_design(subset, covariates=covariates)
    observed = fit_ols(subset[outcome].to_numpy(float), x, names).beta[1]
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    grouped_indices = [group.index.to_numpy() for _, group in subset.groupby(["condition", "date"])]
    original_tail = subset["tail"].to_numpy(int)
    for iteration in range(n_permutations):
        permuted = original_tail.copy()
        for indices in grouped_indices:
            permuted[indices] = rng.permutation(permuted[indices])
        permuted_frame = subset.copy()
        permuted_frame["tail"] = permuted
        x_permuted, _ = build_design(permuted_frame, covariates=covariates)
        null[iteration] = fit_ols(subset[outcome].to_numpy(float), x_permuted, names).beta[1]
    p_value = (1.0 + np.sum(np.abs(null) >= abs(observed))) / (n_permutations + 1.0)
    return observed, float(p_value), len(subset)


def regression_power(effect: float, sigma: float, coefficient_variance_unit: float, df: int, alpha: float) -> float:
    ncp = effect / (sigma * math.sqrt(coefficient_variance_unit))
    critical = stats.t.ppf(1.0 - alpha / 2.0, df)
    return float(stats.nct.cdf(-critical, df, ncp) + stats.nct.sf(critical, df, ncp))


def regression_mde(sigma: float, coefficient_variance_unit: float, df: int, alpha: float, target: float = 0.8) -> float:
    """Smallest tail effect the model detects with `target` power (two-sided t test)."""
    objective = lambda effect: regression_power(effect, sigma, coefficient_variance_unit, df, alpha) - target
    upper = sigma
    while objective(upper) < 0:
        upper *= 1.5
        if upper > 10.0 * sigma:
            raise RuntimeError("Could not bracket the regression MDE")
    return float(optimize.brentq(objective, 1e-9, upper))


def analyze_outcomes(frame: pd.DataFrame, n_permutations: int, seed: int) -> pd.DataFrame:
    """All-trials ANCOVA of the tail effect, one row per outcome in OUTCOMES."""
    x, names = build_design(frame)
    x_no_date, names_no_date = build_design(frame, include_date=False)
    rows: list[dict[str, object]] = []
    for outcome, label in OUTCOMES.items():
        y = frame[outcome].to_numpy(float)
        row: dict[str, object] = {"outcome": outcome, "label": label,
                                  **coefficient_row(fit_ols(y, x, names))}
        no_date = coefficient_row(fit_ols(y, x_no_date, names_no_date))
        row.update({"estimate_no_date": no_date["estimate"], "p_no_date": no_date["p_hc3"]})
        perm_estimate, perm_p, perm_n = overlap_permutation(frame, outcome, n_permutations, seed)
        row.update({"overlap_estimate": perm_estimate, "p_permutation": perm_p, "n_overlap": perm_n})
        rows.append(row)
    results = pd.DataFrame(rows)
    results["p_holm_all_outcomes"] = holm_adjust(results["p_hc3"].astype(float).tolist())
    return results


def analyze_predefined_sweeps(
    frame: pd.DataFrame,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    """One pooled tail contrast for each scientifically defined condition sweep."""
    rows: list[dict[str, object]] = []
    # Pitch sweep first: the row order is kept from the original analysis output.
    for sweep in ("pitch_sweep_at_roll_180", "no_pitch_roll_sweep"):
        definition = SWEEPS[sweep]
        subset = frame.loc[frame["condition"].isin(definition["conditions"])].copy().reset_index(drop=True)
        x, names = build_design(subset)
        model = fit_ols(subset["posture_score"].to_numpy(float), x, names)
        adjusted = coefficient_row(model)
        x_no_date, names_no_date = build_design(subset, include_date=False)
        no_date = coefficient_row(
            fit_ols(subset["posture_score"].to_numpy(float), x_no_date, names_no_date)
        )
        overlap_estimate, p_permutation, n_overlap = overlap_permutation(
            subset, "posture_score", n_permutations, seed
        )
        mde = regression_mde(
            model.sigma, float(model.xtx_inverse[1, 1]), model.df_resid, alpha=0.05
        )
        rows.append(
            {
                "sweep": sweep,
                "label": definition["label"],
                "conditions": ",".join(definition["conditions"]),
                "n": len(subset),
                **adjusted,
                "estimate_no_date": no_date["estimate"],
                "p_no_date": no_date["p_hc3"],
                "overlap_estimate": overlap_estimate,
                "p_permutation": p_permutation,
                "n_overlap": n_overlap,
                "mde_80": mde,
            }
        )
    result = pd.DataFrame(rows)
    result["p_holm_two_sweeps"] = holm_adjust(result["p_hc3"].tolist())
    result["p_permutation_holm_two_sweeps"] = holm_adjust(result["p_permutation"].tolist())
    return result


def analyze_pitch_leveling(
    frame: pd.DataFrame,
    n_permutations: int,
    seed: int,
) -> dict[str, object]:
    """Tail contrast for final absolute whole-cat pitch in the 180-roll pitch sweep."""
    conditions = PITCH_SWEEP
    subset = frame.loc[frame["condition"].isin(conditions)].copy().reset_index(drop=True)
    covariates = ("initial_abs_whole_body_pitch_deg", "initial_angular_speed_deg_s")
    x, names = build_design(subset, covariates=covariates)
    model = fit_ols(subset["abs_whole_body_pitch_deg"].to_numpy(float), x, names)
    adjusted = coefficient_row(model)
    x_no_date, names_no_date = build_design(
        subset, include_date=False, covariates=covariates
    )
    no_date = coefficient_row(
        fit_ols(
            subset["abs_whole_body_pitch_deg"].to_numpy(float),
            x_no_date,
            names_no_date,
        )
    )
    overlap_estimate, permutation_p, n_overlap = overlap_permutation(
        subset,
        "abs_whole_body_pitch_deg",
        n_permutations,
        seed,
        covariates=covariates,
    )
    return {
        "analysis": "pitch_leveling_at_roll_180",
        "label": "Final absolute whole-cat pitch; roll 180, pitch 0/15/30/45 deg",
        "conditions": ",".join(conditions),
        "n": len(subset),
        **adjusted,
        "estimate_no_date": no_date["estimate"],
        "p_no_date": no_date["p_hc3"],
        "overlap_estimate": overlap_estimate,
        "p_permutation": permutation_p,
        "n_overlap": n_overlap,
    }


def build_table2(frame: pd.DataFrame, results: pd.DataFrame, sweep_results: pd.DataFrame) -> pd.DataFrame:
    """Paper Table II: raw whole-cat score per arm plus the adjusted tail effect.

    Raw SDs are sample SDs (ddof=1). The Holm column is p_holm_three_requested: the
    family is {roll sweep, pitch sweep, pitch leveling}, not the two sweeps alone.
    The all-trials row is the primary analysis and has no Holm value.
    """
    rows = []
    for label, sweep in TABLE2_ROWS:
        if sweep is None:
            subset = frame
            model = results.loc[results["outcome"] == "posture_score"].iloc[0]
            p_holm = math.nan
            conditions = ",".join(CONDITION_ORDER)
        else:
            conditions = SWEEPS[sweep]["conditions"]
            subset = frame.loc[frame["condition"].isin(conditions)]
            model = sweep_results.loc[sweep_results["sweep"] == sweep].iloc[0]
            p_holm = float(model["p_holm_three_requested"])
            conditions = ",".join(conditions)
        row = {"analysis": label, "conditions": conditions}
        for morphology, key in (("no_tail", "no_tail"), ("with_tail", "with_tail")):
            scores = subset.loc[subset["morphology"] == morphology, "posture_score"]
            row[f"n_{key}"] = len(scores)
            row[f"mean_{key}"] = scores.mean()
            row[f"sd_{key}"] = scores.std(ddof=1)
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
    """Table II as markdown, at the precision the paper prints."""
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


def write_table2(table: pd.DataFrame, output_dir: Path) -> None:
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
    target_time = math.sqrt(2.0 * target_distance_m / G)
    lines = [
        "# Whole-cat score analysis",
        "",
        f"Generated by `analysis/whole_cat_stats.py`: {len(frame)} trials included, "
        f"{len(excluded)} excluded (impact before {target_distance_m:g} m of fall). Each trial "
        f"is scored at t0 + {target_time:.3f} s, the instant of {target_distance_m:g} m of "
        f"ballistic fall after free-fall detection. Permutation tests: {n_permutations} "
        f"permutations, seed {seed}.",
        "",
        "## Table II",
        "",
        *format_table2(table2),
        "",
        *[f"- {note}" for note in TABLE2_NOTES],
        "",
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
        "## Pitch leveling within the pitch sweep",
        "",
        "Final absolute whole-cat fused pitch (0 deg is level, so a negative effect favors "
        "the tail), adjusted for condition, date, initial absolute whole-cat pitch and initial "
        "angular speed. This is the third member of the Holm family used in Table II.",
        "",
        "| Outcome | n | Tail effect | 95% HC3 CI | HC3 p | Holm p (3 tests) | Permutation p (n) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Absolute final whole-cat pitch | {pitch_level['n']} | {pitch_level['estimate']:+.2f} deg "
        f"| {pitch_level['ci_low']:+.2f} to {pitch_level['ci_high']:+.2f} deg "
        f"| {pitch_level['p_hc3']:.4f} | {pitch_level['p_holm_three_requested']:.4f} "
        f"| {pitch_level['p_permutation']:.4f} ({pitch_level['n_overlap']}) |",
        "",
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
        "'Data notes' of data/README.md and in data/hardware_trials/MANIFEST.csv. "
        "Date is a model term and defines the permutation blocks.",
    ]
    if excluded:
        lines += ["", "## Excluded trials", ""]
        lines += [f"- `{row['source_file']}`: {row['reason']}" for row in excluded]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="folder of trial CSVs, searched recursively "
                             f"(default: {DEFAULT_INPUT.relative_to(ROOT)})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="folder for the CSV/Markdown outputs "
                             f"(default: {DEFAULT_OUTPUT.relative_to(ROOT)})")
    parser.add_argument("--distance", type=float, default=DEFAULT_DISTANCE_M,
                        help="fall distance (m) at which each trial is scored (default: %(default)s)")
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS,
                        help="blocked permutations per test (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="permutation RNG seed (default: %(default)s)")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    excluded: list[dict[str, str]] = []
    for path in find_trials(args.input):
        try:
            records.append(extract_trial(path, args.distance))
        except RuntimeError as error:
            excluded.append({"source_file": display_path(path), "reason": str(error)})
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("No trials passed preprocessing")
    duplicate_key = frame.duplicated(["morphology", "condition", "date", "rep"], keep=False)
    if duplicate_key.any():
        raise ValueError(f"Duplicate canonical trials:\n{frame.loc[duplicate_key].to_string(index=False)}")
    frame.to_csv(args.output / "trial_metrics.csv", index=False)
    pd.DataFrame(excluded, columns=("source_file", "reason")).to_csv(
        args.output / "excluded_trials.csv", index=False)

    results = analyze_outcomes(frame, args.permutations, args.seed)
    results.to_csv(args.output / "model_results.csv", index=False)

    sweep_results = analyze_predefined_sweeps(frame, args.permutations, args.seed)
    pitch_level = analyze_pitch_leveling(frame, args.permutations, args.seed)
    # The Holm family used in the paper: both sweeps plus the pitch-leveling test.
    family_adjusted = holm_adjust(
        sweep_results["p_hc3"].astype(float).tolist() + [float(pitch_level["p_hc3"])]
    )
    sweep_results["p_holm_three_requested"] = family_adjusted[:2]
    pitch_level["p_holm_three_requested"] = family_adjusted[2]
    sweep_results.to_csv(args.output / "sweep_results.csv", index=False)
    pd.DataFrame([pitch_level]).to_csv(args.output / "pitch_level_analysis.csv", index=False)

    table2 = build_table2(frame, results, sweep_results)
    write_table2(table2, args.output)
    write_report(frame, excluded, results, sweep_results, pitch_level, table2,
                 args.output, args.distance, args.permutations, args.seed)

    print("\n".join(format_table2(table2)))
    print(f"\nIncluded {len(frame)}; excluded {len(excluded)} "
          f"short {'drop' if len(excluded) == 1 else 'drops'}")
    print(f"Results written to {args.output}")
    return {"frame": frame, "excluded": excluded, "results": results,
            "sweep_results": sweep_results, "pitch_level": pitch_level, "table2": table2}


if __name__ == "__main__":
    main()
