import os
import re
import glob
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt

BASE_DIR    = './telemetry/Final_Experiment'

# The "outcome" for every trial is measured at this ballistic fall distance (meters) — the point
# used to compute time_at_target, which sets end_idx, which everything else (orientation, rate,
# the report columns) is sliced up to. Change this to sensitivity-check the choice of cutoff
# (e.g. 2.7 or 3.0). 2.5 is the original/baseline value that stats_final.py and both notebooks
# are already wired to read (REPORT_PATH = '.../report_final.csv'); any OTHER value automatically
# writes to a distinctly-named report/plot dir instead, so a sensitivity sweep can't silently
# clobber the baseline run — but analyzing a non-baseline run means pointing stats_final.py (or a
# copy of it) at the tagged REPORT_PATH printed below.
FALL_DISTANCE_TARGET_M = 2.8

def _fmt_target(target_m):
    return f"{target_m}".replace('.', 'p') + 'm'  # 2.7 -> '2p7m', 3.0 -> '3p0m'

if FALL_DISTANCE_TARGET_M == 2.5:
    PLOT_DIR    = './data_analysis/final/plots_final'
    SANITY_DIR  = './data_analysis/final/plots_final/sanity_checks'
    REPORT_PATH = './data_analysis/final/report_final.csv'
else:
    _tag        = _fmt_target(FALL_DISTANCE_TARGET_M)
    PLOT_DIR    = f'./data_analysis/final/plots_final_{_tag}'
    SANITY_DIR  = f'./data_analysis/final/plots_final_{_tag}/sanity_checks'
    REPORT_PATH = f'./data_analysis/final/report_final_{_tag}.csv'

# Report column names keep the historical "_2p5m" suffix regardless of FALL_DISTANCE_TARGET_M —
# they mean "at the fall-distance target," not literally 2.5m. This keeps stats_final.py and the
# notebooks working unmodified against any target's report file; only REPORT_PATH needs updating
# (there, or in a copy) to point at the run you want to analyze.

MORPHOLOGY_COLORS = {'With Tail': '#1A3A5C', 'No Tail': '#8B4513'}


def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
    else:
        return pd.read_csv(file_path)


def impact_detector(accs):
    """Index (frame, column) of the peak acceleration. Column 0=F_ACC, 1=B_ACC."""
    row, col = np.unravel_index(np.argmax(accs), accs.shape)
    return int(row), int(col)


def quat_angvel_deg(rots, dt):
    """Angular velocity (deg/s, xyz body frame) from quaternion finite differences.
    Singularity-free — works correctly even when roll crosses ±180°."""
    angvel = np.zeros((len(dt), 3))
    for i in range(len(dt)):
        dq = rots[i].inv() * rots[i + 1]
        angvel[i] = np.degrees(dq.as_rotvec()) / dt[i]
    return angvel


def low_pass_filter(data, cutoff_freq, fs):
    from scipy.signal import butter, filtfilt
    nyquist_freq   = 0.5 * fs
    normal_cutoff  = cutoff_freq / nyquist_freq
    b, a = butter(N=4, Wn=normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


# =============================================================================
# File discovery
# =============================================================================
# Folder layout: BASE_DIR/{With Tail,No Tail} {No Pitch,With Pitch}/{deg}deg/*.csv
# The "With Pitch/0deg" folders are byte-identical duplicates of the
# "No Pitch/180deg" folders (same physical drop, filed as the shared anchor
# point for both sweeps) — skip them here so each recording is loaded once.

MORPHOLOGY_DIRS = {
    'With Tail': 'With Tail',
    'No Tail':   'No Tail',
}

# Capturing groups: (1) morphology prefix (2) Roll/Pitch (3) degree (4) date (5) rep (6) spare-marker
FILENAME_RE = re.compile(r'^(WithTail|NoTail)(Roll|Pitch)(\d+)_(\d{4})_(\d+)(\(spare\))?\.csv$')

records = []
trajectories = defaultdict(list)  # trial -> list of per-rep trajectory dicts, for the sanity-check plots

for morphology, morph_dir in MORPHOLOGY_DIRS.items():
    expected_prefix = morph_dir.replace(' ', '')  # 'With Tail' -> 'WithTail'

    for sweep, trial_prefix in [('No Pitch', 'r'), ('With Pitch', 'r180_p')]:
        expected_type = 'Roll' if sweep == 'No Pitch' else 'Pitch'
        sweep_dir = os.path.join(BASE_DIR, f'{morph_dir} {sweep}')

        for deg_dir in sorted(glob.glob(os.path.join(sweep_dir, '*deg'))):
            deg = int(os.path.basename(deg_dir).replace('deg', ''))

            if sweep == 'With Pitch' and deg == 0:
                continue  # dedup — identical to the No-Pitch 180deg trial

            trial = f'r{deg}' if sweep == 'No Pitch' else f'{trial_prefix}{deg}'

            for file_path in sorted(glob.glob(os.path.join(deg_dir, '*.csv'))):
                file_name = os.path.basename(file_path)
                m = FILENAME_RE.match(file_name)
                if m is None:
                    print(f"  SKIPPED (unrecognized filename): {file_name}")
                    continue
                if m.group(6):
                    print(f"  SKIPPED (spare rep): {file_name}")
                    continue

                # ── Sanity check: does the filename actually match the folder it's in? ──
                # Catches a file dropped in the wrong morphology/sweep/degree folder, which
                # would otherwise be silently mislabeled using the folder's metadata instead
                # of its own.
                f_prefix, f_type, f_deg = m.group(1), m.group(2), int(m.group(3))
                if f_prefix != expected_prefix or f_type != expected_type or f_deg != deg:
                    print(f"  SKIPPED (filename/folder mismatch): {file_name} in "
                          f"'{morph_dir} {sweep}/{deg}deg/' — filename implies "
                          f"{f_prefix}/{f_type}/{f_deg}deg")
                    continue

                date, rep = m.group(4), int(m.group(5))

                print(file_path)
                df = load_data(file_path)
                if df is None:
                    continue

                # ── Fall detection ────────────────────────────────────
                index_at_impact, _ = impact_detector(df[['F_ACC', 'B_ACC']].to_numpy())
                time_at_impact  = df['Time'].iloc[index_at_impact]
                time_at_initial = df['Time'].iloc[0]
                duration        = time_at_impact - time_at_initial

                fall_distance = 9.81 / 2 * duration ** 2
                if fall_distance <= FALL_DISTANCE_TARGET_M:
                    print(f"  SKIPPED: fall distance too short "
                          f"({fall_distance:.2f} m < {FALL_DISTANCE_TARGET_M} m)")
                    continue

                time_at_target = np.sqrt(2 * FALL_DISTANCE_TARGET_M / 9.81) + time_at_initial
                end_idx = np.argmin(np.abs(df['Time'] - time_at_target))
                # end_idx should generally land before impact (the fall-distance filter targets
                # that), but at ~50 Hz a borderline-duration drop (fall_distance just over target)
                # can have its target-distance point and its detected-impact frame round to the
                # same sample — not a bug, just means "outcome at target" and "outcome at impact"
                # coincide for that rep. A real problem would be end_idx landing well AFTER
                # impact, which this still catches.
                if end_idx == index_at_impact:
                    print(f"  NOTE: {FALL_DISTANCE_TARGET_M}m cutoff coincides with the detected "
                          f"impact frame ({end_idx}) — borderline duration "
                          f"(fall_distance={fall_distance:.2f} m)")
                assert end_idx <= index_at_impact, (
                    f"{file_name}: {FALL_DISTANCE_TARGET_M}m cutoff (frame {end_idx}) landed AFTER "
                    f"the detected impact (frame {index_at_impact}) — investigate this file, "
                    f"something is inconsistent"
                )
                time    = df['Time'].iloc[0:end_idx].to_numpy()
                time_rebased = time - time_at_initial

                # ── Orientation ───────────────────────────────────────
                f_quat = ['F_Q0', 'F_Q1', 'F_Q2', 'F_Q3']
                b_quat = ['B_Q0', 'B_Q1', 'B_Q2', 'B_Q3']

                f_ori  = df[f_quat].iloc[0:end_idx].to_numpy()
                b_ori  = df[b_quat].iloc[0:end_idx].to_numpy()

                f_rot  = R.from_quat(f_ori, scalar_first=True)
                b_rots = R.from_quat(b_ori, scalar_first=True)
                f_ori  = f_rot.as_euler('xyz', degrees=True)
                b_ori  = b_rots.as_euler('xyz', degrees=True)

                # ── Sanity check: gimbal lock in the 'xyz' Euler extraction ──────────
                # When pitch approaches ±90°, roll and yaw become coupled/degenerate — a small
                # real rotation can produce a large, spurious jump in the extracted roll angle
                # (unwrap() only guards against the ±180° wraparound, not this). That spurious
                # jump then propagates into the unwrapped/reported roll value. This doesn't
                # corrupt the quaternion-based rate columns (quat_angvel_deg doesn't go through
                # Euler angles), only the *_roll_*/*_pitch_* angle columns for this trial.
                GIMBAL_LOCK_WARN_DEG = 80
                max_abs_pitch = np.abs(f_ori[:, 1]).max()
                if max_abs_pitch > GIMBAL_LOCK_WARN_DEG:
                    print(f"  ⚠️  NEAR GIMBAL LOCK: front-body pitch reaches {max_abs_pitch:.1f}° "
                          f"(> {GIMBAL_LOCK_WARN_DEG}°) in this trial — the reported roll angle "
                          f"columns may not be physically meaningful for this rep; inspect "
                          f"'{file_name}' manually before trusting F_roll_2p5m/F_roll_initial.")

                _zeros  = np.zeros(end_idx)
                r_rot1  = R.from_rotvec(np.column_stack([df['F_M1'].iloc[0:end_idx].to_numpy(), _zeros, _zeros]))
                r_pitch = R.from_rotvec(np.column_stack([_zeros, df['F_M2'].iloc[0:end_idx].to_numpy(), _zeros]))
                r_rot2  = R.from_rotvec(np.column_stack([df['B_M2'].iloc[0:end_idx].to_numpy(), _zeros, _zeros]))
                fk_rots = f_rot * r_rot1 * r_pitch * r_rot2

                f_ori  = np.degrees(np.unwrap(np.radians(f_ori),  axis=0, discont=np.pi))
                b_ori  = np.degrees(np.unwrap(np.radians(b_ori),  axis=0, discont=np.pi))
                fk_ori = np.degrees(np.unwrap(np.radians(fk_rots.as_euler('xyz', degrees=True)), axis=0, discont=np.pi))

                # ── Angular velocity ──────────────────────────────────
                # Quaternion finite-differencing amplifies high-frequency noise, and this data
                # shows a specific artifact on top of that: an occasional stale/duplicated
                # quaternion sample produces a near-zero rate on one frame and a compensating
                # spike on the next (visible as alternating ~0 / ~1500+ deg/s pairs in the raw
                # signal). A low-pass filter with a cutoff well below Nyquist but well above the
                # tumble's real bandwidth (the whole rotation envelope rises and falls over
                # several hundred ms, i.e. a few Hz) removes that ringing without reshaping the
                # actual dynamics. RATE_FILTER_CUTOFF_HZ is deliberately conservative — closer to
                # Nyquist than to the signal bandwidth — err toward trusting the data.
                RATE_FILTER_CUTOFF_HZ = 15
                dt        = np.diff(time)
                fs        = 1 / np.median(dt)
                f_angvel  = low_pass_filter(quat_angvel_deg(f_rot,   dt), RATE_FILTER_CUTOFF_HZ, fs)
                b_angvel  = low_pass_filter(quat_angvel_deg(b_rots,  dt), RATE_FILTER_CUTOFF_HZ, fs)
                fk_angvel = low_pass_filter(quat_angvel_deg(fk_rots, dt), RATE_FILTER_CUTOFF_HZ, fs)

                # ── Per-rep plot ──────────────────────────────────────
                # Note: B_Q is a constant identity quaternion in this dataset (no physical rear
                # IMU was used) — b_ori is always flat at 0 and carries no information, so it's
                # deliberately left out of this plot (unlike the prelim data_analysis.py, where
                # the rear IMU was real). FK rear is the meaningful second trace.
                plot_dir = os.path.join(PLOT_DIR, date, trial)
                os.makedirs(plot_dir, exist_ok=True)

                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
                ax1.plot(time,      f_ori[:, 0],  label='F IMU')
                ax1.plot(time,      fk_ori[:, 0], label='FK rear', linestyle='--')
                ax1.set_ylabel('Roll (deg)'); ax1.legend()
                ax1.set_title(f'Roll angle — {date} {trial} rep{rep} [{morphology}]')
                ax1.set_ylim(-180, 180)
                ax2.plot(time[:-1], f_angvel[:, 0],  label='F IMU')
                ax2.plot(time[:-1], fk_angvel[:, 0], label='FK rear', linestyle='--')
                ax2.set_ylabel('Roll rate (deg/s)'); ax2.set_xlabel('Time (s)'); ax2.legend()
                plt.tight_layout()
                fig.savefig(os.path.join(plot_dir, f'{morphology.replace(" ", "")}_rep{rep}_roll.png'), dpi=150)
                plt.close(fig)

                # ── Stash trajectory for the combined per-trial sanity-check plot ─────
                trajectories[trial].append({
                    'morphology': morphology, 'date': date, 'rep': rep,
                    'time': time_rebased,
                    'f_roll': f_ori[:, 0], 'f_pitch': f_ori[:, 1],
                    'f_rollrate': f_angvel[:, 0], 'f_pitchrate': f_angvel[:, 1],
                })

                # ── Record ────────────────────────────────────────────
                record = {
                    'date': date, 'trial': trial, 'rep': rep,
                    'morphology': morphology,
                    # at fall-distance target (column name kept as "_2p5m" regardless of FALL_DISTANCE_TARGET_M)
                    'F_roll_2p5m':       round(f_ori[-1, 0],   2),
                    'B_roll_2p5m':       round(fk_ori[-1, 0],  2),
                    'F_rollrate_2p5m':   round(f_angvel[-1, 0], 2),
                    'B_rollrate_2p5m':   round(fk_angvel[-1, 0], 2),
                    # at release
                    'F_roll_initial':    round(f_ori[0, 0],   2),
                    'B_roll_initial':    round(fk_ori[0, 0],  2),
                    'F_rollrate_initial':round(f_angvel[0, 0], 2),
                    'B_rollrate_initial':round(fk_angvel[0, 0], 2),
                    # at fall-distance target
                    'F_pitch_2p5m':      round(f_ori[-1, 1],   2),
                    'B_pitch_2p5m':      round(fk_ori[-1, 1],  2),
                    'F_pitchrate_2p5m':  round(f_angvel[-1, 1], 2),
                    'B_pitchrate_2p5m':  round(fk_angvel[-1, 1], 2),
                    # at release
                    'F_pitch_initial':   round(f_ori[0, 1],   2),
                    'B_pitch_initial':   round(fk_ori[0, 1],  2),
                    'F_pitchrate_initial':round(f_angvel[0, 1], 2),
                    'B_pitchrate_initial':round(fk_angvel[0, 1], 2),
                }
                records.append(record)

# =============================================================================
# Sanity check: duplicate records
# =============================================================================
# Catches accidental double-loading (e.g. a future data drop reintroducing the
# 0deg/180deg overlap, or two files resolving to the same (morphology, trial, date, rep)).

key_counts = defaultdict(int)
for r in records:
    key_counts[(r['morphology'], r['trial'], r['date'], r['rep'])] += 1
duplicates = {k: n for k, n in key_counts.items() if n > 1}
if duplicates:
    print("\n⚠️  DUPLICATE RECORDS DETECTED (same morphology/trial/date/rep loaded more than once):")
    for k, n in duplicates.items():
        print(f"    {k}: loaded {n} times")
else:
    print(f"\n✅ No duplicate (morphology, trial, date, rep) records among {len(records)} rows.")

# =============================================================================
# Sanity-check plots: all reps overlaid on one figure, per trial condition
# =============================================================================
# Lets you eyeball, per condition: are reps within a morphology consistent with
# each other, and does With Tail visually separate from No Tail?

os.makedirs(SANITY_DIR, exist_ok=True)

# Front IMU only — the FK-rear trace is a derived quantity (front IMU + joint encoders), not an
# independent signal, and doubled the line count in these plots without adding much a human
# sanity-checking raw trajectories needs.
#
# Angle panels get a per-rep +-360*n realignment before plotting: unwrap() reports a genuine
# cumulative angle, so two reps of the *same* physical rotation can legitimately differ by an
# exact multiple of 360 (e.g. one rep's Euler extraction round a gimbal-lock passage adds an
# extra revolution). That ambiguity is already harmless for the actual statistics — every
# downstream script wraps through angular_distance(), which is exactly 360-periodic, so it
# doesn't care which multiple was recorded — but it does make the raw trace visually
# incomparable across reps. Picking, per rep, the integer n that pulls its endpoint closest to
# the group's median endpoint (a single constant shift, not a per-frame correction) undoes that
# display-only ambiguity so all reps in a condition plot on the same visual scale. Rate panels
# aren't cumulative and don't need this.
PANELS = [
    ('f_roll',      'Roll (deg)',        True),
    ('f_rollrate',  'Roll rate (deg/s)', False),
    ('f_pitch',     'Pitch (deg)',       True),
    ('f_pitchrate', 'Pitch rate (deg/s)', False),
]

for trial, reps in sorted(trajectories.items()):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.flatten()

    for ax, (key, ylabel, is_angle) in zip(axes, PANELS):
        if is_angle:
            median_end = np.median([r[key][-1] for r in reps])

        seen_morphologies = set()
        for rec in reps:
            color = MORPHOLOGY_COLORS[rec['morphology']]
            label = rec['morphology'] if rec['morphology'] not in seen_morphologies else None
            seen_morphologies.add(rec['morphology'])

            t = rec['time'][:-1] if key.endswith('rate') else rec['time']
            y = rec[key]
            if is_angle:
                n = round((median_end - y[-1]) / 360)
                if n != 0:
                    y = y + 360 * n
            ax.plot(t, y, color=color, alpha=0.6, linewidth=1.2, label=label)

        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    axes[2].set_xlabel('Time since release (s)')
    axes[3].set_xlabel('Time since release (s)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))
    n_with  = sum(1 for r in reps if r['morphology'] == 'With Tail')
    n_no    = sum(1 for r in reps if r['morphology'] == 'No Tail')
    fig.suptitle(f'All reps — trial {trial}  (With Tail: n={n_with}, No Tail: n={n_no}; '
                 f'front IMU only; angle panels realigned by +-360*n per rep)', y=1.06)
    plt.tight_layout()
    fig.savefig(os.path.join(SANITY_DIR, f'{trial}_all_reps.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Sanity-check plot saved: {trial}_all_reps.png ({len(reps)} reps)")

# =============================================================================
# Save
# =============================================================================

report_df = pd.DataFrame(records)

# ── Sanity check: value ranges and missing data ──────────────────────────
print("\n" + "=" * 70)
print("SANITY SUMMARY")
print("=" * 70)
print(f"Total rows: {len(report_df)}")
print(f"NaN values: {report_df.isna().sum().sum()} "
      f"({'none — good' if report_df.isna().sum().sum() == 0 else 'CHECK THIS'})")
print("\nRows per (morphology, trial):")
print(report_df.groupby(['morphology', 'trial']).size().unstack(fill_value=0))

# Angle columns are UNWRAPPED (np.unwrap), so values outside [-180, 180] are expected and
# normal for a robot that tumbles more than half a revolution — that's the whole point of
# using unwrap() instead of raw atan2 output. It is not, by itself, a problem; the downstream
# stats notebooks convert back to shortest-angular-distance before using these values. Only
# flag genuinely implausible multi-revolution counts (a full 360°+ tumble in ~1s is already a
# lot; several full turns would suggest a quaternion/unwrap glitch rather than a real tumble).
angle_cols = [c for c in report_df.columns if 'rate' not in c and ('roll' in c or 'pitch' in c)]
n_multi_rev = (report_df[angle_cols].abs() > 360).sum().sum()
n_implausible = (report_df[angle_cols].abs() > 720).sum().sum()
print(f"\nUnwrapped angle values beyond ±360° (>1 full revolution): {n_multi_rev} "
      f"(expected for hard tumbles, informational only)")
print(f"Unwrapped angle values beyond ±720° (>2 full revolutions): {n_implausible} "
      f"({'none — good' if n_implausible == 0 else 'CHECK THIS — implausible for a ~1s drop'})")

# Angular rate columns SHOULD be bounded — a "moonshot" reading (single-frame rate spike) points
# at a quaternion discontinuity or IMU glitch rather than real motion, unlike the angle columns above.
MOONSHOT_DEG_S = 2000
rate_cols = [c for c in report_df.columns if 'rate' in c]
moonshot_mask = report_df[rate_cols].abs() > MOONSHOT_DEG_S
n_moonshot = moonshot_mask.sum().sum()
print(f"\nRate columns exceeding {MOONSHOT_DEG_S} deg/s ('moonshot', likely sensor glitch): {n_moonshot} "
      f"({'none — good' if n_moonshot == 0 else 'CHECK THIS'})")
if n_moonshot:
    flagged_rows = report_df.loc[moonshot_mask.any(axis=1), ['date', 'trial', 'rep', 'morphology']]
    print(flagged_rows.to_string(index=False))

report_df.to_csv(REPORT_PATH, index=False)
print(f"\nreport_final.csv saved → {REPORT_PATH} ({len(report_df)} trials)")
