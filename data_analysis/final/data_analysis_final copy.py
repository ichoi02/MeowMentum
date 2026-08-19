import os
import re
import glob
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt

BASE_DIR    = './telemetry/Final_Experiment'
PLOT_DIR    = './data_analysis/final/plots_final'
SANITY_DIR  = './data_analysis/final/plots_final/sanity_checks'
REPORT_PATH = './data_analysis/final/report_final.csv'

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

                if duration < 0.7:
                    print(f"  SKIPPED: fall duration too short ({duration:.2f} s < 0.7 s)")
                    continue

                fall_distance = 9.81 / 2 * duration ** 2
                if fall_distance <= 2.5:
                    print(f"  SKIPPED: fall distance too short ({fall_distance:.2f} m < 2.5 m)")
                    continue

                time_at_2p5m = np.sqrt(2 * 2.5 / 9.81) + time_at_initial
                end_idx = np.argmin(np.abs(df['Time'] - time_at_2p5m))
                # end_idx should generally land before impact (the fall-distance filter targets
                # that), but at ~50 Hz a borderline-duration drop (fall_distance just over 2.5 m)
                # can have its 2.5m-point and its detected-impact frame round to the same sample —
                # not a bug, just means "outcome at 2.5m" and "outcome at impact" coincide for that
                # rep. A real problem would be end_idx landing well AFTER impact, which this still catches.
                if end_idx == index_at_impact:
                    print(f"  NOTE: 2.5m cutoff coincides with the detected impact frame ({end_idx}) "
                          f"— borderline duration (fall_distance={fall_distance:.2f} m)")
                assert end_idx <= index_at_impact, (
                    f"{file_name}: 2.5m cutoff (frame {end_idx}) landed AFTER the detected impact "
                    f"(frame {index_at_impact}) — investigate this file, something is inconsistent"
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

                _zeros  = np.zeros(end_idx)
                r_rot1  = R.from_rotvec(np.column_stack([df['F_M1'].iloc[0:end_idx].to_numpy(), _zeros, _zeros]))
                r_pitch = R.from_rotvec(np.column_stack([_zeros, df['F_M2'].iloc[0:end_idx].to_numpy(), _zeros]))
                r_rot2  = R.from_rotvec(np.column_stack([df['B_M2'].iloc[0:end_idx].to_numpy(), _zeros, _zeros]))
                fk_rots = f_rot * r_rot1 * r_pitch * r_rot2

                f_ori  = np.degrees(np.unwrap(np.radians(f_ori),  axis=0, discont=np.pi))
                b_ori  = np.degrees(np.unwrap(np.radians(b_ori),  axis=0, discont=np.pi))
                fk_ori = np.degrees(np.unwrap(np.radians(fk_rots.as_euler('xyz', degrees=True)), axis=0, discont=np.pi))

                # ── Angular velocity ──────────────────────────────────
                dt        = np.diff(time)
                f_angvel  = quat_angvel_deg(f_rot,   dt)
                b_angvel  = quat_angvel_deg(b_rots,  dt)
                fk_angvel = quat_angvel_deg(fk_rots, dt)

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
                    'fk_roll': fk_ori[:, 0], 'fk_pitch': fk_ori[:, 1],
                    'f_rollrate': f_angvel[:, 0], 'f_pitchrate': f_angvel[:, 1],
                })

                # ── Record ────────────────────────────────────────────
                record = {
                    'date': date, 'trial': trial, 'rep': rep,
                    'morphology': morphology,
                    # at 2.5 m
                    'F_roll_2p5m':       round(f_ori[-1, 0],   2),
                    'B_roll_2p5m':       round(fk_ori[-1, 0],  2),
                    'F_rollrate_2p5m':   round(f_angvel[-1, 0], 2),
                    'B_rollrate_2p5m':   round(fk_angvel[-1, 0], 2),
                    # at release
                    'F_roll_initial':    round(f_ori[0, 0],   2),
                    'B_roll_initial':    round(fk_ori[0, 0],  2),
                    'F_rollrate_initial':round(f_angvel[0, 0], 2),
                    'B_rollrate_initial':round(fk_angvel[0, 0], 2),
                    # at 2.5 m
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

PANELS = [
    ('f_roll',      'fk_roll',      'Roll (deg)',       (-180, 180)),
    ('f_rollrate',  None,           'Roll rate (deg/s)', None),
    ('f_pitch',     'fk_pitch',     'Pitch (deg)',       (-180, 180)),
    ('f_pitchrate', None,           'Pitch rate (deg/s)', None),
]

for trial, reps in sorted(trajectories.items()):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.flatten()

    for ax, (f_key, fk_key, ylabel, ylim) in zip(axes, PANELS):
        seen_morphologies = set()
        for rec in reps:
            color = MORPHOLOGY_COLORS[rec['morphology']]
            label = rec['morphology'] if rec['morphology'] not in seen_morphologies else None
            seen_morphologies.add(rec['morphology'])

            t = rec['time'][:-1] if f_key.endswith('rate') else rec['time']
            ax.plot(t, rec[f_key], color=color, alpha=0.6, linewidth=1.2, label=label)
            if fk_key is not None:
                ax.plot(rec['time'], rec[fk_key], color=color, alpha=0.35, linewidth=1.0, linestyle='--')

        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)

    axes[2].set_xlabel('Time since release (s)')
    axes[3].set_xlabel('Time since release (s)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))
    n_with  = sum(1 for r in reps if r['morphology'] == 'With Tail')
    n_no    = sum(1 for r in reps if r['morphology'] == 'No Tail')
    fig.suptitle(f'All reps — trial {trial}  (With Tail: n={n_with}, No Tail: n={n_no}; '
                 f'solid=F IMU, dashed=FK rear)', y=1.06)
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
