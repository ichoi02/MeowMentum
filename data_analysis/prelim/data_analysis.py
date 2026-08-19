import os
import pandas as pd
import numpy as np
from pdb import set_trace as st
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

DATA_DIR   = './telemetry'
PLOT_DIR   = './data_analysis/prelim/plots'
REPORT_PATH     = './data_analysis/prelim/report.csv'      # RL only, both morphologies
REPORT_MBC_PATH = './data_analysis/prelim/report_mbc.csv'  # spine+tail only, RL vs MBC

MUJOCO_PLAYBACK = False


def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
    else:
        return pd.read_csv(file_path)


def impact_detector(accs):
    time_at_impact = np.max(np.where(accs == np.max(np.max(accs, axis=0))))
    which_body     = np.argmax(np.where(accs == np.max(np.max(accs, axis=0))))
    return int(time_at_impact), int(which_body)


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
# Trial combinations
# =============================================================================

date_trial_combinations = {
    'spine+tail': {
        'RL': {
            'Apr20': ['r45', 'r90', 'r180'],
            'Apr22': ['r180_p-15', 'r180_p15']
        },
        'MBC': {
            'Apr21': ['r45', 'r90', 'r180'],
            'Apr27': ['r180_p-15', 'r180_p15'],
        }
    },
    'spine-only': {
        'RL': {
            'Apr22': ['r45', 'r90', 'r180', 'r180_p-15', 'r180_p15']
        }
    }
}

# =============================================================================
# Processing loop
# =============================================================================

records     = []   # all records → report.csv
records_mbc = []   # spine+tail RL vs MBC → report_mbc.csv

for morphology in ['spine+tail', 'spine-only']:
    for controller in ['RL', 'MBC']:

        # MBC only exists for spine+tail
        if controller == 'MBC' and morphology == 'spine-only':
            continue

        if controller not in date_trial_combinations[morphology]:
            continue

        for rep in [1, 2, 3, 4, 5]:
            date_trial_pairs = date_trial_combinations[morphology][controller]
            for date, trials in date_trial_pairs.items():
                for trial in trials:
                    file_name = f"{DATA_DIR}/{controller}/{morphology}/{date}_{trial}_{rep}.csv"
                    print(file_name)
                    df = load_data(file_name)
                    if df is None:
                        continue

                    # ── Fall detection ────────────────────────────────────
                    index_at_impact, which_body = impact_detector(df[['F_ACC', 'B_ACC']].to_numpy())
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
                    time    = df['Time'].iloc[0:end_idx].to_numpy()

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

                    # ── Plots ─────────────────────────────────────────────
                    plot_dir = os.path.join(PLOT_DIR, date, trial)
                    os.makedirs(plot_dir, exist_ok=True)

                    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
                    ax1.plot(time,      f_ori[:, 0],  label='F IMU')
                    ax1.plot(time,      b_ori[:, 0],  label='B IMU')
                    ax1.plot(time,      fk_ori[:, 0], label='FK rear', linestyle='--')
                    ax1.set_ylabel('Roll (deg)'); ax1.legend()
                    ax1.set_title(f'Roll angle — {date} {trial} rep{rep} [{controller}]')
                    ax1.set_ylim(-180, 180)
                    ax2.plot(time[:-1], f_angvel[:, 0],  label='F IMU')
                    ax2.plot(time[:-1], b_angvel[:, 0],  label='B IMU')
                    ax2.plot(time[:-1], fk_angvel[:, 0], label='FK rear', linestyle='--')
                    ax2.set_ylabel('Roll rate (deg/s)'); ax2.set_xlabel('Time (s)'); ax2.legend()
                    plt.tight_layout()
                    fig.savefig(os.path.join(plot_dir, f'{controller}_rep{rep}_roll.png'), dpi=150)
                    plt.close(fig)

                    # ── Record ────────────────────────────────────────────
                    record = {
                        'date': date, 'trial': trial, 'rep': rep,
                        'morphology': morphology, 'controller': controller,
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

                    # report.csv: RL only, both morphologies (original)
                    if controller == 'RL':
                        records.append(record)

                    # report_mbc.csv: spine+tail only, RL vs MBC
                    if morphology == 'spine+tail':
                        records_mbc.append(record)

# =============================================================================
# Save
# =============================================================================

report_df = pd.DataFrame(records)
report_df.to_csv(REPORT_PATH, index=False)
print(f"\nreport.csv saved → {REPORT_PATH} ({len(report_df)} trials)")

report_mbc_df = pd.DataFrame(records_mbc)
report_mbc_df.to_csv(REPORT_MBC_PATH, index=False)
print(f"report_mbc.csv saved → {REPORT_MBC_PATH} ({len(report_mbc_df)} trials)")
