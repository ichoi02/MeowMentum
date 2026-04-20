"""
Robust sensor fusion for rear-body orientation.

Strategy — per-frame cross-sensor confidence weighting:
  At each frame we compare how much the B IMU and FK each changed since the
  previous frame.  When one sensor shows a large step the other does NOT show,
  that step is a misread (encoder glitch or IMU spike).  We SLERP the two
  absolute orientations toward whichever source moved LESS, frame by frame.

  w_fk  = sa_imu / (sa_imu + sa_fk + eps)   → 1 when IMU spiked, 0 when FK spiked
  fused = SLERP(q_imu, q_fk, w_fk)

  A short median filter on the raw step angles prevents the trust signal itself
  from being fooled by a single-frame outlier.
"""
import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import median_filter
from matplotlib import pyplot as plt

DATA_DIR    = './telemetry'
PLOT_DIR    = './data_analysis/plots_fusion'
REPORT_PATH = './data_analysis/report_fusion.csv'

SMOOTH_WINDOW = 3   # median filter width for step-angle trust computation
EPS           = 0.5 # deg — floor to avoid 0/0 when both sensors are still


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist.")
    return pd.read_csv(path)


def impact_detector(accs):
    time_at_impact = np.max(np.where(accs == np.max(np.max(accs, axis=0))))
    which_body     = np.argmax(np.where(accs == np.max(np.max(accs, axis=0))))
    return int(time_at_impact), int(which_body)


def robust_fuse(b_rots: R, fk_rots: R,
                smooth_window: int = SMOOTH_WINDOW,
                eps: float = EPS):
    """
    Per-frame adaptive fusion of B IMU and FK orientations.

    At each frame, the sensor whose orientation changed LESS since the
    previous frame is trusted MORE.  A spike in one source that the other
    does not confirm is treated as a misread and suppressed.

    Returns
    -------
    fused : R   — fused rotation sequence (length N)
    w_fk  : (N,) float  — per-frame weight toward FK (0 = pure IMU, 1 = pure FK)
    """
    n = len(b_rots)

    # Step magnitude (deg) for each source at each frame
    sa_imu = np.zeros(n)
    sa_fk  = np.zeros(n)
    for i in range(1, n):
        sa_imu[i] = np.degrees((b_rots[i - 1].inv()  * b_rots[i]).magnitude())
        sa_fk[i]  = np.degrees((fk_rots[i - 1].inv() * fk_rots[i]).magnitude())

    # Smooth so a 1-frame outlier doesn't corrupt the trust signal itself
    sa_imu_s = median_filter(sa_imu, size=smooth_window)
    sa_fk_s  = median_filter(sa_fk,  size=smooth_window)

    # Trust weight toward FK: large when IMU step is relatively bigger
    w_fk = sa_imu_s / (sa_imu_s + sa_fk_s + eps)

    # Frame-by-frame SLERP: q_imu * (q_imu⁻¹ * q_fk)^w_fk
    fused = [
        b_rots[i] * (b_rots[i].inv() * fk_rots[i]) ** w_fk[i]
        for i in range(n)
    ]
    return R.concatenate(fused), w_fk


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

records = []

for date in ['Apr18']:
    for trial in ['r45', 'r90', 'r180']:
        for rep in ['1', '2', '3']:

            for file_name in os.listdir(DATA_DIR):
                if file_name.startswith(f"{date}_{trial}_{rep}"):
                    print(f"Loading {file_name}...")
                    df = load_data(f"{DATA_DIR}/{file_name}")
                    break
            else:
                raise FileNotFoundError(f"No file found for {date} {trial} {rep}")

            # --- timing ---
            index_at_impact, _ = impact_detector(df[['F_ACC', 'B_ACC']].to_numpy())
            time_at_impact  = df['Time'].iloc[index_at_impact]
            time_at_release = df['Time'].iloc[0]
            duration        = time_at_impact - time_at_release
            fall_distance   = 9.81 / 2 * duration ** 2

            print(f"  fall distance: {fall_distance:.2f} m, duration: {duration:.2f} s")
            if fall_distance <= 2.5:
                print(f"  SKIPPED: fall distance too short ({fall_distance:.2f} m < 2.5 m)")
                continue

            time_at_2p5m = np.sqrt(2 * 2.5 / 9.81) + time_at_release
            end_idx      = np.argmin(np.abs(df['Time'] - time_at_2p5m))
            time         = df['Time'].iloc[0:end_idx].to_numpy()

            # --- raw quaternions ---
            f_quats_raw = df[['F_Q0','F_Q1','F_Q2','F_Q3']].iloc[0:end_idx].to_numpy()
            b_quats_raw = df[['B_Q0','B_Q1','B_Q2','B_Q3']].iloc[0:end_idx].to_numpy()

            f_rot  = R.from_quat(f_quats_raw, scalar_first=True)
            b_rots = R.from_quat(b_quats_raw, scalar_first=True)

            # --- FK rear orientation ---
            _zeros  = np.zeros(end_idx)
            r_rot1  = R.from_rotvec(np.column_stack([df['F_M1'].iloc[0:end_idx].to_numpy(), _zeros, _zeros]))
            r_pitch = R.from_rotvec(np.column_stack([_zeros, df['F_M2'].iloc[0:end_idx].to_numpy(), _zeros]))
            r_rot2  = R.from_rotvec(np.column_stack([df['B_M2'].iloc[0:end_idx].to_numpy(), _zeros, _zeros]))
            fk_rots = f_rot * r_rot1 * r_pitch * r_rot2

            # --- fused ---
            fused_rots, w_fk = robust_fuse(b_rots, fk_rots)

            # --- Euler angles (xyz, degrees) ---
            f_ori     = f_rot.as_euler('xyz', degrees=True)
            b_ori     = b_rots.as_euler('xyz', degrees=True)
            fk_ori    = fk_rots.as_euler('xyz', degrees=True)
            fused_ori = fused_rots.as_euler('xyz', degrees=True)

            # --- angular velocity (deg/s) ---
            dt           = np.diff(time)
            f_angvel     = np.diff(np.unwrap(f_ori,     axis=0, period=360), axis=0) / dt[:, None]
            b_angvel     = np.diff(np.unwrap(b_ori,     axis=0, period=360), axis=0) / dt[:, None]
            fk_angvel    = np.diff(np.unwrap(fk_ori,    axis=0, period=360), axis=0) / dt[:, None]
            fused_angvel = np.diff(np.unwrap(fused_ori, axis=0, period=360), axis=0) / dt[:, None]

            # --- plot ---
            plot_dir = os.path.join(PLOT_DIR, date, trial)
            os.makedirs(plot_dir, exist_ok=True)

            fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 10))
            fig.suptitle(f'Rear-body roll — {date} {trial} rep{rep}')

            ax = axes[0]
            ax.plot(time, b_ori[:, 0],    label='B IMU',  color='tab:orange', alpha=0.5)
            ax.plot(time, fk_ori[:, 0],   label='FK',     color='tab:green',  alpha=0.5, linestyle='--')
            ax.plot(time, fused_ori[:, 0],label='Fused',  color='tab:blue',   linewidth=2)
            ax.set_ylabel('Roll (deg)'); ax.set_title('Roll angle'); ax.legend()

            ax = axes[1]
            ax.plot(time[:-1], b_angvel[:, 0],    label='B IMU',  color='tab:orange', alpha=0.5)
            ax.plot(time[:-1], fk_angvel[:, 0],   label='FK',     color='tab:green',  alpha=0.5, linestyle='--')
            ax.plot(time[:-1], fused_angvel[:, 0],label='Fused',  color='tab:blue',   linewidth=2)
            ax.set_ylabel('Roll rate (deg/s)'); ax.legend()

            ax = axes[2]
            ax.plot(time, w_fk,   label='FK weight',  color='tab:green')
            ax.plot(time, 1-w_fk, label='IMU weight', color='tab:orange')
            ax.set_ylabel('Trust weight'); ax.set_xlabel('Time (s)'); ax.legend()
            ax.set_title('Per-frame sensor trust (1 = fully trusted)')

            plt.tight_layout()
            fig.savefig(os.path.join(plot_dir, f'rep{rep}_fusion.png'), dpi=150)
            plt.close(fig)

            # --- report ---
            records.append({
                'date': date, 'trial': trial, 'rep': rep,
                'F_roll_2p5m':              round(f_ori[-1, 0],       2),
                'B_fused_roll_2p5m':        round(fused_ori[-1, 0],   2),
                'F_rollrate_2p5m':          round(f_angvel[-1, 0],    2),
                'B_fused_rollrate_2p5m':    round(fused_angvel[-1, 0],2),
                'F_roll_initial':           round(f_ori[0, 0],        2),
                'B_fused_roll_initial':     round(fused_ori[0, 0],    2),
                'F_rollrate_initial':       round(f_angvel[0, 0],     2),
                'B_fused_rollrate_initial': round(fused_angvel[0, 0], 2),
            })

report_df = pd.DataFrame(records)
report_df.to_csv(REPORT_PATH, index=False)
print(f"\nFusion report saved to {REPORT_PATH} ({len(report_df)} trials)")
print(report_df.to_string(index=False))
