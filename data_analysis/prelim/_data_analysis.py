import os
import pandas as pd
import numpy as np
from pdb import set_trace as st
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

DATA_DIR = './telemetry'

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data as a DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise Exception(f"An error occurred while loading the data: {e}")
    
def impact_detector(accs):
    """
    Detect the impact of a fall based on the accelerometer data.

    
    """

    time_at_impact = np.max(np.where(accs == np.max(np.max(accs, axis=0))))
    which_body = np.argmax(np.where(accs == np.max(np.max(accs, axis=0))))

    return int(time_at_impact), int(which_body)
    
def low_pass_filter(data, cutoff_freq, fs):
    """
    Apply a low-pass Butterworth filter to the data.

    Parameters:
    data (np.ndarray): The input data to be filtered.
    cutoff_freq (float): The cutoff frequency of the filter in Hz.
    fs (float): The sampling frequency of the data in Hz.

    Returns:
    np.ndarray: The filtered data.
    """
    from scipy.signal import butter, filtfilt

    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(N=4, Wn=normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data, axis=0)

    return filtered_data


# date = 'Apr18'
# trial = 'r180'
# rep = '1'

# file_name = 'Apr18_r45_3'
# df = load_data(f"{DATA_DIR}/{file_name}.csv")

for date in ['Apr18']:
    for trial in ['r45', 'r90', 'r180']:
        for rep in ['1', '2', '3']:
            for file_name in os.listdir(DATA_DIR):
                if file_name.startswith(f"{date}_{trial}_{rep}"):
                    print(f"Loading data from {file_name}...")
                    df = load_data(f"{DATA_DIR}/{file_name}")
                    break
                else:            
                    raise FileNotFoundError(f"No file found for {date} {trial} {rep}")

            #%% Determining the start and end of the fall
            # Fall detection (to determine the duration of the fall)
            index_at_impact, which_body = impact_detector(df[['F_ACC', 'B_ACC']].to_numpy())
            time_at_impact = df['Time'].iloc[index_at_impact]
            time_at_release = df['Time'].iloc[0]
            duration = time_at_impact - time_at_release
            fall_distance = 9.81 / 2 * duration**2
            print(f"fall distance: {fall_distance:.2f} m, duration: {duration:.2f} s")
            assert fall_distance > 2.5, "The fall distance is too short !!"

            time_at_2p5m = np.sqrt(2 * 2.5 / 9.81) + time_at_release

            end_idx = np.argmin(np.abs(df['Time'] - time_at_2p5m))
            start = time_at_release; end = time_at_2p5m
            time = df['Time'].iloc[0:end_idx].to_numpy()

            #%% import the quaternions to determine the orientation of the cat at the moment of impact
            f_quat = ['F_Q0', 'F_Q1', 'F_Q2', 'F_Q3']; b_quat = ['B_Q0', 'B_Q1', 'B_Q2', 'B_Q3']

            f_ori = df[f_quat].iloc[0:end_idx].to_numpy()
            b_ori = df[b_quat].iloc[0:end_idx].to_numpy()

            # axis order: x = roll, y = pitch, z = yaw
            f_rot = R.from_quat(f_ori, scalar_first=True)
            f_ori = f_rot.as_euler('xyz', degrees=True)
            b_ori = R.from_quat(b_ori, scalar_first=True).as_euler('xyz', degrees=True)

            # FK-based rear orientation: front_IMU * rot1(F_M1, x) * pitch(F_M2, y) * rot2(B_M2, x)
            _zeros = np.zeros(end_idx)
            r_rot1  = R.from_rotvec(np.column_stack([df['F_M1'].iloc[0:end_idx].to_numpy(), _zeros, _zeros]))
            r_pitch = R.from_rotvec(np.column_stack([_zeros, df['F_M2'].iloc[0:end_idx].to_numpy(), _zeros]))
            r_rot2  = R.from_rotvec(np.column_stack([df['B_M2'].iloc[0:end_idx].to_numpy(), _zeros, _zeros]))
            fk_ori  = (f_rot * r_rot1 * r_pitch * r_rot2).as_euler('xyz', degrees=True)

            # # filter
            # filt_f_ori = low_pass_filter(f_ori, cutoff_freq=20, fs=50)
            # filt_b_ori = low_pass_filter(b_ori, cutoff_freq=20, fs=50)

            # plt.plot(time, f_ori[:, 0], label='F roll')
            # plt.plot(time, b_ori[:, 1], label='B pitch')

            # plt.plot(time, filt_f_ori[:, 0], label='F roll (filtered)', linestyle='--')
            # plt.plot(time, filt_b_ori[:, 1], label='B pitch (filtered)', linestyle='--')

            # plt.show()

            #%% Checking the performance
            '''
            Main metrics
            1. orientation at 2.5m -> check the average of front and back body orientation at the moment of impact
            2. angular velocity at 2.5m -> check the average of front and back body angular velocity at the moment of impact

            Side metrics
            1. spine angle at impact: check the angle difference between the front and back body at the moment of impact (was it straight or not?)

            Sanity check
            1. Check the orientation at release; was it close to the desired orientation?
            2. Check the angular velocity at release; was it close to zero?
            '''


            def check_imu_reliability(angvel, label):
                roll_rate = angvel[:, 0]
                MOONSHOT  = 2000  # deg/s — physically implausible in a single frame
                SLOPE     = 500   # deg/s — threshold for "sustained high" rate

                issues = []
                moonshot_frames = np.where(np.abs(roll_rate) > MOONSHOT)[0]
                if len(moonshot_frames):
                    issues.append(f"moonshot: {len(moonshot_frames)} frame(s) |rate| > {MOONSHOT} deg/s "
                                f"(max {np.abs(roll_rate).max():.0f} deg/s at frame {moonshot_frames[0]})")

                jitter = np.std(np.diff(roll_rate))  # std of acceleration = jitter proxy
                if jitter > 300:
                    issues.append(f"jitter: roll-rate std-of-diff = {jitter:.0f} deg/s²")

                high_slope_frac = np.mean(np.abs(roll_rate) > SLOPE)
                if high_slope_frac > 0.3:
                    issues.append(f"high slope: {high_slope_frac*100:.0f}% of frames |rate| > {SLOPE} deg/s")

                status = "UNRELIABLE ⚠" if issues else "OK ✓"
                print(f"  {label}: {status}")
                for iss in issues:
                    print(f"    ! {iss}")

            # Angular velocity (deg/s) via numerical differentiation
            dt = np.diff(time)
            f_angvel  = np.diff(np.unwrap(f_ori,  axis=0, period=360), axis=0) / dt[:, None]
            b_angvel  = np.diff(np.unwrap(b_ori,  axis=0, period=360), axis=0) / dt[:, None]
            fk_angvel = np.diff(np.unwrap(fk_ori, axis=0, period=360), axis=0) / dt[:, None]

            # Plot roll angle + roll rate for all three sources
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
            ax1.plot(time,      f_ori[:, 0],  label='F IMU');  ax1.plot(time, b_ori[:, 0], label='B IMU')
            ax1.plot(time,      fk_ori[:, 0], label='FK rear', linestyle='--')
            ax1.set_ylabel('Roll (deg)'); ax1.legend(); ax1.set_title('Roll angle')

            ax2.plot(time[:-1], f_angvel[:, 0],  label='F IMU')
            ax2.plot(time[:-1], b_angvel[:, 0],  label='B IMU')
            ax2.plot(time[:-1], fk_angvel[:, 0], label='FK rear', linestyle='--')
            ax2.set_ylabel('Roll rate (deg/s)'); ax2.set_xlabel('Time (s)'); ax2.legend()
            plt.tight_layout(); plt.show()

            # Main metrics at 2.5m
            print("=== Main Metrics at 2.5m ===")
            print(f"F  IMU  orientation  [roll, pitch, yaw] (deg):   {f_ori[-1].round(1)}")
            print(f"B  IMU  orientation  [roll, pitch, yaw] (deg):   {b_ori[-1].round(1)}")
            print(f"FK rear orientation  [roll, pitch, yaw] (deg):   {fk_ori[-1].round(1)}")
            print(f"F  IMU  angular vel  [roll, pitch, yaw] (deg/s): {f_angvel[-1].round(1)}")
            print(f"B  IMU  angular vel  [roll, pitch, yaw] (deg/s): {b_angvel[-1].round(1)}")
            print(f"FK rear angular vel  [roll, pitch, yaw] (deg/s): {fk_angvel[-1].round(1)}")

            # Side metric: spine bending angle (FK is the ground truth for rear position)
            spine_angle_imu = f_ori[-1] - b_ori[-1]
            spine_angle_fk  = f_ori[-1] - fk_ori[-1]
            print("\n=== Side Metrics ===")
            print(f"Spine angle (B IMU) [roll, pitch, yaw] (deg):   {spine_angle_imu.round(1)}")
            print(f"Spine angle (FK)    [roll, pitch, yaw] (deg):   {spine_angle_fk.round(1)}")

            # Sanity checks at release
            print("\n=== Sanity Checks at Release ===")
            print(f"F  IMU  orientation  [roll, pitch, yaw] (deg):   {f_ori[0].round(1)}")
            print(f"B  IMU  orientation  [roll, pitch, yaw] (deg):   {b_ori[0].round(1)}")
            print(f"FK rear orientation  [roll, pitch, yaw] (deg):   {fk_ori[0].round(1)}")
            print(f"F  IMU  angular vel  [roll, pitch, yaw] (deg/s): {f_angvel[0].round(1)}")
            print(f"B  IMU  angular vel  [roll, pitch, yaw] (deg/s): {b_angvel[0].round(1)}")
            print(f"FK rear angular vel  [roll, pitch, yaw] (deg/s): {fk_angvel[0].round(1)}")

            print("\n=== B IMU Reliability ===")
            check_imu_reliability(b_angvel, "B IMU roll")


            #%% MuJoCo playback (FK + rear IMU ghost) — runs via mjpython subprocess
            import subprocess
            playback_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mujoco_playback.py')
            subprocess.Popen([
                'mjpython', playback_script,
                '--file', f'{DATA_DIR}/{file_name}.csv',
                '--end_idx', str(end_idx),
            ])



