import os
from scipy.spatial.transform import Rotation as R

# Edge / Pi: ONNX probes GPU on load; keep stderr quiet (4 = fatal in typical ORT builds).
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "4")

# Some Pi/conda setups set LD_PRELOAD to a missing OpenBLAS path; drop broken entries
# before NumPy so this process and ctypes-loaded libs don't trip over them.
def _strip_broken_ld_preload():
    p = os.environ.get("LD_PRELOAD", "")
    if not p:
        return
    sep = ":" if ":" in p else " "
    kept = []
    for x in p.split(sep):
        x = x.strip()
        if not x:
            continue
        if x.startswith("-") or x.startswith(" "):
            kept.append(x)
            continue
        if ("/" in x or x.endswith(".so") or ".so." in x) and not os.path.isfile(x):
            continue
        kept.append(x)
    if kept:
        os.environ["LD_PRELOAD"] = sep.join(kept)
    else:
        os.environ.pop("LD_PRELOAD", None)


_strip_broken_ld_preload()

import serial
import serial.tools.list_ports
import termios
import time
import math
import io
import argparse
import numpy as np
import csv
import threading

# --- CONFIGURATION ---
SN_FRONT = "18451300" 
SN_BACK  = "18452630"

BAUD_RATE = 115200
LOOP_HZ = 50

# Bound serial draining per tick (~50Hz telemetry per board); unbounded reads starve the Python loop.
SERIAL_DRAIN_MAX_LINES = 32

# Drop Trigger
USE_DROP_TRIGGER = True
drop_acc_threshold = 1.0   # m/s^2
time_max = 0.85            

# Each motor sign differnece (Front and Rear Roll Sign Difference)
front_roll_sign = -1.0
rear_roll_sign = +1.0
tail_sign = +1.0


# Phase Number
PHASE_BEND_SPINE = 0
PHASE_RIGHTING = 1
PHASE_SETTLE = 2

# Phase Config
spine_target = 88.0   # for PHASE_BEND_SPINE (deg)
spine_target_threshold = 10.0   # How close to target before switching to next phase (deg)
roll_threshold = 10.0   # Below this err move to settle phase. (deg)

# Gains
Kp_spine = 1.0
Kp_settle_roll = 0.8
Kp_settle_roll_diff = 0.3
Kp_tail = 0.8

def _open_teensy_serial(port_path: str) -> serial.Serial:
    # write_timeout=0: do not block indefinitely if USB TX is wedged (pair with non-blocking Teensy TX).
    return serial.Serial(
        port_path,
        BAUD_RATE,
        timeout=0.005,
        write_timeout=0,
    )


debug_action = [0.0, 0.0, 0.0, 0.0]

class TeensyInterface:
    def __init__(self, port_path, name):
        self.port_path = port_path
        self.name = name
        self._rx_remainder = b""
        self.ser = _open_teensy_serial(port_path)

        self.quat = [1.0, 0.0, 0.0, 0.0]       # w x y z
        self.gyro = np.zeros(3, dtype=float)   # Gyro. gx, gy, gz
        self.m1_rad = 0.0                        
        self.m2_rad = 0.0
        self.acc_mag = 0.0

    def reopen_serial(self):
        try:
            self.ser.close()
        except (OSError, AttributeError, ValueError):
            pass
        time.sleep(0.2)
        self.ser = _open_teensy_serial(self.port_path)
        self._rx_remainder = b""
        time.sleep(0.05)

    def _is_serial_gone(self, err):
        en = getattr(err, "errno", None)
        if en is None and getattr(err, "args", None):
            a0 = err.args[0]
            if isinstance(a0, OSError):
                en = a0.errno
        # EIO, EBADF, ENODEV, EPIPE — typical when USB CDC drops or device resets.
        return en in (5, 9, 19, 32)

    def flush_input_safe(self):
        """tcflush can raise EIO if USB CDC reset/re-enumerated; reopen and retry once."""
        try:
            self.ser.reset_input_buffer()
        except (termios.error, OSError) as e:
            print(f"{self.name}: input flush failed ({e!r}); reopening serial…")
            self.reopen_serial()
            try:
                self.ser.reset_input_buffer()
            except (termios.error, OSError):
                pass
        self._rx_remainder = b""
        self.quat = [1.0, 0.0, 0.0, 0.0] 
        self.gyro = np.zeros(3, dtype=float)
        self.m1_rad = 0.0
        self.m2_rad = 0.0
        self.acc_mag = 9.8

    def reset_encoders(self):
        """Sends the command to zero out the hardware encoder counts."""
        try:
            self.ser.write(b"RESET\n")
            print(f"{self.name}: Reset encoders.")
        except (OSError, serial.SerialException) as e:
            if self._is_serial_gone(e):
                self.reopen_serial()
    
    def reset_IMU(self):
        try:
            self.ser.write(b"RESET_IMU\n")
            print(f"{self.name}: Reset IMU.")
        except (OSError, serial.SerialException) as e:
            if self._is_serial_gone(e):
                self.reopen_serial()
    
    def reset_I2C(self):
        try:
            self.ser.write(b"RESET_I2C\n")
            print(f"{self.name}: Reset I2C.")
        except (OSError, serial.SerialException) as e:
            if self._is_serial_gone(e):
                self.reopen_serial()

    def reboot_teensy(self):
        try:
            self.ser.write(b"REBOOT\n")
            print(f"{self.name}: Reboot.")
        except (OSError, serial.SerialException) as e:
            if self._is_serial_gone(e):
                self.reopen_serial()

    def stop_all_motors(self):
        try:
            self.ser.write(b"STOP\n")
            print(f"{self.name}: Stopping motors.")
        except (OSError, serial.SerialException) as e:
            if self._is_serial_gone(e):
                self.reopen_serial()

    def start_all_motors(self):
        try:
            self.ser.write(b"START\n")
            print(f"{self.name}: Starting motors.")
        except (OSError, serial.SerialException) as e:
            if self._is_serial_gone(e):
                self.reopen_serial()

    def update_sensor_data(self):
        """Reads buffered lines to get the freshest IMU and encoder data."""
        try:
            waiting = self.ser.in_waiting
        except (OSError, serial.SerialException) as e:
            if self._is_serial_gone(e):
                print(f"{self.name}: serial I/O in update_sensor_data; reopening…")
                self.reopen_serial()
            return

        if waiting <= 0:
            return

        max_bytes = min(waiting, 4096)
        try:
            raw = self.ser.read(max_bytes)
        except (OSError, serial.SerialException) as e:
            if self._is_serial_gone(e):
                self.reopen_serial()
            return

        if not raw:
            return

        self._rx_remainder += raw
        if b"\n" not in self._rx_remainder:
            if len(self._rx_remainder) > 8192:
                self._rx_remainder = self._rx_remainder[-4096:]
            return

        chunks = self._rx_remainder.split(b"\n")
        self._rx_remainder = chunks[-1]
        complete = chunks[:-1][-SERIAL_DRAIN_MAX_LINES:]
        latest_line = None
        for c in complete:
            s = c.decode("utf-8", errors="ignore").strip()
            if s:
                latest_line = s
        # print(latest_line)
        if latest_line:
            parts = latest_line.split(',')
            if len(parts) == 10:
                try:
                    self.quat = [float(x) for x in parts[:4]]
                    self.quat = self.align_imu_quaternions(np.array([self.quat]), self.name)
                    self.gyro =  np.array([float(x) for x in parts[4:7]], dtype=float) 
                    self.m1_rad = float(parts[7])
                    self.m2_rad = float(parts[8])
                    self.acc_mag = float(parts[9])
                except ValueError:
                    pass

    def set_motors(self, m1, m2):
        """Sends target motor angles (2dp) to Teensy: 'm1,m2\\n'."""
        cmd = f"{m1:.2f},{m2:.2f}\n"
        try:
            self.ser.write(cmd.encode('utf-8'))
        except (OSError, serial.SerialException) as e:
            if self._is_serial_gone(e):
                print(f"{self.name}: serial I/O error in set_motors; reopening…")
                self.reopen_serial()
    
    def align_imu_quaternions(self, quats_wxyz, imu_type):
        r_raw = R.from_quat(quats_wxyz, scalar_first=True)
        
        if imu_type == 'Front':
            r_align = R.from_euler('xyz', [0, 0, 90], degrees=True) # -90은 아니겠지?
            
        elif imu_type == 'Back':
            r_align = R.from_euler('xyz', [180, 0, -90], degrees=True)
            
        r_global = r_raw * r_align
        
        aligned_wxyz = r_global.as_quat(scalar_first=True)
        return aligned_wxyz.squeeze(0)


def get_port_by_sn(serial_number):
    for port in serial.tools.list_ports.comports():
        if port.serial_number == serial_number:
            return port.device
    return None


def keyboard_input_thread():
    global debug_action
    while True:
        try:
            user_in = input().split()
            if len(user_in) == 2:
                motor_num = int(user_in[0])
                target_angle = float(user_in[1])
                debug_action[motor_num - 1] = target_angle
                print(f"Updated Motor {motor_num} to {target_angle}")
        except Exception as e:
            print(f"Invalid input. Try again. ({e})")

######## Will be adding some more func for model-based controller
def get_joint_state(front, back):
    q_front_roll = front.m1_rad
    q_spine      = front.m2_rad
    q_tail       = back.m1_rad
    q_rear_roll  = back.m2_rad

    return {
        "q_front_roll": q_front_roll,
        "q_spine": q_spine,
        "q_tail": q_tail,
        "q_rear_roll": q_rear_roll,
        "quat_front": np.asarray(front.quat, dtype=float),
        "quat_back":  np.asarray(back.quat, dtype=float),
        "gyro_front": np.asarray(front.gyro, dtype=float),
        "gyro_back":  np.asarray(back.gyro, dtype=float),
        "acc_front": float(front.acc_mag),
        "acc_back":  float(back.acc_mag),
    }

# Convert quaternions to euler angles (roll, pitch, yaw)   
# euler(radians)
def quat_wxyz_to_euler_xyz(q_wxyz):
    r = R.from_quat(q_wxyz, scalar_first=True)
    return r.as_euler('xyz', degrees=False)

# Compute Rear roll angle error (world frame) 
# Just getting rear roll angle. Would be equal to error.
def compute_roll_error(state):
    roll_back, _, _ = quat_wxyz_to_euler_xyz(state["quat_back"])
    return roll_back

# 이거 그냥 나중에 롤이랑 합쳐 버리자
# Compute pitch error (world frame)
def compute_pitch_error(state):
    _, pitch_back, _ = quat_wxyz_to_euler_xyz(state["quat_back"])
    return pitch_back


# Coupled roll / Rear and front body roll differnece.
def compute_derived_state(state):
    qf = state["q_front_roll"]
    qr = state["q_rear_roll"]

    phi_coupl = 0.5 * (front_roll_sign * qf + rear_roll_sign * qr)   # Coupled roll component (rad)   
    phi_diff = (front_roll_sign * qf - rear_roll_sign * qr)    # Front and rear body roll difference

    return {**state, "phi_coupl": phi_coupl, "phi_diff": phi_diff}



########


def main():
    parser = argparse.ArgumentParser(description="Teensy Control Loop with Telemetry Logging")
    parser.add_argument(
        '--debug',
        action='store_true',
        help="P-control test: sinusoidal joint targets within ±DEBUG_JOINT_SWING_DEG (see CONFIG)",
    )
    args = parser.parse_args()

    path_front = get_port_by_sn(SN_FRONT)
    path_back = get_port_by_sn(SN_BACK)
    
    if not path_front or not path_back:
        print(f"Error finding boards! Front: {path_front}, Back: {path_back}")
        return

    print("Connecting to boards...")
    front = TeensyInterface(path_front, "Front")
    back = TeensyInterface(path_back, "Back")
    time.sleep(1) # Wait for serial connection to establish

    print("Rebooting Teensy")
    front.reboot_teensy()
    front.ser.close()
    time.sleep(2)
    back.reboot_teensy()
    back.ser.close()
    time.sleep(2)

    path_front = get_port_by_sn(SN_FRONT)
    path_back = get_port_by_sn(SN_BACK)
    front = TeensyInterface(path_front, "Front")
    back = TeensyInterface(path_back, "Back")
    print(f"{path_front}, {path_back}")
    time.sleep(1) # Wait for serial connection to establish

    # Stop all motors
    front.stop_all_motors()
    back.stop_all_motors()

    # --- ZERO THE ENCODERS ---
    print("Zeroing motor encoders...")
    front.reset_encoders()
    back.reset_encoders()
    time.sleep(0.5)  # Let Teensy drain RESET; CDC can return EIO on flush if too early

    print("Zeroing IMU quaternions...")
    front.reset_IMU()
    back.reset_IMU()
    time.sleep(0.5)

    # print("Resetting I2C comms...")
    # front.reset_I2C()
    # back.reset_I2C()
    # time.sleep(0.5)
    
    # Flush any stale data that was transmitted before the reset happened
    front.flush_input_safe()
    back.flush_input_safe()



    # front.start_all_motors()
    # back.start_all_motors()
    print("Press any key to start.")
    input()

    # if not args.debug:
    #     print("Loading ONNX Model...")
    #     import onnxruntime as ort
    #     try:
    #         ort.set_default_logger_severity(3)
    #     except (AttributeError, TypeError):
    #         pass
    #     # FIXME: Replace with your actual model filename if different
    #     ort_session = ort.InferenceSession("cat_controller.onnx")
    
    loop_period = 1.0 / LOOP_HZ
    start_time = time.time()
    log = []

    # Set Initial Phase    
    phase = PHASE_BEND_SPINE

    controller_started = False
    controller_start_time = None

    # Debug print timer
    last_debug_print = 0.0
    
    if args.debug:
        print("Debug mode initiated: enter '[motor num] [target angle]'")
        threading.Thread(target=keyboard_input_thread, daemon=True).start()

    print(f"Starting loop at {LOOP_HZ}Hz. Press Ctrl+C to quit.")
    try:
        while(True): # FIXME: need to limit running time  >>>이거 넣었니 이탁아?
            loop_start = time.time()
            t = loop_start - start_time


            front.update_sensor_data()
            back.update_sensor_data()
            
            action = [0, 0, 0, 0]

            pitch_err_log = float("nan")
            roll_err_log = float("nan")
            phi_diff_log = float("nan")
            phi_coupl_log = float("nan")

            if args.debug:
                action = list(debug_action) 
            else:
                # Before trigger : Hold [0 0 0 0] Pose
                if USE_DROP_TRIGGER and not controller_started:
                    action = [0, 0, 0, 0]

                    # Controller On : Beging Bend Spine Phase
                    #if back.acc_mag < drop_acc_threshold:
                    if True:
                        controller_started = True
                        controller_start_time = loop_start
                        phase = PHASE_BEND_SPINE
                        print(f"DROP TRIGGERED at t={t:.3f}s | acc_trigger_mag={back.acc_mag:.3f}")

                else:
                    # After trigger : Run Controller

                    # timeout measured from controller start
                    ctrl_t = loop_start - controller_start_time if controller_start_time is not None else 0.0
                    if ctrl_t >= time_max:
                        print(f"Timeout reached ({time_max:.1f} s after trigger). Stopping...")
                        front.stop_all_motors()
                        back.stop_all_motors()
                        break

                    ## Phase Controller
                    state = get_joint_state(front, back)
                    state = compute_derived_state(state)
                    roll_err = compute_roll_error(state)
                    pitch_err = compute_pitch_error(state)

                    # Logging for debug
                    pitch_err_log = pitch_err
                    roll_err_log = roll_err
                    phi_diff_log = state["phi_diff"]
                    phi_coupl_log = state["phi_coupl"]

                    cmd_front_roll = 0.0
                    cmd_rear_roll = 0.0
                    cmd_spine = 0.0
                    cmd_tail = 0.0

                    if phase == PHASE_BEND_SPINE:
                        cmd_front_roll = 0.0
                        cmd_rear_roll = 0.0
                        cmd_spine = np.deg2rad(spine_target)
                        cmd_tail = 0.0

                        if abs(state["q_spine"] - np.deg2rad(spine_target)) < np.deg2rad(spine_target_threshold):
                            phase = PHASE_RIGHTING

                    elif phase == PHASE_RIGHTING:
                        u_roll = -1.0 * roll_err 
                        cmd_front_roll = front_roll_sign * u_roll
                        cmd_rear_roll = rear_roll_sign * u_roll
                        cmd_spine = Kp_spine*np.deg2rad(spine_target)
                        cmd_tail = tail_sign * Kp_tail * pitch_err

                        if abs(roll_err) < np.deg2rad(roll_threshold): 
                            phase = PHASE_SETTLE  

                    # If 여기서 개빡츼게 -> Divide settling phase into 2 diff phase.
                    elif phase == PHASE_SETTLE:
                        u_roll = -Kp_settle_roll * roll_err - Kp_settle_roll_diff * state["phi_diff"]
                        cmd_front_roll = front_roll_sign * u_roll
                        cmd_rear_roll = rear_roll_sign * u_roll
                        cmd_spine = 0.0
                        cmd_tail = 0.0

                    cmd_front_roll = np.clip(cmd_front_roll, -6.28, 6.28)
                    cmd_rear_roll  = np.clip(cmd_rear_roll,  -6.28, 6.28)
                    cmd_tail       = np.clip(cmd_tail,       -1.55, 1.55)
                    cmd_spine      = np.clip(cmd_spine,      -1.55, 1.55)

                    action = [cmd_front_roll, cmd_spine, cmd_tail, cmd_rear_roll]  

            # Logging for debug    
            phase_log = phase
            cmd_front_roll_log = action[0]
            cmd_spine_log      = action[1]
            cmd_tail_log       = action[2]
            cmd_rear_roll_log  = action[3]

            front.set_motors(action[0], action[1])
            back.set_motors(action[2], action[3])

            # Debug print
            if t - last_debug_print >= 0.1:   # print every 0.1 s
                print(
                    f"t={t:.2f} | phase={phase_log} | roll_err={roll_err_log:.3f} | pitch_err={pitch_err_log:.3f} | "
                    f"phi_coupl={phi_coupl_log:.3f} | phi_diff={phi_diff_log:.3f} | "
                    f"cmd=[fr={cmd_front_roll_log:.3f}, sp={cmd_spine_log:.3f}, "
                    f"ta={cmd_tail_log:.3f}, rr={cmd_rear_roll_log:.3f}]"
                )
                last_debug_print = t



            log.append([
                round(t, 4),
                front.quat[0], front.quat[1], front.quat[2], front.quat[3],
                front.gyro[0], front.gyro[1], front.gyro[2],
                front.m1_rad, front.m2_rad,
                front.acc_mag,
                action[0], action[1],
                back.quat[0], back.quat[1], back.quat[2], back.quat[3],
                back.gyro[0], back.gyro[1], back.gyro[2],
                back.m1_rad, back.m2_rad,
                back.acc_mag,
                action[2], action[3],

            ])

            # Enforce timing
            elapsed = time.time() - loop_start
            if elapsed < loop_period:
                time.sleep(loop_period - elapsed)
            else:
                print(f"WARNING: Loop missed deadline! Took {elapsed:.4f}s")

    except KeyboardInterrupt:
        print("\nExiting...")
        front.stop_all_motors()
        back.stop_all_motors()
        if log:
            filename = f"telemetry/telemetry_{int(time.time())}.csv"
            print(f"Saving {len(log)} records to {filename}...")
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Time", 
                                 "F_Q0", "F_Q1", "F_Q2", "F_Q3",
                                 "F_GX", "F_GY", "F_GZ",
                                 "F_M1", "F_M2", 
                                 "F_ACC", "Cmd_F1", "Cmd_F2", 
                                 "B_Q0", "B_Q1", "B_Q2", "B_Q3", 
                                 "B_GX", "B_GY", "B_GZ",
                                 "B_M1", "B_M2",
                                 "B_ACC", "Cmd_B1", "Cmd_B2"])
                writer.writerows(log)
            print("Done.")

if __name__ == "__main__":
    main()