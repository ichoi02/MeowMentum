"""Raspberry Pi main loop: fly an exported student policy on the robot.

    python hardware/controller.py --variant tail       # tail attached
    python hardware/controller.py --variant notail     # tail removed

Two Teensy 4.0 boards (hardware/PD_control_{front,back}) run the 1 kHz joint PD and
stream a sensor line every 20 ms. This script tares the IMU and zeroes the encoders,
waits for free fall (front |acc| < 3.5 m/s^2), then runs the ONNX student at 50 Hz
for CONTROL_DURATION, sends joint targets to both boards and writes
telemetry/<variant>_<unix time>.csv plus a .json sidecar naming the policy that flew.
See hardware/README.md for the procedure.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import termios
import threading
import time

import numpy as np
import serial
import serial.tools.list_ports
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Edge / Pi: ONNX probes GPU on load; keep stderr quiet (4 = fatal in typical ORT builds).
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "4")

import cat_env.env_util as util
from variants import VARIANTS, onnx_path

# --- CONFIGURATION ---
# USB serial numbers of the two boards (find yours with hardware/discover_teensy.py).
# flash_pd_control_teensy.sh parses these two lines.
SN_FRONT = "18451300"
SN_BACK  = "18452630"

BAUD_RATE = 115200
LOOP_HZ = 50

CONTROL_DURATION = 1.0    # sim episode: 50 steps x (1/50 s) = 1.0 s

# --- Policy / sim-matching parameters (MUST match cat_env/cat_env.py + model/cat.xml) ---
GRAV_DIM = 3
JOINT_DIM = 4
ACT_DIM = 3
STUDENT_OBS_DIM = GRAV_DIM + JOINT_DIM + ACT_DIM + 1   # 11 (distillation.STUDENT_OBS_DIM)
MAX_STEPS = 50            # cat_env.max_steps; CONTROL_DURATION / (1/LOOP_HZ)

# Per-channel action low-pass, [roll, pitch, tail]. MUST equal cat_env.filter_alpha:
# the policy is trained against this filter, so a mismatch changes the plant it acts on.
FILTER_ALPHA = np.array([0.1, 0.3, 0.1])

# Joint target ranges (rad), from model/cat.xml jnt_range:
ROLL_RANGE = 7.28         # rot1 / rot2   (+/-417 deg)
PITCH_RANGE = 1.57        # pitch         (+/-90 deg)
TAIL_RANGE = 1.9199       # tail          (+/-110 deg)

FREE_FALL_ACC = 3.5       # m/s^2; front |acc| below this starts the control loop

# Bound serial draining per tick (~50Hz telemetry per board); unbounded reads starve the Python loop.
SERIAL_DRAIN_MAX_LINES = 32

TELEMETRY_HEADER = ["Time", "F_Q0", "F_Q1", "F_Q2", "F_Q3", "F_M1", "F_M2", "F_ACC", "Cmd_F1", "Cmd_F2",
                    "B_Q0", "B_Q1", "B_Q2", "B_Q3", "B_M1", "B_M2", "B_ACC", "Cmd_B1", "Cmd_B2"]


def build_student_obs(frame, prev_action, step_frac):
    """[grav(3), joints(4)] + prev action(3) + step(1) -> the 11-dim student input.
    MUST match distillation.build_student_obs exactly."""
    return np.concatenate([frame, np.asarray(prev_action, dtype=float).ravel(),
                           [float(step_frac)]])


def zero_tail(frame, variant):
    """No-tail variant: force the tail joint reading to 0.

    `cat_notail.xml` keeps the tail joint but starves its motor to ctrlrange +-1e-6, so
    the tail never moves in training and the policy's third output gets no gradient.
    On the robot the tail joint still reads whatever the encoder reports, so the
    observation is pinned to the 0 the student was distilled on. The action is zeroed
    too (policy_action): commanding 0 makes the Teensy PD actively hold the tail joint
    at zero rather than letting it swing during the tumble.
    """
    if variant == "notail":
        frame = np.asarray(frame, dtype=float).copy()
        frame[GRAV_DIM + 3] = 0.0        # joints are [rot1, pitch, rot2, tail]
    return frame


def student_obs(front_quat, joints, prev_action, step_i, variant):
    """Student input (1 x 11, float32) from the aligned front IMU quaternion (w,x,y,z),
    the joints [rot1, pitch, rot2, tail] in rad, the previous raw action and the step
    count since release. MUST match distillation.py."""
    # Front IMU orientation -> projected gravity (yaw-invariant, 3-dim).
    front_grav = util.to_projected_gravity(front_quat)
    frame = zero_tail(np.concatenate([front_grav, joints]), variant)
    # Current frame + the action commanded last tick + episode progress.
    obs = build_student_obs(frame, prev_action, min(step_i / MAX_STEPS, 1.0))
    return np.array(obs, dtype=np.float32).reshape(1, -1)


def policy_action(ort_session, input_name, obs, variant):
    """ONNX inference -> raw normalized action [roll, pitch, tail] in [-1, 1].

    This RAW action is what the next observation carries as prev_action: cat_env
    stores prev_action before filtering, so this must match.
    """
    raw_action = ort_session.run(None, {input_name: obs})[0][0]
    raw_action = np.clip(np.asarray(raw_action, dtype=float), -1.0, 1.0)
    if variant == "notail":
        raw_action[2] = 0.0     # untrained channel; hold the tail at zero
    return raw_action


def joint_targets(raw_action, action_filt):
    """Per-channel low-pass, then map to joint targets. Both stages mirror
    cat_env.step exactly, in the same order.

    Returns ([rot1, pitch, tail, rot2] targets in rad, new filter state).
    """
    action_filt = FILTER_ALPHA * raw_action + (1.0 - FILTER_ALPHA) * action_filt

    # Map normalized action -> joint target angles (rad), using the sim
    # jnt_range from model/cat.xml. rot2 mirrors rot1 (rot2 = -rot1).
    roll  = util.map_value(float(action_filt[0]), -1, 1, -ROLL_RANGE, ROLL_RANGE)
    pitch = util.map_value(float(action_filt[1]), -1, 1, -PITCH_RANGE, PITCH_RANGE)
    tail  = util.map_value(float(action_filt[2]), -1, 1, -TAIL_RANGE, TAIL_RANGE)
    return [roll, pitch, tail, -roll], action_filt


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

        self.quat = [1.0, 0.0, 0.0, 0.0]
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
        """Tare the IMU (front board only; the back board ignores it)."""
        try:
            self.ser.write(b"RESET_IMU\n")
            print(f"{self.name}: Reset IMU.")
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
        """Reads buffered lines to get the freshest IMU and encoder data.

        Each board line is 'qr,qi,qj,qk,angle1,angle2,acc'; only the newest complete
        line is used. The back board has no IMU and always sends 1,0,0,0 and acc 0.
        """
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

        if latest_line:
            parts = latest_line.split(',')
            if len(parts) == 7:
                try:
                    self.quat = [float(x) for x in parts[:4]]
                    self.quat = self.align_imu_quaternions(np.array([self.quat]), self.name)
                    self.m1_rad = float(parts[4])
                    self.m2_rad = float(parts[5])
                    self.acc_mag = float(parts[6])
                except ValueError:
                    pass

    def set_motors(self, m1, m2):
        """Sends target joint angles (rad) to the Teensy as 'm1,m2\\n' with 4 decimals,
        finer than the encoder resolution (3.8 mrad on pitch/tail)."""
        cmd = f"{m1:.4f},{m2:.4f}\n"
        try:
            self.ser.write(cmd.encode('utf-8'))
        except (OSError, serial.SerialException) as e:
            if self._is_serial_gone(e):
                print(f"{self.name}: serial I/O error in set_motors; reopening…")
                self.reopen_serial()

    def align_imu_quaternions(self, quats_wxyz, imu_type):
        """Re-express the tared IMU rotation in the robot's axes (r_align^-1 * q * r_align;
        the front sensor axes are rotated 90 deg about z from the body axes). The back
        board's identity placeholder stays identity."""
        r_raw = R.from_quat(quats_wxyz, scalar_first=True)

        if imu_type == 'Front':
            r_align = R.from_euler('xyz', [0, 0, 90], degrees=True)

        elif imu_type == 'Back':
            r_align = R.from_euler('xyz', [180, 0, -90], degrees=True)

        r_global = r_align.inv() * r_raw * r_align

        aligned_wxyz = r_global.as_quat(scalar_first=True)
        return aligned_wxyz.squeeze(0)

def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()

def load_policy(path):
    """Open the ONNX student and check its input width. Returns (session, input name)."""
    import onnxruntime as ort
    try:
        ort.set_default_logger_severity(3)
    except (AttributeError, TypeError):
        pass

    ort_session = ort.InferenceSession(path)
    input_name = ort_session.get_inputs()[0].name

    # Fail loudly here rather than on the first inference mid-drop.
    onnx_dim = ort_session.get_inputs()[0].shape[-1]
    if onnx_dim != STUDENT_OBS_DIM:
        raise ValueError(
            f"{os.path.basename(path)} expects {onnx_dim}-dim input but this "
            f"controller builds {STUDENT_OBS_DIM} "
            f"[grav {GRAV_DIM} | joints {JOINT_DIM} | prev action {ACT_DIM} | step 1].")
    return ort_session, input_name

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

def main():
    parser = argparse.ArgumentParser(
        description="Fly the student policy on the robot and log telemetry.")
    parser.add_argument(
        '--variant',
        choices=list(VARIANTS),
        default=None,
        help="required unless --debug: morphology on the robot right now; selects "
             "policies/cat_controller<suffix>.onnx and the no-tail masking"
    )
    parser.add_argument(
        '--onnx',
        default=None,
        help="fly this ONNX file instead of the variant's file in policies/"
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help="no policy: type '<motor> <target rad>' to command joints by hand "
             "(1 front roll, 2 pitch, 3 tail, 4 rear roll)"
    )
    parser.add_argument(
        '--motoroff',
        action='store_true',
        help="leave the motor drivers disabled (sensor/logging dry run)"
    )
    parser.add_argument(
        '--log-dir',
        default="telemetry",
        help="directory for the telemetry CSV (default: ./telemetry)"
    )
    args = parser.parse_args()
    if not args.debug and args.variant is None:
        parser.error("--variant is required (tail or notail): it decides which policy flies")

    print(f"Variant: {args.variant}")
    input_name = None
    policy_file = policy_sha = None
    if args.debug:
        print("Debug mode: no policy loaded.")
    else:
        policy_file = os.path.abspath(args.onnx or onnx_path(args.variant))
        if not os.path.isfile(policy_file):
            hint = "" if args.onnx else (" Export it with: python onnx_conversion.py "
                                         f"--variant {args.variant}")
            raise FileNotFoundError(f"{policy_file} not found.{hint}")
        policy_sha = sha256sum(policy_file)
        print(f"Policy:  {policy_file}")
        print(f"sha256:  {policy_sha}")
        ort_session, input_name = load_policy(policy_file)

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

    # --- ZERO THE ENCODERS AND TARE THE IMU ---
    # The pose right now becomes the reference: the robot must be upright, level and
    # in its straight home pose.
    print("Zeroing motor encoders...")
    front.reset_encoders()
    back.reset_encoders()
    time.sleep(0.5)  # Let Teensy drain RESET; CDC can return EIO on flush if too early

    print("Zeroing IMU quaternions...")
    front.reset_IMU()
    back.reset_IMU()
    time.sleep(0.5)

    # Flush any stale data that was transmitted before the reset happened
    front.flush_input_safe()
    back.flush_input_safe()
    time.sleep(0.5)

    if not args.motoroff:
        front.start_all_motors()
        back.start_all_motors()
    print("Press enter to start.")
    input()

    loop_period = 1.0 / LOOP_HZ
    start_time = time.time()
    log = []

    # Debug mode never detects a drop, so it runs until Ctrl+C.
    drop_started = float('inf')

    if args.debug:
        print("Debug mode initiated: enter '[motor num] [target angle]'")
        threading.Thread(target=keyboard_input_thread, daemon=True).start()
    else:
        # Prints the front attitude (Euler xyz, deg) while the robot is held for release.
        print("Waiting for drop..")
        while(True):
            loop_start = time.time()
            front.update_sensor_data()
            back.update_sensor_data()
            if front.acc_mag < FREE_FALL_ACC:
                print("Drop")
                drop_started = time.time()
                break

            r_front = R.from_quat(front.quat, scalar_first=True)
            print(r_front.as_euler('xyz', degrees=True))

            elapsed = time.time() - loop_start
            if elapsed < loop_period:
                time.sleep(loop_period - elapsed)
            else:
                print(f"WARNING: Loop missed deadline! Took {elapsed:.4f}s")

    print(f"Starting loop at {LOOP_HZ}Hz. Press Ctrl+C to quit.")

    # Student temporal state (reset per drop), matching cat_env.reset_model: no command
    # history at release, so both the previous action and the filter start neutral.
    prev_action = np.zeros(ACT_DIM)
    action_filt = np.zeros(ACT_DIM)
    step_i = 0

    try:
        while(True):
            loop_start = time.time()
            t = loop_start - start_time

            front.update_sensor_data()
            back.update_sensor_data()

            action = [0, 0, 0, 0]

            if args.debug:
                action = list(debug_action)
            else:
                # Joint angles in sim order [rot1, pitch, rot2, tail].
                joints = np.array([front.m1_rad, front.m2_rad, back.m2_rad, back.m1_rad])
                obs = student_obs(front.quat, joints, prev_action, step_i, args.variant)
                step_i += 1
                raw_action = policy_action(ort_session, input_name, obs, args.variant)
                prev_action = raw_action.copy()
                action, action_filt = joint_targets(raw_action, action_filt)

            front.set_motors(action[0], action[1])
            back.set_motors(action[2], action[3])

            log.append([
                round(t, 4),
                front.quat[0], front.quat[1], front.quat[2], front.quat[3],
                front.m1_rad, front.m2_rad,
                front.acc_mag,
                action[0], action[1],
                back.quat[0], back.quat[1], back.quat[2], back.quat[3],
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

            if time.time() - drop_started > CONTROL_DURATION:
                raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\nExiting...")
        front.stop_all_motors()
        back.stop_all_motors()
        if log:
            os.makedirs(args.log_dir, exist_ok=True)
            tag = "debug" if args.debug else args.variant
            filename = os.path.join(args.log_dir, f"{tag}_{int(time.time())}.csv")
            print(f"Saving {len(log)} records to {filename}...")
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(TELEMETRY_HEADER)
                writer.writerows(log)
            # Sidecar with what flew, so every log can be tied to its policy later.
            with open(filename[:-4] + ".json", "w") as f:
                json.dump({"variant": args.variant, "debug": args.debug, "onnx": policy_file,
                           "onnx_sha256": policy_sha, "argv": sys.argv}, f, indent=1)
            print("Done.")

if __name__ == "__main__":
    main()
