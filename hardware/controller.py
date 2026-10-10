"""Raspberry Pi main loop: fly an exported student policy on the robot.

    python -m hardware.controller --variant tail       # tail attached
    python -m hardware.controller --variant notail     # tail removed

Two Teensy 4.0 boards (hardware/PD_control_{front,back}) run the 1 kHz joint PD and
stream a sensor line every 20 ms. This script tares the IMU and zeroes the encoders,
waits for free fall (front |acc| < 3.5 m/s^2), then runs the ONNX student at 50 Hz
for CONTROL_DURATION, sends joint targets to both boards and writes
telemetry/<variant>_<unix time>.csv plus a .json sidecar naming the policy that flew.
See hardware/README.md for the procedure and the other options.
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

# Keep onnxruntime quiet: it probes for a GPU on load (4 = fatal messages only)
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "4")

import train.cat_env.env_util as util
from model.variants import VARIANTS, onnx_path

# Board USB serial numbers, from hardware.discover_teensy; parsed by flash_pd_control_teensy.sh
SN_FRONT = "18451300"
SN_BACK = "18452630"

# Serial link and control rate
BAUD_RATE = 115200
LOOP_HZ = 50
CONTROL_DURATION = 1.0    # s of control after the drop: one 50-step sim episode

# Student observation layout; must match train/distillation.py
GRAV_DIM = 3
JOINT_DIM = 4
ACT_DIM = 3
STUDENT_OBS_DIM = GRAV_DIM + JOINT_DIM + ACT_DIM + 1
MAX_STEPS = 50            # sim episode length; normalizes the step input

# Action low-pass [roll, pitch, tail]; must equal filter_alpha in train/cat_env/cat_env.py
FILTER_ALPHA = np.array([0.1, 0.3, 0.1])

# Joint target ranges in rad, from jnt_range in model/cat.xml
ROLL_RANGE = 7.28         # rot1 and rot2 (+/-417 deg)
PITCH_RANGE = 1.57        # pitch (+/-90 deg)
TAIL_RANGE = 1.9199       # tail (+/-110 deg)

# Front accelerometer magnitude below which the robot is in free fall, in m/s^2
FREE_FALL_ACC = 3.5

# Newest complete lines examined per read, which bounds the parsing work per tick
SERIAL_DRAIN_MAX_LINES = 32

# Columns of the telemetry CSV
TELEMETRY_HEADER = ["Time",
                    "F_Q0", "F_Q1", "F_Q2", "F_Q3", "F_M1", "F_M2", "F_ACC", "Cmd_F1", "Cmd_F2",
                    "B_Q0", "B_Q1", "B_Q2", "B_Q3", "B_M1", "B_M2", "B_ACC", "Cmd_B1", "Cmd_B2"]

# Joint targets typed by hand in --debug mode, [rot1, pitch, tail, rot2] in rad
debug_action = [0.0, 0.0, 0.0, 0.0]


def build_student_obs(frame, prev_action, step_frac):
    """Concatenate [grav(3), joints(4)], the previous action(3) and the step fraction(1).

    Must match build_student_obs in train/distillation.py exactly.
    """
    return np.concatenate([frame, np.asarray(prev_action, dtype=float).ravel(),
                           [float(step_frac)]])


def zero_tail(frame, variant):
    """Force the tail joint reading of a [grav(3), joints(4)] frame to 0 for no-tail.

    The no-tail model keeps the tail joint but starves its motor, so the student was
    distilled with the tail angle at 0, while the robot's encoder still reports it.
    The tail action is zeroed too (policy_action), so the board's PD holds the tail at 0.
    """
    if variant == "notail":
        frame = np.asarray(frame, dtype=float).copy()
        frame[GRAV_DIM + 3] = 0.0        # joints are [rot1, pitch, rot2, tail]
    return frame


def student_obs(front_quat, joints, prev_action, step_i, variant):
    """Build the 1 x 11 float32 student input; must match train/distillation.py.

    Args:
        front_quat: aligned front IMU quaternion [w, x, y, z].
        joints: joint angles [rot1, pitch, rot2, tail] in rad.
        prev_action: the previous raw action [roll, pitch, tail].
        step_i: control steps since release.
        variant: "tail" or "notail".
    """
    # Current frame: projected gravity from the front IMU, then the joint angles
    front_grav = util.to_projected_gravity(front_quat)
    frame = zero_tail(np.concatenate([front_grav, joints]), variant)

    # Append the action commanded last tick and the episode progress
    obs = build_student_obs(frame, prev_action, min(step_i / MAX_STEPS, 1.0))
    return np.array(obs, dtype=np.float32).reshape(1, -1)


def policy_action(ort_session, input_name, obs, variant):
    """Run the student and return its raw action [roll, pitch, tail], clipped to [-1, 1].

    This unfiltered action is the next observation's prev_action, as in cat_env. For
    no-tail, the untrained tail channel is set to 0 so the tail is held at zero.
    """
    raw_action = ort_session.run(None, {input_name: obs})[0][0]
    raw_action = np.clip(np.asarray(raw_action, dtype=float), -1.0, 1.0)
    if variant == "notail":
        raw_action[2] = 0.0
    return raw_action


def joint_targets(raw_action, action_filt):
    """Low-pass the action and map it to joint targets, exactly as cat_env.step does.

    Returns ([rot1, pitch, tail, rot2] targets in rad, new filter state).
    """
    # Low-pass each channel
    action_filt = FILTER_ALPHA * raw_action + (1.0 - FILTER_ALPHA) * action_filt

    # Map [-1, 1] onto each joint's range; rot2 mirrors rot1
    roll = util.map_value(float(action_filt[0]), -1, 1, -ROLL_RANGE, ROLL_RANGE)
    pitch = util.map_value(float(action_filt[1]), -1, 1, -PITCH_RANGE, PITCH_RANGE)
    tail = util.map_value(float(action_filt[2]), -1, 1, -TAIL_RANGE, TAIL_RANGE)
    return [roll, pitch, tail, -roll], action_filt


def _open_teensy_serial(port_path):
    """Open a board's USB serial port."""
    # write_timeout=0 makes writes non-blocking, so a wedged USB link cannot stall the loop
    return serial.Serial(port_path, BAUD_RATE, timeout=0.005, write_timeout=0)


class TeensyInterface:
    """USB serial link to one Teensy board: sensor readings in, commands and targets out."""

    def __init__(self, port_path, name):
        """Open the board on port_path; name ("Front" or "Back") selects the IMU alignment."""
        # Open the board's serial port
        self.port_path = port_path
        self.name = name
        self._rx_remainder = b""
        self.ser = _open_teensy_serial(port_path)

        # Latest readings: quaternion [w, x, y, z], joint angles in rad, |acc| in m/s^2
        self.quat = [1.0, 0.0, 0.0, 0.0]
        self.m1_rad = 0.0
        self.m2_rad = 0.0
        self.acc_mag = 0.0

    def reopen_serial(self):
        """Close and reopen the port after a USB drop-out, discarding any partial line."""
        try:
            self.ser.close()
        except (OSError, AttributeError, ValueError):
            pass
        time.sleep(0.2)
        self.ser = _open_teensy_serial(self.port_path)
        self._rx_remainder = b""
        time.sleep(0.05)

    def _is_serial_gone(self, err):
        """Return True if err means the USB device dropped or reset.

        Checks for EIO, EBADF, ENODEV and EPIPE, also when pyserial wraps the OSError as
        the first argument of a SerialException.
        """
        code = getattr(err, "errno", None)
        if code is None and getattr(err, "args", None):
            first_arg = err.args[0]
            if isinstance(first_arg, OSError):
                code = first_arg.errno
        return code in (5, 9, 19, 32)

    def _send_command(self, command, message):
        """Write a command line and print message; reopen the port if the device is gone."""
        try:
            self.ser.write(command)
            print(f"{self.name}: {message}")
        except (OSError, serial.SerialException) as e:
            if self._is_serial_gone(e):
                self.reopen_serial()

    def flush_input_safe(self):
        """Discard stale input and forget the old readings."""
        # Flush; tcflush raises EIO if the USB device reset, so reopen and retry once
        try:
            self.ser.reset_input_buffer()
        except (termios.error, OSError) as e:
            print(f"{self.name}: input flush failed ({e!r}); reopening serial…")
            self.reopen_serial()
            try:
                self.ser.reset_input_buffer()
            except (termios.error, OSError):
                pass

        # Reset the readings; acc_mag starts at 1 g so no free fall is seen before new data
        self._rx_remainder = b""
        self.quat = [1.0, 0.0, 0.0, 0.0]
        self.m1_rad = 0.0
        self.m2_rad = 0.0
        self.acc_mag = 9.8

    def reset_encoders(self):
        """Zero the board's encoder counts."""
        self._send_command(b"RESET\n", "Reset encoders.")

    def reset_IMU(self):
        """Tare the IMU (front board only; the back board ignores it)."""
        self._send_command(b"RESET_IMU\n", "Reset IMU.")

    def reboot_teensy(self):
        """Software-reset the board."""
        self._send_command(b"REBOOT\n", "Reboot.")

    def stop_all_motors(self):
        """Disable the board's motor driver."""
        self._send_command(b"STOP\n", "Stopping motors.")

    def start_all_motors(self):
        """Enable the board's motor driver."""
        self._send_command(b"START\n", "Starting motors.")

    def _read_latest_line(self):
        """Read what the board has sent; return the newest complete non-empty line or None."""
        # Read the waiting bytes, reopening the port if the device is gone
        try:
            waiting = self.ser.in_waiting
        except (OSError, serial.SerialException) as e:
            if self._is_serial_gone(e):
                print(f"{self.name}: serial I/O in update_sensor_data; reopening…")
                self.reopen_serial()
            return None
        if waiting <= 0:
            return None
        try:
            raw = self.ser.read(min(waiting, 4096))
        except (OSError, serial.SerialException) as e:
            if self._is_serial_gone(e):
                self.reopen_serial()
            return None
        if not raw:
            return None

        # Wait for a full line, capping the buffer if the stream carries no newline
        self._rx_remainder += raw
        if b"\n" not in self._rx_remainder:
            if len(self._rx_remainder) > 8192:
                self._rx_remainder = self._rx_remainder[-4096:]
            return None

        # Keep the partial last line for the next read
        lines = self._rx_remainder.split(b"\n")
        self._rx_remainder = lines[-1]

        # Return the newest non-empty line among the last SERIAL_DRAIN_MAX_LINES complete ones
        for line in reversed(lines[:-1][-SERIAL_DRAIN_MAX_LINES:]):
            text = line.decode("utf-8", errors="ignore").strip()
            if text:
                return text
        return None

    def update_sensor_data(self):
        """Update the readings from the newest complete line the board has sent.

        Each line is 'qr,qi,qj,qk,angle1,angle2,acc'; the back board has no IMU and always
        sends 1,0,0,0 and acc 0. A line without 7 fields is ignored, and a field that does
        not parse ends the update, keeping the readings stored before it.
        """
        # Get the newest complete line, if any
        line = self._read_latest_line()
        if line is None:
            return

        # Parse the fields, aligning the quaternion to the robot's axes
        parts = line.split(',')
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
        """Send the board's two joint targets in rad as 'm1,m2' with 4 decimals.

        4 decimals is finer than the encoder resolution (3.8 mrad on pitch and tail).
        """
        cmd = f"{m1:.4f},{m2:.4f}\n"
        try:
            self.ser.write(cmd.encode('utf-8'))
        except (OSError, serial.SerialException) as e:
            if self._is_serial_gone(e):
                print(f"{self.name}: serial I/O error in set_motors; reopening…")
                self.reopen_serial()

    def align_imu_quaternions(self, quats_wxyz, imu_type):
        """Re-express a tared IMU rotation (1 x 4, [w, x, y, z]) in the robot's axes.

        Computes r_align^-1 * q * r_align, where r_align is the sensor mounting rotation
        (front: 90 deg about z). The back board's identity placeholder stays identity.
        Returns a length-4 [w, x, y, z] array.
        """
        # Rotation measured by the tared IMU
        r_raw = R.from_quat(quats_wxyz, scalar_first=True)

        # Mounting rotation of each board's sensor axes relative to the body axes
        if imu_type == 'Front':
            r_align = R.from_euler('xyz', [0, 0, 90], degrees=True)
        elif imu_type == 'Back':
            r_align = R.from_euler('xyz', [180, 0, -90], degrees=True)

        # Re-express the rotation in the body axes
        r_global = r_align.inv() * r_raw * r_align
        return r_global.as_quat(scalar_first=True).squeeze(0)


def sha256sum(path):
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def load_policy(path):
    """Open the ONNX student and check its input width; return (session, input name)."""
    # Load onnxruntime, showing only its errors (3) where the installed build supports it
    import onnxruntime as ort
    try:
        ort.set_default_logger_severity(3)
    except (AttributeError, TypeError):
        pass

    # Open the session
    ort_session = ort.InferenceSession(path)
    input_name = ort_session.get_inputs()[0].name

    # Check the input width now rather than on the first inference mid-drop
    onnx_dim = ort_session.get_inputs()[0].shape[-1]
    if onnx_dim != STUDENT_OBS_DIM:
        raise ValueError(
            f"{os.path.basename(path)} expects {onnx_dim}-dim input but this "
            f"controller builds {STUDENT_OBS_DIM} "
            f"[grav {GRAV_DIM} | joints {JOINT_DIM} | prev action {ACT_DIM} | step 1].")
    return ort_session, input_name


def get_port_by_sn(serial_number):
    """Return the device path of the serial port with this USB serial number, or None."""
    for port in serial.tools.list_ports.comports():
        if port.serial_number == serial_number:
            return port.device
    return None


def keyboard_input_thread():
    """Read '<motor> <target rad>' lines from stdin into debug_action, forever."""
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


def parse_args():
    """Parse the command line; --variant is required unless --debug is given."""
    # Define the options
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

    # Parse them, requiring --variant unless --debug
    args = parser.parse_args()
    if not args.debug and args.variant is None:
        parser.error("--variant is required (tail or notail): it decides which policy flies")
    return args


def open_policy(args):
    """Load the policy to fly and print its file and SHA-256.

    Returns (policy file, SHA-256, ONNX session, input name), all None in --debug mode.
    """
    # Fly no policy in --debug mode
    if args.debug:
        print("Debug mode: no policy loaded.")
        return None, None, None, None

    # Use --onnx, or else the variant's export in policies/
    policy_file = args.onnx or onnx_path(args.variant)
    if not os.path.isfile(policy_file):
        hint = "" if args.onnx else (" Export it with: python -m hardware.onnx_conversion "
                                     f"--variant {args.variant}")
        raise FileNotFoundError(f"{policy_file} not found.{hint}")

    # Identify the policy by its file and SHA-256
    policy_sha = sha256sum(policy_file)
    print(f"Policy:  {policy_file}")
    print(f"sha256:  {policy_sha}")

    # Open the ONNX session
    ort_session, input_name = load_policy(policy_file)
    return policy_file, policy_sha, ort_session, input_name


def connect_boards():
    """Find both boards by serial number, reboot them and reconnect.

    Returns (front, back) TeensyInterfaces, or None if a board is not plugged in.
    """
    # Find both boards by serial number
    path_front = get_port_by_sn(SN_FRONT)
    path_back = get_port_by_sn(SN_BACK)
    if not path_front or not path_back:
        print(f"Error finding boards! Front: {path_front}, Back: {path_back}")
        return None

    # Open both boards
    print("Connecting to boards...")
    front = TeensyInterface(path_front, "Front")
    back = TeensyInterface(path_back, "Back")
    time.sleep(1)

    # Reboot both boards
    print("Rebooting Teensy")
    front.reboot_teensy()
    front.ser.close()
    time.sleep(2)
    back.reboot_teensy()
    back.ser.close()
    time.sleep(2)

    # Reconnect to the rebooted boards
    path_front = get_port_by_sn(SN_FRONT)
    path_back = get_port_by_sn(SN_BACK)
    front = TeensyInterface(path_front, "Front")
    back = TeensyInterface(path_back, "Back")
    print(f"{path_front}, {path_back}")
    time.sleep(1)
    return front, back


def sleep_until_next_tick(loop_start):
    """Sleep out the rest of the 1 / LOOP_HZ tick begun at loop_start, or warn of an overrun."""
    loop_period = 1.0 / LOOP_HZ
    elapsed = time.time() - loop_start
    if elapsed < loop_period:
        time.sleep(loop_period - elapsed)
    else:
        print(f"WARNING: Loop missed deadline! Took {elapsed:.4f}s")


def wait_for_drop(front, back):
    """Print the front attitude every tick until free fall; return the time of the drop."""
    # Poll both boards once per tick until free fall
    print("Waiting for drop..")
    while True:
        # Read both boards at the start of the tick
        loop_start = time.time()
        front.update_sensor_data()
        back.update_sensor_data()

        # Return the drop time once the robot is in free fall
        if front.acc_mag < FREE_FALL_ACC:
            print("Drop")
            return time.time()

        # Show the front attitude (Euler xyz, deg) while the robot is held for release
        r_front = R.from_quat(front.quat, scalar_first=True)
        print(r_front.as_euler('xyz', degrees=True))

        # Hold the loop rate
        sleep_until_next_tick(loop_start)


def fly(front, back, args, ort_session, input_name, start_time, drop_started, log):
    """Run the control loop at LOOP_HZ, appending one telemetry row per step to log.

    Ends like Ctrl+C, by raising KeyboardInterrupt, once CONTROL_DURATION has passed
    since drop_started; with drop_started = inf (--debug) only Ctrl+C ends it.
    """
    # Start the student with no command history, as cat_env.reset_model does
    prev_action = np.zeros(ACT_DIM)
    action_filt = np.zeros(ACT_DIM)
    step_i = 0

    # Run one control step per tick
    while True:
        # Read both boards at the start of the tick
        loop_start = time.time()
        t = loop_start - start_time
        front.update_sensor_data()
        back.update_sensor_data()

        # Choose the targets [rot1, pitch, tail, rot2]: typed by hand, or from the policy
        if args.debug:
            action = list(debug_action)
        else:
            joints = np.array([front.m1_rad, front.m2_rad, back.m2_rad, back.m1_rad])
            obs = student_obs(front.quat, joints, prev_action, step_i, args.variant)
            step_i += 1
            raw_action = policy_action(ort_session, input_name, obs, args.variant)
            prev_action = raw_action.copy()
            action, action_filt = joint_targets(raw_action, action_filt)

        # Send the targets to both boards
        front.set_motors(action[0], action[1])
        back.set_motors(action[2], action[3])

        # Log what this step read and sent
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

        # Hold the loop rate
        sleep_until_next_tick(loop_start)

        # End the control window like a Ctrl+C once CONTROL_DURATION has passed since the drop
        if time.time() - drop_started > CONTROL_DURATION:
            raise KeyboardInterrupt


def save_telemetry(log, args, policy_file, policy_sha):
    """Write the telemetry CSV and a JSON sidecar naming the policy that flew."""
    # Pick the output path: <log dir>/<variant or debug>_<Unix time>.csv
    os.makedirs(args.log_dir, exist_ok=True)
    tag = "debug" if args.debug else args.variant
    filename = os.path.join(args.log_dir, f"{tag}_{int(time.time())}.csv")
    print(f"Saving {len(log)} records to {filename}...")

    # Write the CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(TELEMETRY_HEADER)
        writer.writerows(log)

    # Write the sidecar, so every log can be tied to its policy
    with open(filename[:-4] + ".json", "w") as f:
        json.dump({"variant": args.variant, "debug": args.debug, "onnx": policy_file,
                   "onnx_sha256": policy_sha, "argv": sys.argv}, f, indent=1)
    print("Done.")


def main():
    """Fly one drop: set up the boards, wait for free fall, run the policy, save telemetry."""
    # Parse the command line
    args = parse_args()

    # Load the policy for the morphology on the robot
    print(f"Variant: {args.variant}")
    policy_file, policy_sha, ort_session, input_name = open_policy(args)

    # Connect to both boards
    boards = connect_boards()
    if boards is None:
        return
    front, back = boards

    # Stop the motors
    front.stop_all_motors()
    back.stop_all_motors()

    # Zero the encoders with the robot in its straight home pose
    print("Zeroing motor encoders...")
    front.reset_encoders()
    back.reset_encoders()
    time.sleep(0.5)

    # Tare the IMU with the robot upright and level
    print("Zeroing IMU quaternions...")
    front.reset_IMU()
    back.reset_IMU()
    time.sleep(0.5)

    # Drop the data sent before the reset
    front.flush_input_safe()
    back.flush_input_safe()
    time.sleep(0.5)

    # Enable the motors unless --motoroff
    if not args.motoroff:
        front.start_all_motors()
        back.start_all_motors()

    # Wait for Enter
    print("Press enter to start.")
    input()

    # Start the telemetry clock
    start_time = time.time()

    # Take joint commands from the keyboard (--debug), or wait for the drop
    if args.debug:
        print("Debug mode initiated: enter '[motor num] [target angle]'")
        threading.Thread(target=keyboard_input_thread, daemon=True).start()
        drop_started = float('inf')
    else:
        drop_started = wait_for_drop(front, back)

    # Fly until the control window ends or Ctrl+C
    print(f"Starting loop at {LOOP_HZ}Hz. Press Ctrl+C to quit.")
    log = []
    try:
        fly(front, back, args, ort_session, input_name, start_time, drop_started, log)
    except KeyboardInterrupt:
        # Stop the motors
        print("\nExiting...")
        front.stop_all_motors()
        back.stop_all_motors()

        # Save the telemetry, if any step ran
        if log:
            save_telemetry(log, args, policy_file, policy_sha)


if __name__ == "__main__":
    main()
