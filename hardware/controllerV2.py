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

debug_action = [0.0, 0.0, 0.0, 0.0]

class TeensyInterface:
    def __init__(self, port_path, name):
        self.port_path = port_path
        self.name = name
        self._rx_remainder = b""
        self.ser = serial.Serial(port_path, BAUD_RATE, timeout=0.005)

    def reopen_serial(self):
        try:
            self.ser.close()
        except (OSError, AttributeError, ValueError):
            pass
        time.sleep(0.2)
        self.ser = serial.Serial(self.port_path, BAUD_RATE, timeout=0.005)
        self._rx_remainder = b""
        time.sleep(0.05)

    def _is_serial_gone(self, err):
        en = getattr(err, "errno", None)
        return en == 5 or en == 19  # EIO, ENODEV

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

    def reset_encoders(self):
        """Sends the command to zero out the hardware encoder counts."""
        try:
            self.ser.write(b"RESET\n")
            print(f"{self.name}: Reset encoders.")
        except OSError as e:
            if self._is_serial_gone(e):
                self.reopen_serial()

    def reboot_teensy(self):
        try:
            self.ser.write(b"REBOOT\n")
            print(f"{self.name}: Reboot.")
        except OSError as e:
            if self._is_serial_gone(e):
                self.reopen_serial()

    def stop_all_motors(self):
        try:
            self.ser.write(b"STOP\n")
            print(f"{self.name}: Stopping otors.")
        except OSError as e:
            if self._is_serial_gone(e):
                self.reopen_serial()

    def start_all_motors(self):
        try:
            self.ser.write(b"START\n")
            print(f"{self.name}: Starting motors.")
        except OSError as e:
            if self._is_serial_gone(e):
                self.reopen_serial()

    def update_sensor_data(self):
        """Reads buffered lines to get the freshest IMU and encoder data."""
        try:
            waiting = self.ser.in_waiting
        except OSError as e:
            if self._is_serial_gone(e):
                print(f"{self.name}: serial EIO in update_sensor_data; reopening…")
                self.reopen_serial()
            return

        if waiting <= 0:
            return

        max_bytes = min(waiting, 4096)
        try:
            raw = self.ser.read(max_bytes)
        except OSError as e:
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
            if len(parts) == 6:
                try:
                    self.quat = [float(x) for x in parts[:4]]
                    self.quat = self.rotate_quat(self.quat, 0, 0, 90)
                    self.m1_rad = float(parts[4])
                    self.m2_rad = float(parts[5])
                except ValueError:
                    pass

    def set_motors(self, m1, m2):
        """Sends target motor angles (2dp) to Teensy. Assumes format: 'M1,M2\n'"""
        cmd = f"{m1:.2f},{m2:.2f}\n"
        try:
            self.ser.write(cmd.encode('utf-8'))
        except OSError as e:
            if self._is_serial_gone(e):
                print(f"{self.name}: serial EIO in set_motors; reopening…")
                self.reopen_serial()
    
    def rotate_quat(self, q, x, y, z):
        base_rot = R.from_quat(q, scalar_first=True)
        new_rot = R.from_euler('xyz', [x, y, z], degrees=True)
        combined_rot = base_rot * new_rot
        return combined_rot.as_quat(scalar_first=True)

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
    print(f"{path_front}, {path_back}")
    time.sleep(1) # Wait for serial connection to establish

    # Stop all motors
    front.stop_all_motors()
    back.stop_all_motors()

    front.reboot_teensy()
    back.reboot_teensy()
    time.sleep(1)

    # --- ZERO THE ENCODERS ---
    print("Zeroing motor encoders...")
    front.reset_encoders()
    back.reset_encoders()
    time.sleep(0.25)  # Let Teensy drain RESET; CDC can return EIO on flush if too early
    
    # Flush any stale data that was transmitted before the reset happened
    front.flush_input_safe()
    back.flush_input_safe()

    print("Press any key to start.")
    input()
    front.start_all_motors()
    back.start_all_motors()

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
    
    if args.debug:
        print("Debug mode initiated: enter '[motor num] [target angle]'")
        threading.Thread(target=keyboard_input_thread, daemon=True).start()

    print(f"Starting loop at {LOOP_HZ}Hz. Press Ctrl+C to quit.")
    try:
        while(True): # FIXME: need to limit running time
            loop_start = time.time()
            t = loop_start - start_time

            front.update_sensor_data()
            back.update_sensor_data()

            action = [0, 0, 0, 0]

            if args.debug:
                action = list(debug_action) 
            else:
                # TODO: add onnx control here
                pass

            front.set_motors(action[0], action[1])
            back.set_motors(action[2], action[3])

            log.append([
                round(t, 4),
                front.quat[0], front.quat[1], front.quat[2], front.quat[3],
                front.m1_rad, front.m2_rad,
                action[0], action[1],
                back.quat[0], back.quat[1], back.quat[2], back.quat[3],
                back.m1_rad, back.m2_rad,
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
            filename = f"telemetry_{int(time.time())}.csv"
            print(f"Saving {len(log)} records to {filename}...")
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Time", "F_Q0", "F_Q1", "F_Q2", "F_Q3", "F_M1", "F_M2", "Cmd_F1", "Cmd_F2", 
                                 "B_Q0", "B_Q1", "B_Q2", "B_Q3", "B_M1", "B_M2", "Cmd_B1", "Cmd_B2"])
                writer.writerows(log)
            print("Done.")

"""
    # --- SETUP CSV LOGGING ---
    log_filename = f"telemetry_log_{int(time.time())}.csv"
    print(f"Logging telemetry to: {log_filename}")
    
    with open(log_filename, 'w', newline='', buffering=io.DEFAULT_BUFFER_SIZE * 8) as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(headers)

        print(f"Starting loop at {LOOP_HZ}Hz. Press Ctrl+C to quit.")
        try:
            while True:
                loop_start = time.time()
                t = loop_start - start_time

                front.update_sensor_data()
                back.update_sensor_data()

                # --- 1. Control Logic ---
                if args.debug:
                    lim = math.radians(DEBUG_JOINT_SWING_DEG)
                    breach = _debug_encoder_limit_breached(front, back, lim)
                    if breach is not None:
                        b, j, r = breach
                        print(
                            f"DEBUG safety stop: {b} {j} = {math.degrees(r):.2f}° "
                            f"(limit ±{DEBUG_JOINT_SWING_DEG:g}°); stopping motors and exiting."
                        )
                        front.set_motors(0, 0)
                        back.set_motors(0, 0)
                        time.sleep(0.1)
                        return

                    w = 2 * math.pi * DEBUG_TRACK_FREQ_HZ
                    target_m1 = lim * math.sin(w * t)
                    target_m2 = lim * math.cos(w * t)
                    kp = DEBUG_TRACK_KP
                    m1_cmd = kp * (target_m1 - front.m1_rad)
                    m2_cmd = kp * (target_m2 - front.m2_rad)
                    m1b_cmd = kp * (target_m1 - back.m1_rad)
                    m2b_cmd = kp * (target_m2 - back.m2_rad)

                    action = [m1_cmd, m2_cmd, m1b_cmd, m2b_cmd]

                    front.set_motors(action[0], action[1])
                    back.set_motors(action[2], action[3])

                else:
                    # FIXME: Adjust state array based on your actual ONNX model architecture requirements
                    state = np.array(front.quat + [front.m1_rad, front.m2_rad] + 
                                     back.quat + [back.m1_rad, back.m2_rad], 
                                     dtype=np.float32).reshape(1, -1)
                    
                    # FIXME: Ensure "input" matches your ONNX model's exact input node name
                    action = ort_session.run(None, {"input": state})[0][0] 
                    
                    front.set_motors(action[0], action[1])
                    back.set_motors(action[2], action[3])

                # --- 2. Telemetry Logging ---
                log_row = [
                    round(t, 4),
                    front.quat[0],
                    front.quat[1],
                    front.quat[2],
                    front.quat[3],
                    front.m1_rad,
                    front.m2_rad,
                    action[0],
                    action[1],
                    back.quat[0],
                    back.quat[1],
                    back.quat[2],
                    back.quat[3],
                    back.m1_rad,
                    back.m2_rad,
                    action[2],
                    action[3],
                    round(math.degrees(front.m1_rad), 6),
                    round(math.degrees(front.m2_rad), 6),
                    round(math.degrees(back.m1_rad), 6),
                    round(math.degrees(back.m2_rad), 6),
                ]
                csv_writer.writerow(log_row)

                # --- 3. Enforce Timing ---
                elapsed = time.time() - loop_start
                if elapsed < loop_period:
                    time.sleep(loop_period - elapsed)
                else:
                    print(f"WARNING: Loop missed deadline! Took {elapsed:.4f}s")

        except KeyboardInterrupt:
            print("\nStopping motors and exiting...")
            front.set_motors(0, 0)
            back.set_motors(0, 0)
            time.sleep(0.1)
"""
if __name__ == "__main__":
    main()