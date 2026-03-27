import os

# Edge / Pi: ONNX probes GPU on load and logs a warning when there is no CUDA device.
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

# Some Pi images set LD_PRELOAD to a missing OpenBLAS path; ld.so then prints noisy errors.
def _strip_broken_openblas_preload():
    p = os.environ.get("LD_PRELOAD", "")
    if not p or "libopenblas" not in p:
        return
    sep = ":" if ":" in p else " "
    parts = [x.strip() for x in p.split(sep) if x.strip() and "libopenblas" not in x]
    if parts:
        os.environ["LD_PRELOAD"] = sep.join(parts)
    else:
        os.environ.pop("LD_PRELOAD", None)


_strip_broken_openblas_preload()

import serial
import serial.tools.list_ports
import time
import math
import io
import argparse
import numpy as np
import onnxruntime as ort
import csv

# --- CONFIGURATION ---
# FIXME: Replace these dummy serial numbers with the actual ones from your Teensy boards
SN_FRONT = "18451300" 
SN_BACK  = "18452630"

BAUD_RATE = 115200
# FIXME: Adjust this if your ONNX model requires a different control frequency
LOOP_HZ = 50

# Bound serial draining per tick (~50Hz telemetry per board); unbounded reads starve the Python loop.
SERIAL_DRAIN_MAX_LINES = 32

class TeensyInterface:
    def __init__(self, port_path, name):
        self.ser = serial.Serial(port_path, BAUD_RATE, timeout=0.005)
        self.name = name
        self.quat = [1.0, 0.0, 0.0, 0.0] 
        self.accuracy = 0.0
        self.m1_rad = 0.0
        self.m2_rad = 0.0

    def reset_encoders(self):
        """Sends the command to zero out the hardware encoder counts."""
        self.ser.write(b"RESET\n")

    def update_sensor_data(self):
        """Reads waiting lines in the buffer to get the freshest IMU and encoder data."""
        latest_line = None
        n = 0
        while self.ser.in_waiting > 0 and n < SERIAL_DRAIN_MAX_LINES:
            n += 1
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line:
                    latest_line = line
            except (UnicodeDecodeError, ValueError):
                pass
        
        if latest_line:
            parts = latest_line.split(',')
            if len(parts) == 7:
                try:
                    self.quat = [float(x) for x in parts[:4]]
                    self.accuracy = float(parts[4])
                    self.m1_rad = float(parts[5])
                    self.m2_rad = float(parts[6])
                except ValueError:
                    pass

    def set_motors(self, m1, m2):
        """Sends motor speeds to Teensy. Assumes format: 'M1,M2\n'"""
        # Constrained to 1023 to match 10-bit PWM resolution on the Teensy
        m1 = max(min(int(m1), 1023), -1023)
        m2 = max(min(int(m2), 1023), -1023)
        cmd = f"{m1},{m2}\n"
        self.ser.write(cmd.encode('utf-8'))

def get_port_by_sn(serial_number):
    for port in serial.tools.list_ports.comports():
        if port.serial_number == serial_number:
            return port.device
    return None

def main():
    parser = argparse.ArgumentParser(description="Teensy Control Loop with Telemetry Logging")
    parser.add_argument('--debug', action='store_true', help="Run small amplitude sine profile")
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

    # --- ZERO THE ENCODERS ---
    print("Zeroing motor encoders...")
    front.reset_encoders()
    back.reset_encoders()
    time.sleep(0.1) # Give Teensy time to process the reset
    
    # Flush any stale data that was transmitted before the reset happened
    front.ser.reset_input_buffer()
    back.ser.reset_input_buffer()

    if not args.debug:
        print("Loading ONNX Model...")
        # FIXME: Replace with your actual model filename if different
        ort_session = ort.InferenceSession("cat_controller.onnx") 
    
    loop_period = 1.0 / LOOP_HZ
    start_time = time.time()
    
    # --- SETUP CSV LOGGING ---
    log_filename = f"telemetry_log_{int(time.time())}.csv"
    print(f"Logging telemetry to: {log_filename}")
    
    with open(log_filename, 'w', newline='', buffering=io.DEFAULT_BUFFER_SIZE * 8) as csvfile:
        csv_writer = csv.writer(csvfile)
        headers = [
            "timestamp_s", 
            "front_qr", "front_qi", "front_qj", "front_qk", "front_m1_rad", "front_m2_rad", "front_m1_cmd", "front_m2_cmd",
            "back_qr", "back_qi", "back_qj", "back_qk", "back_m1_rad", "back_m2_rad", "back_m1_cmd", "back_m2_cmd"
        ]
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
                    # FIXME: Tune the amplitude and frequency for your debug testing
                    amp = 30.0 
                    freq = 0.5
                    m1_cmd = amp * math.sin(2 * math.pi * freq * t)
                    m2_cmd = amp * math.cos(2 * math.pi * freq * t)
                    
                    # Unify the action array format so logging works the same as ONNX mode
                    action = [m1_cmd, m2_cmd, m1_cmd, m2_cmd]
                    
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
                    front.quat[0], front.quat[1], front.quat[2], front.quat[3], front.m1_rad, front.m2_rad, action[0], action[1],
                    back.quat[0], back.quat[1], back.quat[2], back.quat[3], back.m1_rad, back.m2_rad, action[2], action[3]
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

if __name__ == "__main__":
    main()