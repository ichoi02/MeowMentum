import serial
import serial.tools.list_ports
import time
import math
import argparse
import numpy as np
import onnxruntime as ort

# --- CONFIGURATION ---
# FIXME: Replace these dummy serial numbers with the ones from the Discovery Script
SN_FRONT = "1234560" 
SN_BACK  = "9876540"

BAUD_RATE = 115200
LOOP_HZ = 50  

class TeensyInterface:
    def __init__(self, port_path, name):
        self.ser = serial.Serial(port_path, BAUD_RATE, timeout=0.005)
        self.name = name
        self.quat = [1.0, 0.0, 0.0, 0.0] # [qr, qi, qj, qk]
        self.accuracy = 0.0

    def update_sensor_data(self):
        """Reads all waiting lines in the buffer to get the freshest IMU data."""
        latest_line = None
        while self.ser.in_waiting > 0:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line:
                    latest_line = line
            except (UnicodeDecodeError, ValueError):
                pass 
        
        if latest_line:
            parts = latest_line.split(',')
            if len(parts) == 5:
                try:
                    self.quat = [float(x) for x in parts[:4]]
                    self.accuracy = float(parts[4])
                except ValueError:
                    pass

    def set_motors(self, m1, m2):
        """Sends motor speeds to Teensy. Assumes format: 'M1,M2\n'"""
        # FIXME: Adjust the 400 / -400 limits to match your VNH5019 library's PWM range
        m1 = max(min(int(m1), 400), -400)
        m2 = max(min(int(m2), 400), -400)
        cmd = f"{m1},{m2}\n"
        self.ser.write(cmd.encode('utf-8'))

def get_port_by_sn(serial_number):
    for port in serial.tools.list_ports.comports():
        if port.serial_number == serial_number:
            return port.device
    return None

def main():
    parser = argparse.ArgumentParser(description="Teensy Control Loop")
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
    time.sleep(1) 

    if not args.debug:
        print("Loading ONNX Model...")
        ort_session = ort.InferenceSession("cat_controller.onnx") 
    
    loop_period = 1.0 / LOOP_HZ
    start_time = time.time()
    
    print(f"Starting loop at {LOOP_HZ}Hz. Press Ctrl+C to quit.")
    try:
        while True:
            loop_start = time.time()
            t = loop_start - start_time

            front.update_sensor_data()
            back.update_sensor_data()

            if args.debug:
                # FIXME: Tune the amplitude and frequency for your debug testing
                amp = 50.0
                freq = 0.5
                m1_cmd = amp * math.sin(2 * math.pi * freq * t)
                m2_cmd = amp * math.cos(2 * math.pi * freq * t)
                
                front.set_motors(m1_cmd, m2_cmd)
                back.set_motors(m1_cmd, m2_cmd)
                
            else:
                # FIXME: Ensure this array matches your model's expected input shape (e.g., 8 features: 4 for front quat, 4 for back quat)
                state = np.array(front.quat + back.quat, dtype=np.float32).reshape(1, -1)
                
                # FIXME: Ensure "input" matches your ONNX model's exact input node name
                action = ort_session.run(None, {"input": state})[0][0] 
                
                # FIXME: Ensure the array indices match what your model outputs for each motor
                # Example assumes action = [front_m1, front_m2, back_m1, back_m2]
                front.set_motors(action[0], action[1])
                back.set_motors(action[2], action[3])

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