import numpy as np

class PDController:
    def __init__(self, kp: float, kd: float):
        self.kp = kp
        self.kd = kd

    def get_torque(self, target_pos, current_pos, current_vel):
        raw_torque = self.kp * (target_pos - current_pos) - self.kd * current_vel
        return np.clip(raw_torque, -1.0, 1.0)

def map_value(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def add_gaussian_noise(values, magnitude):
    pass