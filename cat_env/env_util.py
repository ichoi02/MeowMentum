class PDController:
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def get_torque(self, target_pos, current_pos, current_vel):
        return self.kp * (target_pos - current_pos) - self.kd * current_vel

def map_value(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min