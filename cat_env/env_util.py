import numpy as np
from scipy.spatial.transform import Rotation as R

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
    noise = np.random.normal(loc=0.0, scale=magnitude, size=np.array(values).shape)
    return values + noise

def to_rotation_matrix(quat):
    return R.from_quat(quat, scalar_first=True).as_matrix().flatten()

def add_rotational_noise(rot_matrices_flat, std_dev=0.01):
    rot_matrices = rot_matrices_flat.reshape(-1, 3, 3)
    noise_vecs = np.random.normal(scale=std_dev, size=(rot_matrices.shape[0], 3))
    noise_rotations = R.from_rotvec(noise_vecs)
    original_rotations = R.from_matrix(rot_matrices)
    noisy_rotations = noise_rotations * original_rotations
    return noisy_rotations.as_matrix().flatten()