"""Actuator, encoder and orientation helpers shared by the simulator and the robot controller.

The motor, encoder and velocity-filter models reproduce the Teensy joint PD loop in
hardware/PD_control_{front,back}/*.ino; change both sides together.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

# Firmware motor output: off inside the position deadband, else at least the minimum PWM
MOTOR_DEADBAND = 0.03           # rad at the joint, firmware `deadband`
MOTOR_MIN_CMD = 100.0 / 1023.0  # firmware `minPWM` / `PWM_MAX`

# Quadrature encoder on the motor shaft, read at the joint through the gearbox
ENCODER_TICKS_PER_REV = 48
GEAR_RATIO = {"rot1": 9.68, "pitch": 34.014, "rot2": 9.68, "tail": 34.014}

# Per-joint deadband overrides, in rad; joints not listed use MOTOR_DEADBAND
JOINT_DEADBAND = {}

# Gain of the one-pole low-pass on the measured joint velocity, applied every 1 ms
# Must equal VEL_ALPHA in hardware/PD_control_{front,back}/*.ino
VEL_FILTER_ALPHA = 0.1


def encoder_resolution(joint):
    """Return the smallest joint-angle change the encoder can report, in rad."""
    return 2.0 * np.pi / (ENCODER_TICKS_PER_REV * GEAR_RATIO[joint])


def joint_deadband(joint):
    """Return the position deadband of a joint, in rad."""
    return JOINT_DEADBAND.get(joint, MOTOR_DEADBAND)


def apply_motor_deadband(norm_torque, pos_err, deadband=None):
    """Apply the firmware deadband and minimum-command floor to a normalized torque.

    Args:
        norm_torque: PD output in [-1, 1].
        pos_err: target minus measured joint angle, in rad.
        deadband: deadband in rad; MOTOR_DEADBAND when None.
    """
    # Fall back to the global deadband
    if deadband is None:
        deadband = MOTOR_DEADBAND

    # Switch the motor off while the joint is inside the deadband
    if abs(pos_err) <= deadband:
        return 0.0

    # Raise any nonzero command below the minimum PWM to it
    if 0.0 < abs(norm_torque) < MOTOR_MIN_CMD:
        return np.copysign(MOTOR_MIN_CMD, norm_torque)
    return norm_torque


class PDController:
    """Joint PD loop of the Teensy firmware (runController), step for step.

    With `resolution` set, the joint angle is quantized to encoder ticks and the
    velocity is the difference of quantized angles, low-passed, as on the board.
    """

    def __init__(self, kp: float, kd: float, resolution: float = 0.0,
                 vel_alpha: float = VEL_FILTER_ALPHA, dt: float = 0.001):
        """Store the gains and filter settings and clear the loop state.

        Args:
            kp: gain on the position error (rad) to normalized torque.
            kd: gain on the filtered velocity (rad/s) to normalized torque.
            resolution: rad per encoder tick; 0 disables quantization.
            vel_alpha: velocity low-pass gain per step.
            dt: control period, in s.
        """
        self.kp = kp
        self.kd = kd
        self.resolution = resolution
        self.vel_alpha = vel_alpha
        self.dt = dt
        self.reset()

    def _measure(self, current_pos):
        """Return the angle the encoder reports for a true joint angle."""
        if self.resolution <= 0.0:
            return current_pos
        return np.round(current_pos / self.resolution) * self.resolution

    def reset(self, current_pos=0.0):
        """Clear the velocity filter and seed the angle history; call once per episode.

        Pass the starting joint angle, so that the first velocity sample is not a jump
        from 0.
        """
        self.meas_pos = self._measure(current_pos)
        self.vel_filt = 0.0

    def get_torque(self, target_pos, current_pos, current_vel=None):
        """Return the normalized torque in [-1, 1] for one control step.

        `current_vel` is used only when quantization is off; otherwise the velocity
        is differenced from the encoder reading, as on hardware.
        """
        # Read the encoder and estimate the joint velocity
        prev_pos = self.meas_pos
        self.meas_pos = self._measure(current_pos)
        if self.resolution > 0.0:
            vel = (self.meas_pos - prev_pos) / self.dt
        else:
            vel = current_vel

        # Low-pass the velocity
        self.vel_filt += self.vel_alpha * (vel - self.vel_filt)

        # Apply the PD law and saturate to the normalized range
        raw_torque = self.kp * (target_pos - self.meas_pos) - self.kd * self.vel_filt
        return np.clip(raw_torque, -1.0, 1.0)


def map_value(value, in_min, in_max, out_min, out_max):
    """Map a value linearly from [in_min, in_max] onto [out_min, out_max]."""
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def add_gaussian_noise(values, magnitude):
    """Return the values plus zero-mean Gaussian noise drawn from the global NumPy RNG."""
    noise = np.random.normal(loc=0.0, scale=magnitude, size=np.array(values).shape)
    return values + noise


def to_projected_gravity(quat):
    """Return the world gravity direction (-z) in the body frame, a unit 3-vector.

    `quat` is the body-to-world rotation as [w, x, y, z]. The result does not depend
    on the heading about world z, so it is a yaw-invariant orientation cue.
    """
    r = R.from_quat(quat, scalar_first=True)
    return r.apply([0.0, 0.0, -1.0], inverse=True)
