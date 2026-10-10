"""Gymnasium MuJoCo environment of the robot righting itself in free fall.

Registered as Cat-v0 (model/cat.xml) and CatNoTail-v0 (model/cat_notail.xml) in
train/cat_env/__init__.py. An episode is one 1 s drop of 50 control steps. The action
is the normalized [roll, pitch, tail] joint target, tracked by a simulated copy of the
1 kHz firmware PD loop. Every reset redraws the domain randomization and the release
state from the global NumPy RNG.
"""
import os

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle
from scipy.spatial.transform import Rotation as R

import train.cat_env.env_util as util

# MuJoCo model loaded when no model_path is given, relative to the repository root
_DEFAULT_MODEL = "model/cat.xml"


def _w(name, default):
    """Return a reward weight, overridden by the environment variable CAT_W_<NAME> if set."""
    return float(os.environ.get(f"CAT_W_{name}", default))


# Reward weights (see CatEnv._get_reward); the defaults reproduce the released teachers
# The paper reports reward / 3: task term 1/4 (u_f + u_r) + 1/2 u_f u_r, w_av about 6.7e-4
w_pos = _w("POS", 1.5)
PENALTY_WEIGHTS = {
    "tail": {"en": 0.0, "av": 0.002, "jv": 0.0, "time": 0.0},
    "notail": {"en": 0.0, "av": 0.002, "jv": 0.0, "time": 0.0},
}

# Ranges of the random release state
init_ang_vel_max = 0.5      # rad/s per axis, initial tumble
init_joint_pos_max = 0.2    # rad per joint, offset from the home pose
init_pitch_max_deg = 45.0   # deg; roll is drawn over the full circle and yaw is 0

# Per-channel low-pass gain on the normalized action, ordered roll, pitch, tail
# Must equal FILTER_ALPHA in hardware/controller.py
filter_alpha = np.array([0.1, 0.3, 0.1])

# Observation widths; the privileged (teacher) observation appends this episode's DR draw
# The DR block is normalized so that nominal is 0 and the sampled range is about [-1, 1]:
#   [ 0: 5]  body mass multipliers (the 5 bodies, world body excluded)
#   [ 5:20]  body COM offsets, xyz
#   [20:35]  body inertia multipliers, xyz
#   [35:47]  per motor group (2) x [damping, armature, friction, ctrlrange, kp, kd]
#   [   47]  action delay
BASE_OBS_DIM = 25
DR_DIM = 48


def _dr_scale(x, half_range, center=1.0):
    """Normalize a randomization draw so that `center` maps to 0 and the range to [-1, 1]."""
    return (np.asarray(x) - center) / half_range


class CatEnv(MujocoEnv, EzPickle):
    """Two-body robot with a spine and an optional tail, dropped to right itself in 1 s.

    Each step delays and low-passes the action, maps it onto the joint ranges and runs
    the firmware PD loop for 20 physics steps of 1 ms. The observation is yaw-invariant
    (see _get_obs).
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 50}

    def __init__(self, model_path=_DEFAULT_MODEL, render_mode=None, privileged=True):
        """Load the model and set up the index caches, PD loops and reward weights.

        Args:
            model_path: MuJoCo XML file, relative to the repository root or absolute.
            render_mode: Gymnasium render mode, or None.
            privileged: append the domain-randomization block to the observation.
        """
        # Make the path absolute: MujocoEnv joins a bare relative path onto its own assets
        model_path = os.path.abspath(model_path)

        # Define the spaces; only the privileged observation carries the DR block
        self.privileged = privileged
        self._dr_vec = np.zeros(DR_DIM)
        obs_dim = BASE_OBS_DIM + (DR_DIM if privileged else 0)
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,),
                                       dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Load the model (20 physics steps per control step) and record the pickling args
        MujocoEnv.__init__(
            self,
            model_path=model_path,
            frame_skip=20,
            observation_space=observation_space,
            default_camera_config={"distance": 3.0, "lookat": np.array([0.0, 0.0, 2])},
            render_mode=render_mode
        )
        self.action_space = action_space
        EzPickle.__init__(self, model_path=model_path, render_mode=render_mode,
                          privileged=privileged)

        # Look up the MuJoCo id of each body
        self._body_idx = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("front_body", "rear_body", "spine_1", "spine_2", "tail")}

        # Look up each joint's id, its qpos and qvel addresses and the actuator driving it
        joint_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ("rot1", "pitch", "rot2", "tail")}
        self._joint_qpos_idx = {name: self.model.jnt_qposadr[jid]
                                for name, jid in joint_ids.items()}
        self._joint_qvel_idx = {name: self.model.jnt_dofadr[jid]
                                for name, jid in joint_ids.items()}
        self._joint_id = joint_ids
        self._actuator_idx = {
            name: int(np.flatnonzero(self.model.actuator_trnid[:, 0] == jid)[0])
            for name, jid in joint_ids.items()}

        # Group the joints driven by the same motor and gearbox; each group shares one DR draw
        self._motor_groups = (("rot1", "rot2"), ("pitch", "tail"))

        # Keep the nominal physics parameters that each episode's randomization scales
        self.nominal_mass = self.model.body_mass.copy()
        self.nominal_damping = self.model.dof_damping.copy()
        self.nominal_ipos = self.model.body_ipos.copy()
        self.nominal_inertia = self.model.body_inertia.copy()
        self.nominal_armature = self.model.dof_armature.copy()
        self.nominal_frictionloss = self.model.dof_frictionloss.copy()
        self.nominal_ctrlrange = self.model.actuator_ctrlrange.copy()

        # Start the step count and the previous action of a 50-step (1 s) episode
        # max_steps must match MAX_STEPS and CONTROL_DURATION in hardware/controller.py
        self.steps = 0
        self.max_steps = 50
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)

        # Build one firmware PD loop per joint, with that joint's encoder resolution
        # Nominal (kp, kd) = firmware gains / 1024; rot1 and rot2 share gains (counter-twist)
        self.pd_nominal = [(4.0, 0.4), (20.0, 1.0), (4.0, 0.4), (20.0, 1.0)]
        self.joint_names = ("rot1", "pitch", "rot2", "tail")
        self.pd = [util.PDController(kp, kd,
                                     resolution=util.encoder_resolution(name),
                                     dt=self.model.opt.timestep)
                   for (kp, kd), name in zip(self.pd_nominal, self.joint_names)]

        # Read the penalty weights of this model's variant
        variant = "notail" if "notail" in os.path.basename(model_path) else "tail"
        weights = PENALTY_WEIGHTS[variant]
        self.w_en = _w("EN", weights["en"])
        self.w_av = _w("AV", weights["av"])
        self.w_jv = _w("JV", weights["jv"])
        self.w_time = _w("TIME", weights["time"])

        # Start the per-step log of mapped [roll, pitch, tail] targets and joint angles
        self.ctrls = []

    def step(self, action):
        """Run one 20 ms control step; return (obs, reward, terminated, truncated, info)."""
        # Count the step and clip the action to [-1, 1]
        self.steps += 1
        action = np.clip(action, -1, 1)

        # Delay the action by this episode's number of control steps
        if self.action_delay > 0:
            self.action_buffer.append(action.copy())
            executed_action = self.action_buffer.pop(0)
        else:
            executed_action = action.copy()

        # Low-pass the delayed action per channel, in normalized units
        self.action_filt = (filter_alpha * executed_action
                            + (1.0 - filter_alpha) * self.action_filt)
        executed_action = self.action_filt.copy()

        # Map the filtered action onto the ranges of rot1, pitch and tail (joints 1, 2, 4)
        roll_range = self.model.jnt_range[1][1]
        pitch_range = self.model.jnt_range[2][1]
        tail_range = self.model.jnt_range[4][1]
        executed_action[0] = util.map_value(executed_action[0], -1, 1, -roll_range, roll_range)
        executed_action[1] = util.map_value(executed_action[1], -1, 1, -pitch_range, pitch_range)
        executed_action[2] = util.map_value(executed_action[2], -1, 1, -tail_range, tail_range)

        # Order the joint targets as the actuators; rot2 counter-twists rot1
        targets = (executed_action[0], executed_action[1],
                   -executed_action[0], executed_action[2])

        # Run the firmware PD loop for one control period
        for _ in range(self.frame_skip):
            # Compute each joint's PD torque through the firmware deadband
            # The firmware tests the deadband on the encoder reading, not the true angle
            norm_torque = np.zeros(4)
            for i, (name, target) in enumerate(zip(self.joint_names, targets)):
                pos = self.data.qpos[self._joint_qpos_idx[name]]
                tau = self.pd[i].get_torque(target, pos)
                norm_torque[i] = util.apply_motor_deadband(
                    tau, target - self.pd[i].meas_pos, util.joint_deadband(name))

            # Scale the normalized torques onto the actuator control ranges
            physical_torque = np.zeros(4)
            for i in range(4):
                ctrl_min, ctrl_max = self.model.actuator_ctrlrange[i]
                physical_torque[i] = util.map_value(norm_torque[i], -1.0, 1.0,
                                                    ctrl_min, ctrl_max)

            # Advance the physics by one 1 ms step
            self.data.ctrl[:] = physical_torque
            mujoco.mj_step(self.model, self.data)

        # Log the mapped targets and the joint angles reached
        self.ctrls.append(np.hstack([executed_action, self.data.qpos[7:]]))

        # Observe, score and check for the end of the episode
        observation = self._get_obs()
        reward, info = self._get_reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        # Keep the clipped, unfiltered action for the next action-change penalty
        self.prev_action = action

        # Draw the frame in human render mode
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def reset_model(self):
        """Redraw the domain randomization and the release state; return the observation."""
        # Clear the step count, the previous action and the command log
        self.steps = 0
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.ctrls = []

        # Randomize the body masses
        mass_noise = np.random.uniform(0.8, 1.2, size=self.nominal_mass.shape)
        self.model.body_mass[:] = self.nominal_mass * mass_noise

        # Restore the nominal motor parameters, so the undriven root dofs stay nominal
        self.model.dof_damping[:] = self.nominal_damping
        self.model.dof_armature[:] = self.nominal_armature
        self.model.dof_frictionloss[:] = self.nominal_frictionloss
        self.model.actuator_ctrlrange[:] = self.nominal_ctrlrange

        # Randomize each motor group's damping, armature, friction, ctrlrange and PD gains
        dr_motor = []
        for group in self._motor_groups:
            # Draw the group's six scale factors
            damping_noise = np.random.uniform(0.7, 1.3)
            armature_noise = np.random.uniform(0.4, 2.5)
            friction_noise = np.random.uniform(0.7, 1.3)
            ctrlrange_noise = np.random.uniform(0.8, 1.2)
            kp_noise = np.random.uniform(0.8, 1.2)
            kd_noise = np.random.uniform(0.8, 1.2)

            # Normalize the draws; armature in log space, where its range is symmetric
            dr_motor.extend([
                _dr_scale(damping_noise, 0.3),
                np.log(armature_noise) / np.log(2.5),
                _dr_scale(friction_noise, 0.3),
                _dr_scale(ctrlrange_noise, 0.2),
                _dr_scale(kp_noise, 0.2),
                _dr_scale(kd_noise, 0.2),
            ])

            # Scale both joints of the group
            for name in group:
                dof = self._joint_qvel_idx[name]
                actuator = self._actuator_idx[name]
                joint = self.joint_names.index(name)
                self.model.dof_damping[dof] = self.nominal_damping[dof] * damping_noise
                self.model.dof_armature[dof] = self.nominal_armature[dof] * armature_noise
                self.model.dof_frictionloss[dof] = (self.nominal_frictionloss[dof]
                                                    * friction_noise)
                self.model.actuator_ctrlrange[actuator] = (self.nominal_ctrlrange[actuator]
                                                           * ctrlrange_noise)
                self.pd[joint].kp = self.pd_nominal[joint][0] * kp_noise
                self.pd[joint].kd = self.pd_nominal[joint][1] * kd_noise

        # Randomize the centres of mass, leaving the world body (index 0) in place
        ipos_noise = np.random.uniform(-0.04, 0.04, size=self.nominal_ipos.shape)
        ipos_noise[0] = 0.0
        self.model.body_ipos[:] = self.nominal_ipos + ipos_noise

        # Randomize the principal inertias
        inertia_noise = np.random.uniform(0.8, 1.2, size=self.nominal_inertia.shape)
        self.model.body_inertia[:] = self.nominal_inertia * inertia_noise

        # Draw the action delay (0 to 2 control steps) and fill its buffer with zeros
        self.action_delay = np.random.randint(0, 3)
        zero_action = np.zeros(self.action_space.shape)
        self.action_buffer = [zero_action.copy() for _ in range(self.action_delay)]

        # Start the action low-pass from zero: there is no command history at release
        self.action_filt = np.zeros(self.action_space.shape, dtype=np.float64)

        # Freeze this episode's normalized draw as the privileged observation block
        self._dr_vec = np.concatenate([
            _dr_scale(mass_noise[1:], 0.2),
            (ipos_noise[1:] / 0.04).ravel(),
            _dr_scale(inertia_noise[1:], 0.2).ravel(),
            np.array(dr_motor),
            [self.action_delay - 1.0],
        ])
        assert self._dr_vec.shape == (DR_DIM,), \
            f"DR_DIM={DR_DIM} does not match this model ({self._dr_vec.shape[0]})"

        # Recompute the model constants that depend on the randomized parameters
        mujoco.mj_setConst(self.model, self.data)

        # Draw the release attitude with yaw 0
        # Extrinsic xyz Euler angles; MuJoCo quaternions are [w, x, y, z]
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        roll_deg = np.random.uniform(-180.0, 180.0)
        pitch_deg = np.random.uniform(-init_pitch_max_deg, init_pitch_max_deg)
        quat_xyzw = R.from_euler("xyz", [roll_deg, pitch_deg, 0.0], degrees=True).as_quat()
        qpos[3:7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

        # Draw the initial tumble of the root body, conserved in flight
        qvel[3:6] = np.random.uniform(-init_ang_vel_max, init_ang_vel_max, size=3)

        # Offset each joint from the home pose, within its limits
        joint_noise = np.random.uniform(-init_joint_pos_max, init_joint_pos_max,
                                        size=len(self.joint_names))
        for name, noise in zip(self.joint_names, joint_noise):
            adr = self._joint_qpos_idx[name]
            lo, hi = self.model.jnt_range[self._joint_id[name]]
            qpos[adr] = np.clip(qpos[adr] + noise, lo, hi)

        # Place the robot in the release state
        self.set_state(qpos, qvel)

        # Seed each PD loop with its starting joint angle
        for pd, name in zip(self.pd, self.joint_names):
            pd.reset(self.data.qpos[self._joint_qpos_idx[name]])

        return self._get_obs()

    def _body_gyro(self, name):
        """Return a body's angular velocity in its own frame, as its IMU gyro reads it."""
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                 self._body_idx[name], vel, 1)
        return vel[:3]

    def _get_obs(self):
        """Return the yaw-invariant observation as float32.

        Base layout (25): front and rear projected gravity (3 + 3), front and rear gyro
        (3 + 3), joint angles and velocities in the order [rot1, pitch, rot2, tail]
        (4 + 4), actuator torques (4) and episode progress (1). The privileged DR block
        comes last, so that the student's slices of the base layout do not move.
        """
        # Read each body's orientation as world gravity seen in its own frame
        front_proj_grav = util.to_projected_gravity(self.data.xquat[self._body_idx["front_body"]])
        rear_proj_grav = util.to_projected_gravity(self.data.xquat[self._body_idx["rear_body"]])

        # Read each body's angular velocity in its own frame
        front_gyro = self._body_gyro("front_body")
        rear_gyro = self._body_gyro("rear_body")

        # Take the joint angles and velocities, without the free root joint
        joint_qpos = self.data.qpos[7:]
        joint_qvel = self.data.qvel[6:]

        # Take the actuator torques and the episode progress
        ctrl = self.data.ctrl
        step = np.array([self.steps / self.max_steps])

        # Stack the base observation and, for the teacher, this episode's DR draw
        obs = np.concatenate([
            front_proj_grav, rear_proj_grav,
            front_gyro, rear_gyro,
            joint_qpos, joint_qvel,
            ctrl, step
        ])
        if self.privileged:
            obs = np.concatenate([obs, self._dr_vec])
        return obs.astype(np.float32)

    def _get_reward(self, action):
        """Return the step reward and a dict of its terms.

        With u = 1 - tilt/pi per body (front f, rear r), the reward is

            r = w_pos * (0.5 * (u_f + u_r) + u_f * u_r)
                - w_en * m_en - w_av * m_av - w_jv * m_jv - w_time * m_time

        with m_en the mean squared actuator torque, m_av the mean squared body angular
        velocity, m_jv the mean squared joint velocity and m_time the simultaneity of the
        raw action changes across the three channels. The dict holds the weighted terms
        r_*, the unweighted magnitudes m_* and the mean uprightness. Action smoothness is
        penalized in the SAC actor loss (train/smooth_sac.py), not here.
        """
        # Score each body's uprightness, linear in tilt: 1 upright, 0 inverted
        # The clip keeps arccos defined when rounding puts the up-vector's z just above 1
        front_quat = self.data.xquat[self._body_idx["front_body"]]
        rear_quat = self.data.xquat[self._body_idx["rear_body"]]
        front_up = R.from_quat(front_quat, scalar_first=True).apply([0, 0, 1])
        rear_up = R.from_quat(rear_quat, scalar_first=True).apply([0, 0, 1])
        up_f = 1.0 - np.arccos(np.clip(front_up[2], -1.0, 1.0)) / np.pi
        up_r = 1.0 - np.arccos(np.clip(rear_up[2], -1.0, 1.0)) / np.pi

        # Reward each body being upright, plus both being upright together
        up_mean = 0.5 * (up_f + up_r)
        r_pos = w_pos * (up_mean + up_f * up_r)

        # Measure the actuator energy, body angular velocity and joint velocity
        m_en = np.mean(self.data.ctrl ** 2)
        omega = np.concatenate([self._body_gyro("front_body"), self._body_gyro("rear_body")])
        m_av = np.mean(omega ** 2)
        m_jv = np.mean(self.data.qvel[6:] ** 2)

        # Measure the simultaneity of the raw action changes; zero when at most one moves
        d_action = np.abs(action - self.prev_action)
        m_time = (d_action[0] * d_action[1] + d_action[0] * d_action[2]
                  + d_action[1] * d_action[2])

        # Weight the penalties and subtract them from the task reward
        r_en = self.w_en * m_en
        r_av = self.w_av * m_av
        r_jv = self.w_jv * m_jv
        r_time = self.w_time * m_time
        final_reward = r_pos - r_en - r_av - r_jv - r_time

        # Report the weighted terms, the penalty magnitudes and the mean uprightness
        reward_info = {
            "r_pos": r_pos,
            "r_en": r_en,
            "r_av": r_av,
            "r_jv": r_jv,
            "r_time": r_time,
            "m_en": m_en,
            "m_av": m_av,
            "m_jv": m_jv,
            "m_time": m_time,
            "up_mean": up_mean,
        }
        return final_reward, reward_info

    def _is_terminated(self):
        """Return False: an episode ends only by truncation."""
        return False

    def _is_truncated(self):
        """Return True once the episode has run max_steps control steps."""
        return self.steps >= self.max_steps
