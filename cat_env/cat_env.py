import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle
from scipy.spatial.transform import Rotation as R
import cat_env.env_util as util
import mujoco
import os

_DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "cat.xml")

# ---- reward ----
def _w(name, default):
    """Reward weight, overridable as CAT_W_<NAME> (e.g. CAT_W_AV=0.001) without a code edit."""
    return float(os.environ.get(f"CAT_W_{name}", default))

# Per-step reward (see _get_reward), with u = 1 - tilt/pi per body (front f, rear r):
#
#   r = w_pos * (0.5 * (u_f + u_r) + u_f * u_r)
#       - w_en * m_en - w_av * m_av - w_jv * m_jv - w_time * m_time
#
#   m_en   mean squared applied motor torque (energy)
#   m_av   mean squared body angular velocity, front + rear gyro
#   m_jv   mean squared joint velocity
#   m_time simultaneity of action changes across the 3 channels
#
# These defaults reproduce the released teachers (policies/cat_controller{,_notail}.zip,
# training run "av002"): the task term plus a body angular-velocity penalty only. The
# same weights are used for both variants, so the ablation compares morphologies under
# one reward function. In the paper's normalisation (reward divided by 3) the task term
# is 1/4 (u_f + u_r) + 1/2 u_f u_r and w_av corresponds to ~6.7e-4.
w_pos = _w("POS", 1.5)      # uprightness (task) weight

PENALTY_WEIGHTS = {   # keys: en, av, jv, time -> w_en, w_av, w_jv, w_time
    "tail":   {"en": 0.0, "av": 0.002, "jv": 0.0, "time": 0.0},
    "notail": {"en": 0.0, "av": 0.002, "jv": 0.0, "time": 0.0},
}

init_ang_vel_max = 0.5  # rad/s, per-axis range for random initial tumble
init_joint_pos_max = 0.2  # rad, per-joint range for random initial joint angles
init_pitch_max_deg = 45.0  # deg, release pitch range (roll is full; yaw is irrelevant)

# Per-channel first-order low-pass on the NORMALIZED action, applied before the
# joint-range mapping; mirrored by FILTER_ALPHA in hardware/controller.py.
#
# Sized against hardware logs: run unfiltered, the policy commanded ~10x the roll travel
# the roll joint (+-7.28 rad through 9.68:1 gearing) actually delivered, and ~2.1x on
# pitch (+-1.57 rad through 34:1). Replaying the logged actions through these alphas
# brings that to 1.7x (roll, 0.1) and 0.9x (pitch, 0.3). Tail follows roll: it shares
# pitch's gearing but has a wider range, and it is zeroed for the no-tail variant.
#
# The filter state persists within an episode, which is why the student observation
# includes the previous action (distillation.py).
filter_alpha = np.array([0.1, 0.3, 0.1])  # [roll, pitch, tail]
# --------------------------

# ---- privileged (teacher-only) domain-randomization block ----
# The teacher also observes this episode's DR draw. It is appended after the 25-dim
# base observation, so the student's slices of the base block (distillation.py) and
# the hardware observation layout are unaffected.
#
# Layout (all bodies below exclude the world body, index 0; 5 real bodies):
#   [ 0: 5]  body mass multipliers
#   [ 5:20]  body COM (ipos) offsets, xyz
#   [20:35]  body inertia multipliers, xyz
#   [35:47]  per motor GROUP (2) x [damping, armature, friction, ctrlrange, kp, kd]
#   [   47]  action delay
# Both model variants have the same bodies, joints and motor groups, so the width is
# identical for tail and no-tail.
#
# Every entry is normalized to put nominal at exactly 0 and the sampled range at
# about [-1, 1], rather than handing the network raw multipliers near 1.0.
DR_DIM = 48 # domain randomization vector width, teacher-only; see above
BASE_OBS_DIM = 25


def _dr_scale(x, half_range, center=1.0):
    """Multiplier (or offset) -> ~[-1, 1], nominal at 0."""
    return (np.asarray(x) - center) / half_range

class CatEnv(MujocoEnv, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 50}

    def __init__(self, model_path=_DEFAULT_MODEL, render_mode=None, privileged=True):
        model_path = os.path.abspath(model_path)

        # privileged=False gives the 25-dim base observation without the DR block.
        self.privileged = privileged
        self._dr_vec = np.zeros(DR_DIM)
        obs_dim = BASE_OBS_DIM + (DR_DIM if privileged else 0)

        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

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

        # Cache objects
        self._body_idx = {}
        self._joint_qpos_idx = {}
        self._joint_qvel_idx = {}

        for name in ["front_body", "rear_body", "spine_1", "spine_2", "tail"]:
            self._body_idx[name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

        self._joint_id = {}
        self._actuator_idx = {}
        for name in ["rot1", "pitch", "rot2", "tail"]:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._joint_id[name] = jid
            self._joint_qpos_idx[name] = self.model.jnt_qposadr[jid]
            self._joint_qvel_idx[name] = self.model.jnt_dofadr[jid]
            self._actuator_idx[name] = int(np.flatnonzero(self.model.actuator_trnid[:, 0] == jid)[0])

        # Motor groups: joints driven by the SAME motor + gearbox type. rot1/rot2 are
        # both the 9.68:1 roll motor; pitch/tail are both the 34.014:1 motor. The motor
        # randomization (damping, armature, friction, ctrlrange, PD gain) models
        # properties of the motor/gearbox, so it draws ONE multiplier per group, not
        # one per joint: the real robot's two roll motors do not behave differently.
        self._motor_groups = (("rot1", "rot2"), ("pitch", "tail"))

        # Nominal physics parameters; the per-episode DR scales these.
        self.nominal_mass = self.model.body_mass.copy()
        self.nominal_damping = self.model.dof_damping.copy()
        self.nominal_ipos = self.model.body_ipos.copy()
        self.nominal_inertia = self.model.body_inertia.copy()
        self.nominal_armature = self.model.dof_armature.copy()
        self.nominal_frictionloss = self.model.dof_frictionloss.copy()
        self.nominal_ctrlrange = self.model.actuator_ctrlrange.copy()

        # Initialize variables
        self.steps = 0
        # 50 steps x 20 ms = 1.0 s per drop; hardware/controller.py::CONTROL_DURATION
        # matches it.
        self.max_steps = 50
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)

        # Per-joint PD gains. Order: [rot1, pitch, rot2, tail].
        # kp*(err) - kd*filtered_vel -> normalized [-1,1].
        #
        # Set by tools/tune_pd_gains.py: a step response through this exact control
        # path (PD -> deadband + minimum-PWM floor -> ctrlrange -> 1 ms mj_step) on the
        # free-floating robot, taking the fastest settling among gains with no
        # overshoot and no limit cycle in the WORST case over the DR range, not just at
        # nominal (the armature spread alone is 6x). Re-run the tuner after any
        # inertia edit.
        #
        # The roll pair uses kd = 0.4 rather than 0.2 because the PD differentiates the
        # quantized 48 CPR encoder, which leaves kd = 0.2 under-damped. rot1/rot2 must
        # share gains: the policy drives rot2 as -rot1, so asymmetric roll gains would
        # turn the counter-twist into a net torque. Pitch and tail share a gain pair
        # because their step responses land within 1% of each other.
        self.pd_nominal = [(4.0, 0.4), (20.0, 1.0), (4.0, 0.4), (20.0, 1.0)]
        self.joint_names = ("rot1", "pitch", "rot2", "tail")
        # Each controller carries its own encoder resolution, so the inner loop
        # differentiates a quantized angle exactly as the Teensy does.
        self.pd = [util.PDController(kp, kd,
                                     resolution=util.encoder_resolution(name),
                                     dt=self.model.opt.timestep)
                   for (kp, kd), name in zip(self.pd_nominal, self.joint_names)]

        # Penalty weights for this model's variant, each overridable by
        # CAT_W_EN / CAT_W_AV / CAT_W_JV / CAT_W_TIME.
        variant = "notail" if "notail" in os.path.basename(model_path) else "tail"
        w = PENALTY_WEIGHTS[variant]
        self.w_en = _w("EN", w["en"])
        self.w_av = _w("AV", w["av"])
        self.w_jv = _w("JV", w["jv"])
        self.w_time = _w("TIME", w["time"])

        self.ctrls = []

    def step(self, action):
        self.steps += 1

        action = np.clip(action, -1, 1)

        # Random delay
        if self.action_delay > 0:
            self.action_buffer.append(action.copy())
            executed_action = self.action_buffer.pop(0)
        else:
            executed_action = action.copy()

        # Per-channel low-pass on the commanded (normalized) target. Applied AFTER the
        # delay buffer so the filter sees the action in the order the plant does, and
        # BEFORE the range mapping so alpha is expressed in normalized units and does
        # not silently change meaning if a joint range is edited.
        self.action_filt = (filter_alpha * executed_action
                            + (1.0 - filter_alpha) * self.action_filt)
        executed_action = self.action_filt.copy()

        # PD control
        roll_range = self.model.jnt_range[1][1]
        pitch_range = self.model.jnt_range[2][1]
        tail_range = self.model.jnt_range[4][1]
        executed_action[0] = util.map_value(executed_action[0], -1, 1, -roll_range, roll_range) # roll
        executed_action[1] = util.map_value(executed_action[1], -1, 1, -pitch_range, pitch_range) # pitch
        executed_action[2] = util.map_value(executed_action[2], -1, 1, -tail_range, tail_range) # tail

        # Joint targets in actuator order; rot2 counter-twists rot1.
        targets = (executed_action[0], executed_action[1],
                   -executed_action[0], executed_action[2])

        for _ in range(self.frame_skip):
            norm_torque = np.zeros(4)

            # Recalculate torque based on CURRENT micro-state, then apply the
            # Teensy deadband / minimum-PWM floor so the sim cannot exploit
            # fine torques the real inner loop is incapable of commanding.
            for i, (name, target) in enumerate(zip(self.joint_names, targets)):
                pos = self.data.qpos[self._joint_qpos_idx[name]]
                tau = self.pd[i].get_torque(target, pos)
                # The firmware's deadband test uses the encoder reading, not truth,
                # so the error compared here is the quantized one.
                norm_torque[i] = util.apply_motor_deadband(
                    tau, target - self.pd[i].meas_pos, util.joint_deadband(name))


            # Map normalized torque to physical torque
            physical_torque = np.zeros(4)
            for i in range(4):
                ctrl_min, ctrl_max = self.model.actuator_ctrlrange[i]
                physical_torque[i] = util.map_value(norm_torque[i], -1.0, 1.0, ctrl_min, ctrl_max)

            # Apply to MuJoCo and advance physics by exactly 1 ms
            self.data.ctrl[:] = physical_torque
            mujoco.mj_step(self.model, self.data)

        self.ctrls.append(np.hstack([executed_action, self.data.qpos[7:]]))
        observation = self._get_obs()
        reward, reward_info = self._get_reward(action)
        info = reward_info

        terminated = self._is_terminated()
        truncated = self._is_truncated()

        self.prev_action = action

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def reset_model(self):
        self.steps = 0
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        # Per-episode command log: rows of [mapped target (roll, pitch, tail),
        # joint qpos (4)]. Cleared here rather than at episode end so the finished
        # episode stays readable (e.g. for plotting joint tracking).
        self.ctrls = []
        # Domain randomization
        # Mass
        mass_noise = np.random.uniform(0.8, 1.2, size=self.nominal_mass.shape)
        self.model.body_mass[:] = self.nominal_mass * mass_noise

        # Motor/gearbox parameters. One draw per motor GROUP (see _motor_groups).
        # Start from nominal so the root free-joint dofs (which no motor drives)
        # stay untouched.
        self.model.dof_damping[:] = self.nominal_damping
        self.model.dof_armature[:] = self.nominal_armature
        self.model.dof_frictionloss[:] = self.nominal_frictionloss
        self.model.actuator_ctrlrange[:] = self.nominal_ctrlrange

        dr_motor = []
        for group in self._motor_groups:
            # Damping is a nominal estimate (from motor no-load speed / stall torque),
            # so randomize it wider (+/-30%) than the other params.
            damping_noise = np.random.uniform(0.7, 1.3)
            # Armature is reflected rotor inertia estimated from the motor class (no
            # datasheet value), so it gets the widest range: 0.4-2.5x, symmetric in
            # log space (0.4 = 1/2.5). The draw itself is uniform.
            armature_noise = np.random.uniform(0.4, 2.5)
            friction_noise = np.random.uniform(0.7, 1.3)
            # Actuator authority. ctrlrange encodes stall torque, which moves with LiPo
            # sag across the drop, motor temperature and unit-to-unit spread. Scaling
            # both limits keeps the range symmetric, so this is a pure torque scale.
            ctrlrange_noise = np.random.uniform(0.8, 1.2)
            # PD gains. The flashed inner-loop gains equal pd_nominal, but the
            # PWM -> torque constant they act through is not measured, so the
            # effective closed-loop gain is uncertain. Also a motor property, hence
            # per group.
            kp_noise = np.random.uniform(0.8, 1.2)
            kd_noise = np.random.uniform(0.8, 1.2)

            # Armature's range is symmetric in log space, so it is normalized in log
            # space; a linear min-max scaling would put nominal at about -0.43 instead of 0.
            dr_motor.extend([
                _dr_scale(damping_noise, 0.3),
                np.log(armature_noise) / np.log(2.5),
                _dr_scale(friction_noise, 0.3),
                _dr_scale(ctrlrange_noise, 0.2),
                _dr_scale(kp_noise, 0.2),
                _dr_scale(kd_noise, 0.2),
            ])

            for name in group:
                dof = self._joint_qvel_idx[name]
                act = self._actuator_idx[name]
                i = self.joint_names.index(name)
                self.model.dof_damping[dof] = self.nominal_damping[dof] * damping_noise
                self.model.dof_armature[dof] = self.nominal_armature[dof] * armature_noise
                self.model.dof_frictionloss[dof] = self.nominal_frictionloss[dof] * friction_noise
                self.model.actuator_ctrlrange[act] = self.nominal_ctrlrange[act] * ctrlrange_noise
                self.pd[i].kp = self.pd_nominal[i][0] * kp_noise
                self.pd[i].kd = self.pd_nominal[i][1] * kd_noise

        # COM position
        ipos_noise = np.random.uniform(-0.04, 0.04, size=self.nominal_ipos.shape)
        ipos_noise[0] = 0.0  # Crucial: Do not move the world body (index 0)
        self.model.body_ipos[:] = self.nominal_ipos + ipos_noise

        # Inertia tensor
        inertia_noise = np.random.uniform(0.8, 1.2, size=self.nominal_inertia.shape)
        self.model.body_inertia[:] = self.nominal_inertia * inertia_noise

        # Delay
        self.action_delay = np.random.randint(0, 3)
        zero_action = np.zeros(self.action_space.shape)
        self.action_buffer = [zero_action.copy() for _ in range(self.action_delay)]

        # Action low-pass state, neutral at release (the robot is dropped with its
        # joints wherever they were, but with no command history).
        self.action_filt = np.zeros(self.action_space.shape, dtype=np.float64)

        # Freeze this episode's draw as the privileged block of the observation.
        # The DR is resampled only here, so the block is constant within an episode.
        self._dr_vec = np.concatenate([
            _dr_scale(mass_noise[1:], 0.2),
            (ipos_noise[1:] / 0.04).ravel(),
            _dr_scale(inertia_noise[1:], 0.2).ravel(),
            np.array(dr_motor),
            [self.action_delay - 1.0],  # {0,1,2} steps -> {-1,0,1}
        ])
        assert self._dr_vec.shape == (DR_DIM,), \
            f"DR_DIM={DR_DIM} does not match this model ({self._dr_vec.shape[0]})"

        # Set physics params
        mujoco.mj_setConst(self.model, self.data)

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Release attitude: full roll, pitch within +-45 deg, as in a hand drop
        # (rolled about the long axis, roughly level fore-aft). Yaw is fixed at 0: the
        # observation and reward are yaw-invariant. The Euler order matches
        # evaluate.py::force_attitude and view_policy.py, so --roll/--pitch there land
        # on this distribution.
        roll_deg = np.random.uniform(-180.0, 180.0)
        pitch_deg = np.random.uniform(-init_pitch_max_deg, init_pitch_max_deg)
        quat_xyzw = R.from_euler("xyz", [roll_deg, pitch_deg, 0.0], degrees=True).as_quat()
        qpos[3:7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

        # Initial tumble: root free-joint angular velocity lives in qvel[3:6]. Contact
        # is disabled, so the resulting angular momentum is conserved for the episode.
        qvel[3:6] = np.random.uniform(-init_ang_vel_max, init_ang_vel_max, size=3)

        # Initial joint angles: the release pose is never exactly the nominal home pose
        # (zero offsets, backlash, hand placement). Clipped to each joint's limits.
        joint_noise = np.random.uniform(-init_joint_pos_max, init_joint_pos_max,
                                        size=len(self.joint_names))
        for name, noise in zip(self.joint_names, joint_noise):
            adr = self._joint_qpos_idx[name]
            lo, hi = self.model.jnt_range[self._joint_id[name]]
            qpos[adr] = np.clip(qpos[adr] + noise, lo, hi)

        self.set_state(qpos, qvel)

        # Per-episode inner-loop state, seeded from the pose actually being started
        # from: initial joint angles are randomized, so differencing against 0 would
        # read as a large first-sample velocity.
        for pd, name in zip(self.pd, self.joint_names):
            pd.reset(self.data.qpos[self._joint_qpos_idx[name]])

        return self._get_obs()

    def _body_gyro(self, name):
        """Body-frame angular velocity of a body (what its IMU gyro would read)."""
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                 self._body_idx[name], vel, 1)
        return vel[:3]

    def _get_obs(self):
        # The observation is strictly yaw-invariant: nothing below depends on the
        # robot's heading about world-z. Absolute root position/orientation and
        # world-frame root velocities are dropped (heading-dependent and useless
        # for a contact-free mid-air righting task).

        # Projected gravity (yaw-invariant orientation cue): world -z in body frame.
        # Captures exactly the non-yaw part of each body's orientation.
        front_proj_grav = util.to_projected_gravity(self.data.xquat[self._body_idx["front_body"]])
        rear_proj_grav = util.to_projected_gravity(self.data.xquat[self._body_idx["rear_body"]])

        # Gyro (body-frame angular velocity, yaw-invariant)
        front_gyro = self._body_gyro("front_body")
        rear_gyro = self._body_gyro("rear_body")

        # Joint state only (root free-joint qpos[:7]/qvel[:6] dropped): 4 joints in
        # XML order [rot1, pitch, rot2, tail]. Joint-space -> yaw-invariant.
        joint_qpos = self.data.qpos[7:]
        joint_qvel = self.data.qvel[6:]

        # Control signal (joint-space torques, yaw-invariant)
        ctrl = self.data.ctrl
        step = np.array([self.steps / self.max_steps])

        obs = np.concatenate([
            front_proj_grav, rear_proj_grav,
            front_gyro, rear_gyro,
            joint_qpos, joint_qvel,
            ctrl, step
        ])

        # Privileged tail: this episode's domain-randomization draw. The student's obs
        # is built from slices of the block above (distillation.py), so it never sees
        # this, and it must stay LAST so those slice indices never move.
        if self.privileged:
            obs = np.concatenate([obs, self._dr_vec])
        return obs.astype(np.float32)

    def _get_reward(self, action):
        front_quat = self.data.xquat[self._body_idx["front_body"]]
        rear_quat = self.data.xquat[self._body_idx["rear_body"]]

        # World-frame body up-vectors (local +z mapped to world).
        front_up = R.from_quat(front_quat, scalar_first=True).apply([0, 0, 1])
        rear_up = R.from_quat(rear_quat, scalar_first=True).apply([0, 0, 1])

        # Per-body uprightness, LINEAR in tilt angle: 1 at upright, 0 at inverted.
        # The clip is load-bearing: `apply` can return 1.0 + 1e-16, and arccos of that
        # is NaN.
        up_f = 1.0 - np.arccos(np.clip(front_up[2], -1.0, 1.0)) / np.pi
        up_r = 1.0 - np.arccos(np.clip(rear_up[2], -1.0, 1.0)) / np.pi

        # Dense, per-step, no time ramp: every upright step counts, so under the SAC
        # discount the optimal policy rights as early as possible and holds. The sum
        # term gives each body an independent gradient (no vanishing when the other is
        # inverted); the product term rewards both being upright together.
        r_pos = w_pos * (0.5 * (up_f + up_r) + up_f * up_r)

        # --- penalty magnitudes, unweighted ---
        # Reported in `info` as m_* alongside the weighted r_*, so the quantity a
        # weight prices stays visible even when the weight is 0 (evaluate.py --stats).
        # Action smoothness is not a reward term: it lives in the SAC actor loss
        # (smooth_sac.py). d_action is used only by m_time.
        d_action = np.abs(action - self.prev_action)

        # Applied torque: energy.
        m_en = np.mean(self.data.ctrl ** 2)

        # Body angular velocity: an upright pose reached while still tumbling is not a
        # landing. Angular momentum is conserved mid-air, so the policy can only park
        # the bodies by moving the residual rotation into the joints.
        omega = np.concatenate([self._body_gyro("front_body"), self._body_gyro("rear_body")])
        m_av = np.mean(omega ** 2)

        # Joint velocity: flailing and fast reversals a geared motor cannot track.
        m_jv = np.mean(self.data.qvel[6:] ** 2)

        # Simultaneity: pairwise products of the per-channel action change, zero
        # whenever at most one channel moves. It is computed on the raw action,
        # upstream of the action low-pass, so it does not measure physical joint
        # motion; its default weight is 0.
        m_time = (d_action[0] * d_action[1] + d_action[0] * d_action[2]
                  + d_action[1] * d_action[2])

        r_en = self.w_en * m_en
        r_av, r_jv, r_time = self.w_av * m_av, self.w_jv * m_jv, self.w_time * m_time

        final_reward = r_pos - r_en - r_av - r_jv - r_time
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
            "up_mean": 0.5 * (up_f + up_r),
        }

        return final_reward, reward_info

    def _is_terminated(self):
        return False

    def _is_truncated(self):
        return self.steps >= self.max_steps
