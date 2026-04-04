import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle
from scipy.spatial.transform import Rotation as R
import cat_env.env_util as util
import mujoco
import os

np.random.seed(None)

# ---- train parameters ----
w_pos = 1.0 # pose reward weight
w_sm = -0.5 # smoothness penalty weight
w_en = -5.0 # energy penalty weight
k = 0.1  # tanh gain param
# --------------------------

class CatEnv(MujocoEnv, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        model_path = os.path.abspath("model/cat.xml")
        
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(40,), dtype=np.float64)
        action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            model_path=model_path,
            frame_skip=20,
            observation_space=observation_space,
            default_camera_config={"distance": 3.0, "lookat": np.array([0.0, 0.0, 2])},
            render_mode=render_mode
        )
        self.action_space = action_space
        EzPickle.__init__(self)
        
        # Cache objects
        self._body_idx = {}
        self._joint_qpos_idx = {}
        self._joint_qvel_idx = {}
        
        for name in ["front_body", "rear_body", "spine_1", "spine_2", "tail"]:
            self._body_idx[name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            
        for name in ["rot1", "pitch", "rot2", "tail"]:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._joint_qpos_idx[name] = self.model.jnt_qposadr[jid]
            self._joint_qvel_idx[name] = self.model.jnt_dofadr[jid]

        # Cache nomical physics parameters
        self.nominal_mass = self.model.body_mass.copy()
        self.nominal_damping = self.model.dof_damping.copy()
        self.nominal_ipos = self.model.body_ipos.copy()
        self.nominal_inertia = self.model.body_inertia.copy()

        # Initialize variables
        self.steps = 0
        self.max_steps = 37
        self.prev_action = np.zeros_like(action_space.shape)

        self.pd = []
        self.pd.append(util.PDController(2.0, 0.2))
        self.pd.append(util.PDController(20.0, 2.0))
        self.pd.append(util.PDController(1.0, 0.1))
        self.pd.append(util.PDController(1.0, 0.1))

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

        # PD control
        executed_action[0] = util.map_value(executed_action[0], -1, 1, -np.pi*2, np.pi*2) # roll
        executed_action[1] = util.map_value(executed_action[1], -1, 1, -np.pi/2, np.pi/2) # pitch
        executed_action[2] = util.map_value(executed_action[2], -1, 1, -np.pi/2, np.pi/2) # tail

        for _ in range(self.frame_skip):
            norm_torque = np.zeros(4)

            # Recalculate torque based on CURRENT micro-state
            norm_torque[0] = self.pd[0].get_torque(executed_action[0],
                                              self.data.qpos[self._joint_qpos_idx["rot1"]], 
                                              self.data.qvel[self._joint_qvel_idx["rot1"]])
            norm_torque[1] = self.pd[1].get_torque(executed_action[1],
                                              self.data.qpos[self._joint_qpos_idx["pitch"]], 
                                              self.data.qvel[self._joint_qvel_idx["pitch"]])
            norm_torque[2] = self.pd[2].get_torque(-executed_action[0],
                                              self.data.qpos[self._joint_qpos_idx["rot2"]], 
                                              self.data.qvel[self._joint_qvel_idx["rot2"]])
            norm_torque[3] = self.pd[3].get_torque(executed_action[2],
                                              self.data.qpos[self._joint_qpos_idx["tail"]], 
                                              self.data.qvel[self._joint_qvel_idx["tail"]])
            
            # Map normalized torque to physical torque
            physical_torque = np.zeros(4)
            for i in range(4):
                ctrl_min, ctrl_max = self.model.actuator_ctrlrange[i]
                physical_torque[i] = util.map_value(norm_torque[i], -1.0, 1.0, ctrl_min, ctrl_max)
            
            # Apply to MuJoCo and advance physics by exactly 1 ms
            self.data.ctrl[:] = physical_torque
            mujoco.mj_step(self.model, self.data)
        
        # self.ctrls.append(np.hstack([executed_action, self.data.qpos[7:]]))
        # self.ctrls.append(physical_torque)
        observation = self._get_obs()
        reward = self._get_reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = {}

        self.prev_action = action

        if terminated or truncated:
            # np.save(f"control.npy", np.array(self.ctrls))
            self.ctrls = []
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def reset_model(self):
        self.steps = 0
        self.prev_action = np.zeros_like(self.action_space.shape)

        # Domain randomization
        # Mass
        mass_noise = np.random.uniform(0.9, 1.1, size=self.nominal_mass.shape)
        self.model.body_mass[:] = self.nominal_mass * mass_noise

        # Joint Damping
        damping_noise = np.random.uniform(0.8, 1.2, size=self.nominal_damping.shape)
        self.model.dof_damping[:] = self.nominal_damping * damping_noise

        # COM position
        ipos_noise = np.random.uniform(-0.03, 0.03, size=self.nominal_ipos.shape)
        ipos_noise[0] = 0.0  # Crucial: Do not move the world body (index 0)
        self.model.body_ipos[:] = self.nominal_ipos + ipos_noise

        # Inertia tensor
        inertia_noise = np.random.uniform(0.8, 1.2, size=self.nominal_inertia.shape)
        self.model.body_inertia[:] = self.nominal_inertia * inertia_noise

        # Delay
        self.action_delay = np.random.randint(0, 4)
        zero_action = np.zeros(self.action_space.shape)
        self.action_buffer = [zero_action.copy() for _ in range(self.action_delay)]

        # Set physics params
        mujoco.mj_setConst(self.model, self.data)

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # randomize initial orientation
        self.random_roll = np.random.uniform(-np.pi, np.pi)
        random_pitch = np.random.uniform(-np.pi/4, np.pi/4)

        # Set initial rot/pos
        r = R.from_euler("xyz", [self.random_roll, random_pitch, 0], degrees=False)
        quat_xyzw = r.as_quat()
        qpos[3:7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

        self.set_state(qpos, qvel)

        return self._get_obs()

    def _get_obs(self):
        front_body_pos = self.data.xpos[self._body_idx["front_body"]]
        rear_body_pos = self.data.xpos[self._body_idx["rear_body"]]

        front_body_quat = self.data.xquat[self._body_idx["front_body"]]
        rear_body_quat = self.data.xquat[self._body_idx["rear_body"]]

        qpos = self.data.qpos
        qvel = self.data.qvel
        qacc = self.data.qacc

        ctrl = self.data.ctrl
        step = np.array([self.steps / self.max_steps])
        
        obs = np.concatenate([
            front_body_pos, rear_body_pos,
            front_body_quat, rear_body_quat,
            qpos, qvel, #qacc,
            ctrl, step
        ])
        return obs

    def _get_reward(self, action):
        front_quat = self.data.xquat[self._body_idx["front_body"]]
        rear_quat = self.data.xquat[self._body_idx["rear_body"]]

        # rotation matricies
        r_front = R.from_quat(front_quat[[1, 2, 3, 0]])
        r_rear = R.from_quat(rear_quat[[1, 2, 3, 0]])

        # transform local z vectors to global
        front_up = -r_front.apply([0, 0, 1])
        rear_up = -r_rear.apply([0, 0, 1])

        angle_front = np.arccos(np.clip(front_up[2], -1.0, 1.0))
        angle_rear = np.arccos(np.clip(rear_up[2], -1.0, 1.0))
        
        # get z component and scale
        reward_front = 1.0 - (angle_front / np.pi)
        reward_rear = 1.0 - (angle_rear / np.pi)
        
        r_pos = reward_front*reward_rear
        r_pos *= np.tanh(self.steps*k)
        
        # smoothness reward
        delta = action - self.prev_action
        r_sm = np.mean(delta**2)

        # energy consumption reward
        ctrl = self.data.ctrl
        r_en = np.mean(ctrl**2)

        return w_pos*r_pos + w_sm*r_sm + w_en*r_en
    
    def _is_terminated(self):
        return False

    def _is_truncated(self):
        return self.steps >= self.max_steps
