import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle
from scipy.spatial.transform import Rotation as R
import cat_env.env_util as util
import mujoco
import os

class CatEnv(MujocoEnv, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 100}

    def __init__(self, render_mode=None):
        model_path = os.path.abspath("model/cat.xml")
        
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float64)
        action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            model_path=model_path,
            frame_skip=1,
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

        # Initialize variables
        self.steps = 0
        self.max_steps = 75

        self.pd = []
        self.pd.append(util.PDController(10, 1))
        self.pd.append(util.PDController(10, 1))
        self.pd.append(util.PDController(10, 1))
        self.pd.append(util.PDController(10, 1))

    def step(self, action):
        self.steps += 1
        
        action = np.clip(action, -1, 1)

        action[0] = util.map_value(action[0], -1, 1, -np.pi*2, np.pi*2) # roll
        action[1] = util.map_value(action[1], -1, 1, -np.pi*2, np.pi*2) # pitch
        action[2] = util.map_value(action[2], -1, 1, -np.pi/2, np.pi/2) # tail

        torque = np.zeros(4)
        torque[0] = self.pd[0].get_torque(action[0],
                                          self.data.qpos[self._joint_qpos_idx["rot1"]], 
                                          self.data.qvel[self._joint_qvel_idx["rot1"]])
        torque[1] = self.pd[1].get_torque(action[1],
                                          self.data.qpos[self._joint_qpos_idx["pitch"]], 
                                          self.data.qvel[self._joint_qvel_idx["pitch"]])
        torque[2] = self.pd[2].get_torque(-action[0],
                                          self.data.qpos[self._joint_qpos_idx["rot2"]], 
                                          self.data.qvel[self._joint_qvel_idx["rot2"]])
        torque[3] = self.pd[3].get_torque(action[2],
                                          self.data.qpos[self._joint_qpos_idx["tail"]], 
                                          self.data.qvel[self._joint_qvel_idx["tail"]])
        
        self.do_simulation(torque, self.frame_skip)

        observation = self._get_obs()
        reward = self._get_reward()
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def reset_model(self):
        self.steps = 0
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        
        # randomize initial rotation
        random_quat = np.random.rand(4)
        random_quat /= np.linalg.norm(random_quat)
        qpos[3:7] = random_quat

        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        front_body_quat = self.data.xquat[self._body_idx["front_body"]]
        rear_body_quat = self.data.xquat[self._body_idx["rear_body"]]
        
        obs = np.concatenate([
            front_body_quat, 
            rear_body_quat, 
            self.data.qpos, 
            self.data.qvel
        ])
        return obs

    def _get_reward(self):
        front_quat = self.data.xquat[self._body_idx["front_body"]]
        rear_quat = self.data.xquat[self._body_idx["rear_body"]]
        spine1_quat = self.data.xquat[self._body_idx["spine_1"]]
        spine2_quat = self.data.xquat[self._body_idx["spine_2"]]

        # rotation matricies
        r_front = R.from_quat(front_quat[[1, 2, 3, 0]])
        r_rear = R.from_quat(rear_quat[[1, 2, 3, 0]])
        r_spine1 = R.from_quat(spine1_quat[[1, 2, 3, 0]])
        r_spine2 = R.from_quat(spine2_quat[[1, 2, 3, 0]])

        # transform local z vectors to global
        front_up = -r_front.apply([0, 0, 1])
        rear_up = -r_rear.apply([0, 0, 1])
        spine1_up = -r_spine1.apply([0, 0, 1])
        spine2_up = -r_spine2.apply([0, 0, 1])

        # get z component and scale
        reward_front = (front_up[2] + 1) / 2
        reward_rear = (rear_up[2] + 1) / 2
        reward_spine1 = (spine1_up[2] + 1) / 2
        reward_spine2 = (spine2_up[2] + 1) / 2
        
        reward = reward_front*reward_rear*reward_spine1*reward_spine2
        reward *= np.tanh(self.steps*0.05) # lower the reward weights

        return reward
    
    def _is_terminated(self):
        return False

    def _is_truncated(self):
        return self.steps >= self.max_steps


