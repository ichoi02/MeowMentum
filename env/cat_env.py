import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle
import os

class CatEnv(MujocoEnv, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 500}

    def __init__(self, render_mode=None):
        model_path = os.path.abspath("model/cat.urdf")

        dummy_obs = self._get_obs()
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=dummy_obs.shape, dtype=np.float64)

        MujocoEnv.__init__(
            self,
            model_path=model_path,
            frame_skip=1,
            observation_space=observation_space,
            default_camera_config={"distance": 2.0},
            render_mode=render_mode
        )
        EzPickle.__init__(self)

        self.steps = 0

    def step(self, action):
        self.steps += 1
        
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()

        # qpos[0] is the slider position
        position = self.data.qpos[0]
        reward = -np.abs(position)

        terminated = False
        truncated = True if self.steps >= 300 else False
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        return np.array([])

    def reset_model(self):
        # Reset joint positions and velocities to random small values
        qpos = self.init_qpos + self.np_random.uniform(low=-0.05, high=0.05, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.05, high=0.05, size=self.model.nv)
        
        # Set the state
        self.set_state(qpos, qvel)
        return self._get_obs()
