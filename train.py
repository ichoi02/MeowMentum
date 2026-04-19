import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import cat_env
import time

class TensorboardRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # locals["infos"] is a list of info dictionaries from the vectorized environments
        for info in self.locals["infos"]:
            if "r_pos" in info:
                # Log each term under a "rewards/" group in TensorBoard
                self.logger.record("rewards/r_pos", info["r_pos"])
                self.logger.record("rewards/r_sm", info["r_sm"])
                self.logger.record("rewards/r_en", info["r_en"])
                self.logger.record("rewards/penalty_factor", info["penalty_factor"])
        return True

def train():
    num_cpu = 10  # Change this to match the number of logical cores on your CPU
    
    # make_vec_env handles creating multiple instances of your environment
    env = make_vec_env(
        "Cat-v0", 
        n_envs=num_cpu, 
        vec_env_cls=SubprocVecEnv,                # Uses true multiprocessing
        env_kwargs={"render_mode": "rgb_array"}   # Pass your gym.make kwargs here
    )
    
    model = PPO(
        "MlpPolicy", 
        env, 
        policy_kwargs=dict(net_arch=[256, 256]), 
        verbose=1, 
        tensorboard_log="./run_logs/", 
        device="auto",
        seed=None,
    )

    reward_callback = TensorboardRewardCallback()

    model.learn(
        total_timesteps=2000000,
        log_interval=1,
        progress_bar=True,
        callback=reward_callback
    )

    model_path = f"cat_controller_{str(time.time())}.zip"
    model.save(model_path)
    print(f"Model saved")

if __name__ == "__main__":
    train()