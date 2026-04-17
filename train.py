import gymnasium as gym
from stable_baselines3 import PPO
import cat_env
import time

def train():
    env = gym.make("Cat-v0", render_mode="rgb_array")
    
    model = PPO(
        "MlpPolicy", 
        env, 
        policy_kwargs=dict(net_arch=[128, 128]), 
        verbose=1, 
        tensorboard_log="./run_logs/", 
        device="auto",
        seed=7,
        )

    model.learn(
        total_timesteps=1000000,
        log_interval=10,
        progress_bar=True
        )

    model_path = f"cat_controller_{str(time.time())}"
    model.save(model_path)
    print(f"Model saved")

if __name__ == "__main__":
    train()