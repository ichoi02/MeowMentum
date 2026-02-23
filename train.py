import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import cat_env

def train():
    env = gym.make("Cat-v0", render_mode="rgb_array")
    # env = make_vec_env("Cat-v0", n_envs=16)
    
    model = PPO(
        "MlpPolicy", 
        env, 
        policy_kwargs=dict(net_arch=[64, 64]), 
        verbose=1, 
        tensorboard_log="./run_logs/", 
        device="auto",
        )

    model.learn(
        total_timesteps=200000,
        log_interval=1,
        progress_bar=True
        )

    model_path = "cat_controller"
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

if __name__ == "__main__":
    train()