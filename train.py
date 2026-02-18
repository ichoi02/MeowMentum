import gymnasium as gym
from stable_baselines3 import PPO
import env.cat_env as cat_env 

def train():
    env = gym.make("Slider-v0", render_mode=None)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./run_logs/")

    model.learn(total_timesteps=20000)

    model_path = "ppo_slider_model"
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

if __name__ == "__main__":
    train()