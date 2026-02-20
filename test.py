import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
import cat_env

def visualize():
    # Create the environment with human rendering
    env = gym.make("Cat-v0", render_mode="human")

    # Path to the trained model
    model_path = "cat_controller"
    
    if not os.path.exists(f"{model_path}.zip"):
        print(f"Error: Model file '{model_path}.zip' not found.")
        return

    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    # Reset the environment
    obs, _ = env.reset()
    
    print("Starting visualization... Press Ctrl+C to stop.")
    try:
        while True:
            # Get action from the model (deterministic=True for best performance)
            action, _ = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\nVisualization stopped.")
    finally:
        env.close()

if __name__ == "__main__":
    visualize()