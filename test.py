import gymnasium as gym
from stable_baselines3 import PPO
import os
import time
import mujoco
import mujoco.viewer
import cat_env

def visualize():
    env = gym.make("Cat-v0")
    model_path = "cat_controller"
     
    if not os.path.exists(f"{model_path}.zip"):
        print(f"Error: Model file '{model_path}.zip' not found.")
        return

    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    obs, _ = env.reset()
    
    mj_model = env.unwrapped.model
    mj_data = env.unwrapped.data

    print("Starting visualization... Press Ctrl+C or close the viewer to stop.")
    
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # Cam tracking
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "spine_1")
        viewer.cam.trackbodyid = body_id
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90

        slow = 0.1

        try:
            while viewer.is_running():
                step_start = time.time()
                
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    obs, _ = env.reset()
                
                viewer.sync()
                
                time_until_next_step = env.unwrapped.dt / slow - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                    
        except KeyboardInterrupt:
            print("\nVisualization stopped.")
        finally:
            env.close()

if __name__ == "__main__":
    visualize()