import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import cat_env 

# ---- 1. Re-define the Student Policy ----
class StudentPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

def get_student_obs(full_obs):
    """Extracts the 8 quaternion values from the full 40-dim observation."""
    return full_obs[6:14]

# ---- 2. Visualization Logic ----
def visualize_error():
    # Setup Environment
    env = gym.make("Cat-v0")
    
    # Load Models
    print("Loading expert policy...")
    expert = PPO.load("cat_controller")
    
    print("Loading student policy...")
    student_obs_dim = 8
    act_dim = env.action_space.shape[0]
    student = StudentPolicy(student_obs_dim, act_dim)
    student.load_state_dict(torch.load("student_policy_quats_only.pth"))
    student.eval()

    # Rollout one episode
    full_obs, _ = env.reset()
    done = False
    
    expert_actions = []
    student_actions = []
    steps = []
    step_count = 0

    while not done:
        # Get Expert Action (uses full state)
        exp_action, _ = expert.predict(full_obs, deterministic=True)
        
        # Get Student Action (uses partial state)
        student_obs = get_student_obs(full_obs)
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(student_obs).unsqueeze(0)
            stud_action = student(obs_tensor).squeeze(0).numpy()
            
        expert_actions.append(exp_action)
        student_actions.append(stud_action)
        steps.append(step_count)
        
        # Step the environment using the *expert* action to keep the trajectory stable 
        # (You can change this to stud_action to see how the student handles its own drift)
        full_obs, _, terminated, truncated, _ = env.step(exp_action)
        done = terminated or truncated
        step_count += 1

    env.close()

    # Convert to numpy arrays for plotting
    expert_actions = np.array(expert_actions)
    student_actions = np.array(student_actions)
    
    # Calculate MSE per timestep across the 3 action dimensions
    mse_per_step = np.mean((expert_actions - student_actions)**2, axis=1)

    # ---- 3. Plotting ----
    action_names = ["Roll", "Pitch", "Tail"]
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    # Plot individual action dimensions
    for i in range(3):
        axs[i].plot(steps, expert_actions[:, i], label='Expert (Teacher)', linestyle='--', color='blue', alpha=0.7)
        axs[i].plot(steps, student_actions[:, i], label='Student', color='orange', alpha=0.9)
        axs[i].set_ylabel(f"{action_names[i]} Action")
        axs[i].set_title(f"Predicted Action: {action_names[i]}")
        axs[i].legend(loc="upper right")
        axs[i].grid(True)

    # Plot total MSE
    axs[3].plot(steps, mse_per_step, color='red')
    axs[3].set_ylabel("Mean Squared Error")
    axs[3].set_xlabel("Timestep")
    axs[3].set_title("Total Action Error (MSE) over Time")
    axs[3].grid(True)

    plt.tight_layout()
    plt.savefig("teacher_student_error.png")
    print("Graph saved as teacher_student_error.png")
    plt.show()

if __name__ == "__main__":
    visualize_error()