import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from stable_baselines3 import PPO
import cat_env
import cat_env.env_util as util
import time

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

def get_noisy_student_obs(full_obs, rot_noise_std=0.01, joint_noise_std=0.02):
    # Extract clean data
    rots = full_obs[0:18].copy() # two 9D rotation matrices
    joint_angles = full_obs[18+7:18+7+4].copy() # qpos[7:]

    # Apply noise using your env_util function
    noisy_rots = util.add_rotational_noise(rots, rot_noise_std)
    noisy_joints = util.add_gaussian_noise(joint_angles, joint_noise_std)

    return np.concatenate([noisy_rots, noisy_joints])

# ---- 1. Define the Student Policy ----
class StudentPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
            nn.Tanh() # Binds outputs to [-1, 1]
        )

    def forward(self, x):
        return self.net(x)

# ---- 2. Data Collection Logic ----
def collect_data(env, student_policy, expert_policy, num_steps, is_student_acting=False):
    """
    Rolls out a policy. The expert uses the FULL observation. 
    The student uses the PARTIAL observation.
    """
    student_states = []
    expert_actions = []

    full_obs, _ = env.reset()
    
    for _ in range(num_steps):
        # 1. Extract what the student is allowed to see
        noisy_student_obs = get_noisy_student_obs(full_obs)
        student_states.append(noisy_student_obs)

        # 2. The Privileged Expert gets the full observation to generate ground-truth labels
        exp_action, _ = expert_policy.predict(full_obs, deterministic=True)
        expert_actions.append(exp_action)

        # 3. Decide who drives the environment for this step
        if is_student_acting:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(noisy_student_obs).unsqueeze(0)
                # The student predicts the action using quats and joint angles
                act_action = student_policy(obs_tensor).squeeze(0).numpy()
        else:
            act_action = exp_action

        # 4. Step the environment
        full_obs, _, terminated, truncated, _ = env.step(act_action)

        if terminated or truncated:
            full_obs, _ = env.reset()

    return np.array(student_states), np.array(expert_actions)

# ---- 3. Main DAgger Loop ----
def run_dagger():
    env = gym.make("Cat-v0")
    
    # 18 for rotation matrix + 4 for joint angles = 22 total dimensions
    student_obs_dim = 22
    act_dim = env.action_space.shape[0]

    print("Loading privileged expert policy...")
    expert = PPO.load("cat_controller_1776459043.2138839")

    # Initialize Student with restricted observation space
    student = StudentPolicy(student_obs_dim, act_dim)
    optimizer = optim.Adam(student.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    iterations = 100
    steps_per_iter = 2000 
    batch_size = 64
    epochs_per_iter = 10   

    print("Iteration 0: Collecting initial expert data...")
    # Expert drives, Expert labels
    D_states, D_actions = collect_data(env, student, expert, steps_per_iter, is_student_acting=False)

    for i in range(1, iterations + 1):
        print(f"\n--- DAgger Iteration {i}/{iterations} ---")
        
        # Train student mapping: Partial Obs -> Expert Action
        dataset = TensorDataset(torch.FloatTensor(D_states), torch.FloatTensor(D_actions))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        student.train()
        for epoch in range(epochs_per_iter):
            total_loss = 0
            for batch_states, batch_actions in dataloader:
                optimizer.zero_grad()
                pred_actions = student(batch_states)
                loss = criterion(pred_actions, batch_actions)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs_per_iter}, Loss: {total_loss/len(dataloader):.4f}")

        # Student drives (based on partial obs), Expert corrects/labels (based on full obs)
        new_states, new_expert_actions = collect_data(env, student, expert, steps_per_iter, is_student_acting=True)

        D_states = np.concatenate([D_states, new_states], axis=0)
        D_actions = np.concatenate([D_actions, new_expert_actions], axis=0)

        max_buffer_size = 40000
        if len(D_states) > max_buffer_size:
            D_states = D_states[-max_buffer_size:]
            D_actions = D_actions[-max_buffer_size:]

    # Changed save filename to reflect new observation space
    torch.save(student.state_dict(), f"student_policy_{str(time.time())}.pth")

if __name__ == "__main__":
    run_dagger()