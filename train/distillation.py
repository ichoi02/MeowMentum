"""Distill the privileged SAC teacher into the deployable student policy with DAgger.

Usage, from the repository root:

    python -m train.distillation --variant tail
    python -m train.distillation --variant notail
    python -m train.distillation --variant tail --teacher policies/my_teacher.zip --tag _v2
    python -m train.distillation --variant tail --smooth-coef 0.5 --tag _smooth

The teacher defaults to policies/cat_controller<suffix>.zip, and --smooth-coef (default 0)
adds an action-smoothness term to the imitation loss. The student is saved to
policies/student_policy<suffix><tag>_<timestamp>.pth; copy it to
policies/student_policy<suffix>.pth before running `python -m hardware.onnx_conversion`.
The student-observation helpers defined here are shared with tools.view_policy and
hardware.e2e_test.
"""
import argparse
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import SAC
from torch.utils.data import DataLoader, TensorDataset

import train.cat_env  # noqa: F401  (registers Cat-v0 and CatNoTail-v0)
import train.cat_env.env_util as util
from model.variants import VARIANTS, policy_dir, teacher_path
from train.cat_env.cat_env import BASE_OBS_DIM

# Front gravity, joint angles [rot1, pitch, rot2, tail] (rad) and steps / max_steps in the env obs
FRONT_GRAV_SLICE = slice(0, 3)
JOINT_ANGLE_SLICE = slice(12, 16)
STEP_IDX = 24

# Sizes of the parts of the student observation
GRAV_DIM = 3
JOINT_DIM = 4
ACT_DIM = 3
FRAME_DIM = GRAV_DIM + JOINT_DIM

# Student obs [grav(3), joints(4), prev raw action(3), progress(1)], as in hardware/controller.py
STUDENT_OBS_DIM = FRAME_DIM + ACT_DIM + 1

# Tail position among the joint angles; zeroed for notail here and in hardware/controller.py
TAIL_JOINT_IDX = 3

# DAgger schedule and optimizer settings
DAGGER_ITERATIONS = 50
STEPS_PER_ITERATION = 2000
EPOCHS_PER_ITERATION = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_DATASET_SIZE = 40000


def zero_tail(frame_or_obs, variant, joint_offset=GRAV_DIM):
    """Zero the tail joint reading for the no-tail variant.

    Return a float copy with the tail angle at index `joint_offset + TAIL_JOINT_IDX` set
    to 0 for "notail", and the input itself for "tail".
    """
    if variant == "notail":
        frame_or_obs = np.asarray(frame_or_obs, dtype=float).copy()
        frame_or_obs[joint_offset + TAIL_JOINT_IDX] = 0.0
    return frame_or_obs


def build_student_obs(frame, prev_action, step_frac):
    """Build the student observation from a frame, the previous action and the progress.

    Must match build_student_obs in hardware/controller.py.
    """
    return np.concatenate([frame, np.asarray(prev_action, dtype=float).ravel(),
                           [float(step_frac)]])


def sample_sensor_bias(grav_bias_std=0.035, joint_bias_std=0.02):
    """Draw the constant sensor offsets of one drop.

    Returns:
        Dict with "grav", a rotation vector in rad applied to the measured gravity, and
        "joint", offsets in rad added to the joint angles.
    """
    return {
        "grav": np.random.normal(0.0, grav_bias_std, GRAV_DIM),
        "joint": np.random.normal(0.0, joint_bias_std, JOINT_DIM),
    }


def get_noisy_student_frame(full_obs, bias, grav_noise_std=0.02, joint_noise_std=0.02):
    """Return the noisy frame [front gravity(3), joint angles(4)] the student sees.

    `bias` comes from sample_sensor_bias; per-frame noise is added on top of it.
    """
    # Read the true front gravity and joint angles
    front_grav = full_obs[FRONT_GRAV_SLICE].copy()
    joint_angles = full_obs[JOINT_ANGLE_SLICE].copy()

    # Rotate the gravity direction by the IMU mounting bias plus per-frame tilt noise
    grav_rotvec = bias["grav"] + np.random.normal(0.0, grav_noise_std, GRAV_DIM)
    noisy_grav = R.from_rotvec(grav_rotvec).apply(front_grav)

    # Add the encoder zeroing bias plus per-frame noise to the joint angles
    noisy_joints = util.add_gaussian_noise(joint_angles + bias["joint"], joint_noise_std)

    # Join the noisy gravity and joint angles into one frame
    return np.concatenate([noisy_grav, noisy_joints])


class StudentPolicy(nn.Module):
    """MLP from the student observation to a normalized action in [-1, 1]."""

    def __init__(self, obs_dim, act_dim, hidden_size=256):
        """Build two ReLU hidden layers and a tanh output layer."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        """Return the actions for a batch of student observations."""
        return self.net(x)


def _start_episode(env, max_delay):
    """Reset the env and draw the new episode's observation delay and sensor bias."""
    full_obs, _ = env.reset()
    obs_delay = np.random.randint(0, max_delay + 1)
    bias = sample_sensor_bias()
    return full_obs, obs_delay, bias


def collect_data(env, student_policy, expert_policy, num_steps, is_student_acting=False,
                 max_delay=2, variant="tail"):
    """Roll out the expert or the student and label every step with the expert's action.

    The expert sees the full current observation. The student sees a noisy frame delayed
    by 0 to `max_delay` steps (drawn per episode), its previous action and the progress.

    Returns:
        student_states: (num_steps, STUDENT_OBS_DIM) student observations.
        expert_actions: (num_steps, ACT_DIM) expert actions for the same steps.
        pair_valid: (num_steps,) bool, True when the next row is the next step of the
            same episode. The final row is False, so collections can be concatenated.
    """
    # Start the first episode with empty outputs and no command history
    student_states, expert_actions, pair_valid = [], [], []
    full_obs, obs_delay, bias = _start_episode(env, max_delay)
    obs_buffer = []
    prev_action = np.zeros(ACT_DIM)

    # Roll out num_steps steps, starting a new episode whenever one ends
    for _ in range(num_steps):
        # Build the student observation; only the sensor frame is delayed
        obs_buffer.append(zero_tail(get_noisy_student_frame(full_obs, bias), variant))
        frame = obs_buffer[max(len(obs_buffer) - 1 - obs_delay, 0)]
        student_obs = build_student_obs(frame, prev_action, full_obs[STEP_IDX])
        student_states.append(student_obs)

        # Label the step with the expert's action on the full current observation
        expert_action, _ = expert_policy.predict(full_obs, deterministic=True)
        expert_actions.append(expert_action)

        # Choose who drives the env
        if is_student_acting:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(student_obs).unsqueeze(0)
                action = student_policy(obs_tensor).squeeze(0).numpy()
        else:
            action = expert_action

        # Step the env and feed the raw action back as the previous action
        # (quirk kept: the released no-tail student was distilled with the inert tail command in
        # prev_action[2], whereas the robot feeds back 0)
        prev_action = np.asarray(action, dtype=float).ravel().copy()
        full_obs, _, terminated, truncated, _ = env.step(action)
        pair_valid.append(not (terminated or truncated))

        # Start a new episode when this one ends
        if terminated or truncated:
            full_obs, obs_delay, bias = _start_episode(env, max_delay)
            obs_buffer = []
            prev_action = np.zeros(ACT_DIM)

    # Mark the final row, which has no next step
    pair_valid = np.array(pair_valid, dtype=bool)
    pair_valid[-1] = False
    return np.array(student_states), np.array(expert_actions), pair_valid


def _train_student(student, optimizer, states, actions, pair_valid, smooth_coef):
    """Fit the student to the expert actions for EPOCHS_PER_ITERATION epochs.

    With `smooth_coef` > 0 the loss adds smooth_coef times the mean squared change of the
    student's action between consecutive steps, over the pairs marked in `pair_valid`.
    """
    # Build the shuffled dataset of (state, action, next state, pair valid) rows
    states = torch.FloatTensor(states)
    next_states = torch.cat([states[1:], states[-1:]], dim=0)
    dataset = TensorDataset(states, torch.FloatTensor(actions), next_states,
                            torch.FloatTensor(pair_valid.astype(np.float32)))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Train on the dataset with the mean squared error to the expert actions
    criterion = nn.MSELoss()
    student.train()
    for epoch in range(EPOCHS_PER_ITERATION):
        total_loss = 0
        for batch_states, batch_actions, batch_next, batch_valid in dataloader:
            # Imitate the expert actions
            optimizer.zero_grad()
            pred_actions = student(batch_states)
            loss = criterion(pred_actions, batch_actions)

            # Penalize the action change between consecutive steps
            if smooth_coef > 0.0:
                action_change = student(batch_next) - pred_actions
                num_valid = batch_valid.sum()
                if num_valid > 0:
                    loss = loss + smooth_coef * (
                        batch_valid * (action_change ** 2).sum(dim=1)).sum() / num_valid

            # Take a gradient step
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Report the mean batch loss every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/{EPOCHS_PER_ITERATION}, "
                  f"Loss: {total_loss / len(dataloader):.4f}")


def run_dagger(variant="tail", teacher=None, tag="", smooth_coef=0.0):
    """Distill a variant's teacher into a new student with DAgger and save its weights.

    Args:
        variant: "tail" or "notail".
        teacher: teacher .zip; defaults to policies/cat_controller<suffix>.zip.
        tag: extra suffix for the output file name.
        smooth_coef: weight of the action-smoothness term in the loss; 0 disables it.
    """
    # Load the teacher, then build the env at the observation width it expects
    cfg = VARIANTS[variant]
    print("Loading privileged expert policy...")
    expert = SAC.load(teacher or teacher_path(variant))
    privileged = expert.observation_space.shape[0] > BASE_OBS_DIM
    env = gym.make(cfg["env_id"], privileged=privileged)
    print(f"  teacher obs {expert.observation_space.shape[0]}-dim "
          f"(privileged={privileged})")

    # Build the student and its optimizer
    act_dim = env.action_space.shape[0]
    print(f"  student obs {STUDENT_OBS_DIM}-dim "
          f"[grav 3 | joints 4 | prev action {ACT_DIM} | step 1]"
          + ("  (tail joint reading zeroed)" if variant == "notail" else ""))
    student = StudentPolicy(STUDENT_OBS_DIM, act_dim)
    optimizer = optim.Adam(student.parameters(), lr=LEARNING_RATE)

    # Start the dataset with expert-driven data
    print("Iteration 0: Collecting initial expert data...")
    states, actions, pair_valid = collect_data(env, student, expert, STEPS_PER_ITERATION,
                                               is_student_acting=False, variant=variant)

    # Alternate training on all data so far with student-driven collection
    for i in range(1, DAGGER_ITERATIONS + 1):
        print(f"\n--- DAgger Iteration {i}/{DAGGER_ITERATIONS} ---")

        # Fit the student to the expert labels collected so far
        _train_student(student, optimizer, states, actions, pair_valid, smooth_coef)

        # Let the student drive while the expert labels the states it visits
        new_states, new_actions, new_valid = collect_data(
            env, student, expert, STEPS_PER_ITERATION, is_student_acting=True,
            variant=variant)

        # Add the new data, keeping only the newest MAX_DATASET_SIZE samples
        states = np.concatenate([states, new_states], axis=0)
        actions = np.concatenate([actions, new_actions], axis=0)
        pair_valid = np.concatenate([pair_valid, new_valid], axis=0)
        if len(states) > MAX_DATASET_SIZE:
            states = states[-MAX_DATASET_SIZE:]
            actions = actions[-MAX_DATASET_SIZE:]
            pair_valid = pair_valid[-MAX_DATASET_SIZE:]

    # Save the student's weights under a timestamped name
    out = os.path.join(
        policy_dir(),
        f"student_policy{cfg['suffix']}{tag}_{time.strftime('%Y%m%d-%H%M%S')}.pth")
    torch.save(student.state_dict(), out)
    print(f"Student saved to {out}")


def main():
    """Parse the command line and run DAgger."""
    # Parse the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS), default="tail",
                        help="ablation condition to distill (default: tail)")
    parser.add_argument("--teacher", default=None,
                        help="explicit teacher .zip "
                             "(default: policies/cat_controller<suffix>.zip)")
    parser.add_argument("--tag", default="", help="extra suffix for the output filename")
    parser.add_argument("--smooth-coef", type=float, default=0.0,
                        help="weight on ||pi(s_t+1) - pi(s_t)||^2 in the distillation "
                             "loss (default 0.0 = off). The student is otherwise pure "
                             "imitation and inherits none of the teacher's actor-loss "
                             "smoothness.")
    args = parser.parse_args()

    # Distill the chosen variant's teacher into a student
    run_dagger(args.variant, args.teacher, args.tag, args.smooth_coef)


if __name__ == "__main__":
    main()
