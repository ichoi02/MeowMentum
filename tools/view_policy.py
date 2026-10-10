"""Watch a trained policy right the robot in the MuJoCo viewer (closed loop, sim).

    python -m tools.view_policy --variant tail --agent student
    python -m tools.view_policy --variant notail --agent teacher --roll 180 --pitch 45 --slow 0.2

Episodes repeat until the viewer is closed. On macOS the passive viewer needs
`mjpython` instead of `python`.
"""
import argparse
import time

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import SAC

import train.cat_env  # noqa: F401  (registers Cat-v0 and CatNoTail-v0)
from train.cat_env.cat_env import BASE_OBS_DIM
from train.distillation import (
    StudentPolicy, build_student_obs, zero_tail, STUDENT_OBS_DIM, ACT_DIM,
    STEP_IDX, FRONT_GRAV_SLICE, JOINT_ANGLE_SLICE,
)
from model.variants import VARIANTS, teacher_path, student_path


def student_frame(full_obs):
    """Return the noise-free student frame [front_proj_grav(3), joint_angles(4)]."""
    return np.concatenate([full_obs[FRONT_GRAV_SLICE], full_obs[JOINT_ANGLE_SLICE]])


def force_release(env, roll_deg=None, pitch_deg=None, still=False):
    """Re-release the robot at a fixed roll/pitch and return the recomputed observation.

    Overwrites only the root orientation drawn by reset and, with `still`, the root
    angular velocity; the domain-randomization draw and the joint angles stay random.
    A roll or pitch that is not given is 0 and yaw is always 0. The Euler order matches
    CatEnv.reset_model.
    """
    # Start from the state drawn by reset
    base_env = env.unwrapped
    qpos, qvel = base_env.data.qpos.copy(), base_env.data.qvel.copy()

    # Set the root orientation; MuJoCo stores the free-joint quaternion as [w, x, y, z]
    if roll_deg is not None or pitch_deg is not None:
        euler = [roll_deg or 0.0, pitch_deg or 0.0, 0.0]
        x, y, z, w = R.from_euler("xyz", euler, degrees=True).as_quat()
        qpos[3:7] = [w, x, y, z]

    # Release with zero angular velocity instead of the random initial tumble
    if still:
        qvel[3:6] = 0.0

    # Apply the state and recompute the observation, which reset returned for the old state
    base_env.set_state(qpos, qvel)
    return base_env._get_obs()


def start_episode(env, forced, roll_deg, pitch_deg, still):
    """Reset the env, apply the fixed release when `forced` is true, and return the obs."""
    obs, _ = env.reset()
    if forced:
        obs = force_release(env, roll_deg, pitch_deg, still)
    return obs


def student_action(student, obs, prev_action, variant):
    """Return the student's action for the current env observation."""
    # Build the student input from the clean frame, the previous action and the step
    frame = zero_tail(student_frame(obs), variant)
    student_obs = build_student_obs(frame, prev_action, obs[STEP_IDX])

    # Run the network
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(student_obs).unsqueeze(0)
        action = student(obs_tensor).squeeze(0).numpy()

    # Hold the untrained tail channel at zero for the no-tail variant, as the controller does
    if variant == "notail":
        action[2] = 0.0
    return action


def track_robot(viewer, mj_model):
    """Point the viewer camera at the spine and follow it."""
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "spine_1")
    viewer.cam.distance = 1.5
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 90


def visualize(variant="tail", agent="student", roll_deg=None, pitch_deg=None,
              still=False, policy_path=None, slow=1.0):
    """Run episodes of a policy in the passive viewer until the viewer is closed."""
    # Look up the variant's environment, which rejects an unknown variant before any loading
    env_id = VARIANTS[variant]["env_id"]

    # Load the teacher first, since the env keeps the privileged block only if the teacher uses it
    teacher = None
    if agent == "teacher":
        print("Loading teacher policy")
        teacher = SAC.load(policy_path or teacher_path(variant))
        privileged = teacher.observation_space.shape[0] > BASE_OBS_DIM
    else:
        privileged = True
    env = gym.make(env_id, privileged=privileged)

    # Load the student in evaluation mode
    if agent == "student":
        print("Loading student policy")
        student = StudentPolicy(STUDENT_OBS_DIM, env.action_space.shape[0])
        student.load_state_dict(torch.load(
            policy_path or student_path(variant), map_location="cpu"))
        student.eval()

    # Release the robot, at the fixed attitude if one was given, with no command history
    forced = roll_deg is not None or pitch_deg is not None or still
    obs = start_episode(env, forced, roll_deg, pitch_deg, still)
    prev_action = np.zeros(ACT_DIM)

    # Run episodes in the viewer, following the robot, until the viewer is closed
    mj_model = env.unwrapped.model
    mj_data = env.unwrapped.data
    print("Starting visualization")
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        track_robot(viewer, mj_model)
        try:
            while viewer.is_running():
                # Step the env with the policy's action, timing the control step
                step_start = time.time()
                if agent == "teacher":
                    action, _ = teacher.predict(obs, deterministic=True)
                elif agent == "student":
                    action = student_action(student, obs, prev_action, variant)
                    prev_action = np.asarray(action, dtype=float).copy()
                obs, _, terminated, truncated, _ = env.step(action)

                # Start a new episode when this one ends
                if terminated or truncated:
                    obs = start_episode(env, forced, roll_deg, pitch_deg, still)
                    prev_action = np.zeros(ACT_DIM)

                # Draw the frame and wait out the control step, scaled by the playback speed
                viewer.sync()
                time_until_next_step = env.unwrapped.dt / slow - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            env.close()


def main():
    """Parse the command line and run the viewer."""
    # Parse and validate the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS), default="tail",
                        help="ablation condition to view (default: tail)")
    parser.add_argument("--agent", choices=["student", "teacher"], default="student",
                        help="policy to view (default: student)")
    parser.add_argument("--roll", type=float, default=None, metavar="DEG",
                        help="release at a fixed roll about world x instead of a "
                             "random attitude, e.g. 180 (upside-down) or 90 (on its side)")
    parser.add_argument("--pitch", type=float, default=None, metavar="DEG",
                        help="release at a fixed pitch about world y; combines with "
                             "--roll, and either one alone leaves the other at 0")
    parser.add_argument("--still", action="store_true",
                        help="release with zero angular velocity (no initial tumble)")
    parser.add_argument("--policy", default=None,
                        help="explicit policy file to load, instead of the staged "
                             "policies/cat_controller<suffix>.zip or "
                             "policies/student_policy<suffix>.pth")
    parser.add_argument("--slow", type=float, default=1.0, metavar="FACTOR",
                        help="playback speed as a fraction of real time "
                             "(default: 1.0 = real time; 0.2 = 5x slow motion)")
    args = parser.parse_args()
    if args.slow <= 0:
        parser.error("--slow must be > 0")

    # Run the viewer
    visualize(args.variant, args.agent, args.roll, args.pitch, args.still,
              args.policy, args.slow)


if __name__ == "__main__":
    main()
