"""Train the privileged SAC teacher for one variant.

Usage, from the repository root:

    python -m train.train --variant tail
    python -m train.train --variant notail

The defaults are the configuration of the released teachers. The checkpoint is saved as
policies/cat_controller<suffix><tag>_<timestamp>.zip (or at --out); copy it to
policies/cat_controller<suffix>.zip to distill from it. TensorBoard logs go to
run_logs/ (the released teachers' logs are in results/training_logs/).
"""
import argparse
import os
import time

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import train.cat_env  # noqa: F401  (registers Cat-v0 and CatNoTail-v0)
from model.variants import VARIANTS, policy_dir
from train.smooth_sac import SmoothSAC


class TensorboardRewardCallback(BaseCallback):
    """Log the per-step reward terms to TensorBoard under rewards/.

    Each term is averaged over the logging interval. Action smoothness is not a reward
    term; SmoothSAC logs it as train/smooth_loss.
    """

    # Reward terms in the info dict returned by the environment's step
    TERMS = ("r_pos", "r_en", "r_av", "r_jv", "r_time", "up_mean")

    def _on_step(self) -> bool:
        """Add the reward terms of every environment's info dict to the running means."""
        for info in self.locals["infos"]:
            if "r_pos" in info:
                for term in self.TERMS:
                    self.logger.record_mean(f"rewards/{term}", info[term])
        return True


def train(variant="tail", total_timesteps=1_000_000, tag="", n_envs=10, seed=None,
          out=None, gradient_steps=1, run_name=None, privileged=True,
          smooth_coef=2.0):
    """Train a SAC teacher, save it and return the checkpoint path.

    Args:
        variant: key of VARIANTS, "tail" or "notail".
        total_timesteps: environment steps, summed over all environments.
        tag: extra suffix for the timestamped checkpoint name.
        n_envs: number of parallel environments.
        seed: random seed, or None for an unseeded run.
        out: checkpoint path; if empty, a timestamped file in policies/.
        gradient_steps: gradient steps per vectorized environment step.
        run_name: TensorBoard run name; if empty, "SAC".
        privileged: whether the observation includes the domain-randomization block.
        smooth_coef: weight of the action-smoothness term in the actor loss.
    """
    # Build the parallel environments of the variant
    config = VARIANTS[variant]
    env = make_vec_env(
        config["env_id"],
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"privileged": privileged},
    )

    # Build the SAC teacher with the action-smoothness term in its actor loss
    model = SmoothSAC(
        "MlpPolicy",
        env,
        smooth_coef=smooth_coef,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log="./run_logs/",
        device="cpu",
        seed=seed,
        learning_rate=3e-4,
        buffer_size=300_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=gradient_steps,
        ent_coef="auto",
    )

    # Train, logging the reward terms to TensorBoard
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=20,
        progress_bar=True,
        callback=TensorboardRewardCallback(),
        tb_log_name=run_name or "SAC",
    )

    # Save to out, or else to a timestamped checkpoint in policies/
    if out:
        model_path = out
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_name = f"cat_controller{config['suffix']}{tag}_{timestamp}.zip"
        model_path = os.path.join(policy_dir(), file_name)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model_path


def main():
    """Parse the command line and train one teacher."""
    # Parse the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS), default="tail",
                        help="robot variant to train (default: tail)")
    parser.add_argument("--steps", type=int, default=1_000_000,
                        help="total training timesteps (default: 1000000)")
    parser.add_argument("--tag", default="",
                        help="extra suffix for the saved model filename")
    parser.add_argument("--envs", type=int, default=10,
                        help="number of parallel environments (default: 10)")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed (default: unseeded)")
    parser.add_argument("--out", default=None,
                        help="checkpoint path (default: a timestamped .zip in policies/)")
    parser.add_argument("--gradient-steps", type=int, default=1,
                        help="gradient steps per vec-env step (default: 1)")
    parser.add_argument("--run-name", default=None,
                        help="TensorBoard run name (default: SAC)")
    parser.add_argument("--no-privileged", action="store_true",
                        help="train on the 25-dim observation without the privileged "
                             "domain-randomization block")
    parser.add_argument("--smooth-coef", type=float, default=2.0,
                        help="weight of the L2 action-smoothness term in the actor loss "
                             "(default: 2.0; 0 disables it, and train/smooth_loss is "
                             "still logged)")
    args = parser.parse_args()

    # Train with the parsed options
    train(variant=args.variant, total_timesteps=args.steps, tag=args.tag, n_envs=args.envs,
          seed=args.seed, out=args.out, gradient_steps=args.gradient_steps,
          run_name=args.run_name, privileged=not args.no_privileged,
          smooth_coef=args.smooth_coef)


if __name__ == "__main__":
    main()
