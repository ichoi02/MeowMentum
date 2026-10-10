"""SAC with an L2 action-smoothness term in the actor loss.

The actor loss is

    L_actor = L_SAC + smooth_coef * || pi_mu(s_{t+1}) - pi_mu(s_t) ||^2

where pi_mu = tanh(mu) is the deterministic mean action, the one
`predict(deterministic=True)` returns, not a sample with exploration noise. The squared
norm is summed over the 3 action dimensions and averaged over the batch. The released
teachers used smooth_coef = 2.0. The unweighted term is logged as `train/smooth_loss`,
also when smooth_coef is 0.
"""
import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F


class SmoothSAC(SAC):
    """SAC whose actor loss carries an L2 action-smoothness term.

    Only the training objective differs from SAC, so checkpoints load with plain
    `SAC.load`.
    """

    def __init__(self, *args, smooth_coef: float = 2.0, **kwargs):
        """Build SAC from the SAC arguments, with smooth_coef weighting the smoothness term."""
        super().__init__(*args, **kwargs)
        self.smooth_coef = smooth_coef

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """Run SAC gradient steps with the smoothness term added to the actor loss.

        Statement for statement this is stable-baselines3 2.7.1 `SAC.train` plus the
        smoothness term, so it can be diffed against upstream after a version bump.
        """
        # Switch to train mode and update the learning rates
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        # Per-gradient-step values, averaged for logging
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, smooth_losses = [], [], []

        # Take the gradient steps, each on a fresh replay batch
        for gradient_step in range(gradient_steps):
            # Sample a batch; with n-step replay the discount is gamma**n_steps
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            # Resample the gSDE noise, since log_std may have changed
            if self.use_sde:
                self.actor.reset_noise()

            # Sample actions from the current actor for the batch states
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            # Entropy coefficient and, when it is learned, its loss
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef
                                  * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            # Optimize the entropy coefficient
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # Soft TD target from the target critics
            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations)
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = (replay_data.rewards
                                   + (1 - replay_data.dones) * discounts * next_q_values)

            # Optimize the critics on the TD target
            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values)
                                    for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Smoothness term on the deterministic mean action between s_t and s_{t+1}
            # No (1 - dones) mask: the env only truncates, so no pair spans two episodes
            pi_t = self.actor(replay_data.observations, deterministic=True)
            pi_next = self.actor(replay_data.next_observations, deterministic=True)
            smooth_loss = ((pi_next - pi_t) ** 2).sum(dim=1).mean()
            smooth_losses.append(smooth_loss.item())

            # Optimize the actor on the SAC loss plus the weighted smoothness term
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean() + self.smooth_coef * smooth_loss
            actor_losses.append(actor_loss.item())
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Polyak-update the target critics
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        # Log the update count and the mean of each value over the gradient steps
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/smooth_loss", np.mean(smooth_losses))
        self.logger.record("train/smooth_coef", self.smooth_coef)
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
