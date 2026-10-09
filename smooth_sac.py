"""SAC with a temporal action-smoothness term in the ACTOR loss.

    L_actor = L_SAC + smooth_coef * || pi_mu(s_{t+1}) - pi_mu(s_t) ||^2

where `pi_mu` is the DETERMINISTIC mean action, tanh(mu), not a sample from the
squashed Gaussian. Two reasons this matters:

  1. The sampled action carries SAC's exploration noise, which never reaches
     hardware. Penalizing it prices jitter the deployed policy does not have, and
     -- because the noise scale `log_std` is itself a network output -- hands the
     actor a degenerate way to satisfy the term: shrink the entropy, which fights
     the ent_coef auto-tuner.
  2. tanh(mu) is exactly what `predict(deterministic=True)` returns, i.e. the
     action the student is distilled from, so the quantity being smoothed is the
     commanded trajectory itself.

In the actor loss rather than the reward, the term's gradient reaches the actor
directly on every update instead of being learned by the critic and traded against
task reward inside Q, where a smoothness penalty can make holding still look optimal.

The released teachers were trained with smooth_coef = 2.0 (the default here and in
train.py). The term is the squared L2 norm summed over the 3 action dims and averaged
over the batch. It is logged unweighted as `train/smooth_loss` (also at
smooth_coef = 0); `evaluate.py --stats` reports the same quantity (m_dsm) for a
trained policy.
"""
import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update


class SmoothSAC(SAC):
    """SAC whose actor loss carries an L2 action-smoothness term.

    Drop-in for SAC. Checkpoints stay loadable by plain `SAC.load` (the policy is
    unchanged -- only the training objective differs), which is what
    distillation.py, evaluate.py and view_policy.py rely on.
    """

    def __init__(self, *args, smooth_coef: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.smooth_coef = smooth_coef

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Body is stable-baselines3 2.7.1 SAC.train() with the smoothness term
        # added at the actor loss; kept line-for-line otherwise so a version bump
        # diffs cleanly against upstream.
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, smooth_losses = [], [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # --- action smoothness, on the deterministic mean ---
            # `actor(obs, deterministic=True)` is tanh(mu) -- the mean of the
            # squashed Gaussian, with no reparameterized sample in it. Called
            # AFTER action_log_prob above: both share `actor.action_dist`, and
            # the call mutates its stored Normal. Harmless here (log_prob is
            # already a tensor and its graph does not read the object back), but
            # it is why this block cannot be hoisted above the sampling.
            #
            # (s_t, s_{t+1}) is exactly the consecutive pair the term wants: SB3
            # stores the true `terminal_observation` for truncated steps, and
            # this env only ever truncates (_is_terminated is always False), so
            # no sampled pair straddles an episode boundary. Add a `(1 - dones)`
            # mask here if that ever stops being true.
            pi_t = self.actor(replay_data.observations, deterministic=True)
            pi_next = self.actor(replay_data.next_observations, deterministic=True)
            # Squared L2 per transition (summed over the 3 action dims), averaged
            # over the batch.
            smooth_loss = ((pi_next - pi_t) ** 2).sum(dim=1).mean()
            smooth_losses.append(smooth_loss.item())

            # Compute actor loss
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean() + self.smooth_coef * smooth_loss
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        # Unweighted, like the env's m_* magnitudes: what the coefficient prices,
        # readable at any coefficient including zero.
        self.logger.record("train/smooth_loss", np.mean(smooth_losses))
        self.logger.record("train/smooth_coef", self.smooth_coef)
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
