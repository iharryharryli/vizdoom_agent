import torch
import torch.nn as nn
import torch.optim as optim

from .kfac import KFACOptimizer

import torch.nn.functional as F


class A2C_ACKTR():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 mse_coef,
                 kl_coef,
                 a2c_coef,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False,
                 use_adam=False):

        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.mse_coef = mse_coef
        self.kl_coef = kl_coef
        self.a2c_coef = a2c_coef

        self.max_grad_norm = max_grad_norm

        if acktr:
            self.optimizer = KFACOptimizer(actor_critic)
        else:
            if use_adam:
                self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
            else:
                self.optimizer = optim.RMSprop(
                actor_critic.parameters(), lr, eps=eps, alpha=alpha)

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, _, ob_original, ob_reconstructed, mu, logvar, p_mu, p_logvar =\
         self.actor_critic.evaluate_actions(
                rollouts.obs[:-1].view(-1, *obs_shape),
                rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
                rollouts.masks[:-1].view(-1, 1),
                rollouts.prev_action_one_hot[:-1],
                rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        # mse & kl
        reconstuct_mse = F.mse_loss(ob_reconstructed, ob_original)
        kl = p_logvar - logvar + (logvar.exp() + (mu - p_mu).pow(2)) / (p_logvar.exp())
        kl = kl.mean()

        self.optimizer.zero_grad()
        total_loss = self.a2c_coef *  (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef) + self.mse_coef * reconstuct_mse + self.kl_coef * kl
        total_loss.backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item(), kl.item(), \
            reconstuct_mse.item()
