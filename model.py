import torch
import torch.nn as nn
import torch.nn.functional as F

from distributions import Categorical, DiagGaussian
from utils import init, init_normc_


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 32, 7, 7)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, device, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if len(obs_shape) == 3:
            self.base = CNNBase(obs_shape[0], action_space.n, device, **base_kwargs)
        elif len(obs_shape) == 1:
            self.base = MLPBase(obs_shape[0], **base_kwargs)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, prev_action_one_hot, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, prev_action_one_hot)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, prev_action_one_hot):
        value, _, _ = self.base(inputs, rnn_hxs, masks, prev_action_one_hot)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, prev_action_one_hot, action):
        value, actor_features, rnn_hxs, ob_original, ob_reconstructed, q_mu, q_logvar, p_mu, p_logvar = self.base(inputs, 
            rnn_hxs, masks, prev_action_one_hot, is_training=True)
        
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs, ob_original, ob_reconstructed, \
        q_mu, q_logvar, p_mu, p_logvar


class NNBase(nn.Module):

    def __init__(self, recurrent, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size * 3
        self._recurrent = recurrent

        
        self.h_gru = nn.GRUCell(hidden_size, hidden_size)
        nn.init.orthogonal_(self.h_gru.weight_ih.data)
        nn.init.orthogonal_(self.h_gru.weight_hh.data)
        self.h_gru.bias_ih.data.fill_(0)
        self.h_gru.bias_hh.data.fill_(0)

        self.f_gru = nn.GRUCell(hidden_size, hidden_size)
        nn.init.orthogonal_(self.f_gru.weight_ih.data)
        nn.init.orthogonal_(self.f_gru.weight_hh.data)
        self.f_gru.bias_ih.data.fill_(0)
        self.f_gru.bias_hh.data.fill_(0)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    @property
    def output_size(self):
        #return self._hidden_size
        return self.hidden_size

    def _forward_gru(self, c, hxs, masks, prev_action_one_hot):    
        # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
        N = hxs.size(0)
        T = int(c.size(0) / N)
        # unflatten
        c = c.view(T, N, c.size(1))
        masks = masks.view(T, N, 1)
        prev_action_one_hot = prev_action_one_hot.view(T, N, self.num_actions)

        f = hxs[:, : self.hidden_size]
        h = hxs[:, self.hidden_size : 2 * self.hidden_size]
        q_mu = hxs[:, 2 * self.hidden_size : ]

        acc_p_dist = []
        acc_q_dist = []

        for i in range(T):
            # P
            p_input = torch.cat([h, q_mu, prev_action_one_hot[i]], dim=1)
            p_dist = self.p_network(p_input * masks[i])
            p_mu = p_dist[:, : self.hidden_size]

            # Q
            q_input = torch.cat([f, c[i], prev_action_one_hot[i]], dim=1)
            q_dist = self.q_network(q_input * masks[i])
            q_mu = q_dist[:, : self.hidden_size]

            # Update GRU
            f = self.f_gru(c[i], f * masks[i])
            h = self.h_gru(p_mu, h * masks[i])

            # Save Output
            acc_p_dist.append(p_dist)
            acc_q_dist.append(q_dist)

        # assert len(outputs) == T
        # x is a (T, N, -1) tensor
        acc_p_dist = torch.stack(acc_p_dist, dim=0)
        acc_q_dist = torch.stack(acc_q_dist, dim=0)
        # flatten
        acc_p_dist = acc_p_dist.view(T * N, -1)
        acc_q_dist = acc_q_dist.view(T * N, -1)

        hxs = torch.cat([f,h,q_mu], dim=1)

        return acc_p_dist, acc_q_dist, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, num_actions, device, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size)

        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.device = device

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        init2_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        init3_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('sigmoid'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU()
        )        

        self.critic_linear = init2_(nn.Linear(hidden_size, 1))

        self.decoder = nn.Sequential(
            init_(nn.Linear(hidden_size, 32 * 7 * 7)),
            nn.ReLU(),
            UnFlatten(),
            init_(nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            init_(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)),
            nn.ReLU(),
            init3_(nn.ConvTranspose2d(32, num_inputs, kernel_size=8, stride=4)),
            nn.Sigmoid()
        )

        dist_size = hidden_size * 2

        self.p_network = nn.Sequential(
            init_(nn.Linear(hidden_size + hidden_size + num_actions, dist_size)),
            nn.ReLU(),
            init_(nn.Linear(dist_size, dist_size)),
            nn.ReLU(),
            init2_(nn.Linear(dist_size, dist_size))
        )

        self.q_network = nn.Sequential(
            init_(nn.Linear(hidden_size + hidden_size + num_actions, dist_size)),
            nn.ReLU(),
            init_(nn.Linear(dist_size, dist_size)),
            nn.ReLU(),
            init2_(nn.Linear(dist_size, dist_size))
        )

        self.train()

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size(), device=self.device)
        z = mu + std * esp
        return z

    def forward(self, inputs, rnn_hxs, masks, prev_action_one_hot, is_training=False):
        ob_original = inputs / 255.0

        c = self.main(ob_original)
        
        p_dist, q_dist, rnn_hxs = self._forward_gru(c, rnn_hxs, masks, prev_action_one_hot)

        q_mu = q_dist[:, : self.hidden_size]
        q_logvar = q_dist[:, self.hidden_size :]
        p_mu = p_dist[:, : self.hidden_size]
        p_logvar = p_dist[:, self.hidden_size :]

        if is_training:
            # reconstruct
            z = self.reparameterize(q_mu, q_logvar)
            ob_reconstructed = self.decoder(z)
            
            return self.critic_linear(q_mu), q_mu, rnn_hxs, ob_original, ob_reconstructed, q_mu, q_logvar, p_mu, p_logvar
        else:
            return self.critic_linear(q_mu), q_mu, rnn_hxs
