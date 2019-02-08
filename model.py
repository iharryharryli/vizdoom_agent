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
        value, actor_features, rnn_hxs, ob_original, ob_reconstructed, mu, logvar, p_mu, p_logvar, attention = self.base(inputs, 
            rnn_hxs, masks, prev_action_one_hot, is_training=True)
        
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs, ob_original, ob_reconstructed, \
        mu, logvar, p_mu, p_logvar, attention


class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        #return self._hidden_size
        return self.hidden_size

    def _forward_gru(self, x, hxs, attention, masks, prev_action_one_hot):
        # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
        N = hxs.size(0)
        T = int(x.size(0) / N)

        # unflatten
        x = x.view(T, N, x.size(1))

        # Same deal with masks
        masks = masks.view(T, N, 1)

        # Same deal with attention
        attention = attention.view(T, N, attention.size(1))

        p_mu_acc = []
        attended_x_acc = []
        for i in range(T):
            hidden_state = hxs * masks[i] + x[i] * (1 - masks[i])
            p_mu = self.gru(prev_action_one_hot[i] * masks[i], hidden_state)
            attended_x = hxs = torch.mul(x[i], attention[i]) + torch.mul(p_mu, 1 - attention[i])

            p_mu_acc.append(p_mu)
            attended_x_acc.append(attended_x)
            
        # assert len(outputs) == T
        # x is a (T, N, -1) tensor
        p_mu = torch.stack(p_mu_acc, dim=0)
        attended_x = torch.stack(attended_x_acc, dim=0)
        # flatten
        p_mu = p_mu.view(T * N, -1)
        attended_x = attended_x.view(T * N, -1)

        return p_mu, attended_x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, num_actions, device, recurrent=False, hidden_size=512):
        hidden_size = 32 * 7 * 7 

        super(CNNBase, self).__init__(recurrent, num_actions, hidden_size)

        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.device = device
        
        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
        )

        self.attention = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.Sigmoid(),
            Flatten(),
        )

        self.decoder = nn.Sequential(
            UnFlatten(),
            init_(nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            init_(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)),
            nn.ReLU(),
            init_(nn.ConvTranspose2d(32, num_inputs, kernel_size=8, stride=4)),
            nn.Sigmoid()
        )

        self.var_network = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU()
        )

        self.policy_network = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU()
        )

        init2_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init2_(nn.Linear(hidden_size, 1))

        self.train()

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size(), device=self.device)
        z = mu + std * esp
        return z

    def forward(self, inputs, rnn_hxs, masks, prev_action_one_hot, is_training=False):
        ob_original = inputs / 255.0
        rnn_hxs_original = rnn_hxs

        mu = self.main(ob_original)
        attention = self.attention(ob_original)
        
        p_mu, attended_x, rnn_hxs = self._forward_gru(mu, rnn_hxs, attention, masks, prev_action_one_hot)

        policy = self.policy_network(attended_x)

        if is_training:
            # reconstruct
            logvar = self.var_network(mu)
            p_logvar = self.var_network(p_mu)
            z = self.reparameterize(mu, logvar)
            ob_reconstructed = self.decoder(z)
            
            return self.critic_linear(policy), policy, rnn_hxs, ob_original, ob_reconstructed, \
             mu, logvar, p_mu, p_logvar, attention
        else:
            return self.critic_linear(policy), policy, rnn_hxs

        

class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m,
            init_normc_,
            lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
