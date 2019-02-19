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
    nn.init.calculate_gain('tanh'))

init4_ = lambda m: init(m,
    nn.init.orthogonal_,
    lambda x: nn.init.constant_(x, 0),
    nn.init.calculate_gain('sigmoid'))


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
            self.dist = Categorical(self.base.hidden_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn, p_rnn, masks, prev_action_one_hot, deterministic=False):
        value, actor_features, rnn, p_rnn = self.base(inputs, rnn, p_rnn, masks, prev_action_one_hot)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn, p_rnn

    def get_value(self, inputs, rnn, p_rnn, masks, prev_action_one_hot):
        value, _, _, _ = self.base(inputs, rnn, p_rnn, masks, prev_action_one_hot)
        return value

    def evaluate_actions(self, inputs, rnn, p_rnn, masks, prev_action_one_hot, action):
        value, actor_features, ob_original, ob_reconstructed, mu, logvar, p_mu, p_logvar, attention = self.base(inputs, 
            rnn, p_rnn, masks, prev_action_one_hot, is_training=True)
        
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, ob_original, ob_reconstructed, \
        mu, logvar, p_mu, p_logvar, attention


class NNBase(nn.Module):

    def __init__(self, recurrent, num_actions, hidden_size):
        super(NNBase, self).__init__()

        self.hidden_size = hidden_size
        self.p_hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRUCell(hidden_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

            self.p_network = nn.Sequential(
                init_(nn.Linear(hidden_size + num_actions, hidden_size)),
                nn.ReLU(),
                init_(nn.Linear(hidden_size, hidden_size)),
                nn.ReLU(),
                init2_(nn.Linear(hidden_size, 2 * hidden_size)),
            )

    @property
    def is_recurrent(self):
        return self._recurrent

    def _forward_p_gru(self, x, hxs, attention, masks, prev_action_one_hot):
        # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
        N = hxs.size(0)
        T = int(x.size(0) / N)

        # unflatten
        x = x.view(T, N, x.size(1))

        # Same deal with masks
        masks = masks.view(T, N, 1)

        # Same deal with attention
        # attention = attention.view(T, N, attention.size(1))

        p_dist_acc = []
        attended_x_acc = []
        for i in range(T):
            hidden_state = hxs * masks[i] + x[i] * (1 - masks[i])
            cur_input = torch.cat((prev_action_one_hot[i] * masks[i], hidden_state), dim=1)
            p_dist = self.p_network(cur_input)
            p_mu, p_logvar = torch.split(p_dist, self.hidden_size, dim=1)
            # attended_x = hxs = torch.mul(x[i], attention[i]) + torch.mul(p_mu, 1 - attention[i])
            attended_x = hxs = x[i]

            p_dist_acc.append(p_dist)
            attended_x_acc.append(attended_x)
            
        # assert len(outputs) == T
        # x is a (T, N, -1) tensor
        p_dist = torch.stack(p_dist_acc, dim=0)
        attended_x = torch.stack(attended_x_acc, dim=0)
        # flatten
        p_dist = p_dist.view(T * N, -1)
        attended_x = attended_x.view(T * N, -1)

        return p_dist, attended_x, hxs

    def _forward_gru(self, x, hxs, masks):
        # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
        N = hxs.size(0)
        T = int(x.size(0) / N)

        # unflatten
        x = x.view(T, N, x.size(1))

        # Same deal with masks
        masks = masks.view(T, N, 1)

        acc = []
        for i in range(T):
            hidden_state = hxs * masks[i]
            hx = hxs = self.gru(x[i], hidden_state)
            acc.append(hx)
            
        # assert len(outputs) == T
        # x is a (T, N, -1) tensor
        acc = torch.stack(acc, dim=0)
        # flatten
        acc = acc.view(T * N, -1)

        return acc, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, num_actions, device, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, num_actions, hidden_size)

        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.device = device

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init2_(nn.Linear(32 * 7 * 7, hidden_size * 2)),
        )

        # self.attention = nn.Sequential(
        #     init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
        #     nn.ReLU(),
        #     init_(nn.Conv2d(32, 64, 4, stride=2)),
        #     nn.ReLU(),
        #     init_(nn.Conv2d(64, 32, 3, stride=1)),
        #     nn.ReLU(),
        #     Flatten(),
        #     init4_(nn.Linear(32 * 7 * 7, hidden_size)),
        #     nn.Sigmoid()
        # )

        self.decoder = nn.Sequential(
            init_(nn.Linear(hidden_size, 32 * 7 * 7)),
            nn.ReLU(),
            UnFlatten(),
            init_(nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            init_(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)),
            nn.ReLU(),
            init4_(nn.ConvTranspose2d(32, num_inputs, kernel_size=8, stride=4)),
            nn.Sigmoid()
        )

        self.policy_network = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU()
        )

        self.critic_linear = init2_(nn.Linear(hidden_size, 1))

        self.train()

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size(), device=self.device)
        z = mu + std * esp
        return z

    def forward(self, inputs, rnn, p_rnn, masks, prev_action_one_hot, is_training=False):
        ob_original = inputs / 255.0

        dist = self.main(ob_original)
        # attention = self.attention(ob_original)
        attention = None

        mu, logvar = torch.split(dist, self.hidden_size, dim=1)
        
        p_dist, attended_x, p_rnn = self._forward_p_gru(mu, p_rnn, attention, masks, prev_action_one_hot)

        policy = self.policy_network(attended_x)
        policy, rnn = self._forward_gru(policy, rnn, masks)

        if is_training:
            # reconstruct
            p_mu, p_logvar = torch.split(p_dist, self.hidden_size, dim=1)
            z = self.reparameterize(mu, logvar)
            ob_reconstructed = self.decoder(z)
            
            return self.critic_linear(policy), policy, ob_original, ob_reconstructed, \
             mu, logvar, p_mu, p_logvar, attention
        else:
            return self.critic_linear(policy), policy, rnn, p_rnn
