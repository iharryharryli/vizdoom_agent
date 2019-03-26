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

class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        nn.init.orthogonal_(self.reset_gate.weight)
        nn.init.orthogonal_(self.update_gate.weight)
        nn.init.orthogonal_(self.out_gate.weight)
        nn.init.constant_(self.reset_gate.bias, 0.)
        nn.init.constant_(self.update_gate.bias, 0.)
        nn.init.constant_(self.out_gate.bias, 0.)


    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = F.sigmoid(self.update_gate(stacked_inputs))
        reset = F.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = F.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


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

        self._hidden_size = self.conv_rep_size * 3 + hidden_size 
        self._recurrent = recurrent

        self.h_gru = ConvGRUCell(self.conv_rep_shape[0], self.conv_rep_shape[0], 3)
        self.f_gru = ConvGRUCell(self.conv_rep_shape[0], self.conv_rep_shape[0], 3)

        self.policy_gru = nn.GRUCell(hidden_size, hidden_size)
        nn.init.orthogonal_(self.policy_gru.weight_ih.data)
        nn.init.orthogonal_(self.policy_gru.weight_hh.data)
        self.policy_gru.bias_ih.data.fill_(0)
        self.policy_gru.bias_hh.data.fill_(0)


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
        c = c.view(T, N, *self.conv_rep_shape)
        masks = masks.view(T, N, 1)
        prev_action_one_hot = prev_action_one_hot.view(T, N, self.num_actions)

        f = self.unflattener(hxs[:, : self.conv_rep_size])
        h = self.unflattener(hxs[:, self.conv_rep_size : 2 * self.conv_rep_size])
        q_mu = self.unflattener(hxs[:, 2 * self.conv_rep_size : 3 * self.conv_rep_size])
        policy = hxs[:, 3 * self.conv_rep_size : ]

        acc_p_dist = []
        acc_q_dist = []
        acc_policy = []

        for i in range(T):
            broad_prev_action_one_hot = self.broadcast_prev_action_one_hot(prev_action_one_hot[i])
            broad_mask = masks[i].view(N, 1, 1, 1)

            # P
            p_input = torch.cat([h, q_mu, broad_prev_action_one_hot], dim=1)
            p_dist = self.p_network(p_input * broad_mask)
            p_mu = p_dist[:, : self.conv_rep_shape[0]]

            # Q
            q_input = torch.cat([f, c[i], broad_prev_action_one_hot], dim=1)
            q_dist = self.q_network(q_input * broad_mask)
            q_mu = q_dist[:, : self.conv_rep_shape[0]]

            # Update GRU
            f = self.f_gru(c[i], f * broad_mask)
            h = self.h_gru(p_mu, h * broad_mask)

            # Policy
            policy = self.policy_gru(self.policy_net(q_mu), policy * masks[i])

            # Save Output
            acc_p_dist.append(p_dist)
            acc_q_dist.append(q_dist)
            acc_policy.append(policy)

        # assert len(outputs) == T
        # x is a (T, N, -1) tensor
        acc_p_dist = torch.stack(acc_p_dist, dim=0)
        acc_q_dist = torch.stack(acc_q_dist, dim=0)
        acc_policy = torch.stack(acc_policy, dim=0)
        # flatten
        acc_p_dist = acc_p_dist.view(T * N, *self.conv_dist_shape)
        acc_q_dist = acc_q_dist.view(T * N, *self.conv_dist_shape)
        acc_policy = acc_policy.view(T * N, -1)

        hxs = torch.cat([self.flattener(f),self.flattener(h),
            self.flattener(q_mu),policy], dim=1)

        return acc_p_dist, acc_q_dist, acc_policy, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, num_actions, device, recurrent=False, hidden_size=512):
        self.conv_rep_shape = (32, 7, 7)
        self.conv_dist_shape = (64, 7, 7)
        self.conv_rep_size = 32 * 7 * 7
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
        )        

        self.critic_linear = init2_(nn.Linear(hidden_size, 1))

        self.decoder = nn.Sequential(
            init_(nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            init_(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)),
            nn.ReLU(),
            init3_(nn.ConvTranspose2d(32, num_inputs, kernel_size=8, stride=4)),
            nn.Sigmoid()
        )

        dist_size = hidden_size * 2

        self.p_network = nn.Sequential(
            init_(nn.Conv2d(32 + 32 + num_actions, 64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            init2_(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
        )

        self.q_network = nn.Sequential(
            init_(nn.Conv2d(32 + 32 + num_actions, 64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            init2_(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
        )

        self.policy_net = nn.Sequential(
            Flatten(),
            init_(nn.Linear(self.conv_rep_size, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU()
        )

        self.unflattener = UnFlatten()
        self.flattener = Flatten()

        self.train()

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size(), device=self.device)
        z = mu + std * esp
        return z

    def broadcast_prev_action_one_hot(self, prev_action_one_hot):
        wid = self.conv_rep_shape[-1]
        x = prev_action_one_hot[:,:,None]
        x = x.repeat(1, 1, wid * wid)
        x = x.view(x.size(0), x.size(1), wid, wid)
        return x


    def forward(self, inputs, rnn_hxs, masks, prev_action_one_hot, is_training=False):
        ob_original = inputs / 255.0

        c = self.main(ob_original)

        p_dist, q_dist, policy, rnn_hxs = self._forward_gru(c, rnn_hxs, masks, prev_action_one_hot)

        q_mu = q_dist[:, : self.conv_rep_shape[0]]
        q_logvar = q_dist[:, self.conv_rep_shape[0] :]
        p_mu = p_dist[:, : self.conv_rep_shape[0]]
        p_logvar = p_dist[:, self.conv_rep_shape[0] :]

        if is_training:
            # reconstruct
            z = self.reparameterize(q_mu, q_logvar)
            ob_reconstructed = self.decoder(z)
            
            return self.critic_linear(policy), policy, rnn_hxs, ob_original, ob_reconstructed, q_mu, q_logvar, p_mu, p_logvar
        else:
            return self.critic_linear(policy), policy, rnn_hxs
