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

    def act(self, inputs, rnn_hxs, rnn_hys, masks, prev_action_one_hot, deterministic=False):
        value, actor_features, rnn_hxs, rnn_hys = self.base(inputs, rnn_hxs, rnn_hys, masks, prev_action_one_hot)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs, rnn_hys

    def get_value(self, inputs, rnn_hxs, rnn_hys, masks, prev_action_one_hot):
        value, _, _, _ = self.base(inputs, rnn_hxs, rnn_hys, masks, prev_action_one_hot)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, rnn_hys, masks, prev_action_one_hot, action):
        value, actor_features, rnn_hxs, rnn_hys, ob_original, ob_reconstructed, mu, logvar, p_mu, p_logvar = self.base(inputs, 
            rnn_hxs, rnn_hys, masks, prev_action_one_hot, is_training=True)
        
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs, rnn_hys, ob_original, ob_reconstructed, \
        mu, logvar, p_mu, p_logvar


class NNBase(nn.Module):

    def __init__(self, recurrent, ob_encoding_size, num_actions, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRUCell(ob_encoding_size + num_actions, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

            self.gru_p = nn.GRUCell(num_actions, hidden_size)
            nn.init.orthogonal_(self.gru_p.weight_ih.data)
            nn.init.orthogonal_(self.gru_p.weight_hh.data)
            self.gru_p.bias_ih.data.fill_(0)
            self.gru_p.bias_hh.data.fill_(0)

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

    def _forward_gru(self, x, hxs, hys, gru_init, masks, prev_action_one_hot):
        # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
        N = hxs.size(0)
        T = int(x.size(0) / N)

        # unflatten
        x = x.view(T, N, x.size(1))

        # Same deal with masks
        masks = masks.view(T, N, 1)

        # Same deal with gru_init
        gru_init = gru_init.view(T, N, gru_init.size(1))

        outputs = []
        outputs_p = []
        for i in range(T):
            masked_action = prev_action_one_hot[i] * masks[i]
            x2 = torch.cat((x[i], masked_action), dim=1)
            hx = hxs = self.gru(x2, hxs * masks[i])
            outputs.append(hx)

            hy = hys = self.gru_p(masked_action, hys * masks[i] + gru_init[i] * (1 - masks[i]))
            outputs_p.append(hy)

        # assert len(outputs) == T
        # x is a (T, N, -1) tensor
        x = torch.stack(outputs, dim=0)
        y = torch.stack(outputs_p, dim=0)
        # flatten
        x = x.view(T * N, -1)
        y = y.view(T * N, -1)

        return x, hxs, y, hys


class CNNBase(NNBase):
    def __init__(self, num_inputs, num_actions, device, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, num_actions, hidden_size)

        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.device = device

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        self.encoder = nn.Sequential(
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

        init2_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init2_(nn.Linear(hidden_size, 1))

        self.decoder = nn.Sequential(
            init_(nn.Linear(hidden_size, 32 * 7 * 7)),
            nn.ReLU(),
            UnFlatten(),
            init_(nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            init_(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)),
            nn.ReLU(),
            init_(nn.ConvTranspose2d(32, num_inputs, kernel_size=8, stride=4)),
            nn.Sigmoid()
        )

        self.gru_init = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )

        self.p_var = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU()
        )

        self.q_var = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU()
        )

        self.combine = nn.Sequential(
            init_(nn.Linear(hidden_size * 2, hidden_size)),
            nn.ReLU(),
        )

        self.train()

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size(), device=self.device)
        z = mu + std * esp
        return z

    def forward(self, inputs, rnn_hxs, rnn_hys, masks, prev_action_one_hot, is_training=False):
        ob_original = inputs / 255.0

        x = self.encoder(ob_original)

        gru_init = self.gru_init(x)
        
        mu, rnn_hxs, p_mu, rnn_hys = self._forward_gru(x, rnn_hxs, rnn_hys, gru_init, masks, prev_action_one_hot)

        logvar = self.q_var(mu)
        p_logvar = self.p_var(p_mu)

        final = self.combine(torch.cat((mu, p_mu), dim=1))

        if is_training:
            # reconstruct
            z = self.reparameterize(mu, logvar)
            ob_reconstructed = self.decoder(z)
            
            return self.critic_linear(final), final, rnn_hxs, rnn_hys, ob_original, ob_reconstructed, mu, logvar, p_mu, p_logvar
        else:
            return self.critic_linear(final), final, rnn_hxs, rnn_hys

        