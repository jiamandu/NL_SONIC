# Actor-Critic with a frozen pretrained decoder.
#
# Policy architecture:
#   Actor MLP  : [96] -> actor_hidden_dims -> [64]  (token, trainable)
#   Decoder    : [994] -> ... -> [29]               (frozen, pretrained SiLU-MLP)
#   Critic MLP : [num_critic_obs] -> critic_hidden_dims -> [1]  (trainable)
#
# Observation split (env provides 1026-dim actor obs):
#   obs[:, 0:96]   = MLP input  (ang_vel×3, dof_pos×29, dof_vel×29, actions×29, gravity×3, cmd×3)
#   obs[:, 96:126] = his_ang_vel   (10 frames × 3)
#   obs[:, 126:416]= his_dof_pos   (10 frames × 29)
#   obs[:, 416:706]= his_dof_vel   (10 frames × 29)
#   obs[:, 706:996]= his_actions   (10 frames × 29)
#   obs[:, 996:1026]= his_gravity  (10 frames × 3)
#
# Decoder input  = [token(64) | his_ang_vel(30) | his_dof_pos(290) | his_dof_vel(290) | his_actions(290) | his_gravity(30)] = 994
# Decoder output = actions (29)

import torch
import torch.nn as nn
from torch.distributions import Normal

from .actor_critic import get_activation


class DecoderMLP(nn.Module):
    """SiLU-MLP matching the pretrained decoder architecture: 994->2048->2048->1024->1024->512->512->29."""
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ActorCriticNaive(nn.Module):
    """ActorCritic with a trainable MLP encoder and a frozen pretrained decoder."""
    is_recurrent = False

    MLP_INPUT_DIM = 96    # 93 proprio + 3 cmd
    TOKEN_DIM = 64
    DECODER_DIMS = [994, 2048, 2048, 1024, 1024, 512, 512, 29]

    def __init__(self,
                 num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 actor_hidden_dims=[512, 256],
                 critic_hidden_dims=[512, 256, 128],
                 activation='elu',
                 init_noise_std=1.0,
                 decoder_path=None,
                 **kwargs):
        if kwargs:
            print("ActorCriticNaive.__init__ got unexpected arguments, which will be ignored: "
                  + str(list(kwargs.keys())))
        super().__init__()

        # ── Actor MLP: 96 -> hidden -> 64 (token) ────────────────────────────
        act_fn = get_activation(activation)
        mlp_layers = [nn.Linear(self.MLP_INPUT_DIM, actor_hidden_dims[0]),
                      get_activation(activation)]
        for i in range(len(actor_hidden_dims) - 1):
            mlp_layers += [nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]),
                           get_activation(activation)]
        mlp_layers.append(nn.Linear(actor_hidden_dims[-1], self.TOKEN_DIM))
        self.actor_mlp = nn.Sequential(*mlp_layers)

        # ── Frozen decoder: 994 -> ... -> 29 ─────────────────────────────────
        self.decoder = DecoderMLP(self.DECODER_DIMS)
        if decoder_path is not None:
            state = torch.load(decoder_path, map_location='cpu')
            self.decoder.load_state_dict(state)
            print(f"[ActorCriticNaive] Decoder loaded from: {decoder_path}")
        else:
            print("[ActorCriticNaive] WARNING: decoder_path not provided, decoder uses random weights.")
        for p in self.decoder.parameters():
            p.requires_grad = False

        # ── Critic MLP ────────────────────────────────────────────────────────
        critic_layers = [nn.Linear(num_critic_obs, critic_hidden_dims[0]),
                         get_activation(activation)]
        for i in range(len(critic_hidden_dims) - 1):
            critic_layers += [nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]),
                              get_activation(activation)]
        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.critic = nn.Sequential(*critic_layers)

        # ── Action noise ──────────────────────────────────────────────────────
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

        print(f"Actor MLP: {self.actor_mlp}")
        print(f"Critic MLP: {self.critic}")

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    # ── Core forward ──────────────────────────────────────────────────────────

    def _compute_actions(self, observations):
        """MLP -> token, then frozen decoder -> actions."""
        mlp_obs = observations[:, :self.MLP_INPUT_DIM]      # [N, 96]
        history  = observations[:, self.MLP_INPUT_DIM:]     # [N, 930]
        token = self.actor_mlp(mlp_obs)                     # [N, 64]
        decoder_input = torch.cat([token, history], dim=-1) # [N, 994]
        return self.decoder(decoder_input)                  # [N, 29]

    # ── ActorCritic interface ─────────────────────────────────────────────────

    def update_distribution(self, observations):
        mean = self._compute_actions(observations)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        return self._compute_actions(observations)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)
