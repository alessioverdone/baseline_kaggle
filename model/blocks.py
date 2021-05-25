import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from utils.utils_functions import get_features


class SEBlock(nn.Module):
    """Squeeze-Excitation Block"""

    def __init__(self, dim, reduction_ratio=4):
        super(SEBlock, self).__init__()
        self.f1 = nn.Linear(dim, dim // reduction_ratio)
        self.f2 = nn.Linear(dim // reduction_ratio, dim)

    def forward(self, x):
        y = x.mean(axis=(-1, -2))
        y = F.silu(self.f1(y))
        y = torch.sigmoid(self.f2(y))
        return x * y.unsqueeze(-1).unsqueeze(-1)


class BasicBlock(nn.Module):
    """Basic Residual Block"""

    def __init__(self, dim, downscale=False):
        super(BasicBlock, self).__init__()
        if downscale:
            self.conv1 = nn.Conv2d(dim, 2 * dim, 3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(2 * dim, 2 * dim, 3, stride=1, padding=1)
            self.bnorm1 = nn.BatchNorm2d(dim)
            self.bnorm2 = nn.BatchNorm2d(2 * dim)
            self.proj = nn.Conv2d(dim, 2 * dim, 1, stride=2)
            self.se = SEBlock(2 * dim)
        else:
            self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
            self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
            self.bnorm1 = nn.BatchNorm2d(dim)
            self.bnorm2 = nn.BatchNorm2d(dim)
            self.proj = nn.Identity()
            self.se = SEBlock(dim)

    def forward(self, x):
        y = self.conv1(self.bnorm1(F.silu(x)))
        z = self.conv2(self.bnorm2(F.silu(y)))
        return self.se(z) + self.proj(x)


class BottleneckBlock(nn.Module):
    """Bottleneck Residual Block"""

    def __init__(self, dim, downscale=False):
        super(BottleneckBlock, self).__init__()
        if downscale:
            self.conv1 = nn.Conv2d(dim, dim, 1)
            self.conv2 = nn.Conv2d(dim, 2 * dim, 3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(2 * dim, 2 * dim, 1)
            self.bnorm1 = nn.BatchNorm2d(dim)
            self.bnorm2 = nn.BatchNorm2d(dim)
            self.bnorm3 = nn.BatchNorm2d(2 * dim)
            self.proj = nn.Conv2d(dim, 2 * dim, 1, stride=2)
            self.se = SEBlock(2 * dim)
        else:
            self.conv1 = nn.Conv2d(dim, dim, 1)
            self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
            self.conv3 = nn.Conv2d(dim, dim, 1)
            self.bnorm1 = nn.BatchNorm2d(dim)
            self.bnorm2 = nn.BatchNorm2d(dim)
            self.bnorm3 = nn.BatchNorm2d(dim)
            self.proj = nn.Identity()
            self.se = SEBlock(dim)

    def forward(self, x):
        y = self.conv1(self.bnorm1(F.silu(x)))
        z = self.conv2(self.bnorm2(F.silu(y)))
        w = self.conv3(self.bnorm3(F.silu(z)))
        return self.se(w) + self.proj(x)


class ResLayers(nn.Module):
    """Sequential Residual Layers"""

    def __init__(self, block, dim, depth):
        super(ResLayers, self).__init__()
        self.blocks = nn.ModuleList(
            [block(dim, downscale=False) for _ in range(depth - 1)] +
            [block(dim, downscale=True)]
        )

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class Encoder(nn.Module):
    """Res-Net Encoder"""

    def __init__(self, dim_in, depths):
        super(Encoder, self).__init__()
        self.gate = nn.Conv2d(12, dim_in, 1, padding=(3, 5), padding_mode='circular')
        self.layers = nn.ModuleList([
            ResLayers(BasicBlock, dim_in, depths[0]),
            ResLayers(BasicBlock, 2 * dim_in, depths[1]),
            ResLayers(BottleneckBlock, 4 * dim_in, depths[2])
        ])

    def forward(self, x):
        z = self.gate(x)
        for l in self.layers:
            z = l(z)
        return z


class Actor(nn.Module):
    """Actor Head"""

    def __init__(self, dim_in, head_dim):
        super(Actor, self).__init__()
        self.compr = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, head_dim, 1),
            nn.SiLU(inplace=True)
        )
        self.fc = nn.Linear(head_dim, 4)

    def forward(self, state):
        p = self.compr(state)
        p = p.mean(axis=(-1, -2))
        p = self.fc(p)
        return F.log_softmax(p, dim=1)


class Critic(nn.Module):
    """Critic Head"""

    def __init__(self, dim_in, head_dim):
        super(Critic, self).__init__()
        self.compr = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, head_dim, 1),
            nn.SiLU(inplace=True)
        )
        self.fc = nn.utils.weight_norm(nn.Linear(head_dim, 1))

    def forward(self, state):
        v = self.compr(state)
        v = v.mean(axis=(-1, -2))
        v = self.fc(v)
        return torch.tanh(v)


class GNet(nn.Module):
    """G-Net"""

    def __init__(self):
        super(GNet, self).__init__()
        # init hyperparameters
        dim_in = 32
        head_dim = 16
        depths = (2, 2, 2)
        # init modules
        self.encoder = Encoder(dim_in, depths)
        self.actor = Actor(8 * dim_in, head_dim)
        self.critic1 = Critic(8 * dim_in, head_dim)
        self.critic2 = Critic(8 * dim_in, head_dim)

    def forward(self, state):
        latent = self.encoder(state)
        logp = self.actor(latent)
        v1 = self.critic1(latent)
        v2 = self.critic2(latent)
        return logp, (v1, v2)

class RLAgent:
    def __init__(self, net, stochastic):
        self.prev_heads = [-1, -1, -1, -1]
        self.net = net
        self.stochastic = stochastic

    def raw_outputs(self, state):
        with torch.no_grad():
            logits, (v1, v2) = self.net(state.cuda().unsqueeze(0))
            logits = logits.squeeze(0)
            v1 = v1.squeeze(0)
            v2 = v2.squeeze(0)
            if self.stochastic:
                # get probabilities
                probs = torch.exp(logits)
                # convert 2 numpy
                probs = probs.cpu().detach().numpy()
                action = np.random.choice(range(4), p=probs)
            else:
                action = np.argmax(logits.cpu().detach().numpy())
            return action, logits[action], (v1, v2)

    def __call__(self, observation, configuration):
        if observation['step'] == 0:
            self.prev_heads = [-1, -1, -1, -1]
        state = get_features(observation, configuration, self.prev_heads)
        action, _, _ = self.raw_outputs(state)
        self.prev_heads = [goose[0] if len(goose) > 0 else -1 for goose in observation['geese']]
        return ['NORTH', 'EAST', 'SOUTH', 'WEST'][action]