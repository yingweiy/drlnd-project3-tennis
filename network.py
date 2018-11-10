import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Actor, self).__init__()
        fc = [256, 128, 128, 64]
        self.bn = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc[0])
        self.bn1 = nn.BatchNorm1d(fc[0])
        self.fc2 = nn.Linear(fc[0], fc[1])
        self.bn2 = nn.BatchNorm1d(fc[1])
        self.fc3 = nn.Linear(fc[1], fc[2])
        self.bn3 = nn.BatchNorm1d(fc[2])
        self.fc4 = nn.Linear(fc[2], fc[3])
        self.bn4 = nn.BatchNorm1d(fc[3])
        self.fc5 = nn.Linear(fc[-1], action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc5.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(self.bn(state)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        return x

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Critic, self).__init__()
        fc = [512, 256, 128, 64]
        self.bn = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size+action_size, fc[0])
        self.bn1 = nn.BatchNorm1d(fc[0])
        self.fc2 = nn.Linear(fc[0], fc[1])
        self.bn2 = nn.BatchNorm1d(fc[1])
        self.fc3 = nn.Linear(fc[1], fc[2])
        self.bn3 = nn.BatchNorm1d(fc[2])
        self.fc4 = nn.Linear(fc[2], fc[3])
        self.bn4 = nn.BatchNorm1d(fc[3])
        self.fc5 = nn.Linear(fc[3], 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc5.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action1, action2):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.cat((self.bn(state), action1, action2), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)