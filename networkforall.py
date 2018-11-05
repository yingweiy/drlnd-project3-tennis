import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Network(nn.Module):
    def __init__(self, input_dim, hidden_in_dim, hidden_out_dim, output_dim, actor=False):
        super(Network, self).__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""

        self.fc1 = nn.Linear(input_dim,hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim,hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim,output_dim)
        self.actor = actor

    def forward(self, x):
        if self.actor:
            h1 = f.relu(self.fc1(x))
            h2 = f.relu(self.fc2(h1))
            h3 = torch.clamp(self.fc3(h2), min=-1, max=1)
            return h3
        
        else:
            # critic network simply outputs a number
            h1 = f.relu(self.fc1(x))
            h2 = f.relu(self.fc2(h1))
            h3 = self.fc3(h2)
            return h3