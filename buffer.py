from collections import deque, namedtuple
import random
from utilities import transpose_list
import torch
import numpy as np

device = 'cpu'

class ReplayBuffer:
    def __init__(self,size):
        self.size = size
        self.deque = deque(maxlen=self.size)
        self.experience = namedtuple("Experience", field_names=["obs", "obs_full",
                                     "action", "reward",
                                                                "next_obs", "next_obs_full", "done"])

    def push(self,obs, obs_full, action, reward, next_obs, next_obs_full, done):
        """push into the buffer"""
        e = self.experience(obs, obs_full, action, reward, next_obs, next_obs_full, done)
        self.deque.append(e)

    def sample(self, batchsize):
        """sample from the buffer"""
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.deque, k=batchsize)

        obs = torch.from_numpy(np.stack([e.obs for e in experiences if e is not None])).float().to(device)
        obs_full = torch.from_numpy(np.vstack([e.obs_full for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.stack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_obs = torch.from_numpy(np.stack([e.next_obs for e in experiences if e is not None])).float().to(
            device)
        next_obs_full = torch.from_numpy(np.vstack([e.next_obs_full for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (obs, obs_full, actions, rewards, next_obs, next_obs_full, dones)


    def __len__(self):
        return len(self.deque)



