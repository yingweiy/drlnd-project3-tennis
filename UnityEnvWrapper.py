from unityagents import UnityEnvironment
import numpy as np

class TennisEnv:
    def __init__(self):
        self.env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64")

        # get the default brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.env_info = self.reset(True)
        # number of agents
        self.num_agents = len(self.env_info.agents)
        print('Number of agents:', self.num_agents)
        # size of each action
        self.action_size = self.brain.vector_action_space_size
        print('Size of each action:', self.action_size)
        # examine the state space
        states = self.env_info.vector_observations
        self.state_size = states.shape[1]
        print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
        print('The state for the first agent looks like:', states[0])


    def reset(self, train_mode):
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        return env_info

    def step(self, actions):
        env_info = self.env.step(actions)[self.brain_name]  # send all actions to tne environment
        next_states = env_info.vector_observations  # get next state (for each agent)
        rewards = env_info.rewards  # get reward (for each agent)
        dones = env_info.local_done
        return next_states, rewards, dones, env_info

