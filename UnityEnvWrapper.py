from unityagents import UnityEnvironment
import numpy as np

class TennisEnv:
    def __init__(self):
        #self.env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")
        self.env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64")

        # get the default brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        states, full_state, env_info = self.reset(True)
        # number of agents
        self.num_agents = len(env_info.agents)
        print('Number of agents:', self.num_agents)
        # size of each action
        self.action_size = self.brain.vector_action_space_size
        print('Size of each action:', self.action_size)
        # examine the state space
        self.state_size = states.shape[-1]
        print('There are {} agents. Each observes a state with length: {}'.format(2, self.state_size))
        print('The state for the first agent looks like:', states[0, 0, :])
        print('The state for the second agent looks like:', states[0, 1, :])
        print('The full state is:', full_state)

    def reset(self, train_mode=True):
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        states = env_info.vector_observations
        full_state = self.get_full_state(states)
        # leave the first dimension for parallel env (future)
        return np.expand_dims(states, axis=0), \
               full_state, env_info

    def step(self, actions):
        env_info = self.env.step(actions)[self.brain_name]  # send all actions to tne environment
        next_states = env_info.vector_observations
        next_state_full = self.get_full_state(next_states)
        rewards = np.array(env_info.rewards)  # get reward (for each agent)
        dones = np.array(env_info.local_done)
        return np.expand_dims(next_states, axis=0), next_state_full, np.expand_dims(rewards, axis=0), np.expand_dims(dones, axis=0), env_info


    def get_full_state(self, x):
        return np.expand_dims(np.concatenate((x[0], x[1])), axis=0)

    def close(self):
        self.env.close()
