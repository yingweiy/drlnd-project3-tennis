# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
import torch.nn.functional as F

from utilities import soft_update
device = 'cpu'

class MADDPG:
    def __init__(self, discount_factor=0.99, tau=1e-3):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions (both agent) = 24*2 + 2 + 2 = 52
        # 24
        self.maddpg_agent = [DDPGAgent(24, 2,  # actor net: in_actor, hidden, hidden, out_actor
                                       48),   # critic net: in_critic, hidden, hidden
                             DDPGAgent(24, 2, 48)
                             ]
        self.n_agents = len(self.maddpg_agent)
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = []
        for i in range(self.n_agents):
            agent = self.maddpg_agent[i]
            obs = obs_all_agents[i,:]
            action = agent.act(obs, noise)
            actions.append(action)
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = []
        for i in range(self.n_agents):
            agent = self.maddpg_agent[i]
            action = agent.target_act(obs_all_agents[:,i,:], noise)
            target_actions.append(action)
        return target_actions

    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """

        # obs[agent_number]
        obs, obs_full, action, reward, next_obs, next_obs_full, done = samples

        # Shape of objects. N is batch size
        # obs, next_obs: Nx2x24
        # obs_full, next_obs_full: Nx48
        # action: Nx2x2
        # done, reward: Nx2

        agent = self.maddpg_agent[agent_number]


        target_actions = self.target_act(next_obs)
        q_next = agent.target_critic(next_obs_full, target_actions[0], target_actions[1]).squeeze()
        y = reward[:, agent_number].squeeze() + self.discount_factor * q_next * (1 - done[:, agent_number].squeeze())
        q = agent.critic(obs_full, action[:,0,:], action[:,1,:])
        critic_loss = F.mse_loss(q, y)
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
        agent.critic_optimizer.step()

        #update actor network using policy gradient

        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        # shape: N x 4
        # q_input is the actions from both agents
        q_input=[]
        for i in range(self.n_agents):
            acts = self.maddpg_agent[i].actor(obs[:,i,:])
            if i == agent_number:
                q_input.append(acts)
            else:
                q_input.append(acts.detach())

        # get the policy gradient
        actor_loss = -agent.critic(obs_full, q_input[0], q_input[1]).mean()
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1)
        agent.actor_optimizer.step()

        # soft update the target network towards the actual networks
        self.update_targets(agent)

    def update_targets(self, agent):
        """soft update targets"""
        self.iter += 1
        soft_update(agent.target_actor, agent.actor, self.tau)
        soft_update(agent.target_critic, agent.critic, self.tau)
        agent.noise.reset()
            
            
            




