# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

import numpy as np
from ddpg import DDPGAgent
import torch
from utilities import soft_update, transpose_to_tensor
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

#def transpose_to_tensor(x):
#    return torch.tensor(x, dtype=torch.float)

class MADDPG:
    def __init__(self, discount_factor=0.99, tau=1e-3):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions (both agent) = 24*2 + 2 + 2 = 52
        # 24
        self.maddpg_agent = [DDPGAgent(24, 2,  # actor net: in_actor, hidden, hidden, out_actor
                                       52),   # critic net: in_critic, hidden, hidden
                             DDPGAgent(24, 2, 52)
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
            obs = obs_all_agents[0,i,:]
            action = agent.act(obs, noise)
            actions.append(action)
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = []
        for i in range(self.n_agents):
            agent = self.maddpg_agent[i]
            action = agent.target_act(obs_all_agents[i], noise)
            target_actions.append(action)
        return target_actions

    def update(self, samples, agent_number, logger):
        """update the critics and actors of all the agents """

        # obs[agent_number]
        obs, obs_full, action, reward, next_obs, next_obs_full, done = map(transpose_to_tensor, samples)

        obs_full = torch.stack(obs_full)
        next_obs_full = torch.stack(next_obs_full)

        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1)
        
        target_critic_input = torch.cat((next_obs_full.t(),target_actions), dim=1).to(device)
        q_next = agent.target_critic(target_critic_input)
        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))
        action = torch.cat(action, dim=1)
        critic_input = torch.cat((obs_full.t(), action), dim=1).to(device)
        q = agent.critic(critic_input)
        critic_loss = torch.nn.functional.mse_loss(q, y)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
        agent.critic_optimizer.step()

        #update actor network using policy gradient

        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        # shape: 512 x 4
        # q_input is the actions from both agents
        q_input = [self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs) ]
                
        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        # shape: 1000x52
        q_input2 = torch.cat((obs_full.t(), q_input), dim=1)
        
        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses' % agent_number,
                           {'critic loss': cl,
                            'actor_loss': al},
                           self.iter)

    def update_targets(self, a_i):
        """soft update targets"""
        self.iter += 1
        ddpg_agent = self.maddpg_agent[a_i]
        soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
        soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
        ddpg_agent.noise.reset()
            
            
            




