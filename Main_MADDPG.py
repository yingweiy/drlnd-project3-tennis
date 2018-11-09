# main function that sets up environments
# perform training loop

from UnityEnvWrapper import TennisEnv
from buffer import ReplayBuffer
from maddpg import MADDPG
import torch
import numpy as np
from tensorboardX import SummaryWriter
import os
from collections import deque
from random import choice
from utilities import transpose_list, transpose_to_tensor

# for saving gif
import imageio

def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    seeding()
    # number of parallel agents
    number_of_agents = 2
    # number of training episodes.
    # change this to higher number to experiment. say 30000.
    number_of_episodes = 4000
    episode_length = 1000
    batchsize = 512
    
    # amplitude of OU noise
    # this slowly decreases to 0
    noise = 1
    noise_reduction = 0.999

    # how many episodes before update
    episode_per_update = 2

    log_path = os.getcwd()+"/log"
    model_dir= os.getcwd()+"/model_dir"
    
    os.makedirs(model_dir, exist_ok=True)

    # do we need to set multi-thread for this env?
    #torch.set_num_threads(number_of_agents)

    env = TennisEnv()
    
    # keep 5000 episodes worth of replay
    buffer = ReplayBuffer(int(1e6))
    
    # initialize policy and critic
    maddpg = MADDPG()
    logger = SummaryWriter(log_dir=log_path)

    # training loop
    scores_window = deque(maxlen=100)
    for episode in range(0, number_of_episodes):
        reward_this_episode = np.zeros((1, number_of_agents))
        obs, obs_full, env_info = env.reset()
        agent0_reward = []
        agent1_reward = []

        for agent in maddpg.maddpg_agent:
            agent.noise.reset()

        for episode_t in range(episode_length):
            # explore = only explore for a certain number of episodes
            # action input needs to be transposed
            actions = maddpg.act(torch.tensor(obs, dtype=torch.float), noise=noise)
            #print(actions)

            if noise>0.01:
                noise *= noise_reduction
            actions_for_env = torch.stack(actions).detach().numpy()

            # step forward one frame
            next_obs, next_obs_full, rewards, dones, info = env.step(actions_for_env)

            actions_for_buffer = np.expand_dims(actions_for_env, axis=0)
            # add data to buffer
            transition = (obs, obs_full, actions_for_buffer, rewards, next_obs, next_obs_full, dones)
            
            buffer.push(transition)
            #print('Rewards:', rewards)
            reward_this_episode += rewards

            obs = np.copy(next_obs)
            obs_full = np.copy(next_obs_full)

            # update once after every episode_per_update
            if len(buffer) > batchsize and episode>300 and episode % episode_per_update==0:
                a_i = choice(range(number_of_agents))
                samples = buffer.sample(batchsize)
                maddpg.update(samples, a_i, logger)
                maddpg.update_targets(a_i) #soft update the target network towards the actual networks


            if np.any(dones):
                break

        agent0_reward.append(reward_this_episode[0, 0])
        agent1_reward.append(reward_this_episode[0, 1])
        avg_rewards = [np.mean(agent0_reward), np.mean(agent1_reward)]
        scores_window.append(avg_rewards)
        print('\rEpisode {}\tRwd:{:.2f}, {:.2f} Average Score: {:.4f} Noise:{:.2f}'.format(episode,
                                                                              reward_this_episode[0, 0],
                                                                              reward_this_episode[0, 1],
                                                                              np.mean(scores_window),
                                                                                           noise)
              )


        #saving model
        save_dict_list =[]
        save_info = False
        if save_info:
            for i in range(3):
                save_dict = {'actor_params' : maddpg.maddpg_agent[i].actor.state_dict(),
                             'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                             'critic_params' : maddpg.maddpg_agent[i].critic.state_dict(),
                             'critic_optim_params' : maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
                save_dict_list.append(save_dict)

                torch.save(save_dict_list, 
                           os.path.join(model_dir, 'episode-{}.pt'.format(episode)))

    env.close()
    logger.close()

if __name__=='__main__':
    main()
