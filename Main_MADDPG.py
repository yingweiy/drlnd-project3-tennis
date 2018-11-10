# main function that sets up environments
# perform training loop

from UnityEnvWrapper import TennisEnv
from buffer import ReplayBuffer
from maddpg import MADDPG
import torch
import numpy as np
import os
from collections import deque

def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    seeding()
    # number of parallel agents
    number_of_agents = 2
    # number of training episodes.
    # change this to higher number to experiment. say 30000.
    number_of_episodes = 5000
    max_t = 1000
    batchsize = 128
    
    # amplitude of OU noise
    # this slowly decreases to 0
    noise = 1
    noise_reduction = 0.9999

    tau = 1e-3   # soft update factor
    gamma = 0.99 # reward discount factor

    # how many episodes before update
    episode_per_update = 2

    model_dir= os.getcwd()+"/model_dir"
    os.makedirs(model_dir, exist_ok=True)

    # do we need to set multi-thread for this env?
    torch.set_num_threads(number_of_agents*2)

    env = TennisEnv()
    
    # keep 5000 episodes worth of replay
    buffer = ReplayBuffer(int(1e5))
    
    # initialize policy and critic
    maddpg = MADDPG(discount_factor=gamma, tau=tau)

    # training loop
    scores_window = deque(maxlen=100)
    ep_scores = []

    # when to save
    save_on_scores = {5: False, 6: False, 9: False, 10: False, 11:False,
                      12: False, 13: False, 14: False, 15: False, 16:False,
                      17: False, 18: False, 19: False, 20: False}

    for episode in range(0, number_of_episodes):
        reward_this_episode = np.zeros((1, number_of_agents))
        obs, obs_full, env_info = env.reset()
        agent0_reward = []
        agent1_reward = []

        for agent in maddpg.maddpg_agent:
            agent.noise.reset()

        for episode_t in range(max_t):
            # explore = only explore for a certain number of episodes
            # action input needs to be transposed
            #print('Obs:', obs)
            actions = maddpg.act(torch.tensor(obs, dtype=torch.float), noise=noise)
            #print(actions)

            #if noise>0.01:
            noise *= noise_reduction
            actions_for_env = torch.stack(actions).detach().numpy()

            # step forward one frame
            next_obs, next_obs_full, rewards, dones, info = env.step(actions_for_env)

            # add data to buffer
            buffer.push(obs, obs_full, actions_for_env, rewards, next_obs, next_obs_full, dones)

            reward_this_episode += rewards

            obs = np.copy(next_obs)
            obs_full = np.copy(next_obs_full)

            # update once after every episode_per_update
            if len(buffer) > batchsize and episode>0 and episode % episode_per_update==0:
                for a_i in range(number_of_agents):
                    samples = buffer.sample(batchsize)
                    maddpg.update(samples, a_i)

            if np.any(dones):
                break

        agent0_reward.append(reward_this_episode[0, 0])
        agent1_reward.append(reward_this_episode[0, 1])
        avg_rewards = max(reward_this_episode[0, 0], reward_this_episode[0, 1])
        scores_window.append(avg_rewards)
        cur_score = np.mean(scores_window)
        ep_scores.append(cur_score)
        print('\rEpisode:{}, Rwd:{:.3f} vs. {:.3f}, Average Score:{:.4f}, Noise:{:.4f}'.format(episode,
                                                                              reward_this_episode[0, 0],
                                                                              reward_this_episode[0, 1],
                                                                              cur_score,noise))

        #saving model



        save_dict_list =[]
        save_info = False
        score_code = int(cur_score * 10)
        if score_code in save_on_scores.keys():
            if not(save_on_scores[score_code]):
                save_on_scores[score_code] = True
                save_info = True

        if save_info:
            for i in range(number_of_agents):
                save_dict = {'actor_params' : maddpg.maddpg_agent[i].actor.state_dict(),
                             'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                             'critic_params' : maddpg.maddpg_agent[i].critic.state_dict(),
                             'critic_optim_params' : maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
                save_dict_list.append(save_dict)

                torch.save(save_dict_list, 
                           os.path.join(model_dir, 'episode-{}-{}.pt'.format(episode,score_code)))

            np.savez('scores-{}-{}.npz'.format(episode,score_code),  agent0_reward = np.array(agent0_reward),
                                    agent1_reward = np.array(agent1_reward),
                                    avg_max_scores = np.array(ep_scores))

    env.close()

if __name__=='__main__':
    main()
