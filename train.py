import numpy as np
from tqdm import tqdm
import random
import torch

import pyglet #.window import Window
from environment import create_env
from utils import *
from rominfo import *
from agent import DQNAgent
from itertools import count

def train_dqn(num_episodes: int, random_seeds: 'list[int]') -> 'list[list[int]]':
    """Trains the DQN agent with different random seeds.

    Args:
        num_episodes: Total number of episodes
        random_seeds: List of random seeds for training

    Returns:
        rewards_over_seeds: List of rewards for each seed
    """
    rewards_over_seeds = []

    for seed in random_seeds:
        # set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create environment
        env, obs_space_dims, action_space_dims = create_env()
        
        agent = DQNAgent.load(DQNAgent)
        reward_over_episodes = []
        
        episode = 0
        #for episode in tqdm(range(num_episodes)):  #* botei um while porque nem ferrando que eu quero que pare de treinar
        while True:
            episode += 1
            env.reset()
            obs_str, _, _ = getState(getRam(env), 6)
            obs = [float(numero) for numero in obs_str.split(',')]
            done = False
            action_hold = 0
            rewards = 0
            for t in count():
                env.render()

                if t%50 == 0:
                    action = agent.choose_action(obs)
                    action_hold = action
                else:
                    action = action_hold
                _ , reward, done, _ = env.step(dec2bin(action))
                next_obs_str, _, y = getState(getRam(env), 6)
                next_obs = [float(numero) for numero in next_obs_str.split(',')]

                if env.data.is_done() or y > 400:
                    done = True

                rewards += reward
                # store transition and learn
                agent.store_transition(
                    obs, action, reward, next_obs, done
                )
                
                agent.learn()

                # End the episode when either truncated or terminated is true
                done = done

                # Update the observation
                obs = next_obs
                if done:
                    break

            reward_over_episodes.append(rewards)

            # update target network
            if episode % agent.update_freq == 0:
                agent.update_target_network()

            # Print average reward every 100 episodes
            if episode % 100 == 0:
                avg_reward = int(np.mean(reward_over_episodes))  	
                print("Seed:", random_seeds.index(seed) + 1,"Episode:", episode, "Average Reward:", avg_reward)
            pyglet.app.exit()
            agent.save()
        env.close()
        rewards_over_seeds.append(reward_over_episodes)
        
    print('Complete')

    return rewards_over_seeds