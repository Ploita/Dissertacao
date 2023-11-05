from train import *
import random

if __name__ == "__main__":
    num_episodes = int(2)
    seeds = [random.randint(0, 1000)]
    phases = ['YoshiIsland1', 'YoshiIsland2']
    rewards = train_dqn(num_episodes= num_episodes, random_seeds= seeds)