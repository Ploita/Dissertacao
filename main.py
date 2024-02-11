from q_agent import QAgent
from utils import set_seed

import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


env = gym.make('CartPole-v1', render_mode = 'human')

# Good hyper-parameters
# make you feel great!
hparams = {
    'learning_rate': 0.00016151809562265122,
    'discount_factor': 0.99,
    'batch_size': 32,
    'memory_size': 10000,
    'freq_steps_train': 8,
    'freq_steps_update_target': 10,
    'n_steps_warm_up_memory': 1000,
    'n_gradient_steps': 16,
    'nn_hidden_layers': [256, 256],
    'max_grad_norm': 10,
    'normalize_state': False,
    'epsilon_start': 0.9,
    'epsilon_end': 0.14856584122699473,
    'steps_epsilon_decay': 10000,
}

# Lapadula: New hparams from saved_agents/CartPole-v1/298/hparams.json
hparams = {"learning_rate": 0.00045095481485457226, "discount_factor": 0.99, "batch_size": 16, "memory_size": 100000, "freq_steps_update_target": 10, "n_steps_warm_up_memory": 1000, "freq_steps_train": 8, "n_gradient_steps": 16, "nn_hidden_layers": [256, 256], "max_grad_norm": 1, "normalize_state": False, "epsilon_start": 0.9, "epsilon_end": 0.06286625175600052, "steps_epsilon_decay": 10000}

SEED = 2386916045

set_seed(env, SEED)

agent = QAgent(env, **hparams)

from loops import train
train(agent, env, n_episodes=200)

from loops import evaluate
rewards, steps = evaluate(
    agent, env,
    n_episodes=1000,
    epsilon=0.00
)

reward_avg = np.array(rewards).mean()
reward_std = np.array(rewards).std()
print(f'Reward average {reward_avg:.2f}, std {reward_std:.2f}')


fig, ax = plt.subplots(figsize = (10, 4))
ax.set_title("Rewards")    
pd.Series(rewards).plot(kind='hist', bins=100)

plt.show()