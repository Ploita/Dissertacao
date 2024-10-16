from typing import Tuple, List, Callable, Union, Optional
import random
from pathlib import Path
from collections import deque
from pdb import set_trace as stop

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter



def train(
    agent,
    env,
    n_episodes: int,
    log_dir: Optional[Path] = None,
    max_steps: Optional[float] = float("inf"),
    n_episodes_evaluate_agent: int = 100,
    freq_episodes_evaluate_agent: int = 200,
) -> Tuple[List, List]:

    # Tensorborad log writer
    logging = False
    if log_dir is not None:
        
        logging = True

    reward_per_episode = []
    steps_per_episode = []
    global_step_counter = 0

    for i in range(0, n_episodes):

        state = env.reset()[0]

        rewards = 0
        steps = 0
        done = False
        while not done:

            action = agent.act(state)

            # agents takes a step and the environment throws out a new state and
            # a reward
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # agent observes transition and stores it for later use
            agent.observe(state, action, reward, next_state, done)

            # learning happens here, through experience replay
            agent.replay()

            global_step_counter += 1
            steps += 1
            rewards += reward
            state = next_state

        # log to Tensorboard
        if logging:
            writer = SummaryWriter(log_dir)
            writer.add_scalar('train/rewards', rewards, i)
            writer.add_scalar('train/steps', steps, i)
            writer.add_scalar('train/epsilon', agent.epsilon, i)
            writer.add_scalar('train/replay_memory_size', len(agent.memory), i)

        reward_per_episode.append(rewards)
        steps_per_episode.append(steps)

        # if (i > 0) and (i % freq_episodes_evaluate_agent) == 0:
        if (i + 1) % freq_episodes_evaluate_agent == 0:
            # evaluate agent
            eval_rewards, eval_steps = evaluate(agent, env,
                                                n_episodes=n_episodes_evaluate_agent,
                                                epsilon=0.01)

            # from utils import get_success_rate_from_n_steps
            # success_rate = get_success_rate_from_n_steps(env, eval_steps)
            #print(f'Reward mean: {np.mean(eval_rewards):.2f}, std: {np.std(eval_rewards):.2f}')
            #print(f'Num steps mean: {np.mean(eval_steps):.2f}, std: {np.std(eval_steps):.2f}')
            # print(f'Success rate: {success_rate:.2%}')
            if logging:
                writer = SummaryWriter(log_dir)
                writer.add_scalar('eval/avg_reward', np.mean(eval_rewards), i)
                writer.add_scalar('eval/avg_steps', np.mean(eval_steps), i)
            # writer.add_scalar('eval/success_rate', success_rate, i)

        if max_steps is not None and global_step_counter > max_steps:
            break
    return reward_per_episode, steps_per_episode


def evaluate(
    agent,
    env,
    n_episodes: int,
    epsilon: Optional[float] = None,
    seed: Optional[int] = 0
) -> Tuple[List, List]:

    from utils import set_seed
    

    # output metrics
    reward_per_episode = []
    steps_per_episode = []

    for i in range(0, n_episodes):
        seed = np.random.randint(0, 2 ** 30 - 1)
        set_seed(env, seed)
        state = env.reset()[0]
        rewards = 0
        steps = 0
        done = False
        while not done:

            action = agent.act(state, epsilon=epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            rewards += reward
            steps += 1
            state = next_state

        reward_per_episode.append(rewards)
        steps_per_episode.append(steps)

    return reward_per_episode, steps_per_episode