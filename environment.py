from typing import Any
import retro

def create_env() -> 'tuple[Any, int, int]':
    """Creates and wraps the environment for training

    Returns:
        env: Wrapped environment
    """
    env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland1', players=1) # type: ignore #, record = '.')  

    # Records episode-reward
    #wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # type: ignore
    
    # Observation-space of InvertedPendulum-v4
    obs_space_dims = env.observation_space.shape # type: ignore
    
    # Action-space of InvertedPendulum-v4
    action_space_dims = env.action_space.shape[0] # type: ignore
    # todo achar uma nova forma de incorporar essa informação

    return env, obs_space_dims, action_space_dims