import gymnasium as gym
import minihack
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from .Wrappers import WRAPPING_TO_WRAPPER, MiniHackWrap

# List of supported MiniHack environments
MINIHACK_ENV_LST = [
    "MiniHack-Corridor-R2-v0",
]

def get_single_env(env_name, 
                   max_steps=500, 
                   render_mode=None, 
                   env_params={},
                   wrapping_lst=None, 
                   wrapping_params=[]):
    """
    Create a single MiniHack environment with optional wrappers.
    
    Args:
        env_name (str): Name of the MiniHack environment. Must be in MINIHACK_ENV_LST.
        max_steps (int): Maximum number of steps per episode.
        render_mode (str or None): Render mode for the environment.
        wrapping_lst (list or None): List of wrapper names to apply.
        wrapping_params (list): List of parameter dictionaries for each wrapper.
        
    Returns:
        gym.Env: A wrapped MiniHack environment.
    """
    assert env_name in MINIHACK_ENV_LST, f"Environment {env_name} not supported."
    # env = gym.make(env_name, max_episode_steps=max_steps, render_mode=render_mode, **env_params)
    # # Apply each wrapper in the provided list with corresponding parameters.
    # for i, wrapper_name in enumerate(wrapping_lst):
    #     env = WRAPPING_TO_WRAPPER[wrapper_name](env, **wrapping_params[i])
        
        
    base = gym.make(
        env_name,
        observation_keys=(
            "chars",
            "glyphs",
            "blstats",
            "chars_crop",
            "glyphs_crop",
        ),
    )
    env = MiniHackWrap(base, step_reward=-1.0, goal_reward=1000.0, **env_params) #seed=seed, view_size=view_size
    return env

def get_parallel_env(env_name,
                     num_envs, 
                     max_steps=500, 
                     render_mode=None,
                     env_params={},
                     wrapping_lst=None, 
                     wrapping_params=[]):
    """
    Create a vectorized (parallel) MiniHack environment with optional wrappers.
    
    Args:
        env_name (str): Name of the MiniHack environment. Must be in MINIHACK_ENV_LST.
        num_envs (int): Number of parallel environments.
        max_steps (int): Maximum number of steps per episode.
        render_mode (str or None): Render mode for the environments.
        wrapping_lst (list or None): List of wrapper names to apply.
        wrapping_params (list): List of parameter dictionaries for each wrapper.
        
    Returns:
        gym.vector.AsyncVectorEnv: A vectorized MiniHack environment.
    """
    assert env_name in MINIHACK_ENV_LST, f"Environment {env_name} not supported."
    env_fns = []
    for _ in range(num_envs):
        base = gym.make(
        env_name,
            observation_keys=(
                "chars",
                "glyphs",
                "blstats",
                "chars_crop",
                "glyphs_crop",
            ),
        )
        env = MiniHackWrap(base, step_reward=-1.0, goal_reward=1000.0, **env_params) #seed=seed, view_size=view_size
        env_fns.append(lambda: env)
        
    
    envs = SyncVectorEnv(env_fns)
    return envs