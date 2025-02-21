import gymnasium as gym
import minihack

from .Wrappers import WRAPPING_TO_WRAPPER

# List of supported MiniHack environments
MINIHACK_ENV_LST = [
    "MiniHack-Room-5x5-v0",
]

def get_single_env(env_name, 
                   max_steps=500, 
                   render_mode=None, 
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
    # TODO: Complete the environment creation process (e.g., gym.make, applying wrappers)
    # For now, raise NotImplementedError as a placeholder.
    raise NotImplementedError("get_single_env for MiniHack is not implemented yet.")

def get_parallel_env(env_name,
                     num_envs, 
                     max_steps=500, 
                     render_mode=None,
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
    # TODO: Complete the parallel environment creation process (e.g., creating env_fns, applying wrappers)
    # For now, raise NotImplementedError as a placeholder.
    raise NotImplementedError("get_parallel_env for MiniHack is not implemented yet.")