from .MiniGrid import MINIGRID_ENV_LST
from .MiniHack import MINIHACK_ENV_LST

# Combine supported environment lists from both MiniGrid and MiniHack.
ENV_LST = MINIGRID_ENV_LST + MINIHACK_ENV_LST

def get_env(env_name,
            num_envs=1, 
            max_steps=500, 
            render_mode=None,
            env_params={}, 
            wrapping_lst=None, 
            wrapping_params=[]):
    """
    Retrieve an environment (single or vectorized) based on the environment name.
    
    Args:
        env_name (str): Name of the environment; must be in ENV_LST.
        num_envs (int): Number of parallel environments to create.
        max_steps (int): Maximum steps per episode.
        render_mode (str or None): Render mode.
        wrapping_lst (list or None): List of wrappers to apply.
        wrapping_params (list): Parameters for each wrapper.
    
    Returns:
        gym.Env or gym.vector.AsyncVectorEnv: The created (and wrapped) environment.
    """
    assert env_name in ENV_LST, f"Environment {env_name} is not supported."
    
    # Select the proper environment creation functions based on the env type.
    if env_name in MINIGRID_ENV_LST:
        from .MiniGrid import get_single_env, get_parallel_env
        if num_envs == 1:
            env = get_single_env(env_name, max_steps, render_mode, env_params, wrapping_lst, wrapping_params)
        else:
            env = get_parallel_env(env_name, num_envs, max_steps, render_mode, env_params, wrapping_lst, wrapping_params)
    elif env_name in MINIHACK_ENV_LST:
        from .MiniHack import get_single_env, get_parallel_env
        if num_envs == 1:
            env = get_single_env(env_name, max_steps, render_mode, env_params, wrapping_lst, wrapping_params)
        else:
            env = get_parallel_env(env_name, num_envs, max_steps, render_mode, env_params, wrapping_lst, wrapping_params)
    
    return env