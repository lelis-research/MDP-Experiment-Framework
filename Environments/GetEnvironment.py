from Environments.MiniGrid.GetEnvironment import MINIGRID_ENV_LST
from Environments.MiniHack.GetEnvironment import MINIHACK_ENV_LST

ENV_LST = MINIGRID_ENV_LST + MINIHACK_ENV_LST

def get_env(env_name,
            num_envs=1, 
            max_steps=500, 
            render_mode=None, 
            wrapping_lst=None, 
            wrapping_params=[]):
    
    assert env_name in ENV_LST

    if env_name in MINIGRID_ENV_LST:
        from Environments.MiniGrid.GetEnvironment import get_single_env, get_parallel_env
        if num_envs == 1:
            env = get_single_env(env_name, max_steps, render_mode, wrapping_lst, wrapping_params) 
        else:
            env = get_parallel_env(env_name, num_envs, max_steps, render_mode, wrapping_lst, wrapping_params)

    elif env_name in MINIHACK_ENV_LST:
        from Environments.MiniHack.GetEnvironment import get_single_env, get_parallel_env
        if num_envs == 1:
            env = get_single_env(env_name, max_steps, render_mode, wrapping_lst, wrapping_params) 
        else:
            env = get_parallel_env(env_name, num_envs, max_steps, render_mode, wrapping_lst, wrapping_params)
    
    env.custom_config = {
        "env_name": env_name,
        "num_envs": num_envs,
        "max_steps": max_steps,
        "wrapping_lst": wrapping_lst,
        "wrapping_params": wrapping_params,
    }

    return env