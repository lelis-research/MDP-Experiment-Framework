import gymnasium as gym
import minihack

MINIHACK_ENV_LST = [
    "MiniHack-Room-5x5-v0",
]
def get_single_env(env_name, 
                   max_steps=500, 
                   render_mode=None, 
                   wrapping_lst=None, 
                   wrapping_params=[]):
    
    assert env_name in MINIHACK_ENV_LST
    #TODO: Complete Later

def get_parallel_env(env_name,
                     num_envs, 
                     max_steps=500, 
                     render_mode=None,
                     wrapping_lst=None, 
                     wrapping_params=[]):
    
    assert env_name in MINIHACK_ENV_LST
    #TODO: Complete Later