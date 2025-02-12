import gymnasium as gym
from Environments.MiniGrid.Wrappers import *
from gymnasium.vector import AsyncVectorEnv  # or SyncVectorEnv


ENV_LST = [
    "MiniGrid-BlockedUnlockPickup-v0"

    "MiniGrid-LavaCrossingS9N1-v0",
    "MiniGrid-LavaCrossingS9N2-v0",
    "MiniGrid-LavaCrossingS9N3-v0",
    "MiniGrid-LavaCrossingS11N5-v0",

    "MiniGrid-SimpleCrossingS9N1-v0",
    "MiniGrid-SimpleCrossingS9N2-v0",
    "MiniGrid-SimpleCrossingS9N3-v0",
    "MiniGrid-SimpleCrossingS11N15-v0",

    "MiniGrid-DistShift1-v0",
    "MiniGrid-DistShift2-v0",

    "MiniGrid-DoorKey-5x5-v0",
    "MiniGrid-DoorKey-6x6-v0",
    "MiniGrid-DoorKey-8x8-v0",
    "MiniGrid-DoorKey-16x16-v0",

    "MiniGrid-Empty-5x5-v0",
    "MiniGrid-Empty-Random-5x5-v0",
    "MiniGrid-Empty-6x6-v0",
    "MiniGrid-Empty-Random-6x6-v0",
    "MiniGrid-Empty-8x8-v0",
    "MiniGrid-Empty-16x16-v0"]


def get_single_env(env_name, 
                   max_steps=500, 
                   render_mode=None, 
                   wrapping_lst=None, 
                   wrapping_params=[]):
    
    assert env_name in ENV_LST
    env = gym.make(env_name, max_steps=max_steps, render_mode=render_mode)
    for i, wrapper_name in enumerate(wrapping_lst):
        env = WRAPPING_TO_WRAPPER[wrapper_name](env, **wrapping_params[i])
    return env

def get_parallel_env(env_name,
                     num_envs, 
                     max_steps=500, 
                     render_mode=None,
                     wrapping_lst=None, 
                     wrapping_params=[]):
    
    assert env_name in ENV_LST
    
    env_fns = []
    for _ in range(num_envs):
        env = gym.make(env_name, max_steps=max_steps, render_mode=render_mode)
        for i, wrapper_name in enumerate(wrapping_lst):
            env = WRAPPING_TO_WRAPPER[wrapper_name](env, **wrapping_params[i])
        env_fns.append(lambda: env)
    envs = AsyncVectorEnv(env_fns)
    return envs
    