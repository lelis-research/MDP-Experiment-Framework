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
    "MiniGrid-Empty-16x16-v0",
    
    "MiniGrid-Fetch-5x5-N2-v0",
    "MiniGrid-Fetch-6x6-N2-v0",
    "MiniGrid-Fetch-8x8-N3-v0",
    
    "MiniGrid-FourRooms-v0",

    "MiniGrid-GoToDoor-5x5-v0",
    "MiniGrid-GoToDoor-6x6-v0",
    "MiniGrid-GoToDoor-8x8-v0",

    "MiniGrid-GoToObject-6x6-N2-v0",
    "MiniGrid-GoToObject-8x8-N2-v0",

    "MiniGrid-KeyCorridorS3R1-v0",
    "MiniGrid-KeyCorridorS3R2-v0",
    "MiniGrid-KeyCorridorS3R3-v0",
    "MiniGrid-KeyCorridorS4R3-v0",
    "MiniGrid-KeyCorridorS5R3-v0",
    "MiniGrid-KeyCorridorS6R3-v0",

    "MiniGrid-LavaGapS5-v0",
    "MiniGrid-LavaGapS6-v0",
    "MiniGrid-LavaGapS7-v0",

    "MiniGrid-LockedRoom-v0",

    "MiniGrid-MemoryS17Random-v0",
    "MiniGrid-MemoryS13Random-v0",
    "MiniGrid-MemoryS13-v0",
    "MiniGrid-MemoryS11-v0",

    "MiniGrid-MultiRoom-N2-S4-v0",
    "MiniGrid-MultiRoom-N4-S5-v0",
    "MiniGrid-MultiRoom-N6-v0",

    "MiniGrid-ObstructedMaze-1Dlhb-v0",
    "MiniGrid-ObstructedMaze-Full-v0",

    "MiniGrid-PutNear-6x6-N2-v0",
    "MiniGrid-PutNear-8x8-N3-v0",

    "MiniGrid-RedBlueDoors-6x6-v0",
    "MiniGrid-RedBlueDoors-8x8-v0",

    "MiniGrid-Unlock-v0",
    "MiniGrid-UnlockPickup-v0",

    "MiniGrid-Playground-v0",
    
    ]


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
    