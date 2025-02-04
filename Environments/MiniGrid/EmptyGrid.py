import gymnasium as gym
from Environments.MiniGrid.Wrappers import *

env_lst = ["MiniGrid-Empty-5x5-v0",
           "MiniGrid-Empty-Random-5x5-v0",
           "MiniGrid-Empty-6x6-v0",
           "MiniGrid-Empty-Random-6x6-v0",
           "MiniGrid-Empty-8x8-v0",
           "MiniGrid-Empty-16x16-v0"]



def get_empty_grid(env_name="MiniGrid-Empty-5x5-v0", max_steps=500, 
                   render_mode=None, wrapping_lst=None, wrapping_params=[]):
    
    assert env_name in env_lst

    env = gym.make(env_name, max_steps=max_steps, render_mode=render_mode)

    for i, wrapper_name in enumerate(wrapping_lst):
        env = WRAPPING_TO_WRAPPER[wrapper_name](env, **wrapping_params[i])
    return env

if __name__ == "__main__":
    env = get_empty_grid(render_mode="human",
                         wrapping_lst=["ViewSize", "StepReward"], 
                         wrapping_params=[{"agent_view_size": 7},
                                          {"step_reward": -1},
                                          ])
    env.reset(seed=1)