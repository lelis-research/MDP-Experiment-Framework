import gymnasium as gym
import gymnasium_robotics
from .Wrappers import WRAPPING_TO_WRAPPER
from .Maps import BL_TR, BR_TL, TL_BR, TR_BL
from gymnasium_robotics.envs.maze import AntMazeEnv

gym.register_envs(gymnasium_robotics)
gym.register(
    id="AntMaze_BL_TR-v0",  
    entry_point=AntMazeEnv,  # or your custom class
    kwargs={
        "maze_map": BL_TR,
        # "reward_type": "sparse",
        # "eval": False,
    }
)
gym.register(
    id="AntMaze_BR_TL-v0",  
    entry_point=AntMazeEnv,  # or your custom class
    kwargs={
        "maze_map": BR_TL,
        # "reward_type": "sparse",
        # "eval": False,
    }
)
gym.register(
    id="AntMaze_TL_BR-v0",  
    entry_point=AntMazeEnv,  # or your custom class
    kwargs={
        "maze_map": TL_BR,
        # "reward_type": "sparse",
        # "eval": False,
    }
)
gym.register(
    id="AntMaze_TR_BL-v0",  
    entry_point=AntMazeEnv,  # or your custom class
    kwargs={
        "maze_map": TR_BL,
        # "reward_type": "sparse",
        # "eval": False,
    }
)

# List of supported Mujoco environments


MUJOCO_ENV_LST = [
    "Walker2d-v5",
    "Ant-v5",
    "AntMaze_UMaze-v5",
    "AntMaze_BL_TR-v0",
    "AntMaze_BR_TL-v0",
    "AntMaze_TL_BR-v0",
    "AntMaze_TR_BL-v0",
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
    assert env_name in MUJOCO_ENV_LST, f"Environment {env_name} not supported."
    env = gym.make(env_name, render_mode=render_mode, max_episode_steps=max_steps, **env_params)
    # Apply each wrapper in the provided list with corresponding parameters.
    for i, wrapper_name in enumerate(wrapping_lst):
        env = WRAPPING_TO_WRAPPER[wrapper_name](env, **wrapping_params[i])
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
    assert env_name in MUJOCO_ENV_LST, f"Environment {env_name} not supported."
    # TODO: Complete the parallel environment creation process (e.g., creating env_fns, applying wrappers)
    # For now, raise NotImplementedError as a placeholder.
    raise NotImplementedError("get_parallel_env for Mujoco is not implemented yet.")