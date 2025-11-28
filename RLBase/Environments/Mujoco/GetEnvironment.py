import argparse
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import gymnasium_robotics

from . import CustomEnvironments  # noqa: F401 - ensure custom envs are registered on import
from .Wrappers import WRAPPING_TO_WRAPPER

# Common MuJoCo environment ids (extend as needed).
MUJOCO_ENV_LST = [
    "Ant-v5",
    "HalfCheetah-v5",
    "Hopper-v5",
    "Walker2d-v5",
    "Swimmer-v5",
    "Humanoid-v5",
    "HumanoidStandup-v5",
    "InvertedPendulum-v5",
    "InvertedDoublePendulum-v5",
    "Pusher-v5",
    "Reacher-v5",
    "Thrower-v5",
    "Striker-v5",
    
    "AntMaze_UMaze-v5",
]


def get_env(env_name, num_envs, max_steps=1000, render_mode=None, env_params=None, wrapping_lst=None, wrapping_params=None):
    """
    Create a vectorized (parallel) MuJoCo environment.
    
    Args:
        env_name (str): Name of the MuJoCo environment. Must be in MUJOCO_ENV_LST.
        num_envs (int): Number of parallel environments.
        max_steps (int): Maximum steps per episode; forwarded as max_episode_steps unless overridden in env_params.
        render_mode (str or None): Rendering mode for the environments (e.g., "human", "rgb_array"); omitted if None.
        env_params (dict or None): Extra kwargs forwarded to gym.make.
        wrapping_lst (list or None): List of wrapper names to apply.
        wrapping_params (list): List of parameter dictionaries for each wrapper (defaults to empty if not provided).
    
    Returns:
        SyncVectorEnv: A vectorized environment with num_envs instances.
    """
    assert env_name in MUJOCO_ENV_LST, f"Environment {env_name} not supported."
    env_params = {} if env_params is None else env_params
    wrapping_lst = [] if wrapping_lst is None else wrapping_lst
    wrapping_params = [] if wrapping_params is None else wrapping_params

    def make_env():
        env = gym.make(env_name, max_episode_steps=max_steps, render_mode=render_mode, **env_params)
        for i, wrapper_name in enumerate(wrapping_lst):
            params = wrapping_params[i] if i < len(wrapping_params) else {}
            env = WRAPPING_TO_WRAPPER[wrapper_name](env, **params)
        return env

    envs = SyncVectorEnv([make_env for _ in range(num_envs)])
    return envs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Quick MuJoCo vectorized env smoke test.")
    parser.add_argument("--env", type=str, choices=MUJOCO_ENV_LST, default="HalfCheetah-v5", help="MuJoCo env id to create.")
    parser.add_argument("--num_envs", type=int, default=2, help="Number of parallel envs to create.")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max episode steps override passed to gym.make as max_episode_steps.")
    parser.add_argument("--render_mode", type=str, default=None, help="Render mode passed to gym.make.")
    parser.add_argument("--demo_steps", type=int, default=3, help="Number of random steps to run in the demo.")
    args = parser.parse_args()

    envs = get_env(args.env, num_envs=args.num_envs, max_steps=args.max_steps, render_mode=args.render_mode)

    observations, infos = envs.reset()
    print(f"Started {args.env} with {args.num_envs} envs.")

    for step_idx in range(args.demo_steps):
        actions = [envs.single_action_space.sample() for _ in range(args.num_envs)]
        observations, rewards, terminated, truncated, infos = envs.step(actions)
        print(f"Step {step_idx}: actions={actions}, rewards={rewards}, terminated={terminated}, truncated={truncated}")

    envs.close()
