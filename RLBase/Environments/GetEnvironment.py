import argparse

from .MiniGrid import get_env as get_minigrid_env, MINIGRID_ENV_LST
from .Mujoco import get_env as get_mujoco_env, MUJOCO_ENV_LST
from .Atari import get_env as get_atari_env, ATARI_ENV_LST
from .Classic import get_env as get_classic_env, CLASSIC_ENV_LST
# from .MiniHack import get_env as get_minihack_env, MINIHACK_ENV_LST
ENV_SOURCES = (
    ("MiniGrid", MINIGRID_ENV_LST, get_minigrid_env),
    ("Mujoco", MUJOCO_ENV_LST, get_mujoco_env),
    ("Atari", ATARI_ENV_LST, get_atari_env),
    ("Classic", CLASSIC_ENV_LST, get_classic_env),
    # ("MiniHack", MINIHACK_ENV_LST, get_minihack_env),
)

ENV_LST = sorted({env for _, env_list, _ in ENV_SOURCES for env in env_list})


def _resolve_env(env_name):
    for _, env_list, getter in ENV_SOURCES:
        if env_name in env_list:
            return getter
    raise AssertionError(f"Environment {env_name} not supported in any registered family.")


def get_env(env_name, num_envs, max_steps=None, render_mode=None, env_params=None, wrapping_lst=None, wrapping_params=None):
    """
    Dispatch to the correct environment family and build a vectorized env.
    
    Args:
        env_name (str): Environment id; must be present in one of the family lists (MiniGrid, MiniHack, MuJoCo, Atari).
        num_envs (int): Number of parallel environments.
        max_steps (int or None): Optional max steps override (passed through when provided).
        render_mode (str or None): Optional render mode (passed through when provided).
        env_params (dict or None): Extra kwargs forwarded to the family-specific get_env.
        wrapping_lst (list or None): List of wrapper names to apply.
        wrapping_params (list or None): List of parameter dictionaries for each wrapper.
    
    Returns:
        SyncVectorEnv: A vectorized environment with num_envs instances.
    """
    getter = _resolve_env(env_name)

    kwargs = {
        "env_name": env_name,
        "num_envs": num_envs,
        "max_steps": max_steps,
        "render_mode": render_mode,
        "env_params": env_params,
        "wrapping_lst": wrapping_lst,
        "wrapping_params": wrapping_params,
    }

    return getter(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified environment loader for MiniGrid, MiniHack, MuJoCo, Atari.")
    parser.add_argument("--env", type=str, choices=ENV_LST, help="Environment id to create.")
    parser.add_argument("--num_envs", type=int, default=2, help="Number of parallel envs to create.")
    parser.add_argument("--max_steps", type=int, default=None, help="Max episode steps override (family-specific defaults used if None).")
    parser.add_argument("--render_mode", type=str, default=None, help="Render mode passed through to the family env.")
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
