"""Loader for environments designed specifically for this project."""

import argparse

from gymnasium.vector import SyncVectorEnv

from .SimpleGrid import SimpleGridEnv
from .Wrappers import WRAPPING_TO_WRAPPER


SELF_DESIGNED_ENV_LST = [
    "SelfDesigned-SimpleGrid-v0",
]


def get_env(
    env_name,
    num_envs,
    max_steps=None,
    render_mode=None,
    env_params=None,
    wrapping_lst=None,
    wrapping_params=None,
):
    """Create a vectorized self-designed environment."""

    assert env_name in SELF_DESIGNED_ENV_LST, (
        f"Environment {env_name} not supported."
    )

    env_params = {} if env_params is None else dict(env_params)
    wrapping_lst = [] if wrapping_lst is None else wrapping_lst
    wrapping_params = [] if wrapping_params is None else wrapping_params

    def make_env():
        params = dict(env_params)
        if max_steps is not None:
            params["max_steps"] = max_steps
        if render_mode is not None:
            params["render_mode"] = render_mode

        env = SimpleGridEnv(**params)
        for index, wrapper_name in enumerate(wrapping_lst):
            params = (
                wrapping_params[index]
                if index < len(wrapping_params)
                else {}
            )
            env = WRAPPING_TO_WRAPPER[wrapper_name](env, **params)
        return env

    return SyncVectorEnv([make_env for _ in range(num_envs)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quick self-designed environment smoke test."
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=SELF_DESIGNED_ENV_LST,
        default="SelfDesigned-SimpleGrid-v0",
    )
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--demo_steps", type=int, default=3)
    args = parser.parse_args()

    envs = get_env(
        args.env,
        num_envs=args.num_envs,
        max_steps=args.max_steps,
    )
    observations, infos = envs.reset()
    print(f"Started {args.env} with {args.num_envs} envs.")

    for step_index in range(args.demo_steps):
        actions = [
            envs.single_action_space.sample()
            for _ in range(args.num_envs)
        ]
        observations, rewards, terminated, truncated, infos = envs.step(actions)
        print(
            f"Step {step_index}: actions={actions}, rewards={rewards}, "
            f"terminated={terminated}, truncated={truncated}"
        )

    envs.close()
