import argparse
import json
from types import SimpleNamespace
from typing import Any, Optional

import gymnasium as gym
import minihack  # noqa: F401 - ensures envs are registered
import numpy as np

from .Wrappers import WRAPPING_TO_WRAPPER
from ..utils import coerce_wrappers, ensure_namespace, first_non_none, get_attr, merge_env_params

MINIHACK_ENV_LST = [
    "MiniHack-River-v0",
]


def make_minihack_env(full_env_name: str, cfg: Any = None, env_config: Any = None, render_mode: Optional[str] = None) -> gym.Env:
    """Sample Factory-compatible MiniHack constructor."""
    env_name = full_env_name
    assert env_name in MINIHACK_ENV_LST, f"Environment {env_name} is not supported."
    cfg = ensure_namespace(cfg)
    env_config = env_config or {}

    env_params = merge_env_params(cfg, env_config)
    wrappers, wrapper_params = coerce_wrappers(
        get_attr(cfg, "env_wrappers", []) or get_attr(cfg, "wrapping_lst", []),
        get_attr(cfg, "env_wrapper_params", []) or get_attr(cfg, "wrapping_params", []),
    )
    wrappers = get_attr(env_config, "env_wrappers", wrappers) or wrappers
    wrapper_params = get_attr(env_config, "env_wrapper_params", wrapper_params) or wrapper_params
    wrappers, wrapper_params = coerce_wrappers(wrappers, wrapper_params)

    max_steps = first_non_none(
        get_attr(env_config, "max_steps", None),
        get_attr(cfg, "env_max_steps", None),
        get_attr(cfg, "max_steps", None),
    )
    render_mode = first_non_none(
        render_mode,
        get_attr(env_config, "render_mode", None),
        get_attr(cfg, "render_mode", None),
    )

    env = gym.make(
        env_name,
        max_episode_steps=max_steps,
        render_mode=render_mode,
        **env_params,
    )
    for wrapper_name, params in zip(wrappers, wrapper_params):
        wrapper = WRAPPING_TO_WRAPPER.get(wrapper_name)
        if wrapper is None:
            raise KeyError(f"Wrapper '{wrapper_name}' is not registered for MiniHack.")
        env = wrapper(env, **params)

    seed = first_non_none(get_attr(env_config, "seed", None), get_attr(cfg, "seed", None))
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_minihack_sf_env(env_name: str, cfg: Any = None, env_config: Any = None, render_mode: Optional[str] = None, **_) -> gym.Env:
    return make_minihack_env(env_name, cfg, env_config, render_mode=render_mode)


def make_minihack_sf_factory(env_name: str):
    return lambda cfg=None, env_config=None, render_mode=None, **kwargs: make_minihack_env(
        env_name, cfg, env_config, render_mode=render_mode
    )


get_env = make_minihack_env


def _describe_observation(observation):
    if isinstance(observation, dict):
        return {k: _describe_observation(v) for k, v in observation.items()}
    arr = np.asarray(observation)
    return {"shape": list(arr.shape), "dtype": str(arr.dtype)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniHack single-env sanity check (Sample Factory style).")
    parser.add_argument("--env", type=str, default="MiniHack-River-v0", choices=MINIHACK_ENV_LST)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--render-mode", type=str, default=None, choices=[None, "human", "ansi"])
    parser.add_argument("--env-params", type=json.loads, default="{}")
    parser.add_argument("--env-wrappers", type=json.loads, default="[]")
    parser.add_argument("--wrapper-params", type=json.loads, default="[]")
    args = parser.parse_args()

    cfg = SimpleNamespace(
        env_params=args.env_params,
        env_wrappers=args.env_wrappers,
        env_wrapper_params=args.wrapper_params,
        env_max_steps=args.max_steps,
        render_mode=args.render_mode,
        seed=args.seed,
    )
    env = make_minihack_env(args.env, cfg=cfg, env_config={})

    obs, info = env.reset()
    print(f"Reset env '{args.env}'")
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Initial observation info:", _describe_observation(obs))
    print("Initial info:", info)

    for step in range(1, args.steps + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {step}")
        print(f"  Action: {action}")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        print(f"  Observation info: {_describe_observation(obs)}")
        print(f"  Info: {info}")
        if args.render_mode and args.render_mode != "human":
            rendered = env.render()
            if rendered is not None:
                print(f"  Render output type: {type(rendered)}")

    env.close()
