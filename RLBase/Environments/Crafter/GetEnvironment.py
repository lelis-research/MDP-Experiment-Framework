import argparse
from enum import IntEnum
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv
import crafter

from . import CustomEnvironments  # noqa: F401
from .Wrappers import WRAPPING_TO_WRAPPER

CRAFTER_ENV_LST = [
    "CrafterReward-v1",
    "CrafterNoReward-v1",
]


class CrafterGymnasiumWrapper(gym.Env):
    """Thin gymnasium adapter around ``crafter.Env``.

    ``crafter`` uses the old gym API (4-tuple step, bare-obs reset).  This
    wrapper converts those calls to the gymnasium interface (5-tuple step,
    (obs, info) reset) and re-declares the spaces using ``gymnasium.spaces``
    so that ``SyncVectorEnv`` and all standard gymnasium wrappers work
    correctly.

    Args:
        reward (bool): Whether to use the achievement-based reward signal.
            ``True`` → ``CrafterReward-v1``; ``False`` → ``CrafterNoReward-v1``.
        length (int): Maximum episode length in steps (crafter's ``length``).
        seed (int or None): Optional RNG seed forwarded to ``crafter.Env``.
        render_mode (str or None): ``"rgb_array"`` returns frames from
            ``render()``; ``"human"`` displays the frame via pygame.
        **kwargs: Any remaining kwargs forwarded to ``crafter.Env``.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, reward=True, length=10000, seed=None, render_mode=None, **kwargs):
        super().__init__()
        self._env = crafter.Env(reward=reward, length=length, seed=seed, **kwargs)
        self.render_mode = render_mode
        self.observation_space = spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(self._env.action_names))
        self._pygame_display = None
        # IntEnum matching MiniGrid's .actions convention so HumanAgent works unchanged.
        self.actions = IntEnum(
            "Actions", {name: i for i, name in enumerate(self._env.action_names)}
        )

    def reset(self, seed=None, options=None):
        obs = self._env.reset()
        info = {}
        if self.render_mode == "human":
            self._render_human(obs)
        return np.asarray(obs, dtype=np.uint8), info

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = np.asarray(obs, dtype=np.uint8)
        if self.render_mode == "human":
            self._render_human(obs)
        return obs, float(reward), bool(done), False, info

    def render(self):
        obs = self._env.render()
        return np.asarray(obs, dtype=np.uint8)

    def close(self):
        if self._pygame_display is not None:
            import pygame
            pygame.quit()
            self._pygame_display = None
        if hasattr(self._env, "close"):
            self._env.close()

    # ------------------------------------------------------------------
    def _render_human(self, frame):
        import pygame
        frame = np.asarray(frame, dtype=np.uint8)
        if self._pygame_display is None:
            pygame.init()
            h, w = frame.shape[:2]
            scale = max(1, 512 // max(h, w))
            self._scale = scale
            self._pygame_display = pygame.display.set_mode((w * scale, h * scale))
            pygame.display.set_caption("Crafter")
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        h, w = frame.shape[:2]
        surface = pygame.surfarray.make_surface(
            np.transpose(frame, (1, 0, 2))  # (H,W,C) → (W,H,C) for pygame
        )
        scaled = pygame.transform.scale(
            surface,
            (w * self._scale, h * self._scale),
        )
        self._pygame_display.blit(scaled, (0, 0))
        pygame.display.flip()


def get_env(
    env_name,
    num_envs,
    max_steps=10000,
    render_mode=None,
    env_params=None,
    wrapping_lst=None,
    wrapping_params=None,
):
    """
    Create a vectorized (parallel) Crafter environment.

    Args:
        env_name (str): Must be in CRAFTER_ENV_LST.
        num_envs (int): Number of parallel environments.
        max_steps (int or None): Maximum episode length (crafter ``length``).
        render_mode (str or None): ``"human"`` or ``"rgb_array"``.
        env_params (dict or None): Extra kwargs forwarded to ``crafter.Env``
            (e.g. ``area``, ``view``, ``size``).
        wrapping_lst (list or None): Wrapper names to apply (see Wrappers.py).
        wrapping_params (list or None): Parameter dicts per wrapper.

    Returns:
        SyncVectorEnv: Vectorized environment with ``num_envs`` instances.
    """
    assert env_name in CRAFTER_ENV_LST, f"Environment {env_name} not supported."
    env_params = {} if env_params is None else env_params
    wrapping_lst = [] if wrapping_lst is None else wrapping_lst
    wrapping_params = [] if wrapping_params is None else wrapping_params

    use_reward = (env_name == "CrafterReward-v1")
    length = max_steps if max_steps is not None else 10000

    def make_env():
        env = CrafterGymnasiumWrapper(
            reward=use_reward,
            length=length,
            render_mode=render_mode,
            **env_params,
        )
        for i, wrapper_name in enumerate(wrapping_lst):
            params = wrapping_params[i] if i < len(wrapping_params) else {}
            env = WRAPPING_TO_WRAPPER[wrapper_name](env, **params)
        return env

    envs = SyncVectorEnv([make_env for _ in range(num_envs)])
    return envs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick Crafter vectorized env smoke test.")
    parser.add_argument(
        "--env", type=str, choices=CRAFTER_ENV_LST, default="CrafterReward-v1",
    )
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--render_mode", type=str, default=None)
    parser.add_argument("--demo_steps", type=int, default=3)
    args = parser.parse_args()

    envs = get_env(
        args.env,
        num_envs=args.num_envs,
        max_steps=args.max_steps,
        render_mode=args.render_mode,
    )
    observations, infos = envs.reset()
    print(f"Started {args.env} with {args.num_envs} envs.")
    print(f"  obs shape : {observations.shape}")
    print(f"  action space: {envs.single_action_space}")

    for step_idx in range(args.demo_steps):
        actions = [envs.single_action_space.sample() for _ in range(args.num_envs)]
        observations, rewards, terminated, truncated, infos = envs.step(actions)
        print(
            f"Step {step_idx}: actions={actions}, rewards={rewards}, "
            f"terminated={terminated}, truncated={truncated}"
        )

    envs.close()