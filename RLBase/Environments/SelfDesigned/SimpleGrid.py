"""A small grid-navigation environment for testing reinforcement-learning agents."""

from __future__ import annotations

from typing import Optional

import numpy as np
from gymnasium import spaces

from .base import BaseEnvironment


EMPTY = 0
AGENT = 1
GOAL = 2

ACTION_NAMES = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
}

ACTION_DELTAS = {
    0: np.array([-1, 0], dtype=np.int64),
    1: np.array([1, 0], dtype=np.int64),
    2: np.array([0, -1], dtype=np.int64),
    3: np.array([0, 1], dtype=np.int64),
}


class SimpleGridEnv(BaseEnvironment):
    """Simple grid navigation environment for testing RL agents."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        max_steps: int = 100,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if render_mode not in {None, "ansi"}:
            raise ValueError("render_mode must be None or 'ansi'")

        self.size = 7
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=EMPTY,
            high=GOAL,
            shape=(self.size, self.size),
            dtype=np.int64,
        )

        self.agent_pos = np.zeros(2, dtype=np.int64)
        self.goal_pos = np.zeros(2, dtype=np.int64)
        self.steps = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        del options
        super().reset(seed=seed)

        cells = [
            (row, column)
            for row in range(self.size)
            for column in range(self.size)
        ]
        indices = self.np_random.choice(len(cells), size=2, replace=False)

        self.agent_pos = np.array(cells[indices[0]], dtype=np.int64)
        self.goal_pos = np.array(cells[indices[1]], dtype=np.int64)
        self.steps = 0

        return self._get_obs(), {}

    def step(self, action: int):
        if action not in ACTION_DELTAS:
            raise ValueError(f"Invalid action: {action}")

        self.steps += 1
        self._move(action)

        reward = 1 if self._on_goal() else 0
        terminated = bool(reward)
        truncated = self.steps >= self.max_steps and not terminated

        return self._get_obs(), reward, terminated, truncated, {}

    def _move(self, action: int) -> None:
        new_pos = self.agent_pos + ACTION_DELTAS[action]

        if self._in_bounds(new_pos):
            self.agent_pos = new_pos

    def _get_obs(self) -> np.ndarray:
        grid = np.full(
            (self.size, self.size),
            EMPTY,
            dtype=np.int64,
        )

        grid[tuple(self.goal_pos)] = GOAL
        grid[tuple(self.agent_pos)] = AGENT
        return grid

    def _in_bounds(self, pos: np.ndarray) -> bool:
        return bool(np.all((0 <= pos) & (pos < self.size)))

    def _on_goal(self) -> bool:
        return bool(np.array_equal(self.agent_pos, self.goal_pos))

    def render(self) -> str:
        chars = {
            EMPTY: ".",
            AGENT: "@",
            GOAL: "Y",
        }
        grid = self._get_obs()

        return "\n".join(
            " ".join(chars[int(cell)] for cell in row)
            for row in grid
        )


def play_simple_grid() -> None:
    """Play the environment interactively from a terminal."""

    env = SimpleGridEnv()
    observation, _ = env.reset()
    terminated = False
    truncated = False

    print("SimpleGridEnv")
    print("Actions: 0=up, 1=down, 2=left, 3=right, q=quit")
    print("Observation:")
    print(observation)
    print(env.render())

    while not (terminated or truncated):
        raw = input("\naction> ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            break
        if raw not in {"0", "1", "2", "3"}:
            print("Enter 0, 1, 2, 3, or q.")
            continue

        action = int(raw)
        observation, reward, terminated, truncated, _ = env.step(action)
        print(f"action={action} ({ACTION_NAMES[action]}) reward={reward}")
        print("Observation:")
        print(observation)
        print(env.render())

    if terminated:
        print("Episode terminated.")
    elif truncated:
        print("Episode truncated at the step limit.")

    env.close()


if __name__ == "__main__":
    play_simple_grid()
