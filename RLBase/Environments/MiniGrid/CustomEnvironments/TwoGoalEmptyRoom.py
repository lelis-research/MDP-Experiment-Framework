import gymnasium as gym
from gymnasium.envs.registration import register
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import  Goal
from minigrid.utils.rendering import fill_coords, point_in_rect


class RewardGoal(Goal):
    """A Goal that carries a custom reward and optional terminal flag."""
    def __init__(self, reward_value: float = 0.0, is_terminal: bool = False, color: str = "red"):
        super().__init__(color)
        self.reward_value = float(reward_value)
        self.is_terminal = bool(is_terminal)

class TwoGoalEmptyRoom(MiniGridEnv):
    """
    6x6 empty room, red goal in one interior corner and green goal in the opposite.
    +1 for each goal (once). Episode ends only after both are collected.
    """

    def __init__(self, size=6, max_steps=None, agent_start_pos=None, agent_start_dir=0, **kwargs):
        mission_space = MissionSpace(mission_func=lambda: "collect both goals")
        if max_steps is None:
            max_steps = 4 * size * size

        self.size = size
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        self.red_collected = False
        self.green_collected = False
        self.red_pos = None
        self.green_pos = None

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            max_steps=max_steps,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Opposite interior corners
        self.red_pos = (1, 1)
        self.green_pos = (width - 2, height - 2)

        # Place goals (reward stored here, but we’ll control reward/termination in step)
        self.grid.set(*self.red_pos, RewardGoal(reward_value=1.0, is_terminal=False, color="red"))
        self.grid.set(*self.green_pos, RewardGoal(reward_value=1.0, is_terminal=False, color="green"))

        self.red_collected = False
        self.green_collected = False

        # Start agent
        if self.agent_start_pos is None:
            self.agent_pos = (width // 2, height // 2)
        else:
            self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        self.mission = "collect red and green goals"

    def step(self, action):
        # Run MiniGrid dynamics (movement, walls, etc.)
        obs, _, terminated, truncated, info = super().step(action)

        # Ignore MiniGrid’s own goal termination and compute our own reward/termination
        reward = 0.0
        terminated = False

        # Check whether we are standing on a goal tile (by position is simplest/robust)
        if (self.agent_pos == self.red_pos) and (not self.red_collected):
            self.red_collected = True
            reward += 1.0
            self.grid.set(*self.red_pos, None)   # remove so it can't be collected again

        if (self.agent_pos == self.green_pos) and (not self.green_collected):
            self.green_collected = True
            reward += 1.0
            self.grid.set(*self.green_pos, None)

        if self.red_collected and self.green_collected:
            terminated = True

        return obs, reward, terminated, truncated, info

register(
    id="MiniGrid-EmptyTwoGoals-6x6-v0",
    entry_point=TwoGoalEmptyRoom,
    kwargs={"size": 6},
)