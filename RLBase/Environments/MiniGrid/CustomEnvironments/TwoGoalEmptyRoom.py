import gymnasium as gym
from gymnasium.envs.registration import register
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import WorldObj
from minigrid.utils.rendering import fill_coords, point_in_rect


class CollectibleGoal(WorldObj):
    """
    A goal-like tile that the agent can step on (overlap),
    but does NOT automatically terminate the episode (unlike minigrid.core.world_object.Goal).
    """
    def __init__(self, color: str):
        super().__init__(type="collectible_goal", color=color)

    def can_overlap(self):
        return True

    def render(self, img):
        # Solid colored square
        fill_coords(img, point_in_rect(0.0, 1.0, 0.0, 1.0), self.color)


class TwoGoalsCollectEnv(MiniGridEnv):
    """
    6x6 empty room:
      - Red collectible goal at (1, 1)
      - Green collectible goal at (width-2, height-2)
    Reward:
      - +1 when stepping on each goal the first time
      - 0 otherwise
    Termination:
      - only when both goals are collected
    """

    def __init__(self, size=6, max_steps=None, agent_start_pos=None, agent_start_dir=0, **kwargs):
        assert size >= 4, "Need at least 4x4 to have interior cells."

        mission_space = MissionSpace(mission_func=lambda: "collect red and green goals")

        if max_steps is None:
            # Reasonable default for tiny grids
            max_steps = 4 * size * size

        self._size = size
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        # Track collection
        self.red_collected = False
        self.green_collected = False

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

        # Empty interior (no obstacles)

        # Place collectible goals in opposite corners (interior corners)
        self.red_pos = (1, 1)
        self.green_pos = (width - 2, height - 2)

        self.grid.set(*self.red_pos, CollectibleGoal("red"))
        self.grid.set(*self.green_pos, CollectibleGoal("green"))

        # Reset flags each episode
        self.red_collected = False
        self.green_collected = False

        # Agent start (center by default)
        if self.agent_start_pos is None:
            self.agent_pos = (width // 2, height // 2)
        else:
            self.agent_pos = self.agent_start_pos

        self.agent_dir = self.agent_start_dir

        self.mission = "collect red and green goals"

    def step(self, action):
        # Let MiniGrid handle movement, collisions, etc.
        obs, _, terminated, truncated, info = super().step(action)

        # We fully control the reward + termination logic
        reward = 0.0
        terminated = False

        # If agent is standing on a collectible goal, collect it once
        obj = self.grid.get(*self.agent_pos)
        if isinstance(obj, CollectibleGoal):
            if obj.color == "red" and not self.red_collected:
                self.red_collected = True
                reward += 1.0
                self.grid.set(*self.agent_pos, None)
            elif obj.color == "green" and not self.green_collected:
                self.green_collected = True
                reward += 1.0
                self.grid.set(*self.agent_pos, None)

        # End episode only when both collected
        if self.red_collected and self.green_collected:
            terminated = True

        return obs, reward, terminated, truncated, info



register(
    id="MiniGrid-EmptyTwoGoals-6x6-v0",
    entry_point=TwoGoalsCollectEnv,
    kwargs={"size": 6},
)