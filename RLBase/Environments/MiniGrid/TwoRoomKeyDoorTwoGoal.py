from __future__ import annotations
from typing import Any, Optional
from gymnasium.envs.registration import register
import numpy as np
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Key, Goal, Wall

# ---------- Reward-carrying goal wrapper ----------
class RewardGoal(Goal):
    """A Goal that carries a custom reward value."""
    def __init__(self, reward_value: float = 1.0, is_terminal: bool = False, color: str = "red"):
        super().__init__(color)
        self.reward_value = reward_value
        self.is_terminal = is_terminal

    


class TwoRoomKeyDoorTwoGoalEnv(MiniGridEnv):
    """
    Two-room env:
      - Room 1 has a visible goal worth +1 and a key.
      - Locked door to Room 2 (requires key).
      - Room 2 has a goal worth +10.
      - Stepping onto any goal terminates the episode with that reward.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        *,
        grid_size: int = 11,
        key_color: str = "red",
        render_mode: Optional[str] = None,
        max_steps: int = 200,
        see_through_walls: bool = True,
        agent_view_size: int = 7,
        **kwargs: Any,
    ):
        self.key_color = key_color

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            grid_size=grid_size,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            render_mode=render_mode,
            agent_view_size=agent_view_size,
            **kwargs,
        )

        # Keep references for quick checks
        self.goal1: RewardGoal | None = None
        self.goal2: RewardGoal | None = None
        self.door: Door | None = None

    @staticmethod
    def _gen_mission():
        return "Reach a goal: +1 in room one, or unlock the door with the key and reach the +10 goal."

    # ---------- Grid generation ----------
    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)
        # Outer walls
        self.grid.wall_rect(0, 0, width, height)

        # Split into two rooms with a vertical wall in the middle
        mid_x = width // 2
        for y in range(1, height - 1):
            self.grid.set(mid_x, y, Wall())

        # Make a doorway in the middle wall (locked door)
        door_y = height // 2
        self.door = Door(self.key_color, is_open=False, is_locked=True)
        self.grid.set(mid_x, door_y, self.door)

        # Place key in room 1 (left side)
        key_x, key_y = 2, height // 2
        self.grid.set(key_x, key_y, Key(self.key_color))

        # Place Goal A (+1) in room 1 (top-left corner area)
        g1x, g1y = 2, 2
        self.goal1 = RewardGoal(reward_value=1.0, is_terminal=False, color="red")
        self.grid.set(g1x, g1y, self.goal1)

        # Place Goal B (+10) in room 2 (far right)
        g2x, g2y = width - 3, height // 2
        self.goal2 = RewardGoal(reward_value=10.0, is_terminal=True, color="green")
        self.grid.set(g2x, g2y, self.goal2)

        # Agent start in room 1
        self.agent_pos = (1, height // 2)
        self.agent_dir = 0  # facing right

    # ---------- Custom step to assign goal-specific rewards ----------
    def step(self, action):
        # move the agent one step
        obs, reward, terminated, truncated, info = super().step(action)
        

        # Get the contents of the agent cell
        curr_cell = self.grid.get(*self.agent_pos)
        
        if action == self.actions.forward:
            if curr_cell is not None and curr_cell.type == "goal":
                terminated = curr_cell.is_terminal
                reward += curr_cell.reward_value
                self.grid.set(self.agent_pos[0], self.agent_pos[1], None)
                
        
        return obs, reward, terminated, truncated, info
            
    

register(
    id='TwoRoomKeyDoorTwoGoalEnv-v0',
    entry_point=TwoRoomKeyDoorTwoGoalEnv
)