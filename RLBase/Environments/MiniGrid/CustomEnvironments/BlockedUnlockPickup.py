from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid
from minigrid.core.world_object import Ball
from gymnasium.envs.registration import register
import numpy as np


class BlockedUnlockPickupReplaceCarryEnv(RoomGrid):
    """
    Same as BlockedUnlockPickupEnv, except pickup replaces the currently carried
    object instead of requiring a drop first.
    """

    def __init__(self, max_steps: int | None = None, **kwargs):
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES, ["box", "key"]],
        )

        room_size = 6
        if max_steps is None:
            max_steps = 16 * room_size**2

        super().__init__(
            mission_space=mission_space,
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(color: str, obj_type: str):
        return f"pick up the {color} {obj_type}"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Add a box to the room on the right
        obj, _ = self.add_object(1, 0, kind="box")

        # Make sure the two rooms are directly connected by a locked door
        door, pos = self.add_door(0, 0, 0, locked=True)

        # Block the door with a ball
        color = self._rand_color()
        self.grid.set(pos[0] - 1, pos[1], Ball(color))

        # Add a key to unlock the door
        self.add_object(0, 0, "key", door.color)

        self.place_agent(0, 0)

        self.obj = obj
        self.mission = f"pick up the {obj.color} {obj.type}"

    def _pickup_replace(self):
        """
        Custom pickup:
        - if front cell contains a pickup-able object, pick it up
        - replace current carrying instead of requiring drop first
        - previous carried object is discarded
        """
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        if fwd_cell is None:
            return

        if not fwd_cell.can_pickup():
            return

        # Remove object from grid and make it the currently carried object
        self.grid.set(*fwd_pos, None)
        fwd_cell.cur_pos = np.array([-1, -1])
        self.carrying = fwd_cell

    def step(self, action):
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        if action == self.actions.pickup:
            self._pickup_replace()
        else:
            obs, reward, terminated, truncated, info = super().step(action)

            # super().step already incremented step_count, so avoid double counting
            # by returning early for non-pickup actions
            return obs, reward, terminated, truncated, info

        if self.step_count >= self.max_steps:
            truncated = True

        if self.carrying is not None and self.carrying == self.obj:
            reward = self._reward()
            terminated = True

        obs = self.gen_obs()
        info = {}

        return obs, reward, terminated, truncated, info
    
    
register(
    id="MiniGrid-BlockedUnlockPickupReplaceCarry-v0",
    entry_point=BlockedUnlockPickupReplaceCarryEnv,
    kwargs={"max_steps": 100},
)