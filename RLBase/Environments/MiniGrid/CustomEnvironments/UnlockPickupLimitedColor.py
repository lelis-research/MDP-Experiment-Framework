from __future__ import annotations

import random

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid
from gymnasium.envs.registration import register


class UnlockPickupLimitedColorEnv(RoomGrid):
    """
    Same as UnlockPickupEnv but allows restricting the colors
    used for objects in the environment.

    Example:
        env = UnlockPickupLimitedColorEnv(allowed_colors=["red", "green"])
    """

    def __init__(
        self,
        allowed_colors=None,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.allowed_colors = allowed_colors or COLOR_NAMES

        room_size = 6

        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[self.allowed_colors],
        )

        if max_steps is None:
            max_steps = 8 * room_size**2

        super().__init__(
            mission_space=mission_space,
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(color: str):
        return f"pick up the {color} box"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Choose a color from the allowed set
        color = random.choice(self.allowed_colors)

        # Add a box with the chosen color
        obj, _ = self.add_object(1, 0, kind="box", color=color)

        # Add a locked door
        door, _ = self.add_door(0, 0, 0, locked=True, color=color)

        # Add a key with the same color as the door
        self.add_object(0, 0, "key", color)

        self.place_agent(0, 0)

        self.obj = obj
        self.mission = f"pick up the {obj.color} {obj.type}"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.obj:
                reward = self._reward()
                terminated = True

        return obs, reward, terminated, truncated, info
    
    
register(
    id="MiniGrid-UnlockPickupLimitedColor-v0",
    entry_point=UnlockPickupLimitedColorEnv,
    kwargs={"allowed_colors": ['red', 'green'], "max_steps": 100},
)