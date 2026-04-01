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

    Optional behavior:
        auto_drop=True:
            if the agent is already carrying an object and performs pickup on a
            new pickable object, the old carried object is automatically dropped
            onto the front cell and replaced by the new one.

        curriculum_steps (int | None):
            if set, every `curriculum_steps` total environment steps a new color
            is added to allowed_colors (drawn in order from COLOR_NAMES).
            Once all colors are unlocked further crossings are ignored.
            If None, allowed_colors never changes.

    Example:
        env = UnlockPickupLimitedColorEnv(
            allowed_colors=["red"],
            curriculum_steps=10_000,
            auto_drop=True,
        )
    """

    def __init__(
        self,
        allowed_colors=None,
        curriculum_steps: int | None = None,
        auto_drop: bool = False,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.all_colors = list(COLOR_NAMES)
        self.allowed_colors = list(allowed_colors or self.all_colors)
        self.curriculum_steps = curriculum_steps
        self.auto_drop = auto_drop
        
        self._total_steps = 0
        self._curriculum_counter = 0

        room_size = 6

        # Use the full color pool for the mission space so that newly unlocked
        # colors don't cause mission-validation errors.
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[self.all_colors],
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

    def _maybe_auto_drop_before_pickup(self):
        """
        If auto_drop is enabled and the agent is already carrying something,
        try to put the currently carried object onto the front cell so the
        normal MiniGrid pickup can grab the new object there.
        """
        if not self.auto_drop:
            return None

        if self.carrying is None:
            return None

        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        # Need something in front to pick up
        if fwd_cell is None:
            return None

        # Only do this if the front object is pickable
        if not fwd_cell.can_pickup():
            return None
        
        old_carry = self.carrying
        # self.grid.sets(fwd_pos[0], fwd_pos[1], old_carry)
        
        self.carrying = None
        return old_carry

    def _maybe_add_curriculum_color(self):
        """Add the next color from all_colors to allowed_colors if the curriculum threshold is crossed."""
        if self.curriculum_steps is None:
            return
        if self._curriculum_counter < self.curriculum_steps:
            return
        else:
            self._curriculum_counter = 0  # reset step count after crossing threshold
            
        for color in self.all_colors:
            if color not in self.allowed_colors:
                self.allowed_colors.append(color)
                print(f"[Curriculum] step {self._total_steps}: added color '{color}' → allowed_colors={self.allowed_colors}")
                return  # one color per threshold crossing

    def step(self, action):
        if action == self.actions.pickup:
            self._maybe_auto_drop_before_pickup()

        obs, reward, terminated, truncated, info = super().step(action)
        self._total_steps += 1
        self._curriculum_counter += 1
        self._maybe_add_curriculum_color()

        if action == self.actions.pickup:
            if self.carrying is not None and self.carrying == self.obj:
                reward = self._reward()
                terminated = True

        return obs, reward, terminated, truncated, info


register(
    id="MiniGrid-UnlockPickupLimitedColor-v0",
    entry_point=UnlockPickupLimitedColorEnv,
    kwargs={
        "allowed_colors": ["red", "green"], #, "blue", "yellow", "purple", "grey"],
        "auto_drop": False,
        "max_steps": 100,
        "curriculum_steps": None,
    },
)