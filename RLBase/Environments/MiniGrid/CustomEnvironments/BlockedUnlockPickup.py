from __future__ import annotations

import numpy as np
from gymnasium import spaces

from minigrid.core.constants import COLOR_NAMES, OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid
from minigrid.core.world_object import Ball
from gymnasium.envs.registration import register


class BlockedUnlockPickupEnv(RoomGrid):

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

        # Extend observation space
        self.observation_space = spaces.Dict(
            {
                **self.observation_space.spaces,
                "carrying": spaces.Box(
                    low=np.array([-1, -1], dtype=np.int16),
                    high=np.array(
                        [len(OBJECT_TO_IDX) - 1, len(COLOR_TO_IDX) - 1],
                        dtype=np.int16,
                    ),
                    dtype=np.int16,
                    shape=(2,),
                ),
            }
        )

    @staticmethod
    def _gen_mission(color: str, obj_type: str):
        return f"pick up the {color} {obj_type}"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        obj, _ = self.add_object(1, 0, kind="box")

        door, pos = self.add_door(0, 0, 0, locked=True)

        color = self._rand_color()
        self.grid.set(pos[0] - 1, pos[1], Ball(color))

        self.add_object(0, 0, "key", door.color)

        self.place_agent(0, 0)

        self.obj = obj
        self.mission = f"pick up the {obj.color} {obj.type}"

    # -------------------------------
    # Carrying encoding
    # -------------------------------

    def _get_carrying_encoding(self):

        if self.carrying is None:
            return np.array([-1, -1], dtype=np.int16)

        obj = self.carrying

        return np.array(
            [
                OBJECT_TO_IDX[obj.type],
                COLOR_TO_IDX[obj.color],
            ],
            dtype=np.int16,
        )

    def _augment_obs(self, obs):
        obs["carrying"] = self._get_carrying_encoding()
        return obs

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        obs = self._augment_obs(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        obs = self._augment_obs(obs)

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.obj:
                reward = self._reward()
                terminated = True

        return obs, reward, terminated, truncated, info
    
    
register(id="MiniGrid-BlockedUnlockPickupEnvCarry-v0", entry_point=BlockedUnlockPickupEnv)