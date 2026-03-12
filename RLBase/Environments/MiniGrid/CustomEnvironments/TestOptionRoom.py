import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
from typing import Any, Tuple
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Key, Ball, Goal, Door
from gymnasium import spaces
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX


class TestOptionRoom(MiniGridEnv):
    """
    Bigger empty room (outer walls only), no doors.

    Randomly placed each episode:
      - 2 keys:  red, green
      - 2 balls: red, green
      - 2 goals: red, green

    Agent is guaranteed to not start on any object (we sample agent first).
    """

    def __init__(
        self,
        size: int = 11,
        max_steps: int | None = None,
        agent_start_dir: int = 0,
        **kwargs,
    ):
        assert size >= 7, "Size should be >= 7 to comfortably place objects."
        self.size = size
        self.agent_start_dir = agent_start_dir

        if max_steps is None:
            max_steps = 4 * size * size

        mission_space = MissionSpace(mission_func=self._gen_mission)
        
        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            max_steps=max_steps,
            **kwargs,
        )
        self.observation_space = spaces.Dict({
            "image": self.observation_space["image"],          # keep default
            "direction": self.observation_space["direction"],  # keep default
            "mission": self.observation_space["mission"],      # keep default
            "carrying": spaces.Box(
                low=np.array([-1, -1], dtype=np.int16),
                high=np.array([len(OBJECT_TO_IDX) - 1, len(COLOR_TO_IDX) - 1], dtype=np.int16),
                dtype=np.int16,
                shape=(2,),
            ),
        })

    @staticmethod
    def _gen_mission():
        return "collect keys, balls, and reach goals"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Outer walls only
        self.grid.wall_rect(0, 0, width, height)

        def is_empty_cell(x, y):
            # must be empty in the grid AND not the agent position (when agent is set)
            if self.grid.get(x, y) is not None:
                return False
            if hasattr(self, "agent_pos") and self.agent_pos is not None and (x, y) == tuple(self.agent_pos):
                return False
            return True

        def sample_empty_cell(xmin, xmax, ymin, ymax, max_tries=50_000):
            for _ in range(max_tries):
                x = self._rand_int(xmin, xmax + 1)
                y = self._rand_int(ymin, ymax + 1)
                if is_empty_cell(x, y):
                    return (x, y)
            raise RuntimeError("Failed to sample an empty cell (too crowded?).")

        # 1) Sample agent first
        self.agent_pos = sample_empty_cell(1, width - 2, 1, height - 2)
        self.agent_dir = self.agent_start_dir

        # 2) Place objects (cannot overlap agent because sample_empty_cell checks that)
        self.grid.set(*sample_empty_cell(1, width - 2, 1, height - 2), Door("red"))
        self.grid.set(*sample_empty_cell(1, width - 2, 1, height - 2), Door("green"))
        
        self.grid.set(*sample_empty_cell(1, width - 2, 1, height - 2), Ball("red"))
        self.grid.set(*sample_empty_cell(1, width - 2, 1, height - 2), Ball("green"))

        self.grid.set(*sample_empty_cell(1, width - 2, 1, height - 2), Goal("red"))
        self.grid.set(*sample_empty_cell(1, width - 2, 1, height - 2), Goal("green"))
        
        self.grid.set(*sample_empty_cell(1, width - 2, 1, height - 2), Key("red"))
        self.grid.set(*sample_empty_cell(1, width - 2, 1, height - 2), Key("green"))

        self.mission = "collect red/green keys and balls, then reach red/green goals"

    def gen_obs(self):
        # get the normal MiniGrid obs first
        obs = super().gen_obs()

        # attach carrying (type_idx, color_idx) or (-1,-1)
        if self.carrying is not None:
            carry_type = OBJECT_TO_IDX[self.carrying.type]
            carry_color = COLOR_TO_IDX[self.carrying.color]
            obs["carrying"] = np.array([carry_type, carry_color], dtype=np.int16)
        else:
            obs["carrying"] = np.array([-1, -1], dtype=np.int16)

        return obs
    
    
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev_sig = self._world_signature()
        return obs, info

    def _world_signature(self) -> Tuple:
        """
        Signature of the environment *excluding* agent movement.
        Terminates episode if this changes.
        Tracks:
          - carrying (type,color) or (-1,-1)
          - counts of keys/balls remaining on grid
          - door states (pos, color, is_open, is_locked)
        """
        # carrying
        if self.carrying is None:
            carry = (-1, -1)
        else:
            carry = (OBJECT_TO_IDX[self.carrying.type], COLOR_TO_IDX[self.carrying.color])

        # grid scan
        n_keys = 0
        n_balls = 0
        doors = []

        for y in range(self.height):
            for x in range(self.width):
                obj = self.grid.get(x, y)
                if obj is None:
                    continue
                if isinstance(obj, Key):
                    n_keys += 1
                elif isinstance(obj, Ball):
                    n_balls += 1
                elif isinstance(obj, Door):
                    doors.append((x, y, obj.color, bool(obj.is_open), bool(obj.is_locked)))

        doors.sort()
        return (carry, n_keys, n_balls, tuple(doors))
    
    def step(self, action: int):
        # signature before step
        sig_before = self._prev_sig

        obs, reward, terminated, truncated, info = super().step(action)

        # signature after step
        sig_after = self._world_signature()
        self._prev_sig = sig_after

        # terminate if *any* environment change happened
        if sig_after != sig_before:
            terminated = True

        return obs, reward, terminated, truncated, info
    
register(
    id="MiniGrid-TestOptionRoom-v0",
    entry_point=TestOptionRoom,
    kwargs={"size": 11},
)



