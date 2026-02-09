import gymnasium as gym
from gymnasium.envs.registration import register

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Key, Ball, Goal, Door


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


register(
    id="MiniGrid-TestOptionRoom-v0",
    entry_point=TestOptionRoom,
    kwargs={"size": 11},
)



