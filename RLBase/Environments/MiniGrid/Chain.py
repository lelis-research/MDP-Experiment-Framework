from copy import deepcopy

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Lava, Door
from gymnasium.envs.registration import register

class ChainEnv(MiniGridEnv):
    def __init__(
        self,
        chain_length=5,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
    
        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * chain_length**2

        self.chain_length = chain_length
        super().__init__(
            mission_space=mission_space,
            width=chain_length,
            height=3,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return  "Traverse the chain from left to right."
    
    def _gen_grid(self, width, height):
        # Create an empty grid and surround it with walls.
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Place the goal object at the right-most cell of the chain.
        self.put_obj(Goal(), self.chain_length - 2, 1)

        # Set the agent's starting position.
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Traverse the chain from left to right."
    

class ChainEnv2(MiniGridEnv):
    def __init__(
        self,
        chain_length=5,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
    
        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * chain_length**2

        self.chain_length = chain_length
        super().__init__(
            mission_space=mission_space,
            width=chain_length,
            height=3,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return  "Traverse the chain from left to right."
    
    def _gen_grid(self, width, height):
        # Create an empty grid and surround it with walls.
        self.grid = Grid(width, height)

        # Instead of building walls around the grid, fill the border with lava.
        for x in range(width):
            # Top border (y = 0)
            self.grid.set(x, 0, Door("red", is_locked=True))
            # Bottom border (y = height - 1)
            self.grid.set(x, height - 1, Door("red", is_locked=True))
        for y in range(height):
            # Left border (x = 0)
            self.grid.set(0, y, Door("red", is_locked=True))
            # Right border (x = width - 1)
            self.grid.set(width - 1, y, Door("red", is_locked=True))

        # Place the goal object at the right-most cell of the chain.
        self.put_obj(Goal(), self.chain_length - 2, 1)

        # Set the agent's starting position.
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Traverse the chain from left to right."

class ChainEnvLava(MiniGridEnv):
    def __init__(
        self,
        chain_length=5,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * chain_length**2

        self.chain_length = chain_length
        super().__init__(
            mission_space=mission_space,
            width=chain_length,
            height=3,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Traverse the chain from left to right."

    def _gen_grid(self, width, height):
        # Create an empty grid.
        self.grid = Grid(width, height)

        # Instead of building walls around the grid, fill the border with lava.
        for x in range(width):
            # Top border (y = 0)
            self.grid.set(x, 0, Lava())
            # Bottom border (y = height - 1)
            self.grid.set(x, height - 1, Lava())
        for y in range(height):
            # Left border (x = 0)
            self.grid.set(0, y, Lava())
            # Right border (x = width - 1)
            self.grid.set(width - 1, y, Lava())

        # Place the goal object at the right-most cell of the chain.
        self.put_obj(Goal(), self.chain_length - 2, 1)

        # Set the agent's starting position.
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Traverse the chain from left to right."

# Register the environment so it can be created via gym.make.
register(
    id='MiniGrid-ChainEnv-v0',
    entry_point='RLBase.Environments.MiniGrid.Chain:ChainEnv', 
    # kwargs={"chain_length": 5},
)

register(
    id='MiniGrid-ChainEnv-v1',
    entry_point='RLBase.Environments.MiniGrid.Chain:ChainEnv2', 
    # kwargs={"chain_length": 5},
)
register(
    id='MiniGrid-ChainEnvLava-v0',
    entry_point='RLBase.Environments.MiniGrid.Chain:ChainEnvLava',
)