import gymnasium as gym
from gymnasium.envs.registration import register
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import  Goal, Wall
from minigrid.utils.rendering import fill_coords, point_in_rect


class RewardGoal(Goal):
    """A Goal that carries a custom reward and optional terminal flag."""
    def __init__(self, reward_value: float = 0.0, is_terminal: bool = False, color: str = "red"):
        super().__init__(color)
        self.reward_value = float(reward_value)
        self.is_terminal = bool(is_terminal)

class TwoGoalEmptyRoom(MiniGridEnv):
    """
    empty room, red goal in one interior corner and green goal in the opposite.
    +1 for each goal (once). Episode ends only after both are collected.
    Optionally spawns a third (blue) goal at a given step_count.
    """
    
    def __init__(
        self,
        size=6,
        max_steps=None,
        agent_start_pos=None,
        agent_start_dir=0,
        add_goal_step: int | None = None,      # <- NEW
        extra_goal_reward: float = 1.0,        # <- NEW
        extra_goal_color: str = "blue",        # <- NEW
        side_room_w: int = 3,                  # NEW: side room width
        side_room_h: int = 3,                  # NEW: side room height
        **kwargs
    ):
        mission_space = MissionSpace(mission_func=self._gen_mission)
        if max_steps is None:
            max_steps = 4 * size * size

        self.size = size
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        self.red_collected = False
        self.green_collected = False
        self.red_pos = None
        self.green_pos = None

        # NEW: delayed goal config/state
        self.add_goal_step = add_goal_step
        self.extra_goal_reward = float(extra_goal_reward)
        self.extra_goal_color = extra_goal_color
        
        # Blue goal state
        self.extra_goal_pos = None
        self.extra_goal_spawned = False
        self.extra_goal_collected = False
        
        # NEW: side-room geometry + "main room" dimensions
        self.main_w = size
        self.main_h = size
        self.side_room_w = side_room_w
        self.side_room_h = side_room_h

        self.goal_reach_counts = 0   # track total steps for delayed goal spawning
        super().__init__(
            mission_space=mission_space,
            width=size + side_room_w,
            height=size,
            max_steps=max_steps,
            **kwargs,
        )
        
    
    @staticmethod
    def _gen_mission():
        return "collect both goals"
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        # self.grid.wall_rect(0, 0, width, height)
        
        # NEW: fill the entire grid with walls first
        for x in range(width):
            for y in range(height):
                self.grid.set(x, y, Wall())

        # NEW: carve ONLY the main room area (size x size) as empty interior
        mw, mh = self.main_w, self.main_h
        for x in range(1, mw - 1):
            for y in range(1, mh - 1):
                self.grid.set(x, y, None)
                
        

        # Goals in the MAIN room corners (FIXED: don't use full width/height)
        self.red_pos = (1, 1)
        self.green_pos = (mw - 2, mh - 2)

        # Place goals (reward stored here, but we’ll control reward/termination in step)
        self.grid.set(*self.red_pos, RewardGoal(reward_value=0.0, is_terminal=False, color="red"))
        self.grid.set(*self.green_pos, RewardGoal(reward_value=0.0, is_terminal=False, color="green"))

        self.red_collected = False
        self.green_collected = False
        
        # Reset delayed goal state each episode
        self.extra_goal_pos = None
        self.extra_goal_spawned = False
        self.extra_goal_collected = False
        
        # NEW: curriculum — from this episode onward, side room exists from the start
        if (self.add_goal_step is not None) and (self.goal_reach_counts >= self.add_goal_step):
            self._spawn_side_room_with_goal()

        # Start agent
        if self.agent_start_pos is None:
            # self.agent_pos = (mw // 2, mh // 2)
            self.agent_pos = (1, mh - 2)
        else:
            self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        self.mission = "collect red and green goals"
    
    def _spawn_side_room_with_goal(self):
        main_w, main_h = self.main_w, self.main_h
        rw, rh = self.side_room_w, self.side_room_h

        # Side room lives in the extra area to the RIGHT of the main room.
        # Side area starts at x = main_w (since main room uses x in [0..main_w-1]).
        room_x0 = main_w
        room_y0 = main_h // 2 - 1
        room_x1 = room_x0 + rw - 1
        room_y1 = room_y0 + rh - 1

        # Build room boundary walls (the whole side area is already walls, but explicit is fine)
        for x in range(room_x0, room_x1 + 1):
            for y in range(room_y0, room_y1 + 1):
                self.grid.set(x, y, Wall())

        # Carve interior empties (requires rw,rh >= 3 for a non-empty interior)
        for x in range(room_x0 + 1, room_x1):
            for y in range(room_y0 + 1, room_y1):
                self.grid.set(x, y, None)

        # Doorway from main room into side room (open the main room's right wall)
        door_y = room_y0 + 1
        self.grid.set(main_w - 1, door_y, None)  # open the main room right boundary at that y
        self.grid.set(main_w, door_y, None)      # ensure adjacent side cell is empty

        # Place blue goal inside the side room interior
        goal_pos = (room_x0 + rw - 2, room_y0 + rh - 2)
        self.extra_goal_pos = goal_pos
        self.extra_goal_spawned = True
        self.extra_goal_collected = False
        self.grid.set(*goal_pos, RewardGoal(
            reward_value=self.extra_goal_reward,
            is_terminal=False,
            color=self.extra_goal_color
        ))
         
    def step(self, action):        
        # Run MiniGrid dynamics (movement, walls, etc.)
        obs, reward, terminated, truncated, info = super().step(action)

        # Ignore MiniGrid’s own goal termination and compute our own reward/termination
        terminated = False
        
        curr_obj = self.grid.get(*self.agent_pos)
        
        # Check whether we are standing on a goal tile (by position is simplest/robust)
        
        # Collect red
        if (self.agent_pos == self.red_pos) and (not self.red_collected):
            self.red_collected = True
            if isinstance(curr_obj, RewardGoal):
                reward += curr_obj.reward_value
            self.grid.set(*self.red_pos, None)   # remove so it can't be collected again

        # Collect green
        if (self.agent_pos == self.green_pos) and (not self.green_collected):
            self.green_collected = True
            if isinstance(curr_obj, RewardGoal):
                reward += curr_obj.reward_value
            self.grid.set(*self.green_pos, None)
        
        # Collect extra goal
        if self.extra_goal_spawned and (self.agent_pos == self.extra_goal_pos) and (not self.extra_goal_collected):
            self.extra_goal_collected = True
            if isinstance(curr_obj, RewardGoal):
                reward += curr_obj.reward_value
            self.grid.set(*self.extra_goal_pos, None)

        if self.red_collected and self.green_collected:
            self.goal_reach_counts += 1
            terminated = True

        return obs, reward, terminated, truncated, info


register(
    id="MiniGrid-EmptyTwoGoals-v0",
    entry_point=TwoGoalEmptyRoom,
    kwargs={"size": 8, "add_goal_step": None, "extra_goal_reward": 1.0, "side_room_w": 4, "side_room_h": 4},
)