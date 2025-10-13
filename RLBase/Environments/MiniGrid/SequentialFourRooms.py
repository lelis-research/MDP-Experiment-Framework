from gymnasium.envs.registration import register
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Ball, Key, Box, Goal
from typing import Any, Iterable, SupportsFloat, TypeVar
from gymnasium.core import ActType, ObsType
import numpy as np
import matplotlib.pyplot as plt

class SequentialFourRooms(MiniGridEnv):

    def __init__(self, agent_pos=None, goal_pos=None, max_steps=1000, **kwargs):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos

        self.size = 11
        mission_space = MissionSpace(mission_func=self._gen_mission)
        
        self.state = 0

        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            agent_view_size=self.size,
            see_through_walls=True,
            highlight=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Collect the 4 goals in order: red, yellow, blue, green."

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):
            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            # assuming random start direction
            self.agent_dir = self._rand_int(0, 4)
        else:
            self.place_agent()

        self.place_obj(Goal('red'), top=(1,1), size=(width//2-1, height//2-1))
        # top-right room
        self.place_obj(Goal('yellow'), top=(width//2+1,1), size=(width//2-1, height//2-1))
        # bottom-left room
        self.place_obj(Goal('blue'), top=(1,height//2+1), size=(width//2-1, height//2-1))
        # bottom-right room
        self.place_obj(Goal('green'), top=(width//2+1,height//2+1), size=(width//2-1, height//2-1))

    # def compute_shortest_path(self, start, goal):
    #     from collections import deque

    #     queue = deque([start])
    #     visited = {start: None}

    #     while queue:
    #         current = queue.popleft()

    #         if current == goal:
    #             break

    #         for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
    #             neighbor = (current[0] + direction[0], current[1] + direction[1])

    #             if (
    #                 0 <= neighbor[0] < self.size
    #                 and 0 <= neighbor[1] < self.size
    #                 and neighbor not in visited
    #                 and (self.grid.get(*neighbor) is None or self.grid.get(*neighbor).can_overlap())
    #             ):
    #                 visited[neighbor] = current
    #                 queue.append(neighbor)

    #     path = []
    #     step = goal
    #     while step is not None:
    #         path.append(step)
    #         step = visited[step]
    #     path.reverse()

    #     return path

    # def step_option(self, action: int) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
    #     color = None
    #     if action == 3:
    #         color = "yellow"
    #     elif action == 4:
    #         color = "green"
    #     elif action == 5:
    #         color = "red"
    #     elif action == 6:
    #         color = "blue"
        
    #     dst_pos = None
    #     for i in range(self.size):
    #         for j in range(self.size):
    #             cell = self.grid.get(i, j)
    #             if cell is not None and cell.type == "goal" and cell.color == color:
    #                 dst_pos = (i, j)
    #                 break
    #     if dst_pos is None:
    #         return None, None, None, None, None
    #     # print(f"Current position: {self.agent_pos}, Goal position: {dst_pos}")
    #     path = self.compute_shortest_path(self.agent_pos, dst_pos)
        
    #     if len(path) == 1:
    #         return None, None, None, None, None

    #     actions = []
    #     curr = self.agent_pos
    #     curr_dir = self.agent_dir
    #     for pos in path[1:]:
    #         pos_diff = (pos[0] - curr[0], pos[1] - curr[1])
    #         if pos_diff == (1, 0):
    #             desired_dir = 0
    #         elif pos_diff == (0, 1):
    #             desired_dir = 1
    #         elif pos_diff == (-1, 0):
    #             desired_dir = 2
    #         elif pos_diff == (0, -1):
    #             desired_dir = 3
            
    #         while curr_dir != desired_dir:
    #             actions.append(self.actions.right)
    #             curr_dir = (curr_dir + 1) % 4

    #         actions.append(self.actions.forward)
    #         curr = pos

    #     total_reward = 0
    #     for action in actions:
    #         obs, reward, terminated, truncated, info = self.step(action)
    #         total_reward += reward
    #         if terminated or truncated:
    #             break
    #     return obs, total_reward, terminated, truncated, info
    
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        
        # if action >= 3 and action <= 6:
        #     return self.step_option(action)

        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)
        # print(self.grid)
        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            # print(fwd_cell)
            if fwd_cell is not None and self.state == 0 and fwd_cell.type == "goal" and fwd_cell.color == 'red':
                self.state = 1
                self.grid.set(fwd_pos[0], fwd_pos[1], None)
                reward = self._reward()
            if fwd_cell is not None and self.state == 1 and fwd_cell.type == "goal" and fwd_cell.color == 'yellow':
                self.state = 2
                self.grid.set(fwd_pos[0], fwd_pos[1], None)
                reward = self._reward()
            if fwd_cell is not None and self.state == 2 and fwd_cell.type == "goal" and fwd_cell.color == 'blue':
                self.state = 3
                self.grid.set(fwd_pos[0], fwd_pos[1], None)
                reward = self._reward()
            if fwd_cell is not None and self.state == 3 and fwd_cell.type == "goal" and fwd_cell.color == 'green':
                terminated = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}

    def gen_obs(self):       
        grid = self.grid
        vis_mask = np.ones(shape=(self.width, self.height), dtype=bool)
        
        image = grid.encode(vis_mask)

        obs = {"image": image, "direction": self.agent_dir, "mission": self.mission}

        return obs

register(
    id='SequentialFourRooms-v0',
    entry_point=SequentialFourRooms
)