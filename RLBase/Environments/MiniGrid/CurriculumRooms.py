# big_curriculum_env.py
# Massive MiniGrid world (30 snake-connected rooms) with RewardGoal-based sub-goals.
# - Per-room solvability: lava placement is validated by BFS (never blocks paths).
# - Red sub-goals give reward and vanish (non-terminal); final green goal terminates.
# - Boxes used as receptacles are non-toggleable DepositBox (won't vanish).
# - Two combo-door behaviors:
#     * StrictDoor         : order doesn't matter (key before/after balls is OK).
#     * OrderedStrictDoor  : order matters (balls first, then key toggle).
#
# Observations/Renders:
#   - The agent observation is the FULL current room **including** its 1-thick
#     separator walls and the exit door (so 9x9 if room inner is 7x7).
#   - The full-render highlight and the POV render also show that same full room.

from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Set, Deque
from collections import deque
from gymnasium import spaces
import gymnasium as gym
from gymnasium.envs.registration import register
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, COLORS

# MiniGrid
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import (
    WorldObj, Key, Door, Ball, Box, Lava, Wall, Goal
)

# ---------- knobs ----------
ROOM_INNER_H = 7
ROOM_INNER_W = 7
WALL_T = 1 # DO NOT CHANGE THIS
ROOMS_PER_ROW = 6         # 6 x 5 = 30 rooms
ROOM_ROWS = 5
MAX_STEPS = 10_000
FINAL_GOAL_REWARD = 100.0
LAVA_PLACE_MAX_TRIES = 200
OBJ_PLACE_MAX_TRIES = 200

# ---------- Reward-carrying Goal ----------
class RewardGoal(Goal):
    """A Goal that carries a custom reward and optional terminal flag."""
    def __init__(self, reward_value: float = 1.0, is_terminal: bool = False, color: str = "red"):
        super().__init__(color)
        self.reward_value = float(reward_value)
        self.is_terminal = bool(is_terminal)

# ---------- Receptacle that doesn't vanish ----------
class DepositBox(Box):
    """A box that cannot be toggled open (won't vanish)."""
    def toggle(self, env, pos):
        return False

# ---------- Doors for composite requirements ----------
class StrictDoor(Door):
    """
    For composite rooms where order DOESN'T matter.
    - Ignores all toggles while locked.
    - Env unlocks it once ALL its room conditions are satisfied,
      regardless of whether the key was used before or after.
    """
    def __init__(self, color="red", is_locked=True):
        super().__init__(color=color, is_locked=is_locked)
        self.gate_cond_met = False  # balls side (or any non-key side)

    def toggle(self, env, pos):
        if self.is_locked:
            return False
        return super().toggle(env, pos)

class OrderedStrictDoor(StrictDoor):
    """
    For composite rooms where order DOES matter (balls first, THEN key toggle).
    - While locked, any toggle is ignored (same as StrictDoor).
    - The env will only accept/record the key attempt AFTER gate_cond_met=True.
    """
    pass

# -----------------------------------------------------------------------------
# ROOM SPEC SCHEMA (per entry in rooms_spec) – abbreviated here
# -----------------------------------------------------------------------------
#   requirements:
#     - {"open_exit":"none"} | {"open_exit":"key_match"}
#     - {"open_exit":"ball_in_box","pairs":[[ballColor, boxColor],...]}  (any)
#     - {"open_exit":"multi_all","pairs":[[c,c], ...]}                   (all)
#     - {"open_exit":"multi_combo","order":"free"|"after_balls","conditions":[
#           {"type":"key_match","color":"red"},
#           {"type":"ball_in_box"|"multi_all","pairs":[["green","green"],["blue","blue"]]}
#       ]}
#     - Optional "final_goal": True
# -----------------------------------------------------------------------------

rooms_spec: List[Dict[str, Any]] = [
    # ----------------------------- Phase A (1–5): Navigate & subgoals ------------------------------
    {"id": 1, "subgoal": True,
     "exit_door": {"locked": True},
     "requirements": {"open_exit": "none", "final_goal": True}},
    

    # {"id": 1, "subgoal": True,
    #  "exit_door": {"locked": False},
    #  "requirements": {"open_exit": "none"}},
    
    {"id": 2, "subgoal": True,
     "corridors": {"pattern": "L", "material": "door"},
     "exit_door": {"locked": False},
     "requirements": {"open_exit": "none"}},

    {"id": 3, "subgoal": True,
     "corridors": {"pattern": "T", "material": "wall"},
     "exit_door": {"locked": False},
     "requirements": {"open_exit": "none"}},

    {"id": 4, "subgoal": True,
     "corridors": {"pattern": "L", "material": "wall"},
     "lava": {"count": 2},
     "exit_door": {"locked": False},
     "requirements": {"open_exit": "none"}},

    {"id": 5, "subgoal": True,
     "corridors": {"pattern": "T", "material": "door"},
     "lava": {"count": 3},
     "exit_door": {"locked": False},
     "requirements": {"open_exit": "none"}},

    # ----------------------------- Phase B (6–10): Keys & locked exits ------------------------------
    {"id": 6, "subgoal": True,
     "keys": [{"color": "red"}],
     "exit_door": {"locked": True, "color": "red"},
     "requirements": {"open_exit": "key_match"}},

    {"id": 7, "subgoal": True,
     "corridors": {"pattern": "L", "material": "wall"},
     "keys": [{"color": "blue"}],
     "exit_door": {"locked": True, "color": "blue"},
     "requirements": {"open_exit": "key_match"}},

    {"id": 8, "subgoal": True,
     "lava": {"count": 3},
     "keys": [{"color": "green"}],
     "exit_door": {"locked": True, "color": "green"},
     "requirements": {"open_exit": "key_match"}},

    {"id": 9, "subgoal": True,
     "corridors": {"pattern": "T", "material": "wall"},
     "lava": {"count": 4},
     "keys": [{"color": "yellow"}],
     "exit_door": {"locked": True, "color": "yellow"},
     "requirements": {"open_exit": "key_match"}},

    {"id": 10, "subgoal": True,
     "corridors": {"pattern": "L", "material": "door"},
     "lava": {"count": 5},
     "keys": [{"color": "purple"}],
     "exit_door": {"locked": True, "color": "purple"},
     "requirements": {"open_exit": "key_match"}},

    # ----------------------------- Phase C (11–15): Subgoals + navigation hazards ------------------
    {"id": 11, "subgoal": True,
     "lava": {"count": 5},
     "exit_door": {"locked": False},
     "requirements": {"open_exit": "none"}},

    {"id": 12, "subgoal": True,
     "corridors": {"pattern": "T", "material": "wall"},
     "lava": {"count": 6},
     "exit_door": {"locked": False},
     "requirements": {"open_exit": "none"}},

    {"id": 13, "subgoal": True,
     "corridors": {"pattern": "L", "material": "door"},
     "lava": {"count": 7},
     "exit_door": {"locked": False},
     "requirements": {"open_exit": "none"}},

    {"id": 14, "subgoal": True,
     "corridors": {"pattern": "T", "material": "wall"},
     "lava": {"count": 8},
     "exit_door": {"locked": False},
     "requirements": {"open_exit": "none"}},

    {"id": 15, "subgoal": True,
     "corridors": {"pattern": "L", "material": "wall"},
     "lava": {"count": 9},
     "exit_door": {"locked": False},
     "requirements": {"open_exit": "none"}},

    # ----------------------------- Phase D (16–20): Keys amid lava & corridors ----------------------
    {"id": 16, "subgoal": True,
     "lava": {"count": 6},
     "keys": [{"color": "red"}],
     "exit_door": {"locked": True, "color": "red"},
     "requirements": {"open_exit": "key_match"}},

    {"id": 17, "subgoal": True,
     "corridors": {"pattern": "L", "material": "wall"},
     "lava": {"count": 7},
     "keys": [{"color": "blue"}],
     "exit_door": {"locked": True, "color": "blue"},
     "requirements": {"open_exit": "key_match"}},

    {"id": 18, "subgoal": True,
     "corridors": {"pattern": "T", "material": "wall"},
     "lava": {"count": 8},
     "keys": [{"color": "green"}],
     "exit_door": {"locked": True, "color": "green"},
     "requirements": {"open_exit": "key_match"}},

    {"id": 19, "subgoal": True,
     "corridors": {"pattern": "L", "material": "door"},
     "lava": {"count": 9},
     "keys": [{"color": "yellow"}],
     "exit_door": {"locked": True, "color": "yellow"},
     "requirements": {"open_exit": "key_match"}},

    {"id": 20, "subgoal": True,
     "corridors": {"pattern": "T", "material": "wall"},
     "lava": {"count": 10},
     "keys": [{"color": "purple"}],
     "exit_door": {"locked": True, "color": "purple"},
     "requirements": {"open_exit": "key_match"}},

    # ----------------------------- Phase E (21–23): Boxes w/ hidden keys ---------------------------
    {"id": 21, "subgoal": True,
     "corridors": {"pattern": "L", "material": "wall"},
     "lava": {"count": 8},
     "boxes": [{"color": "red", "contains": "key"},
               {"color": "green", "contains": "none"}],
     "exit_door": {"locked": True, "color": "red"},
     "requirements": {"open_exit": "key_match"}},

    {"id": 22, "subgoal": True,
     "corridors": {"pattern": "T", "material": "door"},
     "lava": {"count": 9},
     "boxes": [{"color": "blue", "contains": "key"},
               {"color": "yellow", "contains": "none"},
               {"color": "purple", "contains": "none"}],
     "exit_door": {"locked": True, "color": "blue"},
     "requirements": {"open_exit": "key_match"}},

    {"id": 23, "subgoal": True,
     "corridors": {"pattern": "L", "material": "wall"},
     "lava": {"count": 10},
     "boxes": [{"color": "green", "contains": "key"},
               {"color": "red", "contains": "none"},
               {"color": "blue", "contains": "none"}],
     "exit_door": {"locked": True, "color": "green"},
     "requirements": {"open_exit": "key_match"}},

    # ----------------------------- Phase F (24–26): Learn ball→box deposits ------------------------
    {"id": 24, "subgoal": True,
     "balls": [{"color": "red"}],
     "boxes": [{"color": "red", "contains": "none"}],
     "exit_door": {"locked": True, "color": "red"},
     "requirements": {"open_exit": "ball_in_box", "pairs": [["red", "red"]]}},

    {"id": 25, "subgoal": True,
     "corridors": {"pattern": "T", "material": "wall"},
     "lava": {"count": 7},
     "balls": [{"color": "red"}, {"color": "green"}],
     "boxes": [{"color": "red", "contains": "none"},
               {"color": "green", "contains": "none"}],
     "exit_door": {"locked": True},
     "requirements": {"open_exit": "multi_all",
                      "pairs": [["red","red"],["green","green"]]}},

    {"id": 26, "subgoal": True,
     "corridors": {"pattern": "L", "material": "door"},
     "lava": {"count": 8},
     "balls": [{"color": "red"}, {"color": "blue"}],
     "boxes": [{"color": "red", "contains": "none"},
               {"color": "blue", "contains": "none"}],
     "exit_door": {"locked": True},
     "requirements": {"open_exit": "multi_all",
                      "pairs": [["red","red"],["blue","blue"]]}},

    # ----------------------------- Phase G (27–29): Key + ball combos --------------------------------
    {"id": 27, "subgoal": True,
     "corridors": {"pattern": "T", "material": "wall"},
     "lava": {"count": 8},
     "keys": [{"color": "yellow"}],
     "balls": [{"color": "green"}],
     "boxes": [{"color": "green", "contains": "none"}],
     "exit_door": {"locked": True, "color": "yellow"},
     "requirements": {"open_exit": "multi_combo", "order": "free",
                      "conditions": [
                         {"type": "key_match", "color": "yellow"},
                         {"type": "ball_in_box", "pairs": [["green","green"]]}
                      ]}},

    {"id": 28, "subgoal": True,
     "corridors": {"pattern": "L", "material": "wall"},
     "lava": {"count": 9},
     "keys": [{"color": "blue"}],
     "balls": [{"color": "red"}, {"color": "green"}],
     "boxes": [{"color": "red", "contains": "none"},
               {"color": "green", "contains": "none"}],
     "exit_door": {"locked": True, "color": "blue"},
     "requirements": {"open_exit": "multi_combo", "order": "free",
                      "conditions": [
                         {"type": "key_match", "color": "blue"},
                         {"type": "multi_all", "pairs": [["red","red"],["green","green"]]}
                      ]}},

    # Ordered: balls first, then the key toggle
    {"id": 29, "subgoal": True,
     "corridors": {"pattern": "T", "material": "wall"},
     "lava": {"count": 10},
     "keys": [{"color": "red"}],
     "balls": [{"color": "green"}, {"color": "blue"}, {"color": "yellow"}],
     "boxes": [{"color": "green", "contains": "none"},
               {"color": "blue", "contains": "none"},
               {"color": "yellow", "contains": "none"}],
     "exit_door": {"locked": True, "color": "red"},
     "requirements": {"open_exit": "multi_combo", "order": "after_balls",
                      "conditions": [
                         {"type": "key_match", "color": "red"},
                         {"type": "multi_all", "pairs": [["green","green"],["blue","blue"]]}
                      ]}},

    # ----------------------------- Phase H (30): Final multi-all + terminal goal ---------------------
    {"id": 30, "subgoal": True,
     "corridors": {"pattern": "L", "material": "wall"},
     "lava": {"count": 12},
     "keys": [{"color": "purple"}, {"color": "blue"}],  # distractors
     "balls": [{"color": "red"}, {"color": "green"},
               {"color": "blue"}, {"color": "yellow"}, {"color": "purple"}],
     "boxes": [{"color": "red", "contains": "none"},
               {"color": "green", "contains": "none"},
               {"color": "blue", "contains": "none"},
               {"color": "yellow", "contains": "none"},
               {"color": "purple", "contains": "none"}],
     "exit_door": {"locked": True},
     "requirements": {"open_exit": "multi_all",
                      "pairs": [["red","red"],["green","green"],["blue","blue"],["yellow","yellow"],["purple","purple"]],
                      "final_goal": True}},
]

# ---------- helpers ----------
def room_index_to_rc(i: int) -> Tuple[int, int]:
    r = i // ROOMS_PER_ROW
    c = i % ROOMS_PER_ROW
    if r % 2 == 1:
        c = ROOMS_PER_ROW - 1 - c  # snake on odd rows
    return r, c

def _required_box_colors(requirements: dict) -> Set[str]:
    """Return set of box colors that must accept balls in this room."""
    colors: Set[str] = set()
    mode = requirements.get("open_exit", "none")
    if mode in ("ball_in_box", "multi_all"):
        for c_ball, c_box in requirements.get("pairs", []):
            colors.add(c_box)
    elif mode == "multi_combo":
        for cond in requirements.get("conditions", []):
            if cond.get("type") in ("ball_in_box", "multi_all"):
                for c_ball, c_box in cond.get("pairs", []):
                    colors.add(c_box)
    return colors

# ---------- big env ----------
class BigCurriculumEnv(MiniGridEnv):
    """
    30 rooms with solid separators and one door between consecutive rooms.
    Lava placement and object placement are reachability-safe from the entrance.

    OBSERVATION OVERRIDE:
      - obs["image"] is the FULL current room including separator walls & door.
      - render highlight and POV also show the same 9x9 window.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, *, max_steps: int = MAX_STEPS, **kwargs: Any):
        total_w = ROOMS_PER_ROW * ROOM_INNER_W + (ROOMS_PER_ROW + 1) * WALL_T
        total_h = ROOM_ROWS * ROOM_INNER_H + (ROOM_ROWS + 1) * WALL_T
        mission_space = MissionSpace(mission_func=self._gen_mission)
            
        super().__init__(
            mission_space=mission_space,
            width=total_w,
            height=total_h,
            max_steps=max_steps,
            **kwargs
        )
 
        # Observation is the full room including walls: (7+2, 7+2, 3) = (9,9,3)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255,
                shape=(ROOM_INNER_W + 2*WALL_T, ROOM_INNER_H + 2*WALL_T, 3),
                dtype=np.uint8
            ),
            "direction": spaces.Discrete(4),
            "mission": mission_space,
            # (type_idx, color_idx) or (-1,-1)
            "carrying": spaces.Box(low=-1, high=255, shape=(2,), dtype=np.int16),
        })
        

        self.placed_pairs_by_room: Dict[int, Set[str]] = {}
        self.exit_doors: Dict[int, Tuple[int, int]] = {}
        self.exit_approach: Dict[int, Tuple[int, int]] = {}
        self.entrance_approach: Dict[int, Tuple[int, int]] = {}
        self.final_goal_pos: Optional[Tuple[int, int]] = None
        self.key_used_ok: Dict[int, bool] = {}

    @staticmethod
    def _gen_mission():
        return "Traverse rooms, claim red subgoals, finish at green goal."

    # ----- geometry helpers for doors/approaches -----
    def _door_and_approach_for(self, i: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        cur_r, cur_c = room_index_to_rc(i)
        next_idx = i + 1
        if next_idx >= len(rooms_spec):
            return ((-1, -1), (-1, -1))
        nxt_r, nxt_c = room_index_to_rc(next_idx)
        if nxt_r == cur_r:
            door_y = cur_r * (ROOM_INNER_H + WALL_T) + WALL_T + ROOM_INNER_H // 2
            if nxt_c > cur_c:
                door_x = (cur_c + 1) * (ROOM_INNER_W + WALL_T); approach = (door_x - 1, door_y)
            else:
                door_x = cur_c * (ROOM_INNER_W + WALL_T);       approach = (door_x + 1, door_y)
        else:
            door_x = cur_c * (ROOM_INNER_W + WALL_T) + WALL_T + ROOM_INNER_W // 2
            if nxt_r > cur_r:
                door_y = (cur_r + 1) * (ROOM_INNER_H + WALL_T); approach = (door_x, door_y - 1)
            else:
                door_y = cur_r * (ROOM_INNER_H + WALL_T);       approach = (door_x, door_y + 1)
        return (door_x, door_y), approach

    def _entrance_approach_for(self, i: int) -> Tuple[int, int]:
        if i == 0:
            return (-1, -1)
        prev_r, prev_c = room_index_to_rc(i - 1)
        cur_r, cur_c = room_index_to_rc(i)
        if prev_r == cur_r:
            door_y = cur_r * (ROOM_INNER_H + WALL_T) + WALL_T + ROOM_INNER_H // 2
            if cur_c > prev_c:
                door_x = cur_c * (ROOM_INNER_W + WALL_T);         approach = (door_x + 1, door_y)
            else:
                door_x = (cur_c + 1) * (ROOM_INNER_W + WALL_T);   approach = (door_x - 1, door_y)
        else:
            door_x = cur_c * (ROOM_INNER_W + WALL_T) + WALL_T + ROOM_INNER_W // 2
            if cur_r > prev_r:
                door_y = cur_r * (ROOM_INNER_H + WALL_T);         approach = (door_x, door_y + 1)
            else:
                door_y = (cur_r + 1) * (ROOM_INNER_H + WALL_T);   approach = (door_x, door_y - 1)
        return approach

    @staticmethod
    def _deeper_cell(door: Tuple[int, int], approach: Tuple[int, int]) -> Tuple[int, int]:
        if door == (-1, -1) or approach == (-1, -1):
            return (-1, -1)
        vx = approach[0] - door[0]; vy = approach[1] - door[1]
        vx = 0 if vx == 0 else (1 if vx > 0 else -1)
        vy = 0 if vy == 0 else (1 if vy > 0 else -1)
        return (approach[0] + vx, approach[1] + vy)

    # ----- observation window helpers (room with walls) -----
    def _current_room_bbox(self) -> Tuple[int, int, int, int]:
        """
        Return (x0, y0, w, h) of the current room INCLUDING its surrounding
        separator walls (thickness=WALL_T). This captures walls + the door.
        """
        rid = self._room_id_from_pos(tuple(self.agent_pos))
        if rid is None:
            return 0, 0, ROOM_INNER_W + 2*WALL_T, ROOM_INNER_H + 2*WALL_T
        ry, rx = room_index_to_rc(rid - 1)
        top_x = rx * (ROOM_INNER_W + WALL_T) + WALL_T
        top_y = ry * (ROOM_INNER_H + WALL_T) + WALL_T
        x0 = top_x - WALL_T
        y0 = top_y - WALL_T
        w  = ROOM_INNER_W + 2*WALL_T
        h  = ROOM_INNER_H + 2*WALL_T
        return x0, y0, w, h

    # ----- reachability helpers -----
    @staticmethod
    def _is_passable(cell: Optional[WorldObj]) -> bool:
        if cell is None:
            return True
        if isinstance(cell, (Key, Ball, Box, RewardGoal, DepositBox)):
            return True
        if isinstance(cell, Door):
            return not cell.is_locked
        return False

    def _bfs_room(self, start: Tuple[int, int], top_x: int, top_y: int) -> Set[Tuple[int, int]]:
        sx, sy = start
        if not (top_x <= sx < top_x + ROOM_INNER_W and top_y <= sy < top_y + ROOM_INNER_H):
            return set()
        if not self._is_passable(self.grid.get(sx, sy)):
            return set()

        q: Deque[Tuple[int, int]] = deque([(sx, sy)])
        seen: Set[Tuple[int, int]] = {(sx, sy)}
        while q:
            x, y = q.popleft()
            for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                if not (top_x <= nx < top_x + ROOM_INNER_W and top_y <= ny < top_y + ROOM_INNER_H):
                    continue
                if (nx, ny) in seen:
                    continue
                if self._is_passable(self.grid.get(nx, ny)):
                    seen.add((nx, ny))
                    q.append((nx, ny))
        return seen

    # ----- grid build -----
    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # separator lattice
        for col in range(ROOMS_PER_ROW + 1):
            x = col * (ROOM_INNER_W + WALL_T)
            if 0 < x < width - 1:
                for y in range(1, height - 1):
                    self.grid.set(x, y, Wall())
        for row in range(ROOM_ROWS + 1):
            y = row * (ROOM_INNER_H + WALL_T)
            if 0 < y < height - 1:
                for x in range(1, width - 1):
                    self.grid.set(x, y, Wall())

        # Precompute door/approach lanes
        self.exit_doors.clear(); self.exit_approach.clear(); self.entrance_approach.clear()
        for i, spec in enumerate(rooms_spec):
            room_id = spec["id"]
            door_pos, exit_appr = self._door_and_approach_for(i)
            if door_pos != (-1, -1):
                self.exit_doors[room_id] = door_pos
                self.exit_approach[room_id] = exit_appr
            ent_appr = self._entrance_approach_for(i)
            if ent_appr != (-1, -1):
                self.entrance_approach[room_id] = ent_appr

        # Build rooms
        for i, spec in enumerate(rooms_spec):
            room_id = spec["id"]
            ry, rx = room_index_to_rc(i)
            top_x = rx * (ROOM_INNER_W + WALL_T) + WALL_T
            top_y = ry * (ROOM_INNER_H + WALL_T) + WALL_T

            # carve interior
            for y in range(ROOM_INNER_H):
                for x in range(ROOM_INNER_W):
                    self.grid.set(top_x + x, top_y + y, None)

            # reserved door-lane cells
            reserved: Set[Tuple[int, int]] = set()
            if room_id in self.exit_approach:
                ea = self.exit_approach[room_id]; door = self.exit_doors[room_id]
                reserved.add(ea)
                deeper = self._deeper_cell(door, ea)
                if deeper != (-1, -1):
                    reserved.add(deeper)
            if room_id in self.entrance_approach:
                ia = self.entrance_approach[room_id]
                cx = top_x + ROOM_INNER_W // 2; cy = top_y + ROOM_INNER_H // 2
                vx = 0 if ia[0] == cx else (1 if ia[0] < cx else -1)
                vy = 0 if ia[1] == cy else (1 if ia[1] < cy else -1)
                reserved.add(ia); reserved.add((ia[0] + vx, ia[1] + vy))

            # reachability-aware placement
            start = self.entrance_approach.get(room_id, (top_x + 1, top_y + ROOM_INNER_H // 2))
            placed_targets: List[Tuple[int, int]] = []

            # corridors (optional) before objects
            self._maybe_corridor(spec, top_x, top_y, reserved)

            # boxes
            req = spec.get("requirements", {})
            receptacle_colors = _required_box_colors(req)
            for b in spec.get("boxes", []):
                color = b["color"]; contains = b.get("contains", "none")
                if contains == "key":
                    obj: WorldObj = Box(color=color, contains=Key(color=color))
                elif contains == "ball":
                    obj = Box(color=color, contains=Ball(color=color))
                else:
                    obj = DepositBox(color=color, contains=None) if color in receptacle_colors else Box(color=color, contains=None)
                pos = self._place_reachable(top_x, top_y, obj, reserved, start)
                placed_targets.append(pos)

            for k in spec.get("keys", []):
                pos = self._place_reachable(top_x, top_y, Key(color=k["color"]), reserved, start)
                placed_targets.append(pos)

            for b in spec.get("balls", []):
                pos = self._place_reachable(top_x, top_y, Ball(color=b["color"]), reserved, start)
                placed_targets.append(pos)

            # red subgoal just behind entrance if possible
            if spec.get("subgoal", False):
                sg = RewardGoal(reward_value=float(room_id), is_terminal=False, color="red")
                pos = self._place_reachable(top_x, top_y, sg, reserved, start)
                placed_targets.append(pos)

            # FINAL GOAL (if requested in this room)
            if req.get("final_goal", False):
                fg = RewardGoal(reward_value=FINAL_GOAL_REWARD, is_terminal=True, color="green")
                fg_pos = self._place_reachable(top_x, top_y, fg, reserved, start)
                placed_targets.append(fg_pos)
                self.final_goal_pos = fg_pos

            # include exit approach as a target (lava can't block)
            if room_id in self.exit_approach:
                placed_targets.append(self.exit_approach[room_id])

            # reachability-safe lava
            for _ in range(spec.get("lava", {}).get("count", 0)):
                self._place_lava_with_reachability(top_x, top_y, reserved, start, placed_targets)

            # init trackers
            self.placed_pairs_by_room[room_id] = set()
            self.key_used_ok[room_id] = False

        # Connect rooms and create the proper door object per requirements
        for i, spec in enumerate(rooms_spec):
            room_id = spec["id"]
            if i + 1 >= len(rooms_spec):
                continue
            ex, ey = self.exit_doors[room_id]
            exit_cfg = spec.get("exit_door", {})
            color = exit_cfg.get("color", "red")
            is_locked = exit_cfg.get("locked", False)
            req = spec.get("requirements", {"open_exit": "none"})
            mode = req.get("open_exit", "none")

            if mode == "key_match":
                door_obj: Door = Door(color=color, is_locked=is_locked)
            elif mode == "multi_combo":
                order = req.get("order", "free")
                if order == "after_balls":
                    door_obj = OrderedStrictDoor(color=color, is_locked=is_locked)
                else:
                    door_obj = StrictDoor(color=color, is_locked=is_locked)
            else:
                # ball-only doors (and "none") use StrictDoor to ignore toggles while locked
                door_obj = StrictDoor(color=color, is_locked=is_locked)

            self.grid.set(ex, ey, door_obj)

        # spawn agent (start in room 1 by default)
        sri = getattr(self, "start_room_idx", 0)
        rsy, rsx = room_index_to_rc(sri)
        a_top_x = rsx * (ROOM_INNER_W + WALL_T) + WALL_T
        a_top_y = rsy * (ROOM_INNER_H + WALL_T) + WALL_T
        self.place_agent(top=(a_top_x, a_top_y), size=(ROOM_INNER_W, ROOM_INNER_H))

    # ----- placement helpers -----
    def _random_free_in_room(self, top_x: int, top_y: int) -> Tuple[int, int]:
        while True:
            x = top_x + self._rand_int(0, ROOM_INNER_W - 1)
            y = top_y + self._rand_int(0, ROOM_INNER_H - 1)
            if self.grid.get(x, y) is None:
                return (x, y)

    def _place_in_room(self, top_x: int, top_y: int, obj: WorldObj) -> Tuple[int, int]:
        x, y = self._random_free_in_room(top_x, top_y)
        self.grid.set(x, y, obj)
        return (x, y)

    def _place_reachable(self, top_x: int, top_y: int, obj: WorldObj,
                         avoid: Set[Tuple[int, int]], start: Tuple[int, int]) -> Tuple[int, int]:
        for _ in range(OBJ_PLACE_MAX_TRIES):
            x = top_x + self._rand_int(0, ROOM_INNER_W - 1)
            y = top_y + self._rand_int(0, ROOM_INNER_H - 1)
            if (x, y) in avoid or self.grid.get(x, y) is not None:
                continue
            self.grid.set(x, y, obj)
            if (x, y) in self._bfs_room(start, top_x, top_y):
                return (x, y)
            self.grid.set(x, y, None)
        return self._place_in_room(top_x, top_y, obj)

    def _place_lava_with_reachability(self, top_x: int, top_y: int, avoid: Set[Tuple[int, int]],
                                      start: Tuple[int, int], targets: List[Tuple[int, int]]):
        for _ in range(LAVA_PLACE_MAX_TRIES):
            x = top_x + self._rand_int(0, ROOM_INNER_W - 1)
            y = top_y + self._rand_int(0, ROOM_INNER_H - 1)
            if (x, y) in avoid or self.grid.get(x, y) is not None:
                continue
            self.grid.set(x, y, Lava())
            reachable = self._bfs_room(start, top_x, top_y)
            if all(t in reachable for t in targets):
                return
            self.grid.set(x, y, None)
        # skip if none fits

    # ----- utils -----
    def _room_id_from_pos(self, pos: Tuple[int, int]) -> Optional[int]:
        x, y = pos

        def _room_id_at(px: int, py: int) -> Optional[int]:
            for i, spec in enumerate(rooms_spec):
                room_id = spec["id"]
                ry, rx = room_index_to_rc(i)
                top_x = rx * (ROOM_INNER_W + WALL_T) + WALL_T
                top_y = ry * (ROOM_INNER_H + WALL_T) + WALL_T
                if top_x <= px < top_x + ROOM_INNER_W and top_y <= py < top_y + ROOM_INNER_H:
                    return room_id
            return None

        # 1) Check current position first
        rid_cur = _room_id_at(x, y)
        if rid_cur is not None:
            return rid_cur

        # 2) Then check the front cell
        fx, fy = self.front_pos
        rid_fwd = _room_id_at(fx, fy)
        if rid_fwd is not None:
            return rid_fwd

        # 3) Then check all four neighbors and return the highest id
        candidates = []
        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            rid = _room_id_at(x + dx, y + dy)
            if rid is not None:
                candidates.append(rid)

        if candidates:
            return max(candidates)

        # 4) If nothing found, no valid room nearby
        raise ValueError("Agent is not in any room!")


    def _unlock_exit_for_room(self, room_id: int):
        pos = self.exit_doors.get(room_id)
        if not pos:
            return
        dx, dy = pos
        obj = self.grid.get(dx, dy)
        if isinstance(obj, Door):
            obj.is_locked = False
            obj.is_open = True

    # ----- combo checking -----
    def _balls_ok_for(self, spec: dict, rid: int) -> bool:
        """Evaluate ball-in-box conditions (handles ball_in_box / multi_all / nested)."""
        req = spec.get("requirements", {})
        mode = req.get("open_exit", "none")
        have = self.placed_pairs_by_room.get(rid, set())

        def all_pairs_ok(pairs: List[List[str]]) -> bool:
            # expect (ball_color, box_color); we track by box color
            return all((cb == cx and cx in have) for (cb, cx) in pairs)

        if mode == "ball_in_box":
            pairs = req.get("pairs", [])
            return any((cb == cx and cx in have) for (cb, cx) in pairs) if pairs else False
        if mode == "multi_all":
            pairs = req.get("pairs", [])
            return all_pairs_ok(pairs)
        if mode == "multi_combo":
            ok = True
            for cond in req.get("conditions", []):
                if cond.get("type") in ("ball_in_box", "multi_all"):
                    pairs = cond.get("pairs", [])
                    if cond["type"] == "ball_in_box":
                        this_ok = any((cb == cx and cx in have) for (cb, cx) in pairs) if pairs else False
                    else:
                        this_ok = all_pairs_ok(pairs)
                    ok = ok and this_ok
            return ok
        return False

    def _check_and_maybe_unlock(self, rid: int):
        """Centralized gate logic for unlocking doors."""
        spec = next(s for s in rooms_spec if s["id"] == rid)
        req = spec.get("requirements", {"open_exit": "none"})
        mode = req.get("open_exit", "none")

        if mode == "none":
            self._unlock_exit_for_room(rid)
            return

        if mode == "key_match":
            # normal doors: once unlocked via key, we auto-open here
            pos = self.exit_doors.get(rid)
            if pos:
                dx, dy = pos
                d = self.grid.get(dx, dy)
                if isinstance(d, Door) and not d.is_locked:
                    d.is_open = True
            return

        pos = self.exit_doors.get(rid)
        if not pos:
            return
        dx, dy = pos
        door_obj = self.grid.get(dx, dy)

        # BALL-ONLY doors
        if mode in ("ball_in_box", "multi_all"):
            if self._balls_ok_for(spec, rid):
                self._unlock_exit_for_room(rid)
            return

        # MULTI-COMBO doors (need balls_ok AND key_ok)
        if mode == "multi_combo" and isinstance(door_obj, (StrictDoor, OrderedStrictDoor)):
            balls_ok = self._balls_ok_for(spec, rid)
            door_obj.gate_cond_met = balls_ok  # expose to key filter
            key_ok = bool(self.key_used_ok.get(rid, False))
            if balls_ok and key_ok:
                self._unlock_exit_for_room(rid)

    # ----- custom step -----
    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)

        # RewardGoal pickup
        curr_obj = self.grid.get(*self.agent_pos)
        if isinstance(curr_obj, RewardGoal):
            reward += curr_obj.reward_value
            terminated = bool(curr_obj.is_terminal)
            self.grid.set(*self.agent_pos, None)

        # Ball deposit via TOGGLE while carrying a ball facing a (Deposit)Box of same color
        try:
            toggle_action = self.actions.toggle
        except AttributeError:
            toggle_action = 5

        if action == toggle_action and self.carrying and isinstance(self.carrying, Ball):
            fx, fy = self.front_pos
            obj = self.grid.get(fx, fy)
            if isinstance(obj, (Box, DepositBox)) and obj.color == self.carrying.color:
                obj.contains = Ball(color=obj.color)  # visual fill
                rid = self._room_id_from_pos((fx, fy))
                if rid is not None:
                    self.placed_pairs_by_room.setdefault(rid, set()).add(obj.color)
                    self._check_and_maybe_unlock(rid)
                self.carrying = None

        # Record key toggles at doors (timing depends on door type)
        if action == toggle_action and self.carrying and isinstance(self.carrying, Key):
            fx, fy = self.front_pos
            rid_agent = self._room_id_from_pos(tuple(self.agent_pos))
            if rid_agent is not None and rid_agent in self.exit_doors:
                ex, ey = self.exit_doors[rid_agent]
                if (fx, fy) == (ex, ey):
                    door_obj = self.grid.get(ex, ey)
                    spec = next(s for s in rooms_spec if s["id"] == rid_agent)
                    req = spec.get("requirements", {"open_exit": "none"})
                    mode = req.get("open_exit", "none")

                    if mode == "key_match":
                        # Base MiniGrid Door handles unlock; we auto-open in checker
                        pass

                    elif mode == "multi_combo" and isinstance(door_obj, (StrictDoor, OrderedStrictDoor)):
                        # If door color matches key color:
                        if door_obj.color == self.carrying.color:
                            # OrderedStrictDoor: ignore early keys until balls_ok is True
                            if isinstance(door_obj, OrderedStrictDoor):
                                if door_obj.gate_cond_met:
                                    self.key_used_ok[rid_agent] = True
                            else:
                                # StrictDoor: record keys even before balls are done
                                self.key_used_ok[rid_agent] = True
                            self._check_and_maybe_unlock(rid_agent)

        # For key_match rooms, if someone unlocked via MiniGrid logic,
        # auto-open it here (same behavior as earlier).
        for spec in rooms_spec:
            if spec.get("requirements", {}).get("open_exit") == "key_match":
                rid = spec["id"]
                pos = self.exit_doors.get(rid)
                if pos:
                    dx, dy = pos
                    dobj = self.grid.get(dx, dy)
                    if isinstance(dobj, Door) and not dobj.is_locked:
                        dobj.is_open = True

        return obs, reward, terminated, truncated, info

    # ----- reset -----
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if options and "start_room" in options:
            sr = int(options["start_room"])
            self.start_room_idx = max(0, min(sr - 1, len(rooms_spec) - 1))
        else:
            self.start_room_idx = 0
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    # ---------- OBSERVATION OVERRIDES (room-with-walls) ----------
    def gen_obs_grid(self, agent_view_size=None):
        """
        Observation is the FULL current room including separator walls & door.
        """
        x0, y0, w, h = self._current_room_bbox()
        grid = self.grid.slice(x0, y0, w, h)
        
        # Full visibility inside this room bbox (shape is (W,H))
        vis_mask = np.ones((w, h), dtype=bool)

        return grid, vis_mask
    
    def _encode_room_obs(self, grid, x0, y0, vis_mask):
        """
        Encode the sliced room grid to (W,H,3) and overlay the agent marker.
        Returns (image, carrying_vec), where:
        - image is uint8 (W,H,3)
        - carrying_vec is (type_idx, color_idx), or (-1, -1) if empty
        """
        img = grid.encode(vis_mask)  # (W,H,3), uint8

        # Agent local coords within the sliced room
        ax, ay = self.agent_pos
        lx, ly = ax - x0, ay - y0
        if 0 <= lx < img.shape[0] and 0 <= ly < img.shape[1]:
            # Paint the agent marker (like FullyObsWrapper does)
            img[lx, ly] = np.array([OBJECT_TO_IDX["agent"],
                                    COLOR_TO_IDX["red"],   # any color you prefer
                                    0], dtype=img.dtype)

        # Carrying as a compact vector (Markov)
        if self.carrying is not None:
            carry_type = OBJECT_TO_IDX[self.carrying.type]
            carry_color = COLOR_TO_IDX[self.carrying.color]
            carrying_vec = np.array([carry_type, carry_color], dtype=np.int16)
        else:
            carrying_vec = np.array([-1, -1], dtype=np.int16)

        return img, carrying_vec
    
    def gen_obs(self):
        # 1) Get your room slice + mask
        x0, y0, w, h = self._current_room_bbox()
        grid, vis_mask = self.gen_obs_grid()

        # 2) Encode and overlay agent
        image, carrying_vec = self._encode_room_obs(grid, x0, y0, vis_mask)

        # 3) Assemble observation dict
        obs = {
            "image": image,
            "direction": self.agent_dir,
            "mission": self.mission
        }
        # Add carrying explicitly for Markov
        obs["carrying"] = carrying_vec
        return obs

    # ---------- RENDER OVERRIDES (highlight/POV use room-with-walls) ----------
    def get_pov_render(self, tile_size):
        x0, y0, w, h = self._current_room_bbox()
        subgrid = self.grid.slice(x0, y0, w, h)

        # agent pos relative to subgrid
        ax, ay = self.agent_pos
        local_agent_pos = (ax - x0, ay - y0)

        vis_mask = np.ones((w, h), dtype=bool)

        # Render the room (agent arrow is drawn by Grid.render when agent_pos is provided)
        img = subgrid.render(
            tile_size,
            agent_pos=local_agent_pos,
            agent_dir=self.agent_dir,
            highlight_mask=vis_mask
        )

        # Overlay a small "carrying" badge in the agent's tile
        if self.carrying is not None:
            self._overlay_carrying_marker(img, local_agent_pos, tile_size, self.carrying)
        return img
    
    def get_full_render(self, highlight, tile_size):
        # Highlight the whole current room bbox in the global render
        highlight_mask = None
        if highlight:
            highlight_mask = np.zeros((self.width, self.height), dtype=bool)
            x0, y0, w, h = self._current_room_bbox()
            x1 = max(0, min(self.width,  x0 + w))
            y1 = max(0, min(self.height, y0 + h))
            x0c = max(0, min(self.width,  x0))
            y0c = max(0, min(self.height, y0))
            highlight_mask[x0c:x1, y0c:y1] = True

        # Render full grid (this already draws the agent triangle)
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask
        )

        # Overlay carrying badge in the agent’s global cell
        if self.carrying is not None:
            self._overlay_carrying_marker(img, tuple(self.agent_pos), tile_size, self.carrying)
        return img

    

    def _overlay_carrying_marker(self, canvas: np.ndarray, cell_xy: tuple[int, int],
                             tile_size: int, obj, scale: float = 0.45):
        """
        Overlay a small rendering of the carried object (`obj`) in the top-left
        corner of the agent's cell on the rendered `canvas`.

        Args:
            canvas: The full RGB image array from Grid.render(...).
            cell_xy: (x_cell, y_cell) coordinate of the agent in the rendered grid.
            tile_size: Pixel size of one cell.
            obj: The carried WorldObj (Key, Ball, Box, etc.).
            scale: Fraction of tile_size to use for the miniature.
        """
        if obj is None:
            return

        H, W, _ = canvas.shape
        x_cell, y_cell = cell_xy

        # Compute pixel bounds of the agent's cell
        x0 = x_cell * tile_size
        y0 = y_cell * tile_size
        if x0 < 0 or y0 < 0 or x0 >= W or y0 >= H:
            return

        # Create a small image for the object using its own render()
        sub = max(6, int(tile_size * scale))
        mini = np.zeros((sub, sub, 3), dtype=np.uint8)
        obj.render(mini)

        # Top-left corner placement with a bit of padding
        pad = max(1, tile_size // 16)
        px, py = x0 + pad, y0 + pad

        # Paste safely inside bounds
        px2 = min(W, px + sub)
        py2 = min(H, py + sub)
        if px2 > px and py2 > py:
            canvas[py:py2, px:px2, :] = mini[:py2 - py, :px2 - px, :]
    
    # ----- corridor drawer -----
    def _maybe_corridor(self, spec: Dict[str, Any], top_x: int, top_y: int, reserved: Set[tuple]):
        cor = spec.get("corridors")
        if not cor:
            return

        pattern  = str(cor.get("pattern", "")).upper()     # "L" or "T"
        material = str(cor.get("material", "wall")).lower()  # "wall" or "door"
        if pattern not in ("L", "T"):
            return

        room_id = spec["id"]

        # --- Random spine positions but inside the room interior ---
        cx_candidates = list(range(top_x + 1, top_x + ROOM_INNER_W - 1))
        cy_candidates = list(range(top_y + 1, top_y + ROOM_INNER_H - 1))
        cx = int(self.np_random.choice(cx_candidates))
        cy = int(self.np_random.choice(cy_candidates))

        # Start = entrance approach (or room center fallback)
        start = self.entrance_approach.get(
            room_id,
            (top_x + ROOM_INNER_W // 2, top_y + ROOM_INNER_H // 2)
        )
        # Goal = exit approach if present, else center
        goal = self.exit_approach.get(
            room_id,
            (top_x + ROOM_INNER_W // 2, top_y + ROOM_INNER_H // 2)
        )

        # Clamp helpers to inner bounds
        def clamp_x(x): return max(top_x + 1, min(top_x + ROOM_INNER_W - 2, x))
        def clamp_y(y): return max(top_y + 1, min(top_y + ROOM_INNER_H - 2, y))

        # We force the corridor gaps to lie on a simple Manhattan route from start→goal.
        # For the vertical line at x=cx, place the gap at y aligned with start or goal.
        gap_v = (cx, clamp_y(start[1] if self.np_random.random() < 0.5 else goal[1]))
        # For the horizontal line at y=cy, place the gap at x aligned with start or goal.
        gap_h = (clamp_x(start[0] if self.np_random.random() < 0.5 else goal[0]), cy)

        def safe_set(x, y, obj):
            if (x, y) in reserved:
                return
            if self.grid.get(x, y) is None:
                self.grid.set(x, y, obj)

        def draw_line_with_explicit_gap(axis: str, gap: tuple[int, int]):
            if axis == "v":
                for y in range(top_y, top_y + ROOM_INNER_H):
                    pos = (cx, y)
                    if pos == gap:
                        continue
                    safe_set(pos[0], pos[1], Wall() if material == "wall" else Door(color="grey", is_locked=False))
            else:
                for x in range(top_x, top_x + ROOM_INNER_W):
                    pos = (x, cy)
                    if pos == gap:
                        continue
                    safe_set(pos[0], pos[1], Wall() if material == "wall" else Door(color="grey", is_locked=False))

        # Draw both arms (order random for variety)
        arms = ["v", "h"]
        self.np_random.shuffle(arms)
        for a in arms:
            if a == "v":
                draw_line_with_explicit_gap("v", gap_v)
            else:
                draw_line_with_explicit_gap("h", gap_h)

        # --- Validate reachability; if blocked, carve a second safety gap and re-validate ---
        reachable = self._bfs_room(start, top_x, top_y)
        if goal not in reachable:
            # Open a second gap aligned with the other endpoint (ensures at least one straight pass)
            second_gap_v = (cx, clamp_y(goal[1] if gap_v[1] != goal[1] else start[1]))
            second_gap_h = (clamp_x(goal[0] if gap_h[0] != goal[0] else start[0]), cy)

            # Clear walls/doors at the new gaps
            for gx, gy in (second_gap_v, second_gap_h):
                if (top_x <= gx < top_x + ROOM_INNER_W) and (top_y <= gy < top_y + ROOM_INNER_H):
                    self.grid.set(gx, gy, None)  # remove obstruction
            # If corridor material is doors, put an unlocked door instead of empty
            if material == "door":
                cxv, cyv = second_gap_v; self.grid.set(cxv, cyv, Door(color="grey", is_locked=False))
                cxh, cyh = second_gap_h; self.grid.set(cxh, cyh, Door(color="grey", is_locked=False))

            # Re-check; if still blocked, nuke the blocking cells along the straight Manhattan path
            reachable = self._bfs_room(start, top_x, top_y)
            if goal not in reachable:
                # Carve a direct Manhattan corridor from start to goal
                x, y = start
                # horizontal
                step = 1 if goal[0] >= x else -1
                for xx in range(x, goal[0] + step, step):
                    if (top_x <= xx < top_x + ROOM_INNER_W) and (top_y <= y < top_y + ROOM_INNER_H):
                        self.grid.set(xx, y, None)
                # vertical
                step = 1 if goal[1] >= y else -1
                for yy in range(y, goal[1] + step, step):
                    if (top_x <= goal[0] < top_x + ROOM_INNER_W) and (top_y <= yy < top_y + ROOM_INNER_H):
                        self.grid.set(goal[0], yy, None)
                # If doors requested, place unlocked doors on the carved line intersections
                if material == "door":
                    self.grid.set(cx, gap_v[1], Door(color="grey", is_locked=False))
                    self.grid.set(gap_h[0], cy, Door(color="grey", is_locked=False))



register(id="BigCurriculumEnv-v0", entry_point=BigCurriculumEnv)