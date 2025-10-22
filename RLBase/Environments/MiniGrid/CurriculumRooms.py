# curriculum_env.py
# Copy-paste this whole file and run it.

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional, Set
import random

import gymnasium as gym
from gymnasium.envs.registration import register

# Adjust imports if your MiniGrid version uses different module paths
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Key, Door, Ball, Box, Lava, WorldObj


# ---------- Constants ----------
MAX_STEPS_PER_ROOM_FACTOR = 4  # heuristic: 4 * H * W per room


# ---------- Custom SubGoal ----------
class SubGoal(Goal):
    """A red goal that can be claimed once; gives reward=room_id and then disappears."""
    def __init__(self, color: str = "red"):
        super().__init__(color=color)

    def can_overlap(self):
        # Allow agent to step on it
        return True


# ---------- Room specification (30 rooms) ----------
rooms_spec: List[Dict[str, Any]] = [
    {"id": 1, "size": [7, 7], "lava": {"count": 0},
     "subgoal": {"place": "free"},
     "exit_door": {"locked": False},
     "keys": [], "boxes": [], "balls": [],
     "requirements": {"open_exit": "none"},
     "notes": "Tiny room; learn to grab subgoal and leave."},

    {"id": 2, "size": [9, 9],
     "corridors": {"count": 1, "pattern": "L"},
     "subgoal": {"place": "free"},
     "exit_door": {"locked": False},
     "requirements": {"open_exit": "none"}},

    {"id": 3, "size": [9, 9],
     "subgoal": {"place": "free"},
     "exit_door": {"locked": True, "color": "red"},
     "keys": [{"color": "red", "place": "free", "in_box": False}],
     "requirements": {"open_exit": "key_match"}},

    {"id": 4, "size": [9, 11],
     "subgoal": {"place": "free"},
     "exit_door": {"locked": True, "color": "blue"},
     "keys": [{"color": "blue", "place": "free"},
              {"color": "red",  "place": "free"}],
     "requirements": {"open_exit": "key_match"}},

    {"id": 5, "size": [9, 11],
     "lava": {"count": 8, "density": 0.10},
     "subgoal": {"place": "free"},
     "exit_door": {"locked": False},
     "requirements": {"open_exit": "none"}},

    {"id": 6, "size": [11, 11],
     "lava": {"count": 12, "density": 0.12},
     "subgoal": {"place": "free"},
     "exit_door": {"locked": True, "color": "green"},
     "keys": [{"color": "green", "place": "free"},
              {"color": "red",   "place": "free"},
              {"color": "blue",  "place": "free"}],
     "requirements": {"open_exit": "key_match"}},

    {"id": 7, "size": [11, 11],
     "subgoal": {"place": "free"},
     "boxes": [{"color": "yellow", "contains": "key", "place": "free"}],
     "exit_door": {"locked": True, "color": "yellow"},
     "requirements": {"open_exit": "key_match"}},

    {"id": 8, "size": [11, 13],
     "lava": {"count": 14, "density": 0.10},
     "subgoal": {"place": "free"},
     "boxes": [{"color": "red", "contains": "key"},
               {"color": "blue", "contains": "none"}],
     "keys": [{"color": "blue", "place": "free"}],
     "exit_door": {"locked": True, "color": "red"},
     "requirements": {"open_exit": "key_match"}},

    {"id": 9, "size": [13, 13],
     "lava": {"count": 20, "density": 0.12},
     "subgoal": {"place": "free"},
     "boxes": [{"color": "green", "contains": "key"},
               {"color": "purple", "contains": "key"}],
     "exit_door": {"locked": True, "color": "green"},
     "requirements": {"open_exit": "key_match"}},

    {"id": 10, "size": [13, 15],
     "lava": {"count": 24, "density": 0.12},
     "subgoal": {"place": "free"},
     "boxes": [{"color": "blue", "contains": "key"},
               {"color": "red", "contains": "none"},
               {"color": "green", "contains": "none"}],
     "keys": [{"color": "red", "place": "free"}],
     "exit_door": {"locked": True, "color": "blue"},
     "requirements": {"open_exit": "key_match"}},

    {"id": 11, "size": [11, 11],
     "subgoal": {"place": "free"},
     "balls": [{"color": "red"}],
     "boxes": [{"color": "red", "contains": "none", "place": "free"}],
     "exit_door": {"locked": True, "color": "red"},
     "requirements": {"open_exit": "ball_in_box", "pairs": [["red", "red"]]}},

    {"id": 12, "size": [11, 13],
     "subgoal": {"place": "free"},
     "balls": [{"color": "red"}, {"color": "green"}],
     "boxes": [{"color": "red", "contains": "none"},
               {"color": "green", "contains": "none"}],
     "exit_door": {"locked": True},
     "requirements": {"open_exit": "multi_all",
                      "pairs": [["red", "red"], ["green", "green"]]}},

    {"id": 13, "size": [13, 13],
     "subgoal": {"place": "free"},
     "balls": [{"color": "red"}, {"color": "blue"}],
     "boxes": [{"color": "red", "contains": "none"},
               {"color": "blue", "contains": "none"}],
     "exit_door": {"locked": True},
     "requirements": {"open_exit": "ball_in_box",
                      "pairs": [["red", "red"]]}},

    {"id": 14, "size": [13, 15],
     "subgoal": {"place": "free"},
     "balls": [{"color": "red"}, {"color": "green"}, {"color": "blue"}],
     "boxes": [{"color": "red", "contains": "none"},
               {"color": "green", "contains": "none"},
               {"color": "blue", "contains": "none"}],
     "exit_door": {"locked": True},
     "requirements": {"open_exit": "multi_all",
                      "pairs": [["red", "red"], ["green", "green"], ["blue", "blue"]]}},

    {"id": 15, "size": [15, 15],
     "corridors": {"count": 2, "pattern": "T"},
     "subgoal": {"place": "free"},
     "balls": [{"color": "red"}, {"color": "yellow"}],
     "boxes": [{"color": "red", "contains": "none"},
               {"color": "yellow", "contains": "none"}],
     "exit_door": {"locked": True},
     "requirements": {"open_exit": "multi_all",
                      "pairs": [["red", "red"], ["yellow", "yellow"]]}},

    {"id": 16, "size": [15, 15],
     "lava": {"count": 28, "density": 0.12},
     "subgoal": {"place": "free"},
     "balls": [{"color": "red"}, {"color": "green"}],
     "boxes": [{"color": "red", "contains": "none"},
               {"color": "green", "contains": "none"}],
     "exit_door": {"locked": True},
     "requirements": {"open_exit": "multi_all",
                      "pairs": [["red", "red"], ["green", "green"]]}},

    {"id": 17, "size": [15, 17],
     "lava": {"count": 36, "density": 0.13},
     "subgoal": {"place": "free"},
     "balls": [{"color": "red"}, {"color": "green"}, {"color": "blue"}],
     "boxes": [{"color": "red", "contains": "none"},
               {"color": "green", "contains": "none"},
               {"color": "blue", "contains": "none"}],
     "exit_door": {"locked": True},
     "requirements": {"open_exit": "multi_all",
                      "pairs": [["red", "red"], ["green", "green"], ["blue", "blue"]]}},

    {"id": 18, "size": [17, 17],
     "lava": {"count": 40, "density": 0.14},
     "subgoal": {"place": "free"},
     "balls": [{"color": "yellow"}, {"color": "purple"}],
     "boxes": [{"color": "yellow", "contains": "none"},
               {"color": "purple", "contains": "none"}],
     "keys": [{"color": "yellow"}],  # distractor
     "exit_door": {"locked": True},
     "requirements": {"open_exit": "multi_all",
                      "pairs": [["yellow", "yellow"], ["purple", "purple"]]}},

    {"id": 19, "size": [17, 19],
     "lava": {"count": 44, "density": 0.14},
     "subgoal": {"place": "free"},
     "boxes": [{"color": "red", "contains": "ball"},
               {"color": "green", "contains": "ball"}],
     "balls": [],
     "exit_door": {"locked": True},
     "requirements": {"open_exit": "multi_all",
                      "pairs": [["red", "red"], ["green", "green"]]}},

    {"id": 20, "size": [17, 19],
     "lava": {"count": 48, "density": 0.15},
     "subgoal": {"place": "free"},
     "balls": [{"color": "red"}],
     "boxes": [{"color": "blue", "contains": "key"},
               {"color": "red", "contains": "none"}],
     "keys": [],
     "exit_door": {"locked": True},
     "requirements": {"open_exit": "ball_in_box",
                      "pairs": [["red", "red"]]}},

    {"id": 21, "size": [19, 19],
     "lava": {"count": 52, "density": 0.15},
     "subgoal": {"place": "free"},
     "keys": [{"color": "red"}, {"color": "green"}, {"color": "blue"}],
     "boxes": [{"color": "yellow", "contains": "none"}],
     "balls": [{"color": "yellow"}],
     "exit_door": {"locked": True},
     "requirements": {"open_exit": "ball_in_box",
                      "pairs": [["yellow", "yellow"]]}},

    {"id": 22, "size": [19, 21],
     "lava": {"count": 58, "density": 0.16},
     "subgoal": {"place": "free"},
     "boxes": [{"color": "red", "contains": "ball"},
               {"color": "green", "contains": "ball"},
               {"color": "blue", "contains": "ball"}],
     "balls": [],
     "exit_door": {"locked": True},
     "requirements": {"open_exit": "multi_all",
                      "pairs": [["red", "red"], ["green", "green"], ["blue", "blue"]]}},

    {"id": 23, "size": [21, 21],
     "lava": {"count": 64, "density": 0.17},
     "subgoal": {"place": "free"},
     "balls": [{"color": "purple"}],
     "boxes": [{"color": "purple", "contains": "none"},
               {"color": "red", "contains": "none"},
               {"color": "green", "contains": "none"}],
     "exit_door": {"locked": True},
     "requirements": {"open_exit": "ball_in_box",
                      "pairs": [["purple", "purple"]]}},

    {"id": 24, "size": [21, 23],
     "lava": {"count": 70, "density": 0.18},
     "subgoal": {"place": "free"},
     "keys": [{"color": "red"}, {"color": "blue"}],
     "boxes": [{"color": "yellow", "contains": "none"},
               {"color": "green", "contains": "none"}],
     "balls": [{"color": "yellow"}, {"color": "green"}],
     "exit_door": {"locked": True},
     "requirements": {"open_exit": "multi_all",
                      "pairs": [["yellow", "yellow"], ["green", "green"]]}},

    {"id": 25, "size": [23, 23],
     "lava": {"count": 80, "density": 0.18},
     "subgoal": {"place": "free"},
     "keys": [{"color": "yellow"}, {"color": "purple"}],
     "balls": [{"color": "red"}, {"color": "blue"}],
     "boxes": [{"color": "red", "contains": "none"},
               {"color": "blue", "contains": "none"}],
     "exit_door": {"locked": True},
     "requirements": {"open_exit": "multi_all",
                      "pairs": [["red", "red"], ["blue", "blue"]]},
     "distractors": {"fake_doors": 2}},

    {"id": 26, "size": [23, 25],
     "lava": {"count": 88, "density": 0.19},
     "subgoal": {"place": "free"},
     "keys": [{"color": "green"}, {"color": "blue"}],
     "boxes": [{"color": "red", "contains": "none"},
               {"color": "red", "contains": "ball"},
               {"color": "green", "contains": "ball"}],
     "balls": [{"color": "green"}],
     "exit_door": {"locked": True},
     "requirements": {"open_exit": "multi_all",
                      "pairs": [["red", "red"], ["green", "green"]]}},

    {"id": 27, "size": [25, 25],
     "lava": {"count": 110, "density": 0.20},
     "subgoal": {"place": "free"},
     "balls": [{"color": "red"}, {"color": "green"},
               {"color": "blue"}, {"color": "yellow"}],
     "boxes": [{"color": "red", "contains": "none"},
               {"color": "green", "contains": "none"},
               {"color": "blue", "contains": "none"},
               {"color": "yellow", "contains": "none"}],
     "exit_door": {"locked": True},
     "requirements": {"open_exit": "multi_all",
                      "pairs": [["red", "red"], ["green", "green"],
                                ["blue", "blue"], ["yellow", "yellow"]]}},

    {"id": 28, "size": [25, 27],
     "lava": {"count": 120, "density": 0.20},
     "subgoal": {"place": "free"},
     "boxes": [{"color": "purple", "contains": "ball"},
               {"color": "blue", "contains": "ball"},
               {"color": "red", "contains": "none"},
               {"color": "green", "contains": "none"},
               {"color": "yellow", "contains": "none"}],
     "balls": [],
     "exit_door": {"locked": True},
     "requirements": {"open_exit": "multi_all",
                      "pairs": [["purple", "purple"], ["blue", "blue"]]}},

    {"id": 29, "size": [27, 27],
     "lava": {"count": 140, "density": 0.21},
     "subgoal": {"place": "free"},
     "keys": [{"color": "red"}],
     "balls": [{"color": "green"}, {"color": "blue"}, {"color": "yellow"}],
     "boxes": [{"color": "green", "contains": "none"},
               {"color": "blue", "contains": "none"},
               {"color": "yellow", "contains": "none"}],
     "exit_door": {"locked": True},
     "requirements": {"open_exit": "multi_all",
                      "pairs": [["green", "green"], ["blue", "blue"], ["yellow", "yellow"]]}},

    {"id": 30, "size": [29, 29],
     "lava": {"count": 170, "density": 0.22},
     "subgoal": {"place": "free"},
     "keys": [{"color": "purple"}, {"color": "blue"}],
     "balls": [{"color": "red"}, {"color": "green"},
               {"color": "blue"}, {"color": "yellow"}, {"color": "purple"}],
     "boxes": [{"color": "red", "contains": "none"},
               {"color": "green", "contains": "none"},
               {"color": "blue", "contains": "none"},
               {"color": "yellow", "contains": "none"},
               {"color": "purple", "contains": "none"},
               {"color": "red", "contains": "none"},
               {"color": "green", "contains": "none"}],
     "exit_door": {"locked": True},
     "requirements": {"open_exit": "multi_all",
                      "pairs": [["red", "red"], ["green", "green"],
                                ["blue", "blue"], ["yellow", "yellow"],
                                ["purple", "purple"]]}}
]


# ---------- Environment ----------
class CurriculumRoomsEnv(MiniGridEnv):
    """
    Mechanics:
    - SubGoal: stepping on it grants reward = room_id (once), then it vanishes.
    - Key→Door: standard MiniGrid. Exit door is on the east wall (midpoint).
    - Ball-in-Box: press TOGGLE while carrying a Ball and facing a Box of same color
      to 'deposit' it (box.contains = Ball; carrying cleared). Tracks per-color placements.
    - Corridors: simple 'L' or 'T' internal walls with one-tile openings (optional).
    """

    def __init__(self, rooms_spec: List[Dict[str, Any]], **kwargs):
        self.rooms_spec = rooms_spec
        self.room_idx = 0
        self.claimed_subgoal = False
        self.subgoal_pos: Optional[Tuple[int, int]] = None
        self.exit_pos: Optional[Tuple[int, int]] = None
        self.exit_open = False
        self.placed_pairs: Set[str] = set()

        H, W = self.rooms_spec[0]["size"]
        mission_space = MissionSpace(mission_func=lambda: "Solve room puzzles and exit.")
        super().__init__(mission_space=mission_space, width=W, height=H,
                         max_steps=MAX_STEPS_PER_ROOM_FACTOR * H * W, **kwargs)

    # ---------- Grid generation ----------
    def _gen_grid(self, width: int, height: int):
        spec = self.rooms_spec[self.room_idx]
        H, W = spec["size"]
        assert H == height and W == width, "Env width/height must match spec"

        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.agent_pos = None
        self.agent_dir = 0
        self.exit_open = False
        self.claimed_subgoal = False
        self.placed_pairs.clear()
        self.subgoal_pos = None
        self.exit_pos = None

        # Exit door on east wall (middle)
        exit_locked = spec.get("exit_door", {}).get("locked", False)
        exit_color = spec.get("exit_door", {}).get("color", "red")
        ex_y = height // 2
        ex_x = width - 1
        self.grid.set(ex_x, ex_y, Door(color=exit_color, is_locked=exit_locked))
        self.exit_pos = (ex_x, ex_y)

        # Spawn agent near west side
        self.place_agent(top=(1, 1), size=(max(1, height - 2), 1))

        # Optionally carve simple corridor patterns
        self._maybe_carve_corridors(spec)

        # Lava
        lava_cfg = spec.get("lava", {})
        for _ in range(lava_cfg.get("count", 0)):
            self._place_free(Lava())

        # Keys (some may be placed inside boxes via the 'keys' field with in_box=True;
        # but here we only handle loose keys—embedded ones are handled below in 'boxes')
        for k in spec.get("keys", []):
            if k.get("in_box", False):
                inner_key = Key(color=k["color"])
                self._place_free(Box(color=k["color"], contains=inner_key))
            else:
                self._place_free(Key(color=k["color"]))

        # Boxes (may contain key/ball/none)
        for b in spec.get("boxes", []):
            contains = b.get("contains", "none")
            obj: Optional[WorldObj] = None
            if contains == "key":
                obj = Key(color=b["color"])
            elif contains == "ball":
                obj = Ball(color=b["color"])
            self._place_free(Box(color=b["color"], contains=obj))

        # Balls (loose)
        for b in spec.get("balls", []):
            self._place_free(Ball(color=b["color"]))

        # SubGoal
        self.subgoal_pos = self._place_free(SubGoal(color="red"))

        # Mission & steps
        self.mission = f"Room {spec['id']}: collect subgoal once and open/exit."
        self.max_steps = MAX_STEPS_PER_ROOM_FACTOR * width * height

    # ---------- Helpers ----------
    def _place_free(self, obj: WorldObj) -> Tuple[int, int]:
        """Place object somewhere free inside the inner area."""
        pos = self.place_obj(obj, top=(1, 1), size=(self.width - 2, self.height - 2))
        return pos

    def _agent_on_subgoal(self) -> bool:
        return self.subgoal_pos is not None and tuple(self.agent_pos) == tuple(self.subgoal_pos)

    def _remove_subgoal(self):
        if self.subgoal_pos is not None:
            x, y = self.subgoal_pos
            self.grid.set(x, y, None)
            self.subgoal_pos = None

    def _open_exit_door(self):
        if self.exit_pos:
            ex, ey = self.exit_pos
            obj = self.grid.get(ex, ey)
            if isinstance(obj, Door):
                obj.is_locked = False
                obj.is_open = True
                self.exit_open = True

    def _open_exit_if_now_unlocked(self):
        if not self.exit_pos:
            return
        ex, ey = self.exit_pos
        obj = self.grid.get(ex, ey)
        if isinstance(obj, Door) and (not obj.is_locked):
            obj.is_open = True
            self.exit_open = True

    def _agent_through_exit(self) -> bool:
        if not self.exit_pos:
            return False
        ex, ey = self.exit_pos
        if tuple(self.agent_pos) == (ex, ey):
            obj = self.grid.get(ex, ey)
            return isinstance(obj, Door) and (obj.is_open and not obj.is_locked)
        return False

    def _all_required_pairs_satisfied(self, pairs: List[List[str]]) -> bool:
        for color_ball, color_box in pairs:
            if color_ball != color_box:
                return False
            if color_ball not in self.placed_pairs:
                return False
        return True

    def _maybe_carve_corridors(self, spec: Dict[str, Any]) -> None:
        """Very simple 'L' and 'T' corridor walls with single gaps to create detours."""
        cor = spec.get("corridors")
        if not cor:
            return
        pattern = cor.get("pattern", "none")
        if pattern not in ("L", "T"):
            return

        W, H = self.width, self.height
        cx, cy = W // 2, H // 2

        if pattern == "L":
            # Vertical internal wall with one opening near top
            for y in range(1, H - 1):
                self.grid.set(cx, y, Door(color="grey", is_locked=False))  # thin walls as openable partitions
            self.grid.set(cx, 2, None)  # opening
            # Horizontal internal wall with one opening near left
            for x in range(1, W - 1):
                self.grid.set(x, cy, Door(color="grey", is_locked=False))
            self.grid.set(2, cy, None)

        elif pattern == "T":
            # Vertical stem of T
            for y in range(1, H - 1):
                self.grid.set(cx, y, Door(color="grey", is_locked=False))
            self.grid.set(cx, cy - 1, None)  # gap

            # Horizontal bar of T
            for x in range(1, W - 1):
                self.grid.set(x, cy, Door(color="grey", is_locked=False))
            self.grid.set(cx + 1, cy, None)  # another gap

    # ---------- Step override ----------
    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)

        spec = self.rooms_spec[self.room_idx]
        req = spec.get("requirements", {}).get("open_exit", "none")

        # SubGoal: one-shot reward = room id, then remove the subgoal
        if not self.claimed_subgoal and self._agent_on_subgoal():
            self.claimed_subgoal = True
            reward += float(spec["id"])
            self._remove_subgoal()

        # Ball-in-Box deposit on TOGGLE while carrying Ball and facing same-color Box
        try:
            toggle_action = self.actions.toggle
        except AttributeError:
            toggle_action = 5  # default fallback

        if action == toggle_action and self.carrying and isinstance(self.carrying, Ball):
            fwd_pos = self.front_pos
            obj = self.grid.get(*fwd_pos)
            if isinstance(obj, Box) and obj.color == self.carrying.color:
                # Deposit the ball into the box; mark color as satisfied
                obj.contains = Ball(color=obj.color)
                self.carrying = None
                self.placed_pairs.add(obj.color)

        # Open exit based on requirement
        if req in ("ball_in_box", "multi_all"):
            pairs = spec.get("requirements", {}).get("pairs", [])
            if self._all_required_pairs_satisfied(pairs):
                self._open_exit_door()
        elif req == "key_match":
            # If door got unlocked by key, make sure it's open
            self._open_exit_if_now_unlocked()
        # req == "none": nothing needed

        # Move to next room if stepping onto open exit door cell
        if self._agent_through_exit():
            self.room_idx += 1
            if self.room_idx >= len(self.rooms_spec):
                terminated = True
            else:
                # Resize if next room has different size
                H, W = self.rooms_spec[self.room_idx]["size"]
                if (W != self.width) or (H != self.height):
                    self.width, self.height = W, H
                self._gen_grid(self.width, self.height)
                obs = self.gen_obs()

        return obs, reward, terminated, truncated, info


# ---------- Gymnasium registration ----------
def make_curriculum_env(**kwargs):
    return CurriculumRoomsEnv(rooms_spec=rooms_spec, **kwargs)

register(
    id="CurriculumRoomsEnv-v0",
    entry_point=make_curriculum_env
)


