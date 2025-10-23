# options_minigrid_toolkit_auto.py
# Fully aligned with (x, y) coordinates = (column, row).
# All array access to obs["image"] uses img[y, x], while positions and motion use (x, y).

from __future__ import annotations
from typing import Optional, Callable, List, Dict, Tuple
import numpy as np
from collections import deque

from minigrid.core.constants import (
    COLORS, COLOR_NAMES, COLOR_TO_IDX, IDX_TO_COLOR,
    OBJECT_TO_IDX as OID, IDX_TO_OBJECT, STATE_TO_IDX as SID,
    DIR_TO_VEC,  # (dx, dy) where x = columns (left→right), y = rows (top→down)
)
from ..Utils import BaseOption

# ACTION indices (keep consistent with your env)
A_LEFT, A_RIGHT, A_FORWARD, A_PICKUP, A_DROP, A_TOGGLE, A_DONE = 0, 1, 2, 3, 4, 5, 6

# -----------------------------------------------------------------------------
# Deterministic, explicit direction updates (no arithmetic shortcuts)
# DIR indices follow MiniGrid:
#   0 = right (+x), 1 = down (+y), 2 = left (-x), 3 = up (-y)
# -----------------------------------------------------------------------------
TURN_RIGHT = {0: 1, 1: 2, 2: 3, 3: 0}
TURN_LEFT  = {0: 3, 3: 2, 2: 1, 1: 0}

def turn_action_towards(cur_dir: int, want_dir: int) -> Optional[int]:
    """
    Return the atomic turn action (LEFT/RIGHT) that moves one step toward want_dir.
    Uses explicit transition tables, no modulo arithmetic.
    If already facing want_dir, return None.
    For 180° we choose LEFT deterministically.
    """
    if cur_dir == want_dir:
        return None
    if TURN_RIGHT[cur_dir] == want_dir:
        return A_RIGHT
    if TURN_LEFT[cur_dir] == want_dir:
        return A_LEFT
    # 180° away: choose LEFT (two lefts over two rights is arbitrary but consistent)
    if TURN_RIGHT[TURN_RIGHT[cur_dir]] == want_dir:
        return A_LEFT
    # 270° away: one LEFT reaches want_dir
    if TURN_LEFT[cur_dir] == TURN_LEFT[want_dir]:
        return A_LEFT
    # Fallback (shouldn't happen)
    return A_LEFT


# ============================== Grid / Navigation helpers ==============================

class GridNavMixin:
    """
    Utilities for fully/ego observable MiniGrid with (x, y) world coords:
      - x = column (0..W-1), left → right
      - y = row    (0..H-1), top → down
    The image tensor is still indexed as img[y, x] for numpy correctness.
    """

    # ---------- Primitive reads ----------
    def _img(self, obs) -> np.ndarray:
        return obs["image"]

    def _width_height(self, obs) -> tuple[int, int]:
        img = self._img(obs)
        W, H = int(img.shape[0]), int(img.shape[1])
        return W, H

    def _dir(self, obs) -> int:
        return int(obs["direction"])

    def _dirvec_xy(self, obs) -> np.ndarray:
        """Return direction vector (dx, dy) in world coords."""
        v = DIR_TO_VEC[self._dir(obs)]  # MiniGrid defines (dx, dy)
        return np.array([int(v[0]), int(v[1])], dtype=int)

    def _inside(self, obs, xy: np.ndarray) -> bool:
        W, H = self._width_height(obs)
        x, y = int(xy[0]), int(xy[1])
        return 0 <= x < W and 0 <= y < H

    def _cell_xy(self, obs, xy: np.ndarray) -> np.ndarray:
        """Access a cell using (x, y) world coords; underlying array uses [y, x]."""
        img = self._img(obs)
        x, y = int(xy[0]), int(xy[1])
        return img[x, y]

    def _find_agent(self, obs) -> np.ndarray:
        """
        Fully observable: agent is rendered (OID['agent']).
        Return agent position in (x, y). (np.where gives (ys, xs).)
        """
        img = self._img(obs)
        xs, ys = np.where(img[..., 0] == OID["agent"])
        if len(xs) > 0:
            return np.array([int(xs[0]), int(ys[0])], dtype=int)
        # Fallback center (for egocentric crops)
        W, H = self._width_height(obs)
        return np.array([W // 2, H // 2], dtype=int)

    # ---------- Terrain semantics ----------
    def _passable(self, cell: np.ndarray) -> bool:
        oid, st = int(cell[0]), int(cell[2])
        if oid in (OID["unseen"], OID["wall"], OID["lava"]):
            return False
        if oid == OID["door"] and st != SID["open"]:
            return False
        # empty, floor, goal, key, ball, box, agent => traversable
        return True

    def _forward_is_safe(self, obs) -> bool:
        pos = self._find_agent(obs)
        fwd = pos + self._dirvec_xy(obs)  # explicit forward = pos + dir
        if not self._inside(obs, fwd):
            return False
        return self._passable(self._cell_xy(obs, fwd))

    def _door_is_open(self, obs, xy: np.ndarray) -> bool:
        c = self._cell_xy(obs, xy)
        return int(c[0]) == OID["door"] and int(c[2]) == SID["open"]

    def _door_is_closed(self, obs, xy: np.ndarray) -> bool:
        c = self._cell_xy(obs, xy)
        return int(c[0]) == OID["door"] and int(c[2]) == SID["closed"]
    
    def _door_is_locked(self, obs, xy: np.ndarray) -> bool:
        c = self._cell_xy(obs, xy)
        return int(c[0]) == OID["door"] and int(c[2]) == SID["locked"]

    # ---------- Geometry ----------
    def _ahead_xy(self, obs) -> np.ndarray:
        """
        World coords (x, y) of the tile directly in front of the agent:
        ahead = agent_xy + DIR_TO_VEC[direction]  (DIR_TO_VEC is (dx, dy)).
        """
        return self._find_agent(obs) + self._dirvec_xy(obs)

    def _ahead_cell(self, obs) -> Optional[np.ndarray]:
        """
        The image cell directly ahead of the agent, or None if out of bounds.
        """
        ahead = self._ahead_xy(obs)
        if not self._inside(obs, ahead):
            return None
        return self._cell_xy(obs, ahead)
    
    def _neighbors4(self, obs, xy: Tuple[int, int]) -> list[np.ndarray]:
        W, H = self._width_height(obs)
        x, y = int(xy[0]), int(xy[1])
        cands = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return [np.array(p, dtype=int) for p in cands if 0 <= p[0] < W and 0 <= p[1] < H]

    def _at_cell(self, obs, target_xy: np.ndarray) -> bool:
        return np.array_equal(self._find_agent(obs), target_xy)
    
    def _facing_cell(self, obs, target_xy: np.ndarray) -> bool:
        """
        True iff the agent is *directly facing* the given (x, y) tile.
        In practice: target must be exactly the tile in front of the agent.
        """
        ahead = self._ahead_xy(obs)
        return np.array_equal(ahead, target_xy)
    
    def _facing_is_oid(self, obs, oid: int) -> bool:
        """
        Convenience: True iff the cell directly ahead exists and its object id == oid.
        """
        c = self._ahead_cell(obs)
        return c is not None and int(c[0]) == int(oid)
    
    

    def _face_vec_action(self, obs, target_xy: np.ndarray) -> Optional[int]:
        """Rotate to face the (x,y) target if needed, using explicit turn reasoning."""
        agent = self._find_agent(obs)
        cur_dir = self._dir(obs)
        delta = target_xy - agent
        if delta[0] == 0 and delta[1] == 0:
            return None

        # Map axial deltas → desired DIR indices (no inference by arithmetic)
        # right, down, left, up:
        if   delta[0] > 0 and delta[1] == 0: want_dir = 0
        elif delta[0] == 0 and delta[1] > 0: want_dir = 1
        elif delta[0] < 0 and delta[1] == 0: want_dir = 2
        elif delta[0] == 0 and delta[1] < 0: want_dir = 3
        else:
            # Should not happen with 4-neighborhood BFS paths.
            want_dir = cur_dir
        return turn_action_towards(cur_dir, want_dir)

    # ---------- BFS ----------
    def _bfs_path(
        self,
        obs,
        start_xy: np.ndarray,
        goal_test: Callable[[Tuple[int, int]], bool],
        avoid_lava: bool = True,
        passable_override: Optional[Callable[[Tuple[int, int]], bool]] = None,
    ) -> Optional[list[np.ndarray]]:
        img = self._img(obs)
        start = (int(start_xy[0]), int(start_xy[1]))
        seen: set[Tuple[int, int]] = set([start])
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        q = deque([start])

        def ok(xy: Tuple[int, int]) -> bool:
            if passable_override is not None:
                return passable_override(xy)
            c = img[xy[0], xy[1]]
            if avoid_lava and int(c[0]) == OID["lava"]:
                return False
            return self._passable(c)
        while q:
            xy = q.popleft()
            if goal_test(xy):
                # reconstruct path: list of (x,y) numpy arrays
                path = [np.array([xy[0], xy[1]], dtype=int)]
                p = parent[xy]
                while p is not None:
                    path.append(np.array([p[0], p[1]], dtype=int))
                    p = parent[p]
                path.reverse()
                return path

            for nb in self._neighbors4(obs, xy):
                nb_t = (int(nb[0]), int(nb[1]))
                if nb_t in seen:
                    continue
                
                # If neighbor itself is the goal, accept it even if not passable
                if goal_test(nb_t):
                    parent[nb_t] = xy
                    # reconstruct ending at nb_t
                    path = [np.array([nb_t[0], nb_t[1]], dtype=int)]
                    p = xy
                    while p is not None:
                        path.append(np.array([p[0], p[1]], dtype=int))
                        p = parent[p]
                    path.reverse()
                    return path
            
                if ok(nb_t):
                    seen.add(nb_t)
                    parent[nb_t] = xy
                    q.append(nb_t)
                    
                    
        return None

    def _find_nearest_of_type(
        self,
        obs,
        obj_id_filter: int,
        also_require: Optional[Callable[[Tuple[int, int], np.ndarray], bool]] = None,
        avoid_lava: bool = True,
    ) -> Optional[list[np.ndarray]]:
        img = self._img(obs)
        start = self._find_agent(obs)
        def is_goal(xy: Tuple[int, int]) -> bool:
            c = img[xy[0], xy[1]]
            if int(c[0]) != obj_id_filter:
                return False
            return True if also_require is None else also_require(xy, c)

        return self._bfs_path(obs, start, is_goal, avoid_lava=avoid_lava)

    def _step_towards(self, obs, path):
        """
        Rotate toward the next waypoint and step if aligned; else turn a single step.
        Forward is strictly position + dirvec in (x,y).
        """
        if not path or len(path) < 2:
            return None

        agent = self._find_agent(obs)
        # Find the waypoint immediately after agent
        nxt = None
        if np.array_equal(path[0], agent):
            nxt = path[1]
        else:
            for i in range(len(path) - 1):
                if np.array_equal(path[i], agent):
                    nxt = path[i + 1]
                    break
        if nxt is None:
            return None
        
        # Turn one step toward the waypoint if not facing it yet
        turn = self._face_vec_action(obs, nxt)
        if turn is not None:
            return turn

        # Move forward if the next cell equals the waypoint and is passable
        dirvec = self._dirvec_xy(obs)
        ahead = agent + dirvec  # explicit forward
        if np.array_equal(ahead, nxt) and self._inside(obs, ahead) and self._passable(self._cell_xy(obs, ahead)):
            return A_FORWARD

        # Misaligned or blocked—turn left to search next tick
        return A_LEFT


# ============================== Timed option base ==============================

class TimedOption(BaseOption, GridNavMixin):
    def __init__(self, option_len: int):
        super().__init__(option_len)
        self.step_counter = 0
        self._done = False
        self._phase = "init"
        # scratch
        self._path = None
        self._target = None

    def _tick(self):
        self.step_counter += 1

    def reset(self):
        self.step_counter = 0
        self._done = False
        self._phase = "init"
        self._path = None
        self._target = None

    def is_terminated(self, obs) -> bool:
        term = self._done or (self.step_counter >= self.option_len)
        if term:
            # Make options reusable immediately
            self.reset()
            return True
        return False

    def __repr__(self):
        return f"{self.__class__.__name__}(len={self.option_len})"

# ============================================================
# MiniGrid Options — categorized & lava/wall-aware
# Inheritance: every option -> (TimedOption, GridNavMixin)
# Requires your provided: BaseOption, GridNavMixin, TimedOption, OID, SID,
#   A_LEFT, A_RIGHT, A_FORWARD, A_PICKUP, A_DROP, A_TOGGLE, A_DONE,
#   DIR_TO_VEC, turn_action_towards, TURN_LEFT, TURN_RIGHT
# ============================================================

from typing import Optional
import numpy as np

# ------------------------------------------------------------
# 2) HANDLING DOORS (colors, status)
# ------------------------------------------------------------

class GoAdjFaceNearestClosedDoor(TimedOption, GridNavMixin):
    """Navigate adjacent to the nearest CLOSED door and face it."""
    def select_action(self, obs):
        if self._phase == "init":
            self._phase = "route"
            def is_closed(xy, c): return int(c[0]) == OID["door"] and int(c[2]) == SID["closed"]
            self._path = self._find_nearest_of_type(obs, OID["door"], also_require=is_closed, avoid_lava=True)
            self._target = self._path[-1] if self._path else None

        self._tick()
        if not self._path or self._target is None:
            self._done = True
            return A_DONE

        goal_path = self._path[:-1] if len(self._path) >= 2 else self._path
        act = self._step_towards(obs, goal_path)
        if act is not None: return act

        turn = self._face_vec_action(obs, self._target)
        if turn is not None: return turn

        self._done = True
        return A_DONE

class GoAdjFaceNearestLockedDoor(TimedOption, GridNavMixin):
    """Navigate adjacent to the nearest LOCKED door and face it."""
    def select_action(self, obs):
        if self._phase == "init":
            self._phase = "route"
            def is_locked(xy, c): return int(c[0]) == OID["door"] and int(c[2]) == SID["locked"]
            self._path = self._find_nearest_of_type(obs, OID["door"], also_require=is_locked, avoid_lava=True)
            self._target = self._path[-1] if self._path else None

        self._tick()
        if not self._path or self._target is None:
            self._done = True
            return A_DONE

        goal_path = self._path[:-1] if len(self._path) >= 2 else self._path
        act = self._step_towards(obs, goal_path)
        if act is not None: return act

        turn = self._face_vec_action(obs, self._target)
        if turn is not None: return turn

        self._done = True
        return A_DONE

class GoAdjFaceNearestOpenDoor(TimedOption, GridNavMixin):
    """Navigate adjacent to the nearest OPEN door and face it."""
    def select_action(self, obs):
        if self._phase == "init":
            self._phase = "route"
            def is_open(xy, c): return int(c[0]) == OID["door"] and int(c[2]) == SID["open"]
            self._path = self._find_nearest_of_type(obs, OID["door"], also_require=is_open, avoid_lava=True)
            self._target = self._path[-1] if self._path else None

        self._tick()
        if not self._path or self._target is None:
            self._done = True
            return A_DONE

        goal_path = self._path[:-1] if len(self._path) >= 2 else self._path
        act = self._step_towards(obs, goal_path)
        if act is not None: return act

        turn = self._face_vec_action(obs, self._target)
        if turn is not None: return turn

        self._done = True
        return A_DONE

class OpenNearestClosedDoor(TimedOption, GridNavMixin):
    """Go adjacent to nearest CLOSED door, face it, then toggle to open."""
    def select_action(self, obs):
        if self._phase == "init":
            self._phase = "route"
            def is_closed(xy, c): return int(c[0]) == OID["door"] and int(c[2]) == SID["closed"]
            self._path = self._find_nearest_of_type(obs, OID["door"], also_require=is_closed, avoid_lava=True)
            self._target = self._path[-1] if self._path else None

        self._tick()
        if not self._path or self._target is None:
            self._done = True
            return A_DONE

        goal_path = self._path[:-1] if len(self._path) >= 2 else self._path
        act = self._step_towards(obs, goal_path)
        if act is not None: return act

        turn = self._face_vec_action(obs, self._target)
        if turn is not None: return turn

        self._done = True
        return A_TOGGLE

class UnlockNearestLockedDoor(TimedOption, GridNavMixin):
    """Go adjacent to nearest LOCKED door, face it, then toggle once (env checks key)."""
    def select_action(self, obs):
        if self._phase == "init":
            self._phase = "route"
            def is_locked(xy, c): return int(c[0]) == OID["door"] and int(c[2]) == SID["locked"]
            self._path = self._find_nearest_of_type(obs, OID["door"], also_require=is_locked, avoid_lava=True)
            self._target = self._path[-1] if self._path else None

        self._tick()
        if not self._path or self._target is None:
            self._done = True
            return A_DONE

        goal_path = self._path[:-1] if len(self._path) >= 2 else self._path
        act = self._step_towards(obs, goal_path)
        if act is not None: return act

        turn = self._face_vec_action(obs, self._target)
        if turn is not None: return turn

        self._done = True
        return A_TOGGLE

class GoToNearestGoal(TimedOption, GridNavMixin):
    """Pathfind directly to the nearest goal tile and step onto it."""
    def select_action(self, obs):
        if self._phase == "init":
            self._phase = "route"
            self._path = self._find_nearest_of_type(obs, OID["goal"], also_require=None, avoid_lava=True)
            self._target = self._path[-1] if self._path else None

        self._tick()

        if not self._path or self._target is None:
            self._done = True
            return A_DONE

        # Follow full path (goal is passable, so include last waypoint)
        act = self._step_towards(obs, self._path)
        if act is not None:
            return act

        # Arrived on goal
        self._done = True
        return A_DONE


class FaceAndStepIntoGoal(TimedOption, GridNavMixin):
    """
    If the goal is directly ahead and forward is safe, step in.
    Otherwise, pathfind to the nearest goal then finish when standing on it.
    """
    def select_action(self, obs):
        if self._phase == "init":
            self._phase = "route"
            self._path = self._find_nearest_of_type(obs, OID["goal"], also_require=None, avoid_lava=True)
            self._target = self._path[-1] if self._path else None

        self._tick()

        # Fast path: step if facing a goal and it's safe to enter
        if self._facing_is_oid(obs, OID["goal"]) and self._forward_is_safe(obs):
            return A_FORWARD

        if not self._path or self._target is None:
            self._done = True
            return A_DONE

        act = self._step_towards(obs, self._path)
        if act is not None:
            return act

        # Arrived on goal
        self._done = True
        return A_DONE
    
# Color/state-specific door adjacency base
class _GoAdjFaceDoorByStateColor(TimedOption, GridNavMixin):
    _need_state: Optional[int] = None
    _color_id: Optional[int] = None
    def select_action(self, obs):
        if self._phase == "init":
            self._phase = "route"
            def match(xy, c):
                return int(c[0]) == OID["door"] and \
                       (self._need_state is None or int(c[2]) == int(self._need_state)) and \
                       (self._color_id is None or int(c[1]) == int(self._color_id))
            self._path = self._find_nearest_of_type(obs, OID["door"], also_require=match, avoid_lava=True)
            self._target = self._path[-1] if self._path else None

        self._tick()
        if not self._path or self._target is None:
            self._done = True
            return A_DONE

        goal_path = self._path[:-1] if len(self._path) >= 2 else self._path
        act = self._step_towards(obs, goal_path)
        if act is not None: return act

        turn = self._face_vec_action(obs, self._target)
        if turn is not None: return turn

        self._done = True
        return A_DONE

# Locked doors by color
class GoAdjFaceLockedRedDoor(_GoAdjFaceDoorByStateColor):    _need_state = SID["locked"]; _color_id = 0
class GoAdjFaceLockedGreenDoor(_GoAdjFaceDoorByStateColor):  _need_state = SID["locked"]; _color_id = 1
class GoAdjFaceLockedBlueDoor(_GoAdjFaceDoorByStateColor):   _need_state = SID["locked"]; _color_id = 2
class GoAdjFaceLockedPurpleDoor(_GoAdjFaceDoorByStateColor): _need_state = SID["locked"]; _color_id = 3
class GoAdjFaceLockedYellowDoor(_GoAdjFaceDoorByStateColor): _need_state = SID["locked"]; _color_id = 4
class GoAdjFaceLockedGreyDoor(_GoAdjFaceDoorByStateColor):   _need_state = SID["locked"]; _color_id = 5

# Closed doors by color
class GoAdjFaceClosedRedDoor(_GoAdjFaceDoorByStateColor):    _need_state = SID["closed"]; _color_id = 0
class GoAdjFaceClosedGreenDoor(_GoAdjFaceDoorByStateColor):  _need_state = SID["closed"]; _color_id = 1
class GoAdjFaceClosedBlueDoor(_GoAdjFaceDoorByStateColor):   _need_state = SID["closed"]; _color_id = 2
class GoAdjFaceClosedPurpleDoor(_GoAdjFaceDoorByStateColor): _need_state = SID["closed"]; _color_id = 3
class GoAdjFaceClosedYellowDoor(_GoAdjFaceDoorByStateColor): _need_state = SID["closed"]; _color_id = 4
class GoAdjFaceClosedGreyDoor(_GoAdjFaceDoorByStateColor):   _need_state = SID["closed"]; _color_id = 5


# ------------------------------------------------------------
# 3) HANDLING KEYS
# ------------------------------------------------------------

class PickupNearestKeyAnyColor(TimedOption, GridNavMixin):
    """Go adjacent to nearest key (any color), face it, then pick up."""
    def select_action(self, obs):
        if self._phase == "init":
            self._phase = "route"
            self._path = self._find_nearest_of_type(obs, OID["key"], also_require=None, avoid_lava=True)
            self._target = self._path[-1] if self._path else None

        self._tick()
        if not self._path or self._target is None:
            self._done = True
            return A_DONE

        goal_path = self._path[:-1] if len(self._path) >= 2 else self._path
        act = self._step_towards(obs, goal_path)
        if act is not None: return act

        turn = self._face_vec_action(obs, self._target)
        if turn is not None: return turn

        self._done = True
        return A_PICKUP

class _PickupNearestKeyByColor(TimedOption, GridNavMixin):
    _color_id: Optional[int] = None
    def select_action(self, obs):
        if self._phase == "init":
            self._phase = "route"
            def match(xy, c):
                return int(c[0]) == OID["key"] and \
                       (self._color_id is None or int(c[1]) == int(self._color_id))
            self._path = self._find_nearest_of_type(obs, OID["key"], also_require=match, avoid_lava=True)
            self._target = self._path[-1] if self._path else None

        self._tick()
        if not self._path or self._target is None:
            self._done = True
            return A_DONE

        goal_path = self._path[:-1] if len(self._path) >= 2 else self._path
        act = self._step_towards(obs, goal_path)
        if act is not None: return act

        turn = self._face_vec_action(obs, self._target)
        if turn is not None: return turn

        self._done = True
        return A_PICKUP

class PickupNearestRedKey(_PickupNearestKeyByColor):    _color_id = 0
class PickupNearestGreenKey(_PickupNearestKeyByColor):  _color_id = 1
class PickupNearestBlueKey(_PickupNearestKeyByColor):   _color_id = 2
class PickupNearestPurpleKey(_PickupNearestKeyByColor): _color_id = 3
class PickupNearestYellowKey(_PickupNearestKeyByColor): _color_id = 4
class PickupNearestGreyKey(_PickupNearestKeyByColor):   _color_id = 5


# ------------------------------------------------------------
# 4) HANDLING BALLS
# ------------------------------------------------------------

class PickupNearestBallAnyColor(TimedOption, GridNavMixin):
    """Go adjacent to nearest ball (any color), face it, then pick up."""
    def select_action(self, obs):
        if self._phase == "init":
            self._phase = "route"
            self._path = self._find_nearest_of_type(obs, OID["ball"], also_require=None, avoid_lava=True)
            self._target = self._path[-1] if self._path else None

        self._tick()
        if not self._path or self._target is None:
            self._done = True
            return A_DONE

        goal_path = self._path[:-1] if len(self._path) >= 2 else self._path
        act = self._step_towards(obs, goal_path)
        if act is not None: return act

        turn = self._face_vec_action(obs, self._target)
        if turn is not None: return turn

        self._done = True
        return A_PICKUP

class _PickupNearestBallByColor(TimedOption, GridNavMixin):
    _color_id: Optional[int] = None
    def select_action(self, obs):
        if self._phase == "init":
            self._phase = "route"
            def match(xy, c):
                return int(c[0]) == OID["ball"] and \
                       (self._color_id is None or int(c[1]) == int(self._color_id))
            self._path = self._find_nearest_of_type(obs, OID["ball"], also_require=match, avoid_lava=True)
            self._target = self._path[-1] if self._path else None

        self._tick()
        if not self._path or self._target is None:
            self._done = True
            return A_DONE

        goal_path = self._path[:-1] if len(self._path) >= 2 else self._path
        act = self._step_towards(obs, goal_path)
        if act is not None: return act

        turn = self._face_vec_action(obs, self._target)
        if turn is not None: return turn

        self._done = True
        return A_PICKUP

class PickupNearestRedBall(_PickupNearestBallByColor):    _color_id = 0
class PickupNearestGreenBall(_PickupNearestBallByColor):  _color_id = 1
class PickupNearestBlueBall(_PickupNearestBallByColor):   _color_id = 2
class PickupNearestPurpleBall(_PickupNearestBallByColor): _color_id = 3
class PickupNearestYellowBall(_PickupNearestBallByColor): _color_id = 4
class PickupNearestGreyBall(_PickupNearestBallByColor):   _color_id = 5


# ------------------------------------------------------------
# 5) HANDLING BOXES (open / toggle; pickup revealed)
# ------------------------------------------------------------

class GoAdjFaceNearestBox(TimedOption, GridNavMixin):
    """Navigate adjacent to the nearest box and face it."""
    def select_action(self, obs):
        if self._phase == "init":
            self._phase = "route"
            self._path = self._find_nearest_of_type(obs, OID["box"], also_require=None, avoid_lava=True)
            self._target = self._path[-1] if self._path else None

        self._tick()
        if not self._path or self._target is None:
            self._done = True
            return A_DONE

        goal_path = self._path[:-1] if len(self._path) >= 2 else self._path
        act = self._step_towards(obs, goal_path)
        if act is not None: return act

        turn = self._face_vec_action(obs, self._target)
        if turn is not None: return turn

        self._done = True
        return A_DONE

class OpenNearestBox(TimedOption, GridNavMixin):
    """Navigate adjacent to the nearest box, face it, and toggle once to open."""
    def select_action(self, obs):
        if self._phase == "init":
            self._phase = "route"
            self._path = self._find_nearest_of_type(obs, OID["box"], also_require=None, avoid_lava=True)
            self._target = self._path[-1] if self._path else None

        self._tick()
        if not self._path or self._target is None:
            self._done = True
            return A_DONE

        goal_path = self._path[:-1] if len(self._path) >= 2 else self._path
        act = self._step_towards(obs, goal_path)
        if act is not None: return act

        turn = self._face_vec_action(obs, self._target)
        if turn is not None: return turn

        self._done = True
        return A_TOGGLE

class OpenNearestBoxAndPickupIfItemRevealed(TimedOption, GridNavMixin):
    """
    Open the nearest box; if a key/ball is revealed in front, pick it up.
    """
    def select_action(self, obs):
        if self._phase == "init":
            self._phase = "route"
            self._path = self._find_nearest_of_type(obs, OID["box"], also_require=None, avoid_lava=True)
            self._target = self._path[-1] if self._path else None

        self._tick()

        if self._phase == "route":
            if not self._path or self._target is None:
                self._done = True
                return A_DONE
            goal_path = self._path[:-1] if len(self._path) >= 2 else self._path
            act = self._step_towards(obs, goal_path)
            if act is not None: return act
            self._phase = "face"

        if self._phase == "face":
            turn = self._face_vec_action(obs, self._target)
            if turn is not None: return turn
            self._phase = "toggle"

        if self._phase == "toggle":
            self._phase = "pickup"
            return A_TOGGLE

        if self._phase == "pickup":
            ahead = self._ahead_cell(obs)
            if ahead is not None and int(ahead[0]) in (OID.get("key", -1), OID.get("ball", -1)):
                self._done = True
                return A_PICKUP
            self._done = True
            return A_DONE

        self._done = True
        return A_DONE

class DropHere(TimedOption, GridNavMixin):
    """Single drop attempt on current tile."""
    def select_action(self, obs):
        self._tick()
        if self.step_counter == 1:
            return A_DROP
        self._done = True
        return A_DONE


# ============================================================
# Registry (ordered by your categories: 2→5)
# ============================================================

ALL_OPTIONS = [
    # 2) Doors
    GoAdjFaceNearestClosedDoor,
    GoAdjFaceNearestLockedDoor,
    GoAdjFaceNearestOpenDoor,
    OpenNearestClosedDoor,
    UnlockNearestLockedDoor,
    GoAdjFaceLockedRedDoor,
    GoAdjFaceLockedGreenDoor,
    GoAdjFaceLockedBlueDoor,
    GoAdjFaceLockedPurpleDoor,
    GoAdjFaceLockedYellowDoor,
    GoAdjFaceLockedGreyDoor,
    GoAdjFaceClosedRedDoor,
    GoAdjFaceClosedGreenDoor,
    GoAdjFaceClosedBlueDoor,
    GoAdjFaceClosedPurpleDoor,
    GoAdjFaceClosedYellowDoor,
    GoAdjFaceClosedGreyDoor,

    # 3) Keys
    PickupNearestKeyAnyColor,
    PickupNearestRedKey,
    PickupNearestGreenKey,
    PickupNearestBlueKey,
    PickupNearestPurpleKey,
    PickupNearestYellowKey,
    PickupNearestGreyKey,

    # 4) Balls
    PickupNearestBallAnyColor,
    PickupNearestRedBall,
    PickupNearestGreenBall,
    PickupNearestBlueBall,
    PickupNearestPurpleBall,
    PickupNearestYellowBall,
    PickupNearestGreyBall,

    # 5) Boxes
    GoAdjFaceNearestBox,
    OpenNearestBox,
    OpenNearestBoxAndPickupIfItemRevealed,
    DropHere,
    
    # 6) Goals
    FaceAndStepIntoGoal,
    GoToNearestGoal,
]

def create_all_options(option_len=10):
    """Convenience factory."""
    return [opt(option_len) for opt in ALL_OPTIONS]