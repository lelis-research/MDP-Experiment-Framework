# options_minigrid_toolkit_auto.py
# Fully aligned with (x, y) coordinates = (column, row).
# All array access to obs["image"] uses img[x, y], while positions and motion use (x, y).

from __future__ import annotations

from collections import deque
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from minigrid.core.constants import (
    COLORS,
    COLOR_NAMES,
    COLOR_TO_IDX,
    IDX_TO_COLOR,
    OBJECT_TO_IDX as OID,
    IDX_TO_OBJECT,
    STATE_TO_IDX as SID,
    DIR_TO_VEC,  # (dx, dy) where x = columns (left→right), y = rows (top→down)
)

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
    The image tensor is still indexed as img[x, y] for numpy correctness.
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
        if oid in (OID["box"], OID["ball"], OID["key"]):
            return False
        # empty, floor, goal, agent => traversable
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
    
    def _find_nearest_of_object(
        self,
        obs,
        obj_id: int,
        color_id: Optional[int] = None,
        state_id: Optional[int] = None,
        also_require: Optional[Callable[[Tuple[int, int], np.ndarray], bool]] = None,
        avoid_lava: bool = True,
    ) -> Optional[list[np.ndarray]]:
        """
        Find a BFS path to the nearest cell matching:
        - object id == obj_id
        - (optional) color id == color_id
        - (optional) state id == state_id
        plus any additional predicate `also_require(xy, cell)`.

        Returns:
            path: list of np.array([x, y]) from start to goal (inclusive), or None.
        """
        img = self._img(obs)
        start = self._find_agent(obs)  # np.array([x,y])

        obj_id = int(obj_id)
        color_id = None if color_id is None else int(color_id)
        state_id = None if state_id is None else int(state_id)

        def is_goal(xy: Tuple[int, int]) -> bool:
            # Your convention: img[x, y]
            c = img[int(xy[0]), int(xy[1])]  # [obj, color, state]
            if int(c[0]) != obj_id:
                return False
            if color_id is not None and int(c[1]) != color_id:
                return False
            if state_id is not None and int(c[2]) != state_id:
                return False
            return True if also_require is None else bool(also_require(xy, c))

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

