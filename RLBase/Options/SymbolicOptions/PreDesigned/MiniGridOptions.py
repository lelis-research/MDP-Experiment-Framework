# options_minigrid_library.py
#
# Purpose
# -------
# A comprehensive, ready-to-drop-in library of temporally-extended MiniGrid "Options"
# (skills) that operate purely from the fully observable `obs` dict (obs["image"], obs["direction"]).
#
# Option Categories
# -----------------
# 1) Primitive one-step options:
#    - Turn Left / Turn Right / Move Forward / Pickup / Drop / Toggle / Done
#
# 2) Navigation / positioning options (BFS + safe movement):
#    - Go to nearest object (goal/key/ball/box/door by state/color)
#    - Face a target object (door/key/ball/box/goal)
#    - Step away from lava / avoid hazards
#    - Explore to nearest unseen / frontier (if present in map)
#
# 3) Manipulation options:
#    - Pickup nearest key/ball/box
#    - Toggle a faced door/box/ball
#    - Open a closed door (toggle)
#
# 4) Unlocking / composite skills:
#    - Get key of a color
#    - Go to door of a color (locked/closed)
#    - Unlock locked door (requires matching color key presence in grid)
#    - Fetch key then unlock door
#
# Termination Semantics (Explicit)
# --------------------------------
# Every option terminates explicitly under one of:
#   - Success: desired condition achieved (e.g., standing on goal, object picked up, door opened)
#   - Failure: cannot find target/path, unsafe/blocked, missing prerequisites, inconsistent state
#   - Timeout: exceeds `max_steps` budget (default per option)
#
# Safety Constraints
# ------------------
# - No infinite loops: each option enforces `max_steps`.
# - Deterministic behavior: no random choices unless explicitly using seeded RNG.
# - No mutation of global state: all state is kept on the option instance.
#
# Notes
# -----
# - Coordinates are aligned with the provided helper toolkit:
#     world coords are (x, y) = (column, row)
#     image access uses img[x, y] throughout this toolkit.
# - This file assumes the helper toolkit (constants, GridNavMixin, action indices)
#   is available via: `from .MiniGridHelper import *`
#
# Registration
# ------------
# Uses the existing registry decorator exactly as in examples:
#   from ....registry import register_option
# and decorates each class with @register_option.

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX as OID, COLOR_TO_IDX, STATE_TO_IDX as SID

from ...Base import BaseOption
from .MiniGridHelper import *  # provides GridNavMixin + action constants/helpers
from ....registry import register_option

from minigrid.core.constants import COLOR_NAMES


# ============================================================
# Shared helpers (internal to this module; do not modify mixins)
# ============================================================

class _MiniGridOptionBase(BaseOption, GridNavMixin):
    """
    Internal shared base for consistent counters, max_steps, and small utilities.
    Does not change any external API; options still inherit BaseOption.
    """

    def __init__(self, option_id: Optional[str], hyper_params=None, device: str = "cpu", max_steps: int = 10):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device)
        self.counter = 0
        self.max_steps = int(max_steps)
        self._failed = False

    def _tick(self) -> None:
        self.counter += 1

    def _timeout(self) -> bool:
        return self.counter >= self.max_steps

    def _reset_internal(self):
        self.counter = 0
        self._failed = False

    def reset(self, seed=None):
        super().reset(seed)
        self._reset_internal()

    # ---------- Small read helpers ----------
    def _standing_on_oid_color(self, obs, oid: int, color: Optional[int] = None) -> bool:
        img = self._img(obs)
        agent = self._find_agent(obs)
        c = img[int(agent[0]), int(agent[1])]
        if int(c[0]) != int(oid):
            return False
        if color is None:
            return True
        return int(c[1]) == int(color)

    def _ahead_has_oid(self, obs, oid: int) -> bool:
        c = self._ahead_cell(obs)
        return c is not None and int(c[0]) == int(oid)

    def _ahead_has_oid_color(self, obs, oid: int, color: Optional[int] = None) -> bool:
        c = self._ahead_cell(obs)
        if c is None or int(c[0]) != int(oid):
            return False
        if color is None:
            return True
        return int(c[1]) == int(color)

    def _find_any_oid_exists(self, obs, oid: int, color: Optional[int] = None) -> bool:
        img = self._img(obs)
        xs, ys = np.where(img[..., 0] == int(oid))
        if len(xs) == 0:
            return False
        if color is None:
            return True
        for x, y in zip(xs.tolist(), ys.tolist()):
            if int(img[int(x), int(y), 1]) == int(color):
                return True
        return False

    def _path_to_first_match(
        self,
        obs,
        goal_test_xy: Callable[[Tuple[int, int]], bool],
        avoid_lava: bool = True,
        passable_override: Optional[Callable[[Tuple[int, int]], bool]] = None,
    ):
        start = self._find_agent(obs)
        return self._bfs_path(obs, start, goal_test_xy, avoid_lava=avoid_lava, passable_override=passable_override)

    def _face_target_xy_or_turn(self, obs, target_xy: np.ndarray) -> int:
        turn = self._face_vec_action(obs, target_xy)
        return int(A_LEFT if turn is None else turn)


# ============================================================
# Primitive atomic options (one-step actions)
# ============================================================

@register_option
class ActionLeft(BaseOption):
    """
    Atomic option: turn left once.

    Preconditions:
      - None.

    Termination:
      - Success after exactly 1 primitive step.

    Failure:
      - None (always completes).
    """
    def __init__(self, option_id: Optional[str] = "Turn Left", hyper_params=None, device="cpu"):
        super().__init__(option_id, hyper_params, device)
        self.counter = 0

    def reward_func(self, observation: Any) -> float:
        return 0.0

    def select_action(self, observation):
        self.counter += 1
        return int(A_LEFT)

    def is_terminated(self, observation):
        if self.counter >= 1:
            self.counter = 0
            return True
        return False

    def reset(self, seed=None):
        super().reset(seed)
        self.counter = 0


@register_option
class ActionRight(BaseOption):
    """
    Atomic option: turn right once.

    Preconditions:
      - None.

    Termination:
      - Success after exactly 1 primitive step.

    Failure:
      - None (always completes).
    """
    def __init__(self, option_id: Optional[str] = "Turn Right", hyper_params=None, device="cpu"):
        super().__init__(option_id, hyper_params, device)
        self.counter = 0

    def reward_func(self, observation: Any) -> float:
        return 0.0

    def select_action(self, observation):
        self.counter += 1
        return int(A_RIGHT)

    def is_terminated(self, observation):
        if self.counter >= 1:
            self.counter = 0
            return True
        return False

    def reset(self, seed=None):
        super().reset(seed)
        self.counter = 0


@register_option
class ActionForward(BaseOption):
    """
    Atomic option: move forward once.

    Preconditions:
      - None (environment may block; option still terminates).

    Termination:
      - Success after exactly 1 primitive step.

    Failure:
      - None (always completes).
    """
    def __init__(self, option_id: Optional[str] = "Move Forward", hyper_params=None, device="cpu"):
        super().__init__(option_id, hyper_params, device)
        self.counter = 0

    def reward_func(self, observation: Any) -> float:
        return 0.0

    def select_action(self, observation):
        self.counter += 1
        return int(A_FORWARD)

    def is_terminated(self, observation):
        if self.counter >= 1:
            self.counter = 0
            return True
        return False

    def reset(self, seed=None):
        super().reset(seed)
        self.counter = 0


@register_option
class ActionPickup(BaseOption):
    """
    Atomic option: pickup once.

    Preconditions:
      - None (pickup may fail if nothing to pick up).

    Termination:
      - Success after exactly 1 primitive step.

    Failure:
      - None (always completes).
    """
    def __init__(self, option_id: Optional[str] = "Pickup", hyper_params=None, device="cpu"):
        super().__init__(option_id, hyper_params, device)
        self.counter = 0

    def reward_func(self, observation: Any) -> float:
        return 0.0

    def select_action(self, observation):
        self.counter += 1
        return int(A_PICKUP)

    def is_terminated(self, observation):
        if self.counter >= 1:
            self.counter = 0
            return True
        return False

    def reset(self, seed=None):
        super().reset(seed)
        self.counter = 0


@register_option
class ActionDrop(BaseOption):
    """
    Atomic option: drop once.

    Preconditions:
      - None (drop may fail if nothing carried).

    Termination:
      - Success after exactly 1 primitive step.

    Failure:
      - None (always completes).
    """
    def __init__(self, option_id: Optional[str] = "Drop", hyper_params=None, device="cpu"):
        super().__init__(option_id, hyper_params, device)
        self.counter = 0

    def reward_func(self, observation: Any) -> float:
        return 0.0

    def select_action(self, observation):
        self.counter += 1
        return int(A_DROP)

    def is_terminated(self, observation):
        if self.counter >= 1:
            self.counter = 0
            return True
        return False

    def reset(self, seed=None):
        super().reset(seed)
        self.counter = 0


@register_option
class ActionToggle(BaseOption):
    """
    Atomic option: toggle once.

    Preconditions:
      - None (toggle may or may not have effect).

    Termination:
      - Success after exactly 1 primitive step.

    Failure:
      - None (always completes).
    """
    def __init__(self, option_id: Optional[str] = "Toggle", hyper_params=None, device="cpu"):
        super().__init__(option_id, hyper_params, device)
        self.counter = 0

    def reward_func(self, observation: Any) -> float:
        return 0.0

    def select_action(self, observation):
        self.counter += 1
        return int(A_TOGGLE)

    def is_terminated(self, observation):
        if self.counter >= 1:
            self.counter = 0
            return True
        return False

    def reset(self, seed=None):
        super().reset(seed)
        self.counter = 0


@register_option
class ActionDone(BaseOption):
    """
    Atomic option: done once.

    Preconditions:
      - None.

    Termination:
      - Success after exactly 1 primitive step.

    Failure:
      - None (always completes).
    """
    def __init__(self, option_id: Optional[str] = "Done", hyper_params=None, device="cpu"):
        super().__init__(option_id, hyper_params, device)
        self.counter = 0

    def reward_func(self, observation: Any) -> float:
        return 0.0

    def select_action(self, observation):
        self.counter += 1
        return int(A_DONE)

    def is_terminated(self, observation):
        if self.counter >= 1:
            self.counter = 0
            return True
        return False

    def reset(self, seed=None):
        super().reset(seed)
        self.counter = 0


# ============================================================
# Navigation / positioning options
# ============================================================

@register_option
class GoToNearestObjectOption(_MiniGridOptionBase):
    """
    Navigate to the nearest object of a given type (optionally requiring a color and/or door state).
    Uses BFS over passable terrain, avoids lava by default.

    Parameters:
      - obj_name: str in OBJECT_TO_IDX keys (e.g., "goal", "key", "door", "ball", "box")
      - color_name: Optional[str] in COLOR_TO_IDX keys (e.g., "red") if object has color
      - door_state: Optional[str] in STATE_TO_IDX keys ("open"/"closed"/"locked") only relevant for doors
      - max_steps: int step budget (default 20)

    Preconditions:
      - A path exists to some matching target tile (or to a neighbor if target is non-passable).

    Termination:
      - Success: agent is standing on the target tile (for passable targets)
      - Failure: no path exists / cannot find target
      - Timeout: counter >= max_steps

    Failure Modes:
      - Target not present in observation
      - BFS cannot reach a suitable target
    """

    def __init__(
        self,
        option_id: Optional[str] = "go_to_nearest_object",
        hyper_params=None,
        device: str = "cpu",
        obj_name: str = "goal",
        color_name: Optional[str] = None,
        door_state: Optional[str] = None,
        max_steps: int = 20,
        avoid_lava: bool = True,
    ):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, max_steps=max_steps)
        self.obj_name = str(obj_name)
        self.color_name = None if color_name is None else str(color_name)
        self.door_state = None if door_state is None else str(door_state)
        self.avoid_lava = bool(avoid_lava)

    def _target_oid(self) -> int:
        return int(OID[self.obj_name])

    def _target_color_idx(self) -> Optional[int]:
        if self.color_name is None:
            return None
        return int(COLOR_TO_IDX[self.color_name])

    def _target_state_idx(self) -> Optional[int]:
        if self.door_state is None:
            return None
        return int(SID[self.door_state])

    def _match_cell(self, cell: np.ndarray) -> bool:
        oid = int(cell[0])
        if oid != self._target_oid():
            return False
        col = self._target_color_idx()
        if col is not None and int(cell[1]) != int(col):
            return False
        if oid == int(OID["door"]):
            st = self._target_state_idx()
            if st is not None and int(cell[2]) != int(st):
                return False
        return True

    def _path_to_target(self, obs):
        img = self._img(obs)
        start = self._find_agent(obs)

        def is_goal(xy: Tuple[int, int]) -> bool:
            c = img[int(xy[0]), int(xy[1])]
            return self._match_cell(c)

        return self._bfs_path(obs, start, is_goal, avoid_lava=self.avoid_lava)

    def _standing_on_target(self, obs) -> bool:
        img = self._img(obs)
        agent = self._find_agent(obs)
        c = img[int(agent[0]), int(agent[1])]
        return self._match_cell(c)

    def can_initiate(self, observation: Any) -> bool:
        path = self._path_to_target(observation)
        return path is not None and len(path) >= 2

    def should_initiate(self, observation: Any) -> bool:
        return self.can_initiate(observation)

    def reward_func(self, observation: Any) -> float:
        return 1.0 if self._standing_on_target(observation) else 0.0

    def select_action(self, observation: Any):
        if self._standing_on_target(observation):
            return int(A_DONE)

        path = self._path_to_target(observation)
        if path is None or len(path) < 2:
            self._failed = True
            return int(A_LEFT)

        act = self._step_towards(observation, path)
        self._tick()
        return int(A_LEFT if act is None else act)

    def is_terminated(self, observation: Any) -> bool:
        terminated = self._standing_on_target(observation) or self._failed or (self._path_to_target(observation) is None) or self._timeout()
        if terminated:
            self._reset_internal()
        return terminated


@register_option
class FaceNearestObjectOption(_MiniGridOptionBase):
    """
    Rotate to face the nearest object of a given type/color/state.
    Does not move forward unless necessary for alignment (it will only rotate).

    Parameters:
      - obj_name: object type name
      - color_name: optional color filter
      - door_state: optional state filter for doors
      - max_steps: step budget (default 8)

    Preconditions:
      - A matching object exists somewhere in the observation.

    Termination:
      - Success: the tile directly ahead is a matching object
      - Failure: no matching object present / cannot compute path
      - Timeout: counter >= max_steps
    """

    def __init__(
        self,
        option_id: Optional[str] = "face_nearest_object",
        hyper_params=None,
        device: str = "cpu",
        obj_name: str = "door",
        color_name: Optional[str] = None,
        door_state: Optional[str] = None,
        max_steps: int = 8,
        avoid_lava: bool = True,
    ):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, max_steps=max_steps)
        self.obj_name = str(obj_name)
        self.color_name = None if color_name is None else str(color_name)
        self.door_state = None if door_state is None else str(door_state)
        self.avoid_lava = bool(avoid_lava)

    def _target_oid(self) -> int:
        return int(OID[self.obj_name])

    def _target_color_idx(self) -> Optional[int]:
        if self.color_name is None:
            return None
        return int(COLOR_TO_IDX[self.color_name])

    def _target_state_idx(self) -> Optional[int]:
        if self.door_state is None:
            return None
        return int(SID[self.door_state])

    def _match_cell(self, cell: np.ndarray) -> bool:
        oid = int(cell[0])
        if oid != self._target_oid():
            return False
        col = self._target_color_idx()
        if col is not None and int(cell[1]) != int(col):
            return False
        if oid == int(OID["door"]):
            st = self._target_state_idx()
            if st is not None and int(cell[2]) != int(st):
                return False
        return True

    def _path_to_target(self, obs):
        img = self._img(obs)
        start = self._find_agent(obs)

        def is_goal(xy: Tuple[int, int]) -> bool:
            c = img[int(xy[0]), int(xy[1])]
            return self._match_cell(c)

        return self._bfs_path(obs, start, is_goal, avoid_lava=self.avoid_lava)

    def can_initiate(self, observation: Any) -> bool:
        return self._find_any_oid_exists(
            observation,
            oid=self._target_oid(),
            color=self._target_color_idx(),
        )

    def should_initiate(self, observation: Any) -> bool:
        return self.can_initiate(observation)

    def reward_func(self, observation: Any) -> float:
        return 1.0 if self._ahead_has_oid_color(observation, self._target_oid(), self._target_color_idx()) else 0.0

    def select_action(self, observation: Any):
        if self._ahead_has_oid_color(observation, self._target_oid(), self._target_color_idx()):
            return int(A_DONE)

        path = self._path_to_target(observation)
        if path is None or len(path) < 2:
            self._failed = True
            return int(A_LEFT)

        # Face the final target coordinate
        target_xy = path[-1]
        act = self._face_vec_action(observation, target_xy)
        self._tick()
        return int(A_LEFT if act is None else act)

    def is_terminated(self, observation: Any) -> bool:
        success = self._ahead_has_oid_color(observation, self._target_oid(), self._target_color_idx())
        terminated = success or self._failed or (not self.can_initiate(observation)) or self._timeout()
        if terminated:
            self._reset_internal()
        return terminated


@register_option
class AvoidLavaStepOption(_MiniGridOptionBase):
    """
    Single-step safety option: if the cell ahead is lava or out-of-bounds/impassable, turn away;
    otherwise step forward. This is a "reflex" skill useful inside higher-level policies.

    Preconditions:
      - None.

    Termination:
      - Always terminates after <= 1 step (atomic safety move).

    Failure:
      - None.
    """

    def __init__(self, option_id: Optional[str] = "avoid_lava_step", hyper_params=None, device: str = "cpu"):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, max_steps=1)

    def can_initiate(self, observation: Any) -> bool:
        return True

    def should_initiate(self, observation: Any) -> bool:
        c = self._ahead_cell(observation)
        if c is None:
            return True
        return int(c[0]) == int(OID["lava"]) or (not self._forward_is_safe(observation))

    def reward_func(self, observation: Any) -> float:
        # Small shaping: reward if not facing lava and forward is safe
        c = self._ahead_cell(observation)
        if c is None:
            return 0.0
        if int(c[0]) == int(OID["lava"]):
            return 0.0
        return 1.0 if self._forward_is_safe(observation) else 0.0

    def select_action(self, observation: Any):
        self._tick()

        c = self._ahead_cell(observation)
        if c is None:
            return int(A_LEFT)
        if int(c[0]) == int(OID["lava"]):
            return int(A_LEFT)

        if not self._forward_is_safe(observation):
            return int(A_LEFT)

        return int(A_FORWARD)

    def is_terminated(self, observation: Any) -> bool:
        terminated = self._timeout()
        if terminated:
            self._reset_internal()
        return terminated


@register_option
class ExploreNearestUnseenOption(_MiniGridOptionBase):
    """
    Navigate to the nearest 'unseen' tile (OID['unseen']) if present, using BFS over passable cells.
    This is useful as an exploration fallback in maps that include unseen regions.

    Preconditions:
      - There exists at least one unseen tile, and a passable path to it or to a neighbor.

    Termination:
      - Success: agent is standing on an unseen tile (rare) OR no unseen remains reachable
      - Failure: no unseen exists
      - Timeout: counter >= max_steps

    Failure Modes:
      - No unseen tiles in observation
      - BFS cannot reach any unseen (walled off)
    """

    def __init__(self, option_id: Optional[str] = "explore_nearest_unseen", hyper_params=None, device: str = "cpu", max_steps: int = 25):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, max_steps=max_steps)

    def _path_to_unseen(self, obs):
        img = self._img(obs)
        start = self._find_agent(obs)

        def is_goal(xy: Tuple[int, int]) -> bool:
            c = img[int(xy[0]), int(xy[1])]
            return int(c[0]) == int(OID["unseen"])

        # allow stepping into unseen as goal (BFS already accepts goal even if not passable)
        return self._bfs_path(obs, start, is_goal, avoid_lava=True)

    def can_initiate(self, observation: Any) -> bool:
        img = self._img(observation)
        xs, ys = np.where(img[..., 0] == int(OID["unseen"]))
        if len(xs) == 0:
            return False
        path = self._path_to_unseen(observation)
        return path is not None and len(path) >= 2

    def should_initiate(self, observation: Any) -> bool:
        return self.can_initiate(observation)

    def reward_func(self, observation: Any) -> float:
        # Reward if unseen still exists and we're making progress (simple: 1 if option still viable)
        return 1.0 if self.can_initiate(observation) else 0.0

    def select_action(self, observation: Any):
        path = self._path_to_unseen(observation)
        if path is None or len(path) < 2:
            self._failed = True
            return int(A_LEFT)

        act = self._step_towards(observation, path)
        self._tick()
        return int(A_LEFT if act is None else act)

    def is_terminated(self, observation: Any) -> bool:
        # success if no unseen left reachable or timeout
        terminated = self._failed or (not self.can_initiate(observation)) or self._timeout()
        if terminated:
            self._reset_internal()
        return terminated


# ============================================================
# Manipulation options: pickup / toggle doors/objects
# ============================================================

@register_option
class PickupNearestObjectOption(_MiniGridOptionBase):
    """
    Composite option: Navigate adjacent to an object and pick it up.
    Works for pickable objects: key, ball, box.

    Parameters:
      - obj_name: one of "key", "ball", "box"
      - color_name: optional color filter
      - max_steps: default 20

    Preconditions:
      - A target exists, and a path exists to a cell adjacent to it with a valid facing move.

    Termination:
      - Success: after issuing A_PICKUP while the target is directly ahead
      - Failure: target not found / path not found / timeout

    Failure Modes:
      - Target missing or unreachable
      - Cannot align to face target
    """

    def __init__(
        self,
        option_id: Optional[str] = "pickup_nearest_object",
        hyper_params=None,
        device: str = "cpu",
        obj_name: str = "key",
        color_name: Optional[str] = None,
        max_steps: int = 20,
        avoid_lava: bool = True,
    ):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, max_steps=max_steps)
        self.obj_name = str(obj_name)
        self.color_name = None if color_name is None else str(color_name)
        self.avoid_lava = bool(avoid_lava)

    def _target_oid(self) -> int:
        return int(OID[self.obj_name])

    def _target_color_idx(self) -> Optional[int]:
        if self.color_name is None:
            return None
        return int(COLOR_TO_IDX[self.color_name])

    def _find_target_path(self, obs):
        return self._find_nearest_of_type(
            obs,
            obj_id_filter=self._target_oid(),
            also_require=(None if self._target_color_idx() is None else (lambda xy, c: int(c[1]) == int(self._target_color_idx()))),
            avoid_lava=self.avoid_lava,
        )

    def _is_facing_target(self, obs) -> bool:
        return self._ahead_has_oid_color(obs, self._target_oid(), self._target_color_idx())

    def can_initiate(self, observation: Any) -> bool:
        path = self._find_target_path(observation)
        return path is not None and len(path) >= 2

    def should_initiate(self, observation: Any) -> bool:
        return self.can_initiate(observation)

    def reward_func(self, observation: Any) -> float:
        # Reward when in position to pickup (facing target)
        return 1.0 if self._is_facing_target(observation) else 0.0

    def select_action(self, observation: Any):
        if self._is_facing_target(observation):
            self._tick()
            return int(A_PICKUP)

        path = self._find_target_path(observation)
        if path is None or len(path) < 2:
            self._failed = True
            return int(A_LEFT)

        act = self._step_towards(observation, path)
        self._tick()
        return int(A_LEFT if act is None else act)

    def is_terminated(self, observation: Any) -> bool:
        # Success assumed after attempting pickup when facing target
        terminated = self._failed or self._timeout()
        # Also terminate if target no longer exists (picked up by someone else / disappeared)
        if not self._find_any_oid_exists(observation, self._target_oid(), self._target_color_idx()):
            terminated = True
        if terminated:
            self._reset_internal()
        return terminated


@register_option
class ToggleFacingDoorOption(_MiniGridOptionBase):
    """
    Toggle the door directly in front of the agent (one step), typically to open a closed door or unlock if possible.

    Parameters:
      - color_name: optional door color constraint
      - require_state: optional state ("closed"/"locked"/"open") required before toggling
      - max_steps: default 2 (face/turn is not handled here; this expects you're already facing)

    Preconditions:
      - Agent is facing a door (optionally matching color/state).

    Termination:
      - Success: after issuing A_TOGGLE once while precondition holds
      - Failure: precondition not met
      - Timeout: max_steps
    """

    def __init__(
        self,
        option_id: Optional[str] = "toggle_facing_door",
        hyper_params=None,
        device: str = "cpu",
        color_name: Optional[str] = None,
        require_state: Optional[str] = None,
        max_steps: int = 2,
    ):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, max_steps=max_steps)
        self.color_name = None if color_name is None else str(color_name)
        self.require_state = None if require_state is None else str(require_state)

    def _door_color_idx(self) -> Optional[int]:
        if self.color_name is None:
            return None
        return int(COLOR_TO_IDX[self.color_name])

    def _require_state_idx(self) -> Optional[int]:
        if self.require_state is None:
            return None
        return int(SID[self.require_state])

    def _precond(self, obs) -> bool:
        c = self._ahead_cell(obs)
        if c is None or int(c[0]) != int(OID["door"]):
            return False
        col = self._door_color_idx()
        if col is not None and int(c[1]) != int(col):
            return False
        st = self._require_state_idx()
        if st is not None and int(c[2]) != int(st):
            return False
        return True

    def can_initiate(self, observation: Any) -> bool:
        return self._precond(observation)

    def should_initiate(self, observation: Any) -> bool:
        return self._precond(observation)

    def reward_func(self, observation: Any) -> float:
        # reward if door becomes open
        c = self._ahead_cell(observation)
        if c is None or int(c[0]) != int(OID["door"]):
            return 0.0
        return 1.0 if int(c[2]) == int(SID["open"]) else 0.0

    def select_action(self, observation: Any):
        self._tick()
        if not self._precond(observation):
            self._failed = True
            return int(A_LEFT)
        return int(A_TOGGLE)

    def is_terminated(self, observation: Any) -> bool:
        terminated = self._failed or self._timeout()
        if terminated:
            self._reset_internal()
        return terminated


@register_option
class OpenNearestClosedDoorOption(_MiniGridOptionBase):
    """
    Navigate to the nearest CLOSED door (optionally by color), face it, and toggle to open.

    Parameters:
      - color_name: optional door color
      - max_steps: default 25

    Preconditions:
      - A closed door exists and is reachable.

    Termination:
      - Success: door in front becomes open (or any matching door on path becomes open after toggle)
      - Failure: no closed door / unreachable / timeout
    """

    def __init__(
        self,
        option_id: Optional[str] = "open_nearest_closed_door",
        hyper_params=None,
        device: str = "cpu",
        color_name: Optional[str] = None,
        max_steps: int = 25,
        avoid_lava: bool = True,
    ):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, max_steps=max_steps)
        self.color_name = None if color_name is None else str(color_name)
        self.avoid_lava = bool(avoid_lava)
        self._toggled_once = False

    def _door_color_idx(self) -> Optional[int]:
        if self.color_name is None:
            return None
        return int(COLOR_TO_IDX[self.color_name])

    def _path_to_closed_door(self, obs):
        return self._find_nearest_of_type(
            obs,
            obj_id_filter=OID["door"],
            also_require=lambda xy, c: (int(c[2]) == int(SID["closed"])) and (self._door_color_idx() is None or int(c[1]) == int(self._door_color_idx())),
            avoid_lava=self.avoid_lava,
        )

    def _facing_closed_door(self, obs) -> bool:
        c = self._ahead_cell(obs)
        if c is None:
            return False
        if int(c[0]) != int(OID["door"]):
            return False
        if int(c[2]) != int(SID["closed"]):
            return False
        col = self._door_color_idx()
        if col is not None and int(c[1]) != int(col):
            return False
        return True

    def _facing_open_door(self, obs) -> bool:
        c = self._ahead_cell(obs)
        if c is None:
            return False
        if int(c[0]) != int(OID["door"]):
            return False
        if int(c[2]) != int(SID["open"]):
            return False
        col = self._door_color_idx()
        if col is not None and int(c[1]) != int(col):
            return False
        return True

    def can_initiate(self, observation: Any) -> bool:
        path = self._path_to_closed_door(observation)
        return path is not None and len(path) >= 2

    def should_initiate(self, observation: Any) -> bool:
        return self.can_initiate(observation)

    def reward_func(self, observation: Any) -> float:
        return 1.0 if self._facing_open_door(observation) else 0.0

    def select_action(self, observation: Any):
        # If we are facing a closed door, toggle it
        if self._facing_closed_door(observation):
            self._toggled_once = True
            self._tick()
            return int(A_TOGGLE)

        # If we are facing an open door and we already toggled, we are done
        if self._toggled_once and self._facing_open_door(observation):
            self._tick()
            return int(A_DONE)

        path = self._path_to_closed_door(observation)
        if path is None or len(path) < 2:
            self._failed = True
            return int(A_LEFT)

        # Move toward the door tile; once adjacent, _step_towards will bring us next to it, then we must face it.
        act = self._step_towards(observation, path)
        self._tick()
        return int(A_LEFT if act is None else act)

    def is_terminated(self, observation: Any) -> bool:
        success = self._toggled_once and self._facing_open_door(observation)
        terminated = success or self._failed or (self._path_to_closed_door(observation) is None) or self._timeout()
        if terminated:
            self._toggled_once = False
            self._reset_internal()
        return terminated

    def reset(self, seed=None):
        super().reset(seed)
        self._toggled_once = False


# ============================================================
# Goal-oriented options (colored goals, general goal)
# ============================================================

@register_option
class GoToGoalOption(_MiniGridOptionBase):
    """
    Navigate to the nearest goal (any color).

    Preconditions:
      - A reachable goal exists.

    Termination:
      - Success: standing on a goal tile
      - Failure: no reachable goal
      - Timeout: max_steps
    """

    def __init__(self, option_id: Optional[str] = "go_to_goal", hyper_params=None, device: str = "cpu", max_steps: int = 20):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, max_steps=max_steps)

    def _path_to_goal(self, obs):
        return self._find_nearest_of_type(obs, obj_id_filter=OID["goal"], also_require=None, avoid_lava=True)

    def can_initiate(self, observation: Any) -> bool:
        path = self._path_to_goal(observation)
        return path is not None and len(path) >= 2

    def should_initiate(self, observation: Any) -> bool:
        return self.can_initiate(observation)

    def reward_func(self, observation: Any) -> float:
        return 1.0 if self._standing_on_oid_color(observation, OID["goal"], None) else 0.0

    def select_action(self, observation: Any):
        if self._standing_on_oid_color(observation, OID["goal"], None):
            return int(A_DONE)

        path = self._path_to_goal(observation)
        if path is None or len(path) < 2:
            self._failed = True
            return int(A_LEFT)

        act = self._step_towards(observation, path)
        self._tick()
        return int(A_LEFT if act is None else act)

    def is_terminated(self, observation: Any) -> bool:
        terminated = self._standing_on_oid_color(observation, OID["goal"], None) or self._failed or (self._path_to_goal(observation) is None) or self._timeout()
        if terminated:
            self._reset_internal()
        return terminated


@register_option
class GoToColoredGoalOption(_MiniGridOptionBase):
    """
    Navigate to the nearest goal of a specific color.

    Parameters:
      - color_name: one of COLOR_TO_IDX keys (e.g., "red", "green", ...)

    Preconditions:
      - A reachable colored goal exists.

    Termination:
      - Success: standing on that colored goal tile
      - Failure: no reachable colored goal
      - Timeout: max_steps
    """

    def __init__(self, option_id: Optional[str] = "go_to_colored_goal", hyper_params=None, device: str = "cpu", color_name: str = "green", max_steps: int = 20):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, max_steps=max_steps)
        self.color_name = str(color_name)

    def _color_idx(self) -> int:
        return int(COLOR_TO_IDX[self.color_name])

    def _path_to_colored_goal(self, obs):
        return self._find_nearest_of_type(
            obs,
            obj_id_filter=OID["goal"],
            also_require=lambda xy, c: int(c[1]) == int(self._color_idx()),
            avoid_lava=True,
        )

    def can_initiate(self, observation: Any) -> bool:
        path = self._path_to_colored_goal(observation)
        return path is not None and len(path) >= 2

    def should_initiate(self, observation: Any) -> bool:
        return self.can_initiate(observation)

    def reward_func(self, observation: Any) -> float:
        return 1.0 if self._standing_on_oid_color(observation, OID["goal"], self._color_idx()) else 0.0

    def select_action(self, observation: Any):
        if self._standing_on_oid_color(observation, OID["goal"], self._color_idx()):
            return int(A_DONE)

        path = self._path_to_colored_goal(observation)
        if path is None or len(path) < 2:
            self._failed = True
            return int(A_LEFT)

        act = self._step_towards(observation, path)
        self._tick()
        return int(A_LEFT if act is None else act)

    def is_terminated(self, observation: Any) -> bool:
        terminated = self._standing_on_oid_color(observation, OID["goal"], self._color_idx()) or self._failed or (self._path_to_colored_goal(observation) is None) or self._timeout()
        if terminated:
            self._reset_internal()
        return terminated


# Backward-compatible named colored goal options (stable names)
@register_option
class GoToGreenGoalOption(GoToColoredGoalOption):
    """
    Convenience wrapper: go to green goal.
    """
    def __init__(self, option_id: Optional[str] = "go_to_green_goal", hyper_params=None, device: str = "cpu"):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, color_name="green", max_steps=10)


@register_option
class GoToRedGoalOption(GoToColoredGoalOption):
    """
    Convenience wrapper: go to red goal.
    """
    def __init__(self, option_id: Optional[str] = "go_to_red_goal", hyper_params=None, device: str = "cpu"):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, color_name="red", max_steps=10)


@register_option
class GoToBlueGoalOption(GoToColoredGoalOption):
    """
    Convenience wrapper: go to blue goal.
    """
    def __init__(self, option_id: Optional[str] = "go_to_blue_goal", hyper_params=None, device: str = "cpu"):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, color_name="blue", max_steps=10)


# ============================================================
# Key and Door options (color-aware, composite)
# ============================================================

@register_option
class GoToKeyOption(_MiniGridOptionBase):
    """
    Navigate to the nearest key (optionally of a specific color).

    Parameters:
      - color_name: optional color filter
      - max_steps: default 20

    Preconditions:
      - A reachable key exists.

    Termination:
      - Success: standing on the key tile
      - Failure: no reachable key
      - Timeout: max_steps
    """

    def __init__(self, option_id: Optional[str] = "go_to_key", hyper_params=None, device: str = "cpu", color_name: Optional[str] = None, max_steps: int = 20):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, max_steps=max_steps)
        self.color_name = None if color_name is None else str(color_name)

    def _color_idx(self) -> Optional[int]:
        if self.color_name is None:
            return None
        return int(COLOR_TO_IDX[self.color_name])

    def _path_to_key(self, obs):
        col = self._color_idx()
        return self._find_nearest_of_type(
            obs,
            obj_id_filter=OID["key"],
            also_require=(None if col is None else (lambda xy, c: int(c[1]) == int(col))),
            avoid_lava=True,
        )

    def can_initiate(self, observation: Any) -> bool:
        path = self._path_to_key(observation)
        return path is not None and len(path) >= 2

    def should_initiate(self, observation: Any) -> bool:
        return self.can_initiate(observation)

    def reward_func(self, observation: Any) -> float:
        col = self._color_idx()
        return 1.0 if self._standing_on_oid_color(observation, OID["key"], col) else 0.0

    def select_action(self, observation: Any):
        col = self._color_idx()
        if self._standing_on_oid_color(observation, OID["key"], col):
            return int(A_DONE)

        path = self._path_to_key(observation)
        if path is None or len(path) < 2:
            self._failed = True
            return int(A_LEFT)

        act = self._step_towards(observation, path)
        self._tick()
        return int(A_LEFT if act is None else act)

    def is_terminated(self, observation: Any) -> bool:
        col = self._color_idx()
        terminated = self._standing_on_oid_color(observation, OID["key"], col) or self._failed or (self._path_to_key(observation) is None) or self._timeout()
        if terminated:
            self._reset_internal()
        return terminated


@register_option
class PickupKeyOption(PickupNearestObjectOption):
    """
    Convenience wrapper: pickup nearest key (optionally by color).

    Preconditions:
      - A reachable key exists.

    Termination:
      - Success: after issuing pickup while facing a key
      - Failure/Timeout: as in PickupNearestObjectOption
    """
    def __init__(self, option_id: Optional[str] = "pickup_key", hyper_params=None, device: str = "cpu", color_name: Optional[str] = None):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, obj_name="key", color_name=color_name, max_steps=20, avoid_lava=True)


@register_option
class PickupBallOption(PickupNearestObjectOption):
    """
    Convenience wrapper: pickup nearest ball (optionally by color).
    """
    def __init__(self, option_id: Optional[str] = "pickup_ball", hyper_params=None, device: str = "cpu", color_name: Optional[str] = None):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, obj_name="ball", color_name=color_name, max_steps=20, avoid_lava=True)


@register_option
class PickupBoxOption(PickupNearestObjectOption):
    """
    Convenience wrapper: pickup nearest box (optionally by color).
    """
    def __init__(self, option_id: Optional[str] = "pickup_box", hyper_params=None, device: str = "cpu", color_name: Optional[str] = None):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, obj_name="box", color_name=color_name, max_steps=20, avoid_lava=True)


@register_option
class GoToLockedDoorOption(_MiniGridOptionBase):
    """
    Navigate to the nearest LOCKED door (optionally by color).

    Parameters:
      - color_name: optional door color filter
      - max_steps: default 25

    Preconditions:
      - A locked door exists and is reachable.

    Termination:
      - Success: agent stands on the door tile (unlikely, door is not passable) OR can no longer find path
               (this option is primarily used as sub-skill for 'face/toggle' composites)
      - Failure: no locked door reachable
      - Timeout: max_steps

    Notes:
      - Doors are not passable when locked, so BFS may still return a path that ends at the door as goal
        (helper BFS accepts goal even if not passable). Movement will stop adjacent to it.
    """

    def __init__(self, option_id: Optional[str] = "go_to_locked_door", hyper_params=None, device: str = "cpu", color_name: Optional[str] = None, max_steps: int = 25):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, max_steps=max_steps)
        self.color_name = None if color_name is None else str(color_name)

    def _color_idx(self) -> Optional[int]:
        if self.color_name is None:
            return None
        return int(COLOR_TO_IDX[self.color_name])

    def _path_to_locked_door(self, obs):
        col = self._color_idx()
        return self._find_nearest_of_type(
            obs,
            obj_id_filter=OID["door"],
            also_require=lambda xy, c: (int(c[2]) == int(SID["locked"])) and (col is None or int(c[1]) == int(col)),
            avoid_lava=True,
        )

    def can_initiate(self, observation: Any) -> bool:
        path = self._path_to_locked_door(observation)
        return path is not None and len(path) >= 2

    def should_initiate(self, observation: Any) -> bool:
        return self.can_initiate(observation)

    def reward_func(self, observation: Any) -> float:
        # reward when adjacent and facing the locked door
        col = self._color_idx()
        c = self._ahead_cell(observation)
        if c is None:
            return 0.0
        if int(c[0]) != int(OID["door"]) or int(c[2]) != int(SID["locked"]):
            return 0.0
        if col is not None and int(c[1]) != int(col):
            return 0.0
        return 1.0

    def select_action(self, observation: Any):
        path = self._path_to_locked_door(observation)
        if path is None or len(path) < 2:
            self._failed = True
            return int(A_LEFT)

        act = self._step_towards(observation, path)
        self._tick()
        return int(A_LEFT if act is None else act)

    def is_terminated(self, observation: Any) -> bool:
        terminated = self._failed or (self._path_to_locked_door(observation) is None) or self._timeout()
        if terminated:
            self._reset_internal()
        return terminated


@register_option
class FaceLockedDoorOption(FaceNearestObjectOption):
    """
    Convenience wrapper: face nearest locked door (optionally by color).
    """
    def __init__(self, option_id: Optional[str] = "face_locked_door", hyper_params=None, device: str = "cpu", color_name: Optional[str] = None):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, obj_name="door", color_name=color_name, door_state="locked", max_steps=8, avoid_lava=True)


@register_option
class FaceClosedDoorOption(FaceNearestObjectOption):
    """
    Convenience wrapper: face nearest closed door (optionally by color).
    """
    def __init__(self, option_id: Optional[str] = "face_closed_door", hyper_params=None, device: str = "cpu", color_name: Optional[str] = None):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, obj_name="door", color_name=color_name, door_state="closed", max_steps=8, avoid_lava=True)


@register_option
class UnlockFacingLockedDoorOption(_MiniGridOptionBase):
    """
    Unlock the locked door directly in front of the agent by toggling it.

    Preconditions:
      - Agent is facing a locked door.
      - A matching-color key is assumed to be held by the agent in the environment mechanics.
        (We cannot reliably inspect the carried object from obs["image"] alone; this option is
         designed to be used after acquiring the correct key.)

    Termination:
      - Success: door in front becomes open or closed (i.e., no longer locked) after toggle
      - Failure: not facing locked door
      - Timeout: max_steps
    """

    def __init__(self, option_id: Optional[str] = "unlock_facing_locked_door", hyper_params=None, device: str = "cpu", max_steps: int = 3):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, max_steps=max_steps)
        self._attempted = False

    def _facing_locked_door(self, obs) -> bool:
        c = self._ahead_cell(obs)
        return c is not None and int(c[0]) == int(OID["door"]) and int(c[2]) == int(SID["locked"])

    def _facing_not_locked_door(self, obs) -> bool:
        c = self._ahead_cell(obs)
        return c is not None and int(c[0]) == int(OID["door"]) and int(c[2]) != int(SID["locked"])

    def can_initiate(self, observation: Any) -> bool:
        return self._facing_locked_door(observation)

    def should_initiate(self, observation: Any) -> bool:
        return self._facing_locked_door(observation)

    def reward_func(self, observation: Any) -> float:
        return 1.0 if self._facing_not_locked_door(observation) else 0.0

    def select_action(self, observation: Any):
        if not self._facing_locked_door(observation):
            self._failed = True
            self._tick()
            return int(A_LEFT)

        self._attempted = True
        self._tick()
        return int(A_TOGGLE)

    def is_terminated(self, observation: Any) -> bool:
        success = self._attempted and self._facing_not_locked_door(observation)
        terminated = success or self._failed or self._timeout()
        if terminated:
            self._attempted = False
            self._reset_internal()
        return terminated

    def reset(self, seed=None):
        super().reset(seed)
        self._attempted = False


@register_option
class FetchKeyThenUnlockNearestDoorOption(_MiniGridOptionBase):
    """
    Composite skill:
      1) Navigate to nearest locked door (optionally by color)
      2) Navigate to key of same color (if specified)
      3) Return to locked door, face it, toggle to unlock

    This is a robust, general-purpose "solve door" option for many puzzle domains.
    Because we cannot reliably observe carried objects from obs["image"], this option
    infers feasibility by requiring that a matching key exists somewhere in the grid.

    Parameters:
      - color_name: Optional[str] specifying the door/key color to target; if None,
                    it will pick the nearest locked door of any color and then seek
                    a key of that same color if detectable from the door cell.
      - max_steps: default 60

    Preconditions:
      - A locked door exists (reachable adjacency).
      - A matching key exists somewhere in the observation.

    Termination:
      - Success: door becomes not-locked after toggle attempt
      - Failure: cannot find door or key path at some phase
      - Timeout: max_steps

    Failure Modes:
      - No locked doors
      - Key missing (for selected door color)
      - Unreachable sequences due to walls/lava
    """

    def __init__(
        self,
        option_id: Optional[str] = "fetch_key_then_unlock_nearest_door",
        hyper_params=None,
        device: str = "cpu",
        color_name: Optional[str] = None,
        max_steps: int = 60,
    ):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, max_steps=max_steps)
        self.color_name = None if color_name is None else str(color_name)
        self.phase = 0  # 0: go to door, 1: go to key, 2: return to door+unlock
        self._selected_color_idx: Optional[int] = None
        self._toggled = False

    def _color_idx(self) -> Optional[int]:
        if self.color_name is None:
            return None
        return int(COLOR_TO_IDX[self.color_name])

    def _path_to_locked_door(self, obs, color_idx: Optional[int]):
        return self._find_nearest_of_type(
            obs,
            obj_id_filter=OID["door"],
            also_require=lambda xy, c: (int(c[2]) == int(SID["locked"])) and (color_idx is None or int(c[1]) == int(color_idx)),
            avoid_lava=True,
        )

    def _path_to_key(self, obs, color_idx: Optional[int]):
        return self._find_nearest_of_type(
            obs,
            obj_id_filter=OID["key"],
            also_require=(None if color_idx is None else (lambda xy, c: int(c[1]) == int(color_idx))),
            avoid_lava=True,
        )

    def _facing_locked_door_with_color(self, obs, color_idx: Optional[int]) -> bool:
        c = self._ahead_cell(obs)
        if c is None:
            return False
        if int(c[0]) != int(OID["door"]) or int(c[2]) != int(SID["locked"]):
            return False
        if color_idx is not None and int(c[1]) != int(color_idx):
            return False
        return True

    def _facing_not_locked_door_with_color(self, obs, color_idx: Optional[int]) -> bool:
        c = self._ahead_cell(obs)
        if c is None:
            return False
        if int(c[0]) != int(OID["door"]):
            return False
        if color_idx is not None and int(c[1]) != int(color_idx):
            return False
        return int(c[2]) != int(SID["locked"])

    def can_initiate(self, observation: Any) -> bool:
        # Determine a target door color (either specified, or inferred from nearest locked door)
        want_col = self._color_idx()
        path_d = self._path_to_locked_door(observation, want_col)
        if path_d is None or len(path_d) < 2:
            return False

        # Infer color from the door cell if not specified
        if want_col is None:
            img = self._img(observation)
            door_xy = path_d[-1]
            door_cell = img[int(door_xy[0]), int(door_xy[1])]
            want_col = int(door_cell[1])

        # Require that a matching key exists and is reachable
        path_k = self._path_to_key(observation, want_col)
        return path_k is not None and len(path_k) >= 2

    def should_initiate(self, observation: Any) -> bool:
        return self.can_initiate(observation)

    def reward_func(self, observation: Any) -> float:
        # reward if we successfully unlocked (door ahead not locked)
        col = self._selected_color_idx if self._selected_color_idx is not None else self._color_idx()
        return 1.0 if self._facing_not_locked_door_with_color(observation, col) else 0.0

    def select_action(self, observation: Any):
        # Initialize selection if needed
        if self._selected_color_idx is None:
            want_col = self._color_idx()
            path_d = self._path_to_locked_door(observation, want_col)
            if path_d is None or len(path_d) < 2:
                self._failed = True
                self._tick()
                return int(A_LEFT)

            if want_col is None:
                img = self._img(observation)
                door_xy = path_d[-1]
                door_cell = img[int(door_xy[0]), int(door_xy[1])]
                want_col = int(door_cell[1])
            self._selected_color_idx = want_col

        col = self._selected_color_idx

        # Phase 2: unlock if facing locked door
        if self.phase == 2:
            if self._facing_locked_door_with_color(observation, col):
                self._toggled = True
                self._tick()
                return int(A_TOGGLE)

            # If after toggling the door is not locked anymore, finish
            if self._toggled and self._facing_not_locked_door_with_color(observation, col):
                self._tick()
                return int(A_DONE)

            # Navigate back to locked door
            path_d = self._path_to_locked_door(observation, col)
            if path_d is None or len(path_d) < 2:
                self._failed = True
                self._tick()
                return int(A_LEFT)

            act = self._step_towards(observation, path_d)
            self._tick()
            return int(A_LEFT if act is None else act)

        # Phase 1: go to key
        if self.phase == 1:
            # If standing on key tile, attempt to pickup (best effort) then move to phase 2
            if self._standing_on_oid_color(observation, OID["key"], col):
                self.phase = 2
                self._tick()
                return int(A_PICKUP)

            path_k = self._path_to_key(observation, col)
            if path_k is None or len(path_k) < 2:
                self._failed = True
                self._tick()
                return int(A_LEFT)

            act = self._step_towards(observation, path_k)
            self._tick()
            return int(A_LEFT if act is None else act)

        # Phase 0: approach door first (to select it / be near it)
        path_d = self._path_to_locked_door(observation, col)
        if path_d is None or len(path_d) < 2:
            self._failed = True
            self._tick()
            return int(A_LEFT)

        # If we are already adjacent/facing-ish, switch to key phase
        # We detect adjacency by checking if door is directly ahead OR within 1 step ahead from current pos.
        if self._facing_locked_door_with_color(observation, col):
            self.phase = 1
            self._tick()
            return int(A_LEFT)

        act = self._step_towards(observation, path_d)
        self._tick()
        # If stepping got us adjacent (door ahead), proceed to phase 1 next tick
        if act == A_FORWARD or act in (A_LEFT, A_RIGHT):
            # keep phase unless we are now facing the door
            if self._facing_locked_door_with_color(observation, col):
                self.phase = 1
        return int(A_LEFT if act is None else act)

    def is_terminated(self, observation: Any) -> bool:
        col = self._selected_color_idx if self._selected_color_idx is not None else self._color_idx()
        success = self._toggled and self._facing_not_locked_door_with_color(observation, col)
        # Fail if we can no longer find either key or door during execution
        door_ok = self._path_to_locked_door(observation, col) is not None
        key_ok = self._path_to_key(observation, col) is not None
        terminated = success or self._failed or (not door_ok) or (not key_ok) or self._timeout()
        if terminated:
            self.phase = 0
            self._selected_color_idx = None
            self._toggled = False
            self._reset_internal()
        return terminated

    def reset(self, seed=None):
        super().reset(seed)
        self.phase = 0
        self._selected_color_idx = None
        self._toggled = False


# ============================================================
# Object-interaction options: balls/boxes toggles
# ============================================================

@register_option
class ToggleFacingObjectOption(_MiniGridOptionBase):
    """
    Toggle the object directly in front of the agent if it matches type/color.
    Intended for objects that react to toggle (ball/box/door).

    Parameters:
      - obj_name: "ball" or "box" or "door"
      - color_name: optional color constraint
      - max_steps: default 2

    Preconditions:
      - Agent is facing the target object.

    Termination:
      - Success: after toggling once with precondition met
      - Failure: precondition not met
      - Timeout: max_steps
    """

    def __init__(
        self,
        option_id: Optional[str] = "toggle_facing_object",
        hyper_params=None,
        device: str = "cpu",
        obj_name: str = "box",
        color_name: Optional[str] = None,
        max_steps: int = 2,
    ):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, max_steps=max_steps)
        self.obj_name = str(obj_name)
        self.color_name = None if color_name is None else str(color_name)

    def _target_oid(self) -> int:
        return int(OID[self.obj_name])

    def _color_idx(self) -> Optional[int]:
        if self.color_name is None:
            return None
        return int(COLOR_TO_IDX[self.color_name])

    def _precond(self, obs) -> bool:
        c = self._ahead_cell(obs)
        if c is None or int(c[0]) != int(self._target_oid()):
            return False
        col = self._color_idx()
        if col is not None and int(c[1]) != int(col):
            return False
        return True

    def can_initiate(self, observation: Any) -> bool:
        return self._precond(observation)

    def should_initiate(self, observation: Any) -> bool:
        return self._precond(observation)

    def reward_func(self, observation: Any) -> float:
        return 1.0 if self._precond(observation) else 0.0

    def select_action(self, observation: Any):
        self._tick()
        if not self._precond(observation):
            self._failed = True
            return int(A_LEFT)
        return int(A_TOGGLE)

    def is_terminated(self, observation: Any) -> bool:
        terminated = self._failed or self._timeout()
        if terminated:
            self._reset_internal()
        return terminated


@register_option
class FaceNearestBoxOption(FaceNearestObjectOption):
    """
    Convenience wrapper: face nearest box (optionally by color).
    """
    def __init__(self, option_id: Optional[str] = "face_box", hyper_params=None, device: str = "cpu", color_name: Optional[str] = None):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, obj_name="box", color_name=color_name, door_state=None, max_steps=8, avoid_lava=True)


@register_option
class FaceNearestBallOption(FaceNearestObjectOption):
    """
    Convenience wrapper: face nearest ball (optionally by color).
    """
    def __init__(self, option_id: Optional[str] = "face_ball", hyper_params=None, device: str = "cpu", color_name: Optional[str] = None):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, obj_name="ball", color_name=color_name, door_state=None, max_steps=8, avoid_lava=True)


@register_option
class GoToBoxOption(GoToNearestObjectOption):
    """
    Convenience wrapper: go to nearest box (optionally by color).
    """
    def __init__(self, option_id: Optional[str] = "go_to_box", hyper_params=None, device: str = "cpu", color_name: Optional[str] = None):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, obj_name="box", color_name=color_name, door_state=None, max_steps=20, avoid_lava=True)


@register_option
class GoToBallOption(GoToNearestObjectOption):
    """
    Convenience wrapper: go to nearest ball (optionally by color).
    """
    def __init__(self, option_id: Optional[str] = "go_to_ball", hyper_params=None, device: str = "cpu", color_name: Optional[str] = None):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, obj_name="ball", color_name=color_name, door_state=None, max_steps=20, avoid_lava=True)


# ============================================================
# Positioning / alignment utilities
# ============================================================

@register_option
class FaceGoalOption(_MiniGridOptionBase):
    """
    Rotate to face the nearest goal (any color).
    This option rotates only, no forward motion.

    Preconditions:
      - A goal exists in observation.

    Termination:
      - Success: goal is directly ahead
      - Failure: no goal exists / cannot compute path
      - Timeout: max_steps
    """

    def __init__(self, option_id: Optional[str] = "face_goal", hyper_params=None, device: str = "cpu", max_steps: int = 8):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, max_steps=max_steps)

    def _path_to_goal(self, obs):
        return self._find_nearest_of_type(obs, obj_id_filter=OID["goal"], also_require=None, avoid_lava=True)

    def can_initiate(self, observation: Any) -> bool:
        return self._find_any_oid_exists(observation, OID["goal"], None)

    def should_initiate(self, observation: Any) -> bool:
        return self.can_initiate(observation)

    def reward_func(self, observation: Any) -> float:
        return 1.0 if self._ahead_has_oid(observation, OID["goal"]) else 0.0

    def select_action(self, observation: Any):
        if self._ahead_has_oid(observation, OID["goal"]):
            return int(A_DONE)

        path = self._path_to_goal(observation)
        if path is None or len(path) < 2:
            self._failed = True
            return int(A_LEFT)

        target_xy = path[-1]
        act = self._face_vec_action(observation, target_xy)
        self._tick()
        return int(A_LEFT if act is None else act)

    def is_terminated(self, observation: Any) -> bool:
        terminated = self._ahead_has_oid(observation, OID["goal"]) or self._failed or (not self.can_initiate(observation)) or self._timeout()
        if terminated:
            self._reset_internal()
        return terminated


@register_option
class StepToClearAheadOption(_MiniGridOptionBase):
    """
    A small positioning option:
      - If forward is safe, move forward.
      - Otherwise rotate left once (to search for a safe direction next tick).

    Preconditions:
      - None.

    Termination:
      - Always terminates after 1 step.

    Failure:
      - None.
    """

    def __init__(self, option_id: Optional[str] = "step_to_clear_ahead", hyper_params=None, device: str = "cpu"):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device, max_steps=1)

    def can_initiate(self, observation: Any) -> bool:
        return True

    def should_initiate(self, observation: Any) -> bool:
        return not self._forward_is_safe(observation)

    def reward_func(self, observation: Any) -> float:
        return 1.0 if self._forward_is_safe(observation) else 0.0

    def select_action(self, observation: Any):
        self._tick()
        return int(A_FORWARD if self._forward_is_safe(observation) else A_LEFT)

    def is_terminated(self, observation: Any) -> bool:
        terminated = self._timeout()
        if terminated:
            self._reset_internal()
        return terminated
    
def custom_build_options():
    options = [
        # Movement
        ActionLeft(),
        ActionRight(),
        ActionForward(),
        ActionPickup(),
        ActionDrop(),
        ActionToggle(),
        
        # Goal
        GoToColoredGoalOption(color_name="green"),
        GoToColoredGoalOption(color_name="red"),
        
        # # Door and Key
        OpenNearestClosedDoorOption(color_name="green"),
        OpenNearestClosedDoorOption(color_name="red"),
        OpenNearestClosedDoorOption(color_name="blue"),
        OpenNearestClosedDoorOption(color_name="purple"),
        OpenNearestClosedDoorOption(color_name="yellow"),
        OpenNearestClosedDoorOption(color_name="grey"),
        
        GoToLockedDoorOption(color_name="green"),
        GoToLockedDoorOption(color_name="red"),
        GoToLockedDoorOption(color_name="blue"),
        GoToLockedDoorOption(color_name="purple"),
        GoToLockedDoorOption(color_name="yellow"),
        GoToLockedDoorOption(color_name="grey"),
        
        PickupKeyOption(color_name="green"),
        PickupKeyOption(color_name="red"),
        PickupKeyOption(color_name="blue"),
        PickupKeyOption(color_name="purple"),
        PickupKeyOption(color_name="yellow"),
        PickupKeyOption(color_name="grey"),
        
    ]
    return options

def build_all_options(hyper_params=None, device: str = "cpu") -> List:
    opts = []

    # ---------------- Atomic (single-instance each) ----------------
    opts += [
        ActionLeft(hyper_params=hyper_params, device=device),
        ActionRight(hyper_params=hyper_params, device=device),
        ActionForward(hyper_params=hyper_params, device=device),
        ActionPickup(hyper_params=hyper_params, device=device),
        ActionDrop(hyper_params=hyper_params, device=device),
        ActionToggle(hyper_params=hyper_params, device=device),
        ActionDone(hyper_params=hyper_params, device=device),
    ]

    # ---------------- Simple fixed ones ----------------
    opts += [
        AvoidLavaStepOption(hyper_params=hyper_params, device=device),
        ExploreNearestUnseenOption(hyper_params=hyper_params, device=device),
        GoToGoalOption(hyper_params=hyper_params, device=device),
        FaceGoalOption(hyper_params=hyper_params, device=device),
        StepToClearAheadOption(hyper_params=hyper_params, device=device),
    ]

    # ---------------- Colored goals ----------------
    # Generic colored-goal with ALL colors
    for c in COLOR_NAMES:
        opts.append(GoToColoredGoalOption(hyper_params=hyper_params, device=device, color_name=c))

    # Also include your named convenience wrappers (these are duplicates vs above for green/red/blue,
    # but they have stable option_id names you might rely on)
    opts += [
        GoToGreenGoalOption(hyper_params=hyper_params, device=device),
        GoToRedGoalOption(hyper_params=hyper_params, device=device),
        GoToBlueGoalOption(hyper_params=hyper_params, device=device),
    ]

    # ---------------- GoToNearestObjectOption (systematic variants) ----------------
    # Objects that have colors in MiniGrid: door, key, ball, box, goal (we cover goal separately too, but fine)
    colored_objects = ["door", "key", "ball", "box", "goal"]
    # Objects that are not meaningfully colored for targeting: wall/lava/floor/empty/agent/unseen (skip)
    uncolored_objects = []

    # Door states we care about for navigation/face
    door_states = ["open", "closed", "locked"]

    # 1) Doors: all colors x all states
    for c in COLOR_NAMES:
        for st in door_states:
            opts.append(
                GoToNearestObjectOption(
                    hyper_params=hyper_params, device=device,
                    obj_name="door", color_name=c, door_state=st
                )
            )

    # 2) Non-door colored objects: all colors, no state
    for obj in ["key", "ball", "box", "goal"]:
        for c in COLOR_NAMES:
            opts.append(
                GoToNearestObjectOption(
                    hyper_params=hyper_params, device=device,
                    obj_name=obj, color_name=c, door_state=None
                )
            )

    # 3) Also include color-agnostic versions (often useful)
    for obj in ["door", "key", "ball", "box", "goal"]:
        if obj == "door":
            for st in door_states:
                opts.append(
                    GoToNearestObjectOption(
                        hyper_params=hyper_params, device=device,
                        obj_name="door", color_name=None, door_state=st
                    )
                )
        else:
            opts.append(
                GoToNearestObjectOption(
                    hyper_params=hyper_params, device=device,
                    obj_name=obj, color_name=None, door_state=None
                )
            )

    # ---------------- FaceNearestObjectOption (systematic variants) ----------------
    # 1) Doors: all colors x all states
    for c in COLOR_NAMES:
        for st in door_states:
            opts.append(
                FaceNearestObjectOption(
                    hyper_params=hyper_params, device=device,
                    obj_name="door", color_name=c, door_state=st
                )
            )

    # 2) Non-door colored objects: all colors
    for obj in ["key", "ball", "box", "goal"]:
        for c in COLOR_NAMES:
            opts.append(
                FaceNearestObjectOption(
                    hyper_params=hyper_params, device=device,
                    obj_name=obj, color_name=c, door_state=None
                )
            )

    # 3) Color-agnostic versions
    for obj in ["door", "key", "ball", "box", "goal"]:
        if obj == "door":
            for st in door_states:
                opts.append(
                    FaceNearestObjectOption(
                        hyper_params=hyper_params, device=device,
                        obj_name="door", color_name=None, door_state=st
                    )
                )
        else:
            opts.append(
                FaceNearestObjectOption(
                    hyper_params=hyper_params, device=device,
                    obj_name=obj, color_name=None, door_state=None
                )
            )

    # ---------------- Keys (go-to and pickup) ----------------
    # GoToKeyOption all colors + agnostic
    opts.append(GoToKeyOption(hyper_params=hyper_params, device=device, color_name=None))
    for c in COLOR_NAMES:
        opts.append(GoToKeyOption(hyper_params=hyper_params, device=device, color_name=c))

    # PickupKey/Ball/Box all colors + agnostic
    opts.append(PickupKeyOption(hyper_params=hyper_params, device=device, color_name=None))
    opts.append(PickupBallOption(hyper_params=hyper_params, device=device, color_name=None))
    opts.append(PickupBoxOption(hyper_params=hyper_params, device=device, color_name=None))
    for c in COLOR_NAMES:
        opts.append(PickupKeyOption(hyper_params=hyper_params, device=device, color_name=c))
        opts.append(PickupBallOption(hyper_params=hyper_params, device=device, color_name=c))
        opts.append(PickupBoxOption(hyper_params=hyper_params, device=device, color_name=c))

    # Also include the generic PickupNearestObjectOption variants explicitly (sometimes useful)
    for obj in ["key", "ball", "box"]:
        opts.append(PickupNearestObjectOption(hyper_params=hyper_params, device=device, obj_name=obj, color_name=None))
        for c in COLOR_NAMES:
            opts.append(PickupNearestObjectOption(hyper_params=hyper_params, device=device, obj_name=obj, color_name=c))

    # ---------------- Door manipulation ----------------
    # ToggleFacingDoorOption: all colors x require_state + agnostic
    require_states = [None, "open", "closed", "locked"]
    for st in require_states:
        opts.append(ToggleFacingDoorOption(hyper_params=hyper_params, device=device, color_name=None, require_state=st))
    for c in COLOR_NAMES:
        for st in require_states:
            opts.append(ToggleFacingDoorOption(hyper_params=hyper_params, device=device, color_name=c, require_state=st))

    # OpenNearestClosedDoorOption: all colors + agnostic
    opts.append(OpenNearestClosedDoorOption(hyper_params=hyper_params, device=device, color_name=None))
    for c in COLOR_NAMES:
        opts.append(OpenNearestClosedDoorOption(hyper_params=hyper_params, device=device, color_name=c))

    # Locked-door navigation/facing: all colors + agnostic
    opts.append(GoToLockedDoorOption(hyper_params=hyper_params, device=device, color_name=None))
    opts.append(FaceLockedDoorOption(hyper_params=hyper_params, device=device, color_name=None))
    opts.append(FaceClosedDoorOption(hyper_params=hyper_params, device=device, color_name=None))
    for c in COLOR_NAMES:
        opts.append(GoToLockedDoorOption(hyper_params=hyper_params, device=device, color_name=c))
        opts.append(FaceLockedDoorOption(hyper_params=hyper_params, device=device, color_name=c))
        opts.append(FaceClosedDoorOption(hyper_params=hyper_params, device=device, color_name=c))

    # UnlockFacingLockedDoorOption: single instance (no params)
    opts.append(UnlockFacingLockedDoorOption(hyper_params=hyper_params, device=device))

    # FetchKeyThenUnlockNearestDoorOption: all colors + agnostic
    opts.append(FetchKeyThenUnlockNearestDoorOption(hyper_params=hyper_params, device=device, color_name=None))
    for c in COLOR_NAMES:
        opts.append(FetchKeyThenUnlockNearestDoorOption(hyper_params=hyper_params, device=device, color_name=c))

    # ---------------- Box/Ball toggles ----------------
    # ToggleFacingObjectOption for box/ball: all colors + agnostic
    for obj in ["box", "ball"]:
        opts.append(ToggleFacingObjectOption(hyper_params=hyper_params, device=device, obj_name=obj, color_name=None))
        for c in COLOR_NAMES:
            opts.append(ToggleFacingObjectOption(hyper_params=hyper_params, device=device, obj_name=obj, color_name=c))

    # Convenience face/go-to for box/ball
    opts.append(FaceNearestBoxOption(hyper_params=hyper_params, device=device, color_name=None))
    opts.append(FaceNearestBallOption(hyper_params=hyper_params, device=device, color_name=None))
    opts.append(GoToBoxOption(hyper_params=hyper_params, device=device, color_name=None))
    opts.append(GoToBallOption(hyper_params=hyper_params, device=device, color_name=None))
    for c in COLOR_NAMES:
        opts.append(FaceNearestBoxOption(hyper_params=hyper_params, device=device, color_name=c))
        opts.append(FaceNearestBallOption(hyper_params=hyper_params, device=device, color_name=c))
        opts.append(GoToBoxOption(hyper_params=hyper_params, device=device, color_name=c))
        opts.append(GoToBallOption(hyper_params=hyper_params, device=device, color_name=c))

    return opts