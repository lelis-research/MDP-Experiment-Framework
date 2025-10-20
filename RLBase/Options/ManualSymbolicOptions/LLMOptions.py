from ..Utils import BaseOption
from ...registry import register_option

import numpy as np
import torch, os, shutil
from typing import Optional, Tuple, List

from minigrid.core.constants import (
    COLOR_NAMES, IDX_TO_OBJECT, DIR_TO_VEC, OBJECT_TO_IDX
)

# ============================================================
# Shared helpers and constants
# ============================================================

COLOR_TO_IDX = {name: i for i, name in enumerate(COLOR_NAMES)}

A_LEFT, A_RIGHT, A_FWD, A_PICK, A_DROP, A_TOG, A_DONE = 0, 1, 2, 3, 4, 5, 6


# Cardinal headings used by _turn_toward
_DIRS = [np.array([-1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, -1])]


class _NavHelperMixin:
    """Reusable utilities for fully observable MiniGrid options.
    Assumes obs["image"] is HxWx3: (type, color, state) and obs["direction"] is 0..3.
    """
    def __init__(self):
        # Basic IDs (guard accesses with .get where needed)
        self.wall_id   = OBJECT_TO_IDX.get("wall")
        self.agent_id  = OBJECT_TO_IDX.get("agent")
        self.door_id   = OBJECT_TO_IDX.get("door")
        self.key_id    = OBJECT_TO_IDX.get("key")
        self.box_id    = OBJECT_TO_IDX.get("box")
        self.ball_id   = OBJECT_TO_IDX.get("ball")
        self.goal_id   = OBJECT_TO_IDX.get("goal")
        self.lava_id   = OBJECT_TO_IDX.get("lava")
        self._origin   = None  # used by ReturnToStart / Fetch

    # ------------------------- Low-level state helpers -------------------------
    def _img(self, obs):
        return obs["image"]

    def _agent_pos_dir(self, obs) -> Tuple[np.ndarray, np.ndarray]:
        img = self._img(obs)
        agent_pos = np.argwhere(img[..., 0] == self.agent_id)[0]
        agent_dir = np.array(DIR_TO_VEC[obs["direction"]], dtype=int)
        return agent_pos, agent_dir

    def _front_cell(self, pos, dir_vec):
        return pos + dir_vec

    def _is_inside(self, img, p):
        H, W = img.shape[:2]
        return 0 <= p[0] < H and 0 <= p[1] < W

    def _is_wall(self, img, p):
        return self.wall_id is not None and self._is_inside(img, p) and (img[tuple(p)][0] == self.wall_id)

    # ------------------------- Object queries -------------------------
    def _nearest_object_pos(self, obs, type_idx: Optional[int], color_idx: Optional[int] = None):
        img = self._img(obs)
        if type_idx is None:
            return None
        mask = (img[..., 0] == type_idx)
        if color_idx is not None:
            mask &= (img[..., 1] == color_idx)
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        agent_pos, _ = self._agent_pos_dir(obs)
        dists = np.abs(coords - agent_pos).sum(axis=1)
        return coords[np.argmin(dists)]

    def _all_object_positions(self, obs, type_idx: int, color_idx: Optional[int] = None) -> np.ndarray:
        img = self._img(obs)
        if color_idx is None:
            return np.argwhere(img[..., 0] == type_idx)
        return np.argwhere((img[..., 0] == type_idx) & (img[..., 1] == color_idx))

    def _any_object_exists(self, obs, type_idx: int, color_idx: Optional[int] = None) -> bool:
        return self._all_object_positions(obs, type_idx, color_idx).size > 0

    def _is_facing_object(self, obs, obj_type_idx: Optional[int] = None, color_idx: Optional[int] = None) -> bool:
        img = self._img(obs)
        pos, dir_vec = self._agent_pos_dir(obs)
        front = self._front_cell(pos, dir_vec)
        if not self._is_inside(img, front):
            return False
        if self._is_wall(img, front):
            return False
        t = img[tuple(front)][0]
        c = img[tuple(front)][1]
        if obj_type_idx is None:
            return True
        if t != obj_type_idx:
            return False
        if color_idx is not None and c != color_idx:
            return False
        return True

    # ------------------------- Steering -------------------------
    def _turn_toward(self, cur_dir: np.ndarray, delta: np.ndarray) -> Optional[int]:
        td = np.array(delta, dtype=int)
        if np.all(td == 0):
            return None
        # Prefer axis of greatest magnitude
        if abs(td[0]) >= abs(td[1]):
            want = np.array([np.sign(td[0]), 0], dtype=int)
        else:
            want = np.array([0, np.sign(td[1])], dtype=int)
        i = next(k for k, d in enumerate(_DIRS) if np.array_equal(d, cur_dir))
        j = next(k for k, d in enumerate(_DIRS) if np.array_equal(d, want))
        if (i - 1) % 4 == j:
            return A_LEFT
        if (i + 1) % 4 == j:
            return A_RIGHT
        # 180° away -> choose left
        return A_LEFT

    def _step_toward_point(self, obs, target_pos: np.ndarray):
        img = self._img(obs)
        agent_pos, agent_dir = self._agent_pos_dir(obs)
        delta = target_pos - agent_pos
        if np.all(delta == 0):
            return A_DONE
        turn = self._turn_toward(agent_dir, delta)
        if turn is not None:
            return turn
        front = self._front_cell(agent_pos, agent_dir)
        if not self._is_inside(img, front) or self._is_wall(img, front):
            return A_LEFT
        return A_FWD

    def _save_checkpoint(self, file_path, data):
        if file_path is not None:
            torch.save(data, f"{file_path}_options.t")
            src = os.path.abspath(__file__)
            dst = f"{file_path}_{self.__class__.__name__}.py"
            shutil.copyfile(src, dst)
        return data


# ============================================================
# NAVIGATION OPTIONS
# ============================================================

@register_option
class GoToPosOption(BaseOption, _NavHelperMixin):
    """Move to a fixed grid coordinate (row, col)."""
    def __init__(self, target_pos, option_len=200):
        _NavHelperMixin.__init__(self)
        self.target_pos = np.array(target_pos, dtype=int)
        self.option_len = option_len
        self.step_counter = 0

    def select_action(self, observation):
        self.step_counter += 1
        return self._step_toward_point(observation, self.target_pos)

    def is_terminated(self, observation):
        agent_pos, _ = self._agent_pos_dir(observation)
        done = np.all(agent_pos == self.target_pos) or self.step_counter >= self.option_len
        if done:
            self.step_counter = 0
        return done

    def save(self, file_path=None):
        data = {"option_class": self.__class__.__name__,
                "target_pos": self.target_pos.tolist(),
                "option_len": self.option_len}
        return self._save_checkpoint(file_path, data)

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        return cls(checkpoint["target_pos"], checkpoint["option_len"])


@register_option
class GoToNearestObjectOption(BaseOption, _NavHelperMixin):
    """Navigate to the nearest object of a given type and optional color."""
    def __init__(self, type_name: str, color: Optional[str] = None, option_len=200):
        _NavHelperMixin.__init__(self)
        self.type_idx = OBJECT_TO_IDX[type_name]
        self.color_idx = None if color is None else COLOR_TO_IDX[color]
        self.option_len = option_len
        self.step_counter = 0

    def select_action(self, observation):
        self.step_counter += 1
        tgt = self._nearest_object_pos(observation, self.type_idx, self.color_idx)
        if tgt is None:
            return A_DONE
        return self._step_toward_point(observation, tgt)

    def is_terminated(self, observation):
        nearby = self._is_facing_object(observation, self.type_idx, self.color_idx)
        done = nearby or self.step_counter >= self.option_len
        if done:
            self.step_counter = 0
        return done

    def save(self, file_path=None):
        data = {"option_class": self.__class__.__name__,
                "type_idx": int(self.type_idx),
                "color_idx": None if self.color_idx is None else int(self.color_idx),
                "option_len": self.option_len}
        return self._save_checkpoint(file_path, data)

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        type_name = IDX_TO_OBJECT[checkpoint["type_idx"]]
        color = None if checkpoint["color_idx"] is None else COLOR_NAMES[checkpoint["color_idx"]]
        return cls(type_name, color, checkpoint["option_len"])


@register_option
class LeaveRoomOption(BaseOption, _NavHelperMixin):
    """Go to nearest door (any) and stand facing it (to prepare to exit)."""
    def __init__(self, option_len=200):
        _NavHelperMixin.__init__(self)
        self.option_len = option_len
        self.step_counter = 0

    def select_action(self, observation):
        self.step_counter += 1
        tgt = self._nearest_object_pos(observation, self.door_id, None)
        if tgt is None:
            return A_DONE
        return self._step_toward_point(observation, tgt)

    def is_terminated(self, observation):
        done = self._is_facing_object(observation, self.door_id, None) or self.step_counter >= self.option_len
        if done:
            self.step_counter = 0
        return done

    def save(self, file_path=None):
        data = {"option_class": self.__class__.__name__, "option_len": self.option_len}
        return self._save_checkpoint(file_path, data)

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        return cls(checkpoint["option_len"])


@register_option
class EnterNextRoomOption(BaseOption, _NavHelperMixin):
    """Approach a door, toggle it if needed, and step through to enter the next room."""
    def __init__(self, option_len=400):
        _NavHelperMixin.__init__(self)
        self.option_len = option_len
        self.step_counter = 0
        self._crossing = False

    def select_action(self, observation):
        self.step_counter += 1
        img = self._img(observation)
        pos, dir_vec = self._agent_pos_dir(observation)
        front = self._front_cell(pos, dir_vec)
        if self._is_facing_object(observation, self.door_id, None):
            self._crossing = True
            return A_TOG
        if self._crossing:
            if not self._is_inside(img, front) or self._is_wall(img, front):
                self._crossing = False
                return A_LEFT
            return A_FWD
        tgt = self._nearest_object_pos(observation, self.door_id, None)
        if tgt is None:
            return A_DONE
        return self._step_toward_point(observation, tgt)

    def is_terminated(self, observation):
        done = self.step_counter >= self.option_len or (self._crossing and not self._is_facing_object(observation, self.door_id, None))
        if done:
            self.step_counter = 0
            self._crossing = False
        return done

    def save(self, file_path=None):
        data = {"option_class": self.__class__.__name__, "option_len": self.option_len}
        return self._save_checkpoint(file_path, data)

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        return cls(checkpoint["option_len"])


# ============================================================
# OBJECT INTERACTION OPTIONS
# ============================================================

@register_option
class PickUpNearestKeyOption(BaseOption, _NavHelperMixin):
    def __init__(self, option_len=200):
        _NavHelperMixin.__init__(self)
        self.option_len = option_len
        self.step_counter = 0

    def select_action(self, observation):
        self.step_counter += 1
        if self._is_facing_object(observation, self.key_id, None):
            return A_PICK
        tgt = self._nearest_object_pos(observation, self.key_id, None)
        if tgt is None:
            return A_DONE
        return self._step_toward_point(observation, tgt)

    def is_terminated(self, observation):
        any_keys = self._any_object_exists(observation, self.key_id, None)
        done = (not any_keys) or self.step_counter >= self.option_len
        if done:
            self.step_counter = 0
        return done

    def save(self, file_path=None):
        data = {"option_class": self.__class__.__name__, "option_len": self.option_len}
        return self._save_checkpoint(file_path, data)

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        return cls(checkpoint["option_len"])


@register_option
class PickUpColorKeyOption(BaseOption, _NavHelperMixin):
    def __init__(self, color: str, option_len=200):
        _NavHelperMixin.__init__(self)
        self.color = color
        self.color_idx = COLOR_TO_IDX[color]
        self.option_len = option_len
        self.step_counter = 0

    def select_action(self, observation):
        self.step_counter += 1
        if self._is_facing_object(observation, self.key_id, self.color_idx):
            return A_PICK
        tgt = self._nearest_object_pos(observation, self.key_id, self.color_idx)
        if tgt is None:
            return A_DONE
        return self._step_toward_point(observation, tgt)

    def is_terminated(self, observation):
        has_color = self._any_object_exists(observation, self.key_id, self.color_idx)
        done = (not has_color) or self.step_counter >= self.option_len
        if done:
            self.step_counter = 0
        return done

    def save(self, file_path=None):
        data = {"option_class": self.__class__.__name__, "color": self.color, "option_len": self.option_len}
        return self._save_checkpoint(file_path, data)

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        return cls(checkpoint["color"], checkpoint["option_len"])


@register_option
class ToggleNearestDoorOption(BaseOption, _NavHelperMixin):
    def __init__(self, option_len=200):
        _NavHelperMixin.__init__(self)
        self.option_len = option_len
        self.step_counter = 0

    def select_action(self, observation):
        self.step_counter += 1
        if self._is_facing_object(observation, self.door_id, None):
            return A_TOG
        tgt = self._nearest_object_pos(observation, self.door_id, None)
        if tgt is None:
            return A_DONE
        return self._step_toward_point(observation, tgt)

    def is_terminated(self, observation):
        done = self.step_counter >= self.option_len or self._is_facing_object(observation, self.door_id, None)
        if done:
            self.step_counter = 0
        return done

    def save(self, file_path=None):
        data = {"option_class": self.__class__.__name__, "option_len": self.option_len}
        return self._save_checkpoint(file_path, data)

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        return cls(checkpoint["option_len"])


@register_option
class UnlockAndOpenNearestDoorOption(BaseOption, _NavHelperMixin):
    def __init__(self, option_len=400):
        _NavHelperMixin.__init__(self)
        self.option_len = option_len
        self.step_counter = 0

    def select_action(self, observation):
        self.step_counter += 1
        if self._is_facing_object(observation, self.door_id, None):
            return A_TOG
        tgt = self._nearest_object_pos(observation, self.door_id, None)
        if tgt is None:
            return A_DONE
        return self._step_toward_point(observation, tgt)

    def is_terminated(self, observation):
        done = self.step_counter >= self.option_len or self._is_facing_object(observation, self.door_id, None)
        if done:
            self.step_counter = 0
        return done

    def save(self, file_path=None):
        data = {"option_class": self.__class__.__name__, "option_len": self.option_len}
        return self._save_checkpoint(file_path, data)

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        return cls(checkpoint["option_len"])


@register_option
class ToggleNearestObjectOption(BaseOption, _NavHelperMixin):
    """Generic: move to nearest object type and toggle it (e.g., door, box)."""
    def __init__(self, type_name: str, option_len=200):
        _NavHelperMixin.__init__(self)
        self.type_idx = OBJECT_TO_IDX[type_name]
        self.option_len = option_len
        self.step_counter = 0

    def select_action(self, observation):
        self.step_counter += 1
        if self._is_facing_object(observation, self.type_idx, None):
            return A_TOG
        tgt = self._nearest_object_pos(observation, self.type_idx, None)
        if tgt is None:
            return A_DONE
        return self._step_toward_point(observation, tgt)

    def is_terminated(self, observation):
        done = self.step_counter >= self.option_len or self._is_facing_object(observation, self.type_idx, None)
        if done:
            self.step_counter = 0
        return done

    def save(self, file_path=None):
        data = {"option_class": self.__class__.__name__, "type_idx": int(self.type_idx), "option_len": self.option_len}
        return self._save_checkpoint(file_path, data)

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        type_name = IDX_TO_OBJECT[checkpoint["type_idx"]]
        return cls(type_name, checkpoint["option_len"])


@register_option
class DropHeldOption(BaseOption, _NavHelperMixin):
    def __init__(self, option_len=20):
        _NavHelperMixin.__init__(self)
        self.option_len = option_len
        self.step_counter = 0
        self._dropped = False

    def select_action(self, observation):
        self.step_counter += 1
        if not self._dropped:
            self._dropped = True
            return A_DROP
        # small shimmy forward if possible
        img = self._img(observation)
        pos, dir_vec = self._agent_pos_dir(observation)
        front = self._front_cell(pos, dir_vec)
        if self._is_inside(img, front) and not self._is_wall(img, front):
            return A_FWD
        return A_LEFT

    def is_terminated(self, observation):
        done = self._dropped or self.step_counter >= self.option_len
        if done:
            self.step_counter = 0
            self._dropped = False
        return done

    def save(self, file_path=None):
        data = {"option_class": self.__class__.__name__, "option_len": self.option_len}
        return self._save_checkpoint(file_path, data)

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        return cls(checkpoint["option_len"])


# ============================================================
# GOAL / TASK OPTIONS
# ============================================================

@register_option
class GoToGoalOption(BaseOption, _NavHelperMixin):
    def __init__(self, option_len=300):
        _NavHelperMixin.__init__(self)
        self.option_len = option_len
        self.step_counter = 0

    def select_action(self, observation):
        self.step_counter += 1
        # Lazy goal id resolution if missing
        if self.goal_id is None:
            for t_idx, name in enumerate(IDX_TO_OBJECT):
                if name == "goal":
                    self.goal_id = t_idx
                    break
        if self.goal_id is None:
            return A_DONE
        tgt = self._nearest_object_pos(observation, self.goal_id, None)
        if tgt is None:
            return A_DONE
        return self._step_toward_point(observation, tgt)

    def is_terminated(self, observation):
        img = self._img(observation)
        pos, _ = self._agent_pos_dir(observation)
        on_goal = self.goal_id is not None and img[tuple(pos)][0] == self.goal_id
        done = on_goal or self.step_counter >= self.option_len
        if done:
            self.step_counter = 0
        return done

    def save(self, file_path=None):
        data = {"option_class": self.__class__.__name__, "option_len": self.option_len}
        return self._save_checkpoint(file_path, data)

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        return cls(checkpoint["option_len"])


@register_option
class FetchObjectToOriginOption(BaseOption, _NavHelperMixin):
    """Go to object type (optional color), pick it up, and return to origin (spawn) to drop."""
    def __init__(self, type_name: str, color: Optional[str] = None, option_len=600):
        _NavHelperMixin.__init__(self)
        self.type_idx = OBJECT_TO_IDX[type_name]
        self.color_idx = None if color is None else COLOR_TO_IDX[color]
        self.option_len = option_len
        self.step_counter = 0
        self._phase = "init"  # init -> go -> pickup -> return -> drop

    def select_action(self, observation):
        self.step_counter += 1
        pos, _ = self._agent_pos_dir(observation)
        if self._origin is None:
            self._origin = pos.copy()
        if self._phase == "init":
            self._phase = "go"
        if self._phase == "go":
            if self._is_facing_object(observation, self.type_idx, self.color_idx):
                self._phase = "pickup"
                return A_PICK
            tgt = self._nearest_object_pos(observation, self.type_idx, self.color_idx)
            if tgt is None:
                return A_DONE
            return self._step_toward_point(observation, tgt)
        if self._phase == "pickup":
            # After pickup, start returning
            self._phase = "return"
            return A_FWD  # small move to clear space if possible
        if self._phase == "return":
            act = self._step_toward_point(observation, self._origin)
            if act == A_DONE:
                self._phase = "drop"
                return A_DROP
            return act
        if self._phase == "drop":
            return A_DROP
        return A_DONE

    def is_terminated(self, observation):
        done = self.step_counter >= self.option_len or self._phase == "drop"
        if done:
            self.step_counter = 0
            self._phase = "init"
            self._origin = None
        return done

    def save(self, file_path=None):
        data = {"option_class": self.__class__.__name__, "type_idx": int(self.type_idx), "color_idx": self.color_idx, "option_len": self.option_len}
        return self._save_checkpoint(file_path, data)

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        type_name = IDX_TO_OBJECT[checkpoint["type_idx"]]
        color = None if checkpoint["color_idx"] is None else COLOR_NAMES[checkpoint["color_idx"]]
        return cls(type_name, color, checkpoint["option_len"])


# ============================================================
# EXPLORATION / MOTION PRIMITIVES
# ============================================================

@register_option
class FollowWallOption(BaseOption, _NavHelperMixin):
    """Follow wall using left-hand (clockwise=False) or right-hand (clockwise=True) rule."""
    def __init__(self, clockwise=True, option_len=300):
        _NavHelperMixin.__init__(self)
        self.clockwise = clockwise
        self.option_len = option_len
        self.step_counter = 0

    def select_action(self, observation):
        self.step_counter += 1
        img = self._img(observation)
        pos, dir_vec = self._agent_pos_dir(observation)
        left_vec  = np.array([-dir_vec[1],  dir_vec[0]], dtype=int)
        right_vec = np.array([ dir_vec[1], -dir_vec[0]], dtype=int)
        front = self._front_cell(pos, dir_vec)
        side_vec = right_vec if self.clockwise else left_vec
        side_cell = pos + side_vec
        if self._is_inside(img, side_cell) and not self._is_wall(img, side_cell):
            # turn toward side, then move forward
            act = self._turn_toward(dir_vec, side_vec)
            if act is None:
                if not self._is_inside(img, front) or self._is_wall(img, front):
                    return A_LEFT
                return A_FWD
            return act
        if self._is_inside(img, front) and not self._is_wall(img, front):
            return A_FWD
        return A_RIGHT if self.clockwise else A_LEFT

    def is_terminated(self, observation):
        done = self.step_counter >= self.option_len
        if done:
            self.step_counter = 0
        return done

    def save(self, file_path=None):
        data = {"option_class": self.__class__.__name__, "clockwise": self.clockwise, "option_len": self.option_len}
        return self._save_checkpoint(file_path, data)

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        return cls(checkpoint["clockwise"], checkpoint["option_len"])


@register_option
class StepForwardUntilObstacleOption(BaseOption, _NavHelperMixin):
    def __init__(self, option_len=200):
        _NavHelperMixin.__init__(self)
        self.option_len = option_len
        self.step_counter = 0

    def select_action(self, observation):
        self.step_counter += 1
        img = self._img(observation)
        pos, dir_vec = self._agent_pos_dir(observation)
        front = self._front_cell(pos, dir_vec)
        if self._is_inside(img, front) and not self._is_wall(img, front):
            return A_FWD
        return A_LEFT

    def is_terminated(self, observation):
        done = self.step_counter >= self.option_len
        if done:
            self.step_counter = 0
        return done

    def save(self, file_path=None):
        data = {"option_class": self.__class__.__name__, "option_len": self.option_len}
        return self._save_checkpoint(file_path, data)

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        return cls(checkpoint["option_len"])


@register_option
class RandomWalkOption(BaseOption, _NavHelperMixin):
    def __init__(self, p_turn=0.3, option_len=200):
        _NavHelperMixin.__init__(self)
        self.p_turn = float(p_turn)
        self.option_len = option_len
        self.step_counter = 0

    def select_action(self, observation):
        # Simple stochastic policy: occasionally turn, else forward if clear
        self.step_counter += 1
        img = self._img(observation)
        pos, dir_vec = self._agent_pos_dir(observation)
        front = self._front_cell(pos, dir_vec)
        if np.random.rand() < self.p_turn:
            return A_LEFT if np.random.rand() < 0.5 else A_RIGHT
        if self._is_inside(img, front) and not self._is_wall(img, front):
            return A_FWD
        return A_LEFT

    def is_terminated(self, observation):
        done = self.step_counter >= self.option_len
        if done:
            self.step_counter = 0
        return done

    def save(self, file_path=None):
        data = {"option_class": self.__class__.__name__, "p_turn": self.p_turn, "option_len": self.option_len}
        return self._save_checkpoint(file_path, data)

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        return cls(checkpoint["p_turn"], checkpoint["option_len"])


@register_option
class ZigZagExploreOption(BaseOption, _NavHelperMixin):
    def __init__(self, stride=3, option_len=250):
        _NavHelperMixin.__init__(self)
        self.stride = int(stride)
        self.option_len = option_len
        self.step_counter = 0
        self._phase = 0
        self._steps_in_leg = 0

    def select_action(self, observation):
        self.step_counter += 1
        img = self._img(observation)
        pos, dir_vec = self._agent_pos_dir(observation)
        front = self._front_cell(pos, dir_vec)
        if self._steps_in_leg >= self.stride or not (self._is_inside(img, front) and not self._is_wall(img, front)):
            # turn and reset leg
            self._phase ^= 1
            self._steps_in_leg = 0
            return A_LEFT
        self._steps_in_leg += 1
        return A_FWD

    def is_terminated(self, observation):
        done = self.step_counter >= self.option_len
        if done:
            self.step_counter = 0
            self._phase = 0
            self._steps_in_leg = 0
        return done

    def save(self, file_path=None):
        data = {"option_class": self.__class__.__name__, "stride": self.stride, "option_len": self.option_len}
        return self._save_checkpoint(file_path, data)

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        return cls(checkpoint["stride"], checkpoint["option_len"])


@register_option
class PerimeterSweepOption(BaseOption, _NavHelperMixin):
    """Trace the boundary of the map by hugging outer walls."""
    def __init__(self, option_len=350):
        _NavHelperMixin.__init__(self)
        self.option_len = option_len
        self.step_counter = 0
        self._fw = FollowWallOption(clockwise=True, option_len=10)

    def select_action(self, observation):
        self.step_counter += 1
        # Delegate to short-horizon wall-follow to avoid recursion issues
        return self._fw.select_action(observation)

    def is_terminated(self, observation):
        done = self.step_counter >= self.option_len
        if done:
            self.step_counter = 0
        return done

    def save(self, file_path=None):
        return self._save_checkpoint(file_path, {"option_class": self.__class__.__name__, "option_len": self.option_len})

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        return cls(checkpoint["option_len"])


# ============================================================
# ROOM-LEVEL OPTIONS
# ============================================================

@register_option
class CenterRoomOption(BaseOption, _NavHelperMixin):
    """Move toward the centroid of current free cells as a proxy for room center."""
    def __init__(self, option_len=200):
        _NavHelperMixin.__init__(self)
        self.option_len = option_len
        self.step_counter = 0

    def _room_center(self, obs):
        img = self._img(obs)
        # Heuristic: free = not wall and not door (walkable or empty types)
        free_mask = img[..., 0] != self.wall_id
        coords = np.argwhere(free_mask)
        if coords.size == 0:
            return None
        center = coords.mean(axis=0)
        return np.rint(center).astype(int)

    def select_action(self, observation):
        self.step_counter += 1
        tgt = self._room_center(observation)
        if tgt is None:
            return A_DONE
        return self._step_toward_point(observation, tgt)

    def is_terminated(self, observation):
        done = self.step_counter >= self.option_len
        if done:
            self.step_counter = 0
        return done

    def save(self, file_path=None):
        return self._save_checkpoint(file_path, {"option_class": self.__class__.__name__, "option_len": self.option_len})

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        return cls(checkpoint["option_len"])


@register_option
class SearchRoomForOption(BaseOption, _NavHelperMixin):
    """Search current room for an object type; if not visible, perform wall-follow exploration."""
    def __init__(self, type_name: str, option_len=400):
        _NavHelperMixin.__init__(self)
        self.type_idx = OBJECT_TO_IDX[type_name]
        self.option_len = option_len
        self.step_counter = 0
        self._fallback = FollowWallOption(clockwise=False, option_len=10)

    def select_action(self, observation):
        self.step_counter += 1
        tgt = self._nearest_object_pos(observation, self.type_idx, None)
        if tgt is not None:
            return self._step_toward_point(observation, tgt)
        return self._fallback.select_action(observation)

    def is_terminated(self, observation):
        near = self._is_facing_object(observation, self.type_idx, None)
        done = near or self.step_counter >= self.option_len
        if done:
            self.step_counter = 0
        return done

    def save(self, file_path=None):
        data = {"option_class": self.__class__.__name__, "type_idx": int(self.type_idx), "option_len": self.option_len}
        return self._save_checkpoint(file_path, data)

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        type_name = IDX_TO_OBJECT[checkpoint["type_idx"]]
        return cls(type_name, checkpoint["option_len"])


@register_option
class ClearRoomOption(BaseOption, _NavHelperMixin):
    """Systematically visit free cells with simple sweeping pattern until time cap."""
    def __init__(self, option_len=500):
        _NavHelperMixin.__init__(self)
        self.option_len = option_len
        self.step_counter = 0
        self._sweeper = ZigZagExploreOption(stride=4, option_len=10)

    def select_action(self, observation):
        self.step_counter += 1
        return self._sweeper.select_action(observation)

    def is_terminated(self, observation):
        done = self.step_counter >= self.option_len
        if done:
            self.step_counter = 0
        return done

    def save(self, file_path=None):
        return self._save_checkpoint(file_path, {"option_class": self.__class__.__name__, "option_len": self.option_len})

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        return cls(checkpoint["option_len"])


# ============================================================
# HEADING / ORIENTATION OPTIONS
# ============================================================

@register_option
class AlignHeadingOption(BaseOption, _NavHelperMixin):
    """Rotate until facing a global heading: one of {"up","right","down","left"}."""
    _NAME_TO_DIR = {
        "up": np.array([-1, 0]),
        "right": np.array([0, 1]),
        "down": np.array([1, 0]),
        "left": np.array([0, -1]),
    }

    def __init__(self, heading: str, option_len=40):
        _NavHelperMixin.__init__(self)
        assert heading in self._NAME_TO_DIR
        self.want_dir = self._NAME_TO_DIR[heading]
        self.option_len = option_len
        self.step_counter = 0

    def select_action(self, observation):
        self.step_counter += 1
        _, cur_dir = self._agent_pos_dir(observation)
        if np.array_equal(cur_dir, self.want_dir):
            return A_DONE
        return self._turn_toward(cur_dir, self.want_dir)

    def is_terminated(self, observation):
        _, cur_dir = self._agent_pos_dir(observation)
        done = np.array_equal(cur_dir, self.want_dir) or self.step_counter >= self.option_len
        if done:
            self.step_counter = 0
        return done

    def save(self, file_path=None):
        return self._save_checkpoint(file_path, {"option_class": self.__class__.__name__, "want_dir": self.want_dir.tolist(), "option_len": self.option_len})

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        inv = {(-1,0): "up", (0,1): "right", (1,0): "down", (0,-1): "left"}
        key = tuple(checkpoint["want_dir"])
        return cls(inv[tuple(key)], checkpoint["option_len"])


@register_option
class TurnTowardObjectOption(BaseOption, _NavHelperMixin):
    """Rotate to face nearest object of a given type/color."""
    def __init__(self, type_name: str, color: Optional[str] = None, option_len=60):
        _NavHelperMixin.__init__(self)
        self.type_idx = OBJECT_TO_IDX[type_name]
        self.color_idx = None if color is None else COLOR_TO_IDX[color]
        self.option_len = option_len
        self.step_counter = 0

    def select_action(self, observation):
        self.step_counter += 1
        pos, cur_dir = self._agent_pos_dir(observation)
        tgt = self._nearest_object_pos(observation, self.type_idx, self.color_idx)
        if tgt is None:
            return A_DONE
        delta = tgt - pos
        act = self._turn_toward(cur_dir, delta)
        return A_DONE if act is None else act

    def is_terminated(self, observation):
        done = self.step_counter >= self.option_len or self._is_facing_object(observation, self.type_idx, self.color_idx)
        if done:
            self.step_counter = 0
        return done

    def save(self, file_path=None):
        data = {"option_class": self.__class__.__name__, "type_idx": int(self.type_idx), "color_idx": self.color_idx, "option_len": self.option_len}
        return self._save_checkpoint(file_path, data)

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        type_name = IDX_TO_OBJECT[checkpoint["type_idx"]]
        color = None if checkpoint["color_idx"] is None else COLOR_NAMES[checkpoint["color_idx"]]
        return cls(type_name, color, checkpoint["option_len"])


@register_option
class Scan360Option(BaseOption, _NavHelperMixin):
    """Do a full rotation (4 turns) to normalize orientation invariance (mostly for debugging)."""
    def __init__(self, option_len=10):
        _NavHelperMixin.__init__(self)
        self.option_len = option_len
        self.step_counter = 0
        self._turns = 0

    def select_action(self, observation):
        self.step_counter += 1
        if self._turns < 4:
            self._turns += 1
            return A_LEFT
        return A_DONE

    def is_terminated(self, observation):
        done = self._turns >= 4 or self.step_counter >= self.option_len
        if done:
            self.step_counter = 0
            self._turns = 0
        return done

    def save(self, file_path=None):
        return self._save_checkpoint(file_path, {"option_class": self.__class__.__name__, "option_len": self.option_len})

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        return cls(checkpoint["option_len"])


# ============================================================
# SAFETY / AVOIDANCE OPTIONS
# ============================================================

@register_option
class AvoidLavaOption(BaseOption, _NavHelperMixin):
    """If lava exists, try to step away from nearest lava cell; else do nothing."""
    def __init__(self, option_len=80):
        _NavHelperMixin.__init__(self)
        self.option_len = option_len
        self.step_counter = 0

    def select_action(self, observation):
        self.step_counter += 1
        if self.lava_id is None:
            return A_DONE
        lava_coords = self._all_object_positions(observation, self.lava_id)
        if lava_coords.size == 0:
            return A_DONE
        pos, dir_vec = self._agent_pos_dir(observation)
        # Push away from closest lava
        dists = np.abs(lava_coords - pos).sum(axis=1)
        closest = lava_coords[np.argmin(dists)]
        delta = pos - closest  # move away
        # If delta is zero (on lava?), just turn
        turn = self._turn_toward(dir_vec, delta if not np.all(delta == 0) else np.array([1, 0]))
        if turn is not None:
            return turn
        img = self._img(observation)
        front = self._front_cell(pos, dir_vec)
        if self._is_inside(img, front) and not self._is_wall(img, front):
            return A_FWD
        return A_LEFT

    def is_terminated(self, observation):
        done = self.step_counter >= self.option_len
        if done:
            self.step_counter = 0
        return done

    def save(self, file_path=None):
        return self._save_checkpoint(file_path, {"option_class": self.__class__.__name__, "option_len": self.option_len})

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        return cls(checkpoint["option_len"])


# ============================================================
# COMPOSITE / MACRO OPTIONS
# ============================================================

@register_option
class GetKeyAndOpenNearestDoorOption(BaseOption, _NavHelperMixin):
    """Find nearest key -> pick up -> go to nearest door -> toggle (open/unlock)."""
    def __init__(self, option_len=800):
        _NavHelperMixin.__init__(self)
        self.option_len = option_len
        self.step_counter = 0
        self._phase = "find_key"

    def select_action(self, observation):
        self.step_counter += 1
        if self._phase == "find_key":
            if self._is_facing_object(observation, self.key_id, None):
                self._phase = "open_door"
                return A_PICK
            tgt = self._nearest_object_pos(observation, self.key_id, None)
            if tgt is None:
                return A_DONE
            return self._step_toward_point(observation, tgt)
        if self._phase == "open_door":
            if self._is_facing_object(observation, self.door_id, None):
                return A_TOG
            tgt = self._nearest_object_pos(observation, self.door_id, None)
            if tgt is None:
                return A_DONE
            return self._step_toward_point(observation, tgt)
        return A_DONE

    def is_terminated(self, observation):
        done = self.step_counter >= self.option_len or (self._phase == "open_door" and self._is_facing_object(observation, self.door_id, None))
        if done:
            self.step_counter = 0
            self._phase = "find_key"
        return done

    def save(self, file_path=None):
        return self._save_checkpoint(file_path, {"option_class": self.__class__.__name__, "option_len": self.option_len})

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        return cls(checkpoint["option_len"])


@register_option
class GoThroughNearestDoorwayOption(BaseOption, _NavHelperMixin):
    """Approach nearest door, toggle, then step forward across it; end once crossed."""
    def __init__(self, option_len=300):
        _NavHelperMixin.__init__(self)
        self.option_len = option_len
        self.step_counter = 0
        self._crossing = False

    def select_action(self, observation):
        self.step_counter += 1
        img = self._img(observation)
        pos, dir_vec = self._agent_pos_dir(observation)
        front = self._front_cell(pos, dir_vec)
        if self._is_facing_object(observation, self.door_id, None):
            self._crossing = True
            return A_TOG
        if self._crossing:
            if not self._is_inside(img, front) or self._is_wall(img, front):
                self._crossing = False
                return A_LEFT
            return A_FWD
        tgt = self._nearest_object_pos(observation, self.door_id, None)
        if tgt is None:
            return A_DONE
        return self._step_toward_point(observation, tgt)

    def is_terminated(self, observation):
        done = self.step_counter >= self.option_len or (self._crossing and not self._is_facing_object(observation, self.door_id, None))
        if done:
            self.step_counter = 0
            self._crossing = False
        return done

    def save(self, file_path=None):
        return self._save_checkpoint(file_path, {"option_class": self.__class__.__name__, "option_len": self.option_len})

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        return cls(checkpoint["option_len"])


@register_option
class ReturnToStartOption(BaseOption, _NavHelperMixin):
    """Record first seen position as origin and return to it."""
    def __init__(self, option_len=250):
        _NavHelperMixin.__init__(self)
        self.option_len = option_len
        self.step_counter = 0

    def select_action(self, observation):
        self.step_counter += 1
        pos, _ = self._agent_pos_dir(observation)
        if self._origin is None:
            self._origin = pos.copy()
            return A_DONE
        return self._step_toward_point(observation, self._origin)

    def is_terminated(self, observation):
        pos, _ = self._agent_pos_dir(observation)
        done = (self._origin is not None and np.all(pos == self._origin)) or self.step_counter >= self.option_len
        if done:
            self.step_counter = 0
            self._origin = None
        return done

    def save(self, file_path=None):
        return self._save_checkpoint(file_path, {"option_class": self.__class__.__name__, "option_len": self.option_len})

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        return cls(checkpoint["option_len"])


# ============================================================
# ANALYSIS / DIAGNOSTIC OPTIONS
# ============================================================

@register_option
class EvaluateExplorationCoverageOption(BaseOption, _NavHelperMixin):
    """Greedy novelty: prefer moving toward farthest unvisited free cell (approx via perimeter sweep)."""
    def __init__(self, option_len=400):
        _NavHelperMixin.__init__(self)
        self.option_len = option_len
        self.step_counter = 0
        self._inner = PerimeterSweepOption(option_len=10)

    def select_action(self, observation):
        self.step_counter += 1
        return self._inner.select_action(observation)

    def is_terminated(self, observation):
        done = self.step_counter >= self.option_len
        if done:
            self.step_counter = 0
        return done

    def save(self, file_path=None):
        return self._save_checkpoint(file_path, {"option_class": self.__class__.__name__, "option_len": self.option_len})

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        return cls(checkpoint["option_len"])
