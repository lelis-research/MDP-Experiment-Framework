import numpy as np
from typing import Any, Optional, Tuple
from minigrid.core.constants import OBJECT_TO_IDX as OID, COLOR_TO_IDX, STATE_TO_IDX

from ...Base import BaseOption
from ....registry import register_option
from .MiniGridHelper import *
from ....Agents.Utils.HyperParams import HyperParameters

@register_option
class ActionLeft(BaseOption):
    def __init__(self, option_id: Optional[str] = "Turn Left", hyper_params = None, device = "cpu"):
        super().__init__(option_id, hyper_params, device)
        self.counter = 0
    
    def select_action(self, observation, internal_state = None):
        self.counter += 1
        return A_LEFT
    
    def is_terminated(self, observation, internal_state = None):
        if self.counter >= 1:
            self.counter = 0
            return True
        return False
    
    def can_initiate(self, observation: Any) -> bool:
        return True
    
    def reset(self):
        self.counter = 0

@register_option
class ActionRight(BaseOption):
    def __init__(self, option_id: Optional[str] = "Turn Right", hyper_params = None, device = "cpu"):
        super().__init__(option_id, hyper_params, device)
        self.counter = 0
    
    def select_action(self, observation, internal_state = None):
        self.counter += 1
        return A_RIGHT
    
    def is_terminated(self, observation, internal_state = None):
        if self.counter >= 1:
            self.counter = 0
            return True
        return False
    
    def can_initiate(self, observation: Any) -> bool:
        return True
    
    def reset(self):
        self.counter = 0

@register_option
class ActionForward(BaseOption):
    def __init__(self, option_id: Optional[str] = "Move Forward", hyper_params = None, device = "cpu"):
        super().__init__(option_id, hyper_params, device)
        self.counter = 0
    
    def select_action(self, observation, internal_state = None):
        self.counter += 1
        return A_FORWARD
    
    def is_terminated(self, observation, internal_state = None):
        if self.counter >= 1:
            self.counter = 0
            return True
        return False
    
    def can_initiate(self, observation: Any) -> bool:
        return True
    
    def reset(self):
        self.counter = 0

@register_option
class ActionPickup(BaseOption):
    def __init__(self, option_id: Optional[str] = "Pick Up", hyper_params = None, device = "cpu"):
        super().__init__(option_id, hyper_params, device)
        self.counter = 0
    
    def select_action(self, observation, internal_state = None):
        self.counter += 1
        return A_PICKUP
    
    def is_terminated(self, observation, internal_state = None):
        if self.counter >= 1:
            self.counter = 0
            return True
        return False
    
    def can_initiate(self, observation: Any) -> bool:
        return True
    
    def reset(self):
        self.counter = 0


@register_option
class ActionDrop(BaseOption):
    def __init__(self, option_id: Optional[str] = "Drop", hyper_params = None, device = "cpu"):
        super().__init__(option_id, hyper_params, device)
        self.counter = 0
    
    def select_action(self, observation, internal_state = None):
        self.counter += 1
        return A_DROP
    
    def is_terminated(self, observation, internal_state = None):
        if self.counter >= 1:
            self.counter = 0
            return True
        return False
    
    def can_initiate(self, observation: Any) -> bool:
        return True
    
    def reset(self):
        self.counter = 0
        
@register_option
class ActionToggle(BaseOption):
    def __init__(self, option_id: Optional[str] = "Toggle", hyper_params = None, device = "cpu"):
        super().__init__(option_id, hyper_params, device)
        self.counter = 0
    
    def select_action(self, observation, internal_state = None):
        self.counter += 1
        return A_TOGGLE
    
    def is_terminated(self, observation, internal_state = None):
        if self.counter >= 1:
            self.counter = 0
            return True
        return False
    
    def can_initiate(self, observation: Any) -> bool:
        return True
    
    def reset(self):
        self.counter = 0
        
@register_option
class ActionDone(BaseOption):
    def __init__(self, option_id: Optional[str] = "Done", hyper_params = None, device = "cpu"):
        super().__init__(option_id, hyper_params, device)
        self.counter = 0
    
    def select_action(self, observation, internal_state = None):
        self.counter += 1
        return A_DONE
    
    def is_terminated(self, observation, internal_state = None):
        if self.counter >= 1:
            self.counter = 0
            return True
        return False
    
    def can_initiate(self, observation: Any) -> bool:
        return True
    
    def reset(self):
        self.counter = 0


@register_option
class GetNearestGoalOption(BaseOption, GridNavMixin):
    def __init__(self, option_id: Optional[str] = "get_nearest_goal", hyper_params=None, device: str = "cpu"):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device)
        self.counter = 0
        if not hasattr(self.hp, 'goal_color'):
            self.hp.goal_color = None

    def can_initiate(self, observation: Any) -> bool:
        path = self._find_nearest_of_object(
            observation,
            obj_id=OID["goal"],
            color_id=COLOR_TO_IDX.get(self.hp.goal_color, None),
            avoid_lava=True,
        )
        return path is not None

    def should_initiate(self, observation: Any) -> bool:
        return self.can_initiate(observation)

    def reward_func(self, observation: Any, action) -> float:
        path = self._find_nearest_of_object(
            observation,
            obj_id=OID["goal"],
            color_id=COLOR_TO_IDX.get(self.hp.goal_color, None),
            avoid_lava=True,
        )
        reward = -len(path) if path is not None else 0
        return reward

    def select_action(self, observation: Any, internal_state: Optional[Any] = None):
        path = self._find_nearest_of_object(
            observation,
            obj_id=OID["goal"],
            color_id=COLOR_TO_IDX.get(self.hp.goal_color, None),
            avoid_lava=True,
        )
        
        if path is None:
            return int(A_DONE)

        action = self._step_towards(observation, path)
        action = A_DONE if action is None else action
        self.counter += 1
        
        return action

    def is_terminated(self, observation: Any, internal_state: Optional[Any] = None) -> bool:
        path = self._find_nearest_of_object(
            observation,
            obj_id=OID["goal"],
            color_id=COLOR_TO_IDX.get(self.hp.goal_color, None),
            avoid_lava=True,
        )
        terminated = path is None or self.counter >= self.hp.option_max_len
    
        if terminated:
            self.counter = 0
        return terminated

    def reset(self, seed=None):
        super().reset(seed)
        self.counter = 0

@register_option
class ToggleNearestDoorOption(BaseOption, GridNavMixin):
    def __init__(self, option_id: Optional[str] = "toggle_nearest_door", hyper_params=None, device: str = "cpu"):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device)
        self.counter = 0
        self.toggled = False
        if not hasattr(self.hp, 'door_color'):
            self.hp.door_color = None
        if not hasattr(self.hp, 'door_state'):
            self.hp.door_state = None  # 'open' or 'closed' or 'locked'

    def can_initiate(self, observation: Any) -> bool:
        path = self._find_nearest_of_object(
            observation,
            obj_id=OID["door"],
            color_id=COLOR_TO_IDX.get(self.hp.door_color, None),
            state_id=STATE_TO_IDX.get(self.hp.door_state, None),
            avoid_lava=True,
        )
        return path is not None

    def should_initiate(self, observation: Any) -> bool:
        return self.can_initiate(observation)

    def reward_func(self, observation: Any, action) -> float:
        path = self._find_nearest_of_object(
            observation,
            obj_id=OID["door"],
            color_id=COLOR_TO_IDX.get(self.hp.door_color, None),
            state_id=STATE_TO_IDX.get(self.hp.door_state, None),
            avoid_lava=True,
        )
        reward = -len(path) if path is not None else 0
        return reward

    def select_action(self, observation: Any, internal_state: Optional[Any] = None):
        path = self._find_nearest_of_object(
            observation,
            obj_id=OID["door"],
            color_id=COLOR_TO_IDX.get(self.hp.door_color, None),
            state_id=STATE_TO_IDX.get(self.hp.door_state, None),
            avoid_lava=True,
        )
        
        if path is None:
            return int(A_DONE)
        
        door_xy = path[-1]
        agent = self._find_agent(observation)
        
        # Adjacent check using existing helper (no Manhattan)
        adjacent = any(np.array_equal(nb, door_xy) for nb in self._neighbors4(observation, (int(agent[0]), int(agent[1]))))

        if adjacent:
            # Face the door using existing helper
            turn = self._face_vec_action(observation, door_xy)
            if turn is not None:
                self.counter += 1
                return int(turn)

            # Now we are facing the door tile
            # (equivalently: self._facing_cell(observation, door_xy) should be True)
            if not self.toggled:
                self.toggled = True
                self.counter += 1
                return int(A_TOGGLE)

            # Already toggled -> finish
            return int(A_DONE)
        

        action = self._step_towards(observation, path)
        action = A_DONE if action is None else action
        self.counter += 1
        
        return action

    def is_terminated(self, observation: Any, internal_state: Optional[Any] = None) -> bool:
        if self.toggled:
            self.toggled = False
            self.counter = 0
            return True
        
        path = self._find_nearest_of_object(
            observation,
            obj_id=OID["door"],
            color_id=COLOR_TO_IDX.get(self.hp.door_color, None),
            state_id=STATE_TO_IDX.get(self.hp.door_state, None),
            avoid_lava=True,
        )
        terminated = path is None or self.counter >= self.hp.option_max_len
    
        if terminated:
            self.counter = 0
            self.toggled = False
        return terminated

    def reset(self, seed=None):
        super().reset(seed)
        self.counter = 0
        self.toggled = False

@register_option
class PickupNearestKeyOption(BaseOption, GridNavMixin):
    def __init__(self, option_id: Optional[str] = "pickup_nearest_key", hyper_params=None, device: str = "cpu"):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device)
        self.counter = 0
        self.picked = False

        if not hasattr(self.hp, "key_color"):
            self.hp.key_color = None  # e.g. "red", "green", ...

    def can_initiate(self, observation: Any) -> bool:
        path = self._find_nearest_of_object(
            observation,
            obj_id=OID["key"],
            color_id=COLOR_TO_IDX.get(self.hp.key_color, None),
            avoid_lava=True,
        )
        return path is not None

    def should_initiate(self, observation: Any) -> bool:
        return self.can_initiate(observation)

    def reward_func(self, observation: Any, action) -> float:
        path = self._find_nearest_of_object(
            observation,
            obj_id=OID["key"],
            color_id=COLOR_TO_IDX.get(self.hp.key_color, None),
            avoid_lava=True,
        )
        return -len(path) if path is not None else 0

    def select_action(self, observation: Any, internal_state: Optional[Any] = None):
        path = self._find_nearest_of_object(
            observation,
            obj_id=OID["key"],
            color_id=COLOR_TO_IDX.get(self.hp.key_color, None),
            avoid_lava=True,
        )

        if path is None:
            return int(A_DONE)

        key_xy = path[-1]
        agent = self._find_agent(observation)

        # Adjacent check using existing helper (no Manhattan)
        adjacent = any(
            np.array_equal(nb, key_xy)
            for nb in self._neighbors4(observation, (int(agent[0]), int(agent[1])))
        )

        if adjacent:
            # Face the key
            turn = self._face_vec_action(observation, key_xy)
            if turn is not None:
                self.counter += 1
                return int(turn)

            # (Now facing the key tile)
            if not self.picked:
                self.picked = True
                self.counter += 1
                return int(A_PICKUP)

            # Already attempted pickup -> finish
            return int(A_DONE)

        # Otherwise keep navigating
        action = self._step_towards(observation, path)
        action = A_DONE if action is None else action
        self.counter += 1
        return int(action)

    def is_terminated(self, observation: Any, internal_state: Optional[Any] = None) -> bool:
        if self.picked:
            self.picked = False
            self.counter = 0
            return True

        path = self._find_nearest_of_object(
            observation,
            obj_id=OID["key"],
            color_id=COLOR_TO_IDX.get(self.hp.key_color, None),
            avoid_lava=True,
        )
        terminated = path is None or self.counter >= self.hp.option_max_len

        if terminated:
            self.counter = 0
            self.picked = False
        return terminated

    def reset(self, seed=None):
        super().reset(seed)
        self.counter = 0
        self.picked = False

@register_option
class PickupNearestBallOption(BaseOption, GridNavMixin):
    def __init__(self, option_id: Optional[str] = "pickup_nearest_ball", hyper_params=None, device: str = "cpu"):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device)
        self.counter = 0
        self.picked = False

        if not hasattr(self.hp, "ball_color"):
            self.hp.ball_color = None  # e.g. "red", "green", ...

    def can_initiate(self, observation: Any) -> bool:
        path = self._find_nearest_of_object(
            observation,
            obj_id=OID["ball"],
            color_id=COLOR_TO_IDX.get(self.hp.ball_color, None),
            avoid_lava=True,
        )
        return path is not None

    def should_initiate(self, observation: Any) -> bool:
        return self.can_initiate(observation)

    def reward_func(self, observation: Any, action) -> float:
        path = self._find_nearest_of_object(
            observation,
            obj_id=OID["ball"],
            color_id=COLOR_TO_IDX.get(self.hp.ball_color, None),
            avoid_lava=True,
        )
        return -len(path) if path is not None else 0

    def select_action(self, observation: Any, internal_state: Optional[Any] = None):
        path = self._find_nearest_of_object(
            observation,
            obj_id=OID["ball"],
            color_id=COLOR_TO_IDX.get(self.hp.ball_color, None),
            avoid_lava=True,
        )

        if path is None:
            return int(A_DONE)

        ball_xy = path[-1]
        agent = self._find_agent(observation)

        adjacent = any(
            np.array_equal(nb, ball_xy)
            for nb in self._neighbors4(observation, (int(agent[0]), int(agent[1])))
        )

        if adjacent:
            turn = self._face_vec_action(observation, ball_xy)
            if turn is not None:
                self.counter += 1
                return int(turn)

            if not self.picked:
                self.picked = True
                self.counter += 1
                return int(A_PICKUP)

            return int(A_DONE)

        action = self._step_towards(observation, path)
        action = A_DONE if action is None else action
        self.counter += 1
        return int(action)

    def is_terminated(self, observation: Any, internal_state: Optional[Any] = None) -> bool:
        if self.picked:
            self.picked = False
            self.counter = 0
            return True

        path = self._find_nearest_of_object(
            observation,
            obj_id=OID["ball"],
            color_id=COLOR_TO_IDX.get(self.hp.ball_color, None),
            avoid_lava=True,
        )
        terminated = path is None or self.counter >= self.hp.option_max_len

        if terminated:
            self.counter = 0
            self.picked = False
        return terminated

    def reset(self, seed=None):
        super().reset(seed)
        self.counter = 0
        self.picked = False

@register_option
class PickupNearestBoxOption(BaseOption, GridNavMixin):
    def __init__(self, option_id: Optional[str] = "pickup_nearest_box", hyper_params=None, device: str = "cpu"):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device)
        self.counter = 0
        self.picked = False

        if not hasattr(self.hp, "box_color"):
            self.hp.box_color = None  # e.g. "red", "green", ...

    def can_initiate(self, observation: Any) -> bool:
        path = self._find_nearest_of_object(
            observation,
            obj_id=OID["box"],
            color_id=COLOR_TO_IDX.get(self.hp.box_color, None),
            avoid_lava=True,
        )
        return path is not None

    def should_initiate(self, observation: Any) -> bool:
        return self.can_initiate(observation)

    def reward_func(self, observation: Any, action) -> float:
        path = self._find_nearest_of_object(
            observation,
            obj_id=OID["box"],
            color_id=COLOR_TO_IDX.get(self.hp.box_color, None),
            avoid_lava=True,
        )
        return -len(path) if path is not None else 0

    def select_action(self, observation: Any, internal_state: Optional[Any] = None):
        path = self._find_nearest_of_object(
            observation,
            obj_id=OID["box"],
            color_id=COLOR_TO_IDX.get(self.hp.box_color, None),
            avoid_lava=True,
        )

        if path is None:
            return int(A_DONE)

        box_xy = path[-1]
        agent = self._find_agent(observation)

        adjacent = any(
            np.array_equal(nb, box_xy)
            for nb in self._neighbors4(observation, (int(agent[0]), int(agent[1])))
        )

        if adjacent:
            turn = self._face_vec_action(observation, box_xy)
            if turn is not None:
                self.counter += 1
                return int(turn)

            if not self.picked:
                self.picked = True
                self.counter += 1
                return int(A_PICKUP)

            return int(A_DONE)

        action = self._step_towards(observation, path)
        action = A_DONE if action is None else action
        self.counter += 1
        return int(action)

    def is_terminated(self, observation: Any, internal_state: Optional[Any] = None) -> bool:
        if self.picked:
            self.picked = False
            self.counter = 0
            return True

        path = self._find_nearest_of_object(
            observation,
            obj_id=OID["box"],
            color_id=COLOR_TO_IDX.get(self.hp.box_color, None),
            avoid_lava=True,
        )
        terminated = path is None or self.counter >= self.hp.option_max_len

        if terminated:
            self.counter = 0
            self.picked = False
        return terminated

    def reset(self, seed=None):
        super().reset(seed)
        self.counter = 0
        self.picked = False
        
@register_option
class ToggleNearestBoxOption(BaseOption, GridNavMixin):
    def __init__(self, option_id: Optional[str] = "toggle_nearest_box", hyper_params=None, device: str = "cpu"):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device)
        self.counter = 0
        self.toggled = False

        if not hasattr(self.hp, "box_color"):
            self.hp.box_color = None  # e.g. "red", "green", ...

    def can_initiate(self, observation: Any) -> bool:
        path = self._find_nearest_of_object(
            observation,
            obj_id=OID["box"],
            color_id=COLOR_TO_IDX.get(self.hp.box_color, None),
            avoid_lava=True,
        )
        return path is not None

    def should_initiate(self, observation: Any) -> bool:
        return self.can_initiate(observation)

    def reward_func(self, observation: Any, action) -> float:
        path = self._find_nearest_of_object(
            observation,
            obj_id=OID["box"],
            color_id=COLOR_TO_IDX.get(self.hp.box_color, None),
            avoid_lava=True,
        )
        return -len(path) if path is not None else 0

    def select_action(self, observation: Any, internal_state: Optional[Any] = None):
        path = self._find_nearest_of_object(
            observation,
            obj_id=OID["box"],
            color_id=COLOR_TO_IDX.get(self.hp.box_color, None),
            avoid_lava=True,
        )

        if path is None:
            return int(A_DONE)

        box_xy = path[-1]
        agent = self._find_agent(observation)

        # Adjacent check using existing helper (no Manhattan)
        adjacent = any(
            np.array_equal(nb, box_xy)
            for nb in self._neighbors4(observation, (int(agent[0]), int(agent[1])))
        )

        if adjacent:
            # Face the box
            turn = self._face_vec_action(observation, box_xy)
            if turn is not None:
                self.counter += 1
                return int(turn)

            # Now facing the box tile -> toggle once
            if not self.toggled:
                self.toggled = True
                self.counter += 1
                return int(A_TOGGLE)

            # Already toggled -> finish
            return int(A_DONE)

        # Otherwise keep navigating
        action = self._step_towards(observation, path)
        action = A_DONE if action is None else action
        self.counter += 1
        return int(action)

    def is_terminated(self, observation: Any, internal_state: Optional[Any] = None) -> bool:
        if self.toggled:
            self.toggled = False
            self.counter = 0
            return True

        path = self._find_nearest_of_object(
            observation,
            obj_id=OID["box"],
            color_id=COLOR_TO_IDX.get(self.hp.box_color, None),
            avoid_lava=True,
        )
        terminated = path is None or self.counter >= self.hp.option_max_len

        if terminated:
            self.counter = 0
            self.toggled = False
        return terminated

    def reset(self, seed=None):
        super().reset(seed)
        self.counter = 0
        self.toggled = False


K = len(COLOR_NAMES)
S = len(STATE_TO_IDX)

actions = [
    ActionLeft(), 
    ActionRight(), 
    ActionForward(), 
    ActionPickup(), 
    ActionDrop(), 
    ActionToggle(), 
    ActionDone()
    ]
action_embeddings = np.array([
    [ 0.0, -1.0,  0.0,  0.0,  0.0,  0.0, -0.8,  0.0],  # ActionLeft
    [ 0.0,  1.0,  0.0,  0.0,  0.0,  0.0, -0.8,  0.0],  # ActionRight
    [ 0.8,  0.0,  0.0,  0.0,  0.0,  0.0, -0.8,  0.0],  # ActionForward
    [ 0.0,  0.0,  0.9,  0.0,  0.0,  0.0, -0.8,  0.0],  # ActionPickup
    [ 0.0,  0.0,  0.0,  0.0,  0.9,  0.0, -0.8,  0.0],  # ActionDrop
    [ 0.0,  0.0,  0.0,  0.9,  0.0,  0.0, -0.8,  0.0],  # ActionToggle
    [ 0.0,  0.0,  0.0,  0.0,  0.0,  1.0, -0.8,  0.0],  # ActionDone
], dtype=np.float32)

goal_options = [
    GetNearestGoalOption(
        option_id=f"get_nearest_goal_{color}",
        hyper_params=HyperParameters(option_max_len=20, goal_color=color),
    )
    for color in COLOR_NAMES + [None]
]
goal_embeddings = np.array([
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.8, (-1.0 + 2.0 * i / (K - 1)) if K > 1 else 0.0] for i in range(K)] + 
    [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 0.0]  # color = None
    ], dtype=np.float32)

key_options = [
    PickupNearestKeyOption(
        option_id=f"pickup_nearest_key_{color}",
        hyper_params=HyperParameters(option_max_len=20, key_color=color),
    )
    for color in COLOR_NAMES + [None]
]
key_embeddings = np.array([
    [0.9, 0.0, 0.9, 0.0, 0.0, 0.2, 0.4, (-1.0 + 2.0 * i / (K - 1)) if K > 1 else 0.0] for i in range(K)] + 
    [
    [0.9, 0.0, 0.9, 0.0, 0.0, 0.2, 0.4, 0.0]  # color = None
    ], dtype=np.float32)

ball_options = [
    PickupNearestBallOption(
        option_id=f"pickup_nearest_ball_{color}",
        hyper_params=HyperParameters(option_max_len=20, ball_color=color),
    )
    for color in COLOR_NAMES + [None]
]
ball_embeddings = np.array([
    [0.9, 0.0, 0.9, 0.0, 0.0, 0.2, 0.2, (-1.0 + 2.0 * i / (K - 1)) if K > 1 else 0.0] for i in range(K)] + 
    [
    [0.9, 0.0, 0.9, 0.0, 0.0, 0.2, 0.2, 0.0]  # color = None
    ], dtype=np.float32)

box_pickup_options = [
    PickupNearestBoxOption(
        option_id=f"pickup_nearest_box_{color}",
        hyper_params=HyperParameters(option_max_len=20, box_color=color),
    )
    for color in COLOR_NAMES + [None]
]
box_pickup_embeddings = np.array([
    [0.9, 0.0, 0.9, 0.0, 0.0, 0.2, 0.0, (-1.0 + 2.0 * i / (K - 1)) if K > 1 else 0.0] for i in range(K)] + 
    [
    [0.9, 0.0, 0.9, 0.0, 0.0, 0.2, 0.0, 0.0]  # color = None
    ], dtype=np.float32)

box_toggle_options = [
    ToggleNearestBoxOption(
        option_id=f"toggle_nearest_box_{color}",
        hyper_params=HyperParameters(option_max_len=20, box_color=color),
    )
    for color in COLOR_NAMES + [None]
]
box_toggle_embeddings = np.array([
    [0.9, 0.0, 0.0, 0.9, 0.0, 0.2, 0.0, (-1.0 + 2.0 * i / (K - 1)) if K > 1 else 0.0] for i in range(K)] + 
    [
    [0.9, 0.0, 0.0, 0.9, 0.0, 0.2, 0.0, 0.0]  # color = None
    ], dtype=np.float32)


door_options = [
    ToggleNearestDoorOption(
        option_id=f"toggle_nearest_door_{color}_{state}",
        hyper_params=HyperParameters(
            option_max_len=20,
            door_color=color,
            door_state=state,
        ),
    )
    for color in COLOR_NAMES + [None]
    for state in STATE_TO_IDX.keys()
]
door_embeddings = np.array(
    [
        [0.9, 0.0, 0.0, 0.9, 0.0, 0.2, -0.4, 0.5 * ((-1.0 + 2.0 * i / (K - 1)) if K > 1 else 0.0) + 0.5 * ((-0.6 + 1.2 * j / (S - 1)) if S > 1 else 0.0)]
        for i in range(K)
        for j in range(S)
    ] + 
    [
        [0.9, 0.0, 0.0, 0.9, 0.0, 0.2, -0.4, (-0.6 + 1.2 * j / (S - 1)) if S > 1 else 0.0]
        for j in range(S)
    ],
    dtype=np.float32
)

manual_options = actions + goal_options + key_options + door_options
manual_embeddings = np.concatenate([
    action_embeddings,
    goal_embeddings,
    key_embeddings,
    door_embeddings,
], axis=0)

# print("action")
# print(action_embeddings)

# print("goal")
# print(goal_embeddings)

# print("key")
# print(key_embeddings)

# print("ball")
# print(ball_embeddings)

# print("box-pickup")
# print(box_pickup_embeddings)

# print("box-toggle")
# print(box_toggle_embeddings)

# print("door")
# print(door_embeddings)

