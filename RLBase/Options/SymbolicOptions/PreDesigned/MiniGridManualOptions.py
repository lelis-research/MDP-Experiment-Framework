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

    def _passable(self, cell: np.ndarray) -> bool:
        flag = super()._passable(cell)
        oid, color, st = int(cell[0]), int(cell[1]),int(cell[2])
        if self.hp.goal_color is None:
            if oid == OID["goal"]:
                return True
        else:
            if oid == OID["goal"] and color == COLOR_TO_IDX[self.hp.goal_color]:
                return True
        return flag
    
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
        return not self._is_carrying(observation) and path is not None

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
        if self._is_carrying(observation) or path is None:
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
        if self.picked or self._is_carrying(observation):
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
        return not self._is_carrying(observation) and path is not None

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

        if self._is_carrying(observation) or path is None:
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
        if self.picked or self._is_carrying(observation):
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
        return not self._is_carrying(observation) and path is not None

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

        if self._is_carrying(observation) or path is None:
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
        if self.picked or self._is_carrying(observation):
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


goal_options = [
    GetNearestGoalOption(
        option_id=f"get_nearest_goal_{color}",
        hyper_params=HyperParameters(option_max_len=20, goal_color=color),
    )
    for color in COLOR_NAMES + [None]
]


key_options = [
    PickupNearestKeyOption(
        option_id=f"pickup_nearest_key_{color}",
        hyper_params=HyperParameters(option_max_len=20, key_color=color),
    )
    for color in COLOR_NAMES #+ [None]
]


ball_options = [
    PickupNearestBallOption(
        option_id=f"pickup_nearest_ball_{color}",
        hyper_params=HyperParameters(option_max_len=20, ball_color=color),
    )
    for color in COLOR_NAMES #+ [None]
]


box_pickup_options = [
    PickupNearestBoxOption(
        option_id=f"pickup_nearest_box_{color}",
        hyper_params=HyperParameters(option_max_len=20, box_color=color),
    )
    for color in COLOR_NAMES #+ [None]
]


box_toggle_options = [
    ToggleNearestBoxOption(
        option_id=f"toggle_nearest_box_{color}",
        hyper_params=HyperParameters(option_max_len=20, box_color=color),
    )
    for color in COLOR_NAMES + [None]
]


door_options = [
    ToggleNearestDoorOption(
        option_id=f"toggle_nearest_door_{color}_{state}",
        hyper_params=HyperParameters(
            option_max_len=20,
            door_color=color,
            door_state=state,
        ),
    )
    for color in COLOR_NAMES #+ [None]
    for state in STATE_TO_IDX.keys()
]


manual_options = actions + goal_options + key_options + door_options
manual_option_lst1 = [
    GetNearestGoalOption(option_id=f"get_nearest_goal_red",hyper_params=HyperParameters(option_max_len=20, goal_color="red")),
    GetNearestGoalOption(option_id=f"get_nearest_goal_red",hyper_params=HyperParameters(option_max_len=20, goal_color="red")),
    GetNearestGoalOption(option_id=f"get_nearest_goal_red",hyper_params=HyperParameters(option_max_len=20, goal_color="red")),

    GetNearestGoalOption(option_id=f"get_nearest_goal_green",hyper_params=HyperParameters(option_max_len=20, goal_color="green")),
    GetNearestGoalOption(option_id=f"get_nearest_goal_green",hyper_params=HyperParameters(option_max_len=20, goal_color="green")),
    GetNearestGoalOption(option_id=f"get_nearest_goal_green",hyper_params=HyperParameters(option_max_len=20, goal_color="green")),
    ]

manual_option_lst2 = [
    GetNearestGoalOption(option_id=f"get_nearest_goal_red",hyper_params=HyperParameters(option_max_len=50, goal_color="red")), #0 
    GetNearestGoalOption(option_id=f"get_nearest_goal_red",hyper_params=HyperParameters(option_max_len=50, goal_color="red")), #1
    
    GetNearestGoalOption(option_id=f"get_nearest_goal_green",hyper_params=HyperParameters(option_max_len=50, goal_color="green")), #2
    GetNearestGoalOption(option_id=f"get_nearest_goal_green",hyper_params=HyperParameters(option_max_len=50, goal_color="green")), #3
    
    ToggleNearestDoorOption(option_id=f"toggle_nearest_door_red",hyper_params=HyperParameters(option_max_len=50, door_color="red")), #4
    ToggleNearestDoorOption(option_id=f"toggle_nearest_door_red",hyper_params=HyperParameters(option_max_len=50, door_color="red")), #5

    ToggleNearestDoorOption(option_id=f"toggle_nearest_door_green",hyper_params=HyperParameters(option_max_len=50, door_color="green")), #6
    ToggleNearestDoorOption(option_id=f"toggle_nearest_door_green",hyper_params=HyperParameters(option_max_len=50, door_color="green")), #7
    
    PickupNearestKeyOption(option_id=f"pickup_nearest_key_red",hyper_params=HyperParameters(option_max_len=50, key_color="red")), #8
    PickupNearestKeyOption(option_id=f"pickup_nearest_key_red",hyper_params=HyperParameters(option_max_len=50, key_color="red")), #9
    
    PickupNearestKeyOption(option_id=f"pickup_nearest_key_green",hyper_params=HyperParameters(option_max_len=50, key_color="green")), #10
    PickupNearestKeyOption(option_id=f"pickup_nearest_key_green",hyper_params=HyperParameters(option_max_len=50, key_color="green")), #11
    
    PickupNearestBallOption(option_id=f"pickup_nearest_ball_red",hyper_params=HyperParameters(option_max_len=50, ball_color="red")), #12
    PickupNearestBallOption(option_id=f"pickup_nearest_ball_red",hyper_params=HyperParameters(option_max_len=50, ball_color="red")), #13
    
    PickupNearestBallOption(option_id=f"pickup_nearest_ball_green",hyper_params=HyperParameters(option_max_len=50, ball_color="green")), #14
    PickupNearestBallOption(option_id=f"pickup_nearest_ball_green",hyper_params=HyperParameters(option_max_len=50, ball_color="green")), #15
    
    ]
manual_option_lst2_nodup = [
    # GetNearestGoalOption(option_id=f"get_nearest_goal_red",hyper_params=HyperParameters(option_max_len=20, goal_color="red")), #0 
    
    # GetNearestGoalOption(option_id=f"get_nearest_goal_green",hyper_params=HyperParameters(option_max_len=20, goal_color="green")), #2
    
    PickupNearestKeyOption(option_id=f"pickup_nearest_key_red",hyper_params=HyperParameters(option_max_len=20, key_color="red")), #8
    
    PickupNearestKeyOption(option_id=f"pickup_nearest_key_green",hyper_params=HyperParameters(option_max_len=20, key_color="green")), #10
    
    PickupNearestBallOption(option_id=f"pickup_nearest_ball_red",hyper_params=HyperParameters(option_max_len=20, ball_color="red")), #12
    
    PickupNearestBallOption(option_id=f"pickup_nearest_ball_green",hyper_params=HyperParameters(option_max_len=20, ball_color="green")), #14
    
    ToggleNearestDoorOption(option_id=f"toggle_nearest_door_red",hyper_params=HyperParameters(option_max_len=20, door_color="red")), #4

    ToggleNearestDoorOption(option_id=f"toggle_nearest_door_green",hyper_params=HyperParameters(option_max_len=20, door_color="green")), #6
    
    ]


manual_option_lst3_nodup = [
    PickupNearestKeyOption(option_id=f"pickup_nearest_key_red",hyper_params=HyperParameters(option_max_len=20, key_color="red")), #8
    PickupNearestKeyOption(option_id=f"pickup_nearest_key_green",hyper_params=HyperParameters(option_max_len=20, key_color="green")), #10
    PickupNearestKeyOption(option_id=f"pickup_nearest_key_blue",hyper_params=HyperParameters(option_max_len=20, key_color="blue")), #10
    PickupNearestKeyOption(option_id=f"pickup_nearest_key_purple",hyper_params=HyperParameters(option_max_len=20, key_color="purple")), #10
    PickupNearestKeyOption(option_id=f"pickup_nearest_key_yellow",hyper_params=HyperParameters(option_max_len=20, key_color="yellow")), #10
    PickupNearestKeyOption(option_id=f"pickup_nearest_key_grey",hyper_params=HyperParameters(option_max_len=20, key_color="grey")), #10

    ToggleNearestDoorOption(option_id=f"toggle_nearest_door_red",hyper_params=HyperParameters(option_max_len=20, door_color="red", door_state="locked")), #4
    ToggleNearestDoorOption(option_id=f"toggle_nearest_door_green",hyper_params=HyperParameters(option_max_len=20, door_color="green", door_state="locked")), #6
    ToggleNearestDoorOption(option_id=f"toggle_nearest_door_blue",hyper_params=HyperParameters(option_max_len=20, door_color="blue", door_state="locked")), #6
    ToggleNearestDoorOption(option_id=f"toggle_nearest_door_purple",hyper_params=HyperParameters(option_max_len=20, door_color="purple", door_state="locked")), #6
    ToggleNearestDoorOption(option_id=f"toggle_nearest_door_yellow",hyper_params=HyperParameters(option_max_len=20, door_color="yellow", door_state="locked")), #6
    ToggleNearestDoorOption(option_id=f"toggle_nearest_door_grey",hyper_params=HyperParameters(option_max_len=20, door_color="grey", door_state="locked")), #6
    ]

manual_option_lst4_nodup = [
    PickupNearestKeyOption(option_id=f"pickup_nearest_key_red",hyper_params=HyperParameters(option_max_len=20, key_color="red")), #8
    PickupNearestKeyOption(option_id=f"pickup_nearest_key_green",hyper_params=HyperParameters(option_max_len=20, key_color="green")), #10
    PickupNearestKeyOption(option_id=f"pickup_nearest_key_blue",hyper_params=HyperParameters(option_max_len=20, key_color="blue")), #10
    PickupNearestKeyOption(option_id=f"pickup_nearest_key_purple",hyper_params=HyperParameters(option_max_len=20, key_color="purple")), #10
    PickupNearestKeyOption(option_id=f"pickup_nearest_key_yellow",hyper_params=HyperParameters(option_max_len=20, key_color="yellow")), #10
    PickupNearestKeyOption(option_id=f"pickup_nearest_key_grey",hyper_params=HyperParameters(option_max_len=20, key_color="grey")), #10

    ToggleNearestDoorOption(option_id=f"toggle_nearest_door_red",hyper_params=HyperParameters(option_max_len=20, door_color="red", door_state="locked")), #4
    ToggleNearestDoorOption(option_id=f"toggle_nearest_door_green",hyper_params=HyperParameters(option_max_len=20, door_color="green", door_state="locked")), #6
    ToggleNearestDoorOption(option_id=f"toggle_nearest_door_blue",hyper_params=HyperParameters(option_max_len=20, door_color="blue", door_state="locked")), #6
    ToggleNearestDoorOption(option_id=f"toggle_nearest_door_purple",hyper_params=HyperParameters(option_max_len=20, door_color="purple", door_state="locked")), #6
    ToggleNearestDoorOption(option_id=f"toggle_nearest_door_yellow",hyper_params=HyperParameters(option_max_len=20, door_color="yellow", door_state="locked")), #6
    ToggleNearestDoorOption(option_id=f"toggle_nearest_door_grey",hyper_params=HyperParameters(option_max_len=20, door_color="grey", door_state="locked")), #6
    
    PickupNearestBoxOption(option_id=f"pickup_nearest_box_red",hyper_params=HyperParameters(option_max_len=20, box_color="red")), #8
    PickupNearestBoxOption(option_id=f"pickup_nearest_box_green",hyper_params=HyperParameters(option_max_len=20, box_color="green")), #10
    PickupNearestBoxOption(option_id=f"pickup_nearest_box_blue",hyper_params=HyperParameters(option_max_len=20, box_color="blue")), #10
    PickupNearestBoxOption(option_id=f"pickup_nearest_box_purple",hyper_params=HyperParameters(option_max_len=20, box_color="purple")), #10
    PickupNearestBoxOption(option_id=f"pickup_nearest_box_yellow",hyper_params=HyperParameters(option_max_len=20, box_color="yellow")), #10
    PickupNearestBoxOption(option_id=f"pickup_nearest_box_grey",hyper_params=HyperParameters(option_max_len=20, box_color="grey")), #10 
    ] + actions

manual_option_lst5_nodup = [
    PickupNearestKeyOption(option_id=f"pickup_nearest_key",hyper_params=HyperParameters(option_max_len=20, key_color=None)), #8
    ToggleNearestDoorOption(option_id=f"toggle_nearest_door",hyper_params=HyperParameters(option_max_len=20, door_color=None, door_state="locked")), #4
    PickupNearestBoxOption(option_id=f"pickup_nearest_box",hyper_params=HyperParameters(option_max_len=20, box_color=None)), #8

    ] + actions