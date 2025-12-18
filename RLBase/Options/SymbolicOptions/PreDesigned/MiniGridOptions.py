import numpy as np
from typing import Any, Optional, Tuple
from minigrid.core.constants import OBJECT_TO_IDX as OID, COLOR_TO_IDX

from ...Base import BaseOption
from ....registry import register_option
from .MiniGridHelper import *

@register_option
class test_option(BaseOption):
    def __init__(self):
        self.counter = 0
        self.option_id = "test"
        self.hp = None
        self.device = None

    def select_action(self, obsersvation):
        self.counter += 1
        return 1

    def is_terminated(self, observation):
        if self.counter >= 5:
            self.counter = 0
            return True
        return False
    
    def reset(self):
        self.counter = 0



@register_option
class GoToGreenGoalOption(BaseOption, GridNavMixin):
    def __init__(self, option_id: Optional[str] = "go_to_green_goal", hyper_params=None, device: str = "cpu"):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device)
        self.counter = 0

    def _path_to_green_goal(self, obs):
        return self._find_nearest_of_type(
            obs,
            obj_id_filter=OID["goal"],
            also_require=lambda xy, c: int(c[1]) == int(COLOR_TO_IDX["green"]),
            avoid_lava=True,
        )

    def _standing_on_green_goal(self, obs) -> bool:
        img = self._img(obs)
        agent = self._find_agent(obs)  # (x,y)
        c = img[int(agent[0]), int(agent[1])]
        return int(c[0]) == int(OID["goal"]) and int(c[1]) == int(COLOR_TO_IDX["green"])

    def can_initiate(self, observation: Any) -> bool:
        path = self._path_to_green_goal(observation)
        return path is not None and len(path) >= 2

    def should_initiate(self, observation: Any) -> bool:
        return self.can_initiate(observation)

    def reward_func(self, observation: Any) -> float:
        return 1.0 if self._standing_on_green_goal(observation) else 0.0

    def select_action(self, observation: Any, internal_state: Optional[Any] = None):
        if self._standing_on_green_goal(observation):
            return int(A_DONE)

        path = self._path_to_green_goal(observation)
        if path is None or len(path) < 2:
            return int(A_LEFT)

        act = self._step_towards(observation, path)

        self.counter += 1

        return int(A_LEFT if act is None else act)

    def is_terminated(self, observation: Any, internal_state: Optional[Any] = None) -> bool:
        terminated = (
            self._standing_on_green_goal(observation)
            or (self._path_to_green_goal(observation) is None)
            or self.counter >= 10
        )
        if terminated:
            self.counter = 0
        return terminated

    def reset(self, seed=None):
        super().reset(seed)
        self.counter = 0

@register_option
class GoToRedGoalOption(BaseOption, GridNavMixin):
    def __init__(self, option_id: Optional[str] = "go_to_red_goal", hyper_params=None, device: str = "cpu"):
        super().__init__(option_id=option_id, hyper_params=hyper_params, device=device)
        self.counter = 0

    def _path_to_red_goal(self, obs):
        return self._find_nearest_of_type(
            obs,
            obj_id_filter=OID["goal"],
            also_require=lambda xy, c: int(c[1]) == int(COLOR_TO_IDX["red"]),
            avoid_lava=True,
        )

    def _standing_on_red_goal(self, obs) -> bool:
        img = self._img(obs)
        agent = self._find_agent(obs)  # (x,y)
        c = img[int(agent[0]), int(agent[1])]
        return int(c[0]) == int(OID["goal"]) and int(c[1]) == int(COLOR_TO_IDX["red"])

    def can_initiate(self, observation: Any) -> bool:
        path = self._path_to_red_goal(observation)
        return path is not None and len(path) >= 2

    def should_initiate(self, observation: Any) -> bool:
        return self.can_initiate(observation)

    def reward_func(self, observation: Any) -> float:
        return 1.0 if self._standing_on_red_goal(observation) else 0.0

    def select_action(self, observation: Any, internal_state: Optional[Any] = None):
        if self._standing_on_red_goal(observation):
            return int(A_DONE)

        path = self._path_to_red_goal(observation)
        if path is None or len(path) < 2:
            return int(A_LEFT)

        act = self._step_towards(observation, path)
        
        self.counter += 1
        
        return int(A_LEFT if act is None else act)

    def is_terminated(self, observation: Any, internal_state: Optional[Any] = None) -> bool:
        terminated = self._standing_on_red_goal(observation) or (self._path_to_red_goal(observation) is None) or self.counter >= 10
        if terminated:
            self.counter = 0
        return terminated
    def reset(self, seed = None):
        super().reset(seed)
        self.counter = 0