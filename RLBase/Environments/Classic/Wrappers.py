import gymnasium as gym
import numpy as np
from gymnasium.core import ActionWrapper, ObservationWrapper, RewardWrapper, Wrapper
from gymnasium.wrappers import (
    ClipAction,
    ClipReward,
    FrameStackObservation,
    NormalizeObservation,
    NormalizeReward,
    RescaleAction,
    RescaleObservation,
    TransformAction,
    TransformObservation,
)
from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar
ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")

class IdentityWrapper(gym.Wrapper):
    """No-op wrapper to keep the wrapper chain composable."""
    def __init__(self, env):
        super().__init__(env)

class RecordActualRewardWrapper(Wrapper):
    """Wrapper to record the actual reward before any modification."""
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        info["actual_reward"] = reward
        return observation, reward, terminated, truncated, info
     

WRAPPING_TO_WRAPPER = {
    "identity": IdentityWrapper,
    # Observation wrappers
    "FrameStackObservation": FrameStackObservation,
    "TransformObservation": TransformObservation,
    "NormalizeObservation": NormalizeObservation,
    "RescaleObservation": RescaleObservation,
    
    # Action wrappers
    "ClipAction": ClipAction,
    "TransformAction": TransformAction,
    "RescaleAction": RescaleAction,
   
    # Reward wrappers
    "ClipReward": ClipReward,
    "NormalizeReward": NormalizeReward,
    "RecordActualReward": RecordActualRewardWrapper,
}
