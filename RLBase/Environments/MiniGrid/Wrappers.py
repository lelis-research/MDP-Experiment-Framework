import gymnasium as gym
import numpy as np
from gymnasium.core import ActionWrapper, ObservationWrapper, RewardWrapper
from minigrid.wrappers import (
    ViewSizeWrapper,
    FullyObsWrapper,
)




WRAPPING_TO_WRAPPER = {
    "ViewSize": ViewSizeWrapper,
    "FullyObs": FullyObsWrapper,
}
