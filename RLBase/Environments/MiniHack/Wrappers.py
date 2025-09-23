import copy
import importlib

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActionWrapper, ObservationWrapper, RewardWrapper
import minihack

class OneHotCharsWrapper(ObservationWrapper):
    def __init__(self, env, char_vocab=(" ", "-", "|", "#", ".", "<", ">", "@", "+")):
        super().__init__(env)
        self.char_vocab = char_vocab
        # Reserve last index for unknown
        self.char_to_idx = {c: i for i, c in enumerate(char_vocab)}
        self.unknown_idx = len(char_vocab)

        char_shape = env.observation_space["chars"].shape  # (H, W)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=char_shape + (len(char_vocab) + 1,),  # +1 for unknown
            dtype=np.int8,
        )

    def observation(self, obs):
        chars = obs["chars"]  # (H, W) integers (ASCII codes)
        # Convert ASCII -> characters
        char_array = np.vectorize(chr)(chars)
        # Map characters to indices (default → unknown_idx)
        idx_array = np.vectorize(self.char_to_idx.get)(char_array, self.unknown_idx)
        # One-hot encode
        one_hot = np.eye(len(self.char_vocab) + 1, dtype=np.int8)[idx_array]
        return one_hot
    
class FixedSeedWrapper(gym.Wrapper):
    def __init__(self, env, seed):
        super().__init__(env)
        self._seed = seed

    def reset(self, **kwargs):
        # force the same seed every reset
        kwargs.pop('seed', None)
        return super().reset(seed=self._seed, **kwargs)  


WRAPPING_TO_WRAPPER = {
    "OneHotChars": OneHotCharsWrapper,
    "FixedSeed": FixedSeedWrapper,

}