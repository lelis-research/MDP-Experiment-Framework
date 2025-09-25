import copy
import importlib

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActionWrapper, ObservationWrapper, RewardWrapper
from gymnasium.vector import VectorWrapper

import minihack
from nle import nethack

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
    """Always reset MiniHack with the same seed so the map layout is identical."""
    def __init__(self, env, seed: int):
        super().__init__(env)
        self._seed = int(seed)

    # Gymnasium API: match the signature exactly
    def reset(self, *, seed=None, options=None):
        # Always force the same seed (ignore caller's seed)
        base = self.env.unwrapped
        base.seed(self._seed)
        return self.env.reset(seed=self._seed, options=options)

class MovementActionWrapper(gym.Wrapper):
    """
    Restrict the action space to only movement actions in MiniHack/NLE.
    """
    def __init__(self, env):
        super().__init__(env)
        base = self.env.unwrapped
        # All 8 compass directions
        movement_actions = list(nethack.CompassDirection)
        base.actions = movement_actions

        # Reduced discrete space
        self.action_space = gym.spaces.Discrete(len(movement_actions))

class LastActionWrapper(ObservationWrapper):
    """
    Augment MiniHack observations with the last action taken by the agent.
    Adds a new key 'last_action' to the observation dict containing a one-hot vector.
    """

    def __init__(self, env):
        super().__init__(env)
        self.n_actions = env.action_space.n
        self.last_action = None

        if not isinstance(env.observation_space, spaces.Dict):
            raise TypeError("MiniHack env must have Dict observation space")

        # Extend observation_space with new entry for last_action
        new_spaces = dict(env.observation_space.spaces)
        new_spaces["last_action"] = spaces.Box(
            low=0, high=1, shape=(self.n_actions,), dtype=np.int8
        )
        self.observation_space = spaces.Dict(new_spaces)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.last_action = None
        return self.observation(obs), info

    def step(self, action):
        self.last_action = action
        obs, reward, terminated, truncated, info = super().step(action)
        return self.observation(obs), reward, terminated, truncated, info

    def observation(self, obs):
        one_hot = np.zeros(self.n_actions, dtype=np.int8)
        if self.last_action is not None:
            one_hot[self.last_action] = 1
        obs = dict(obs)  # copy to avoid mutating original
        obs["last_action"] = one_hot
        return obs


WRAPPING_TO_WRAPPER = {
    "OneHotChars": OneHotCharsWrapper,
    "FixedSeed": FixedSeedWrapper,
    "MovementAction": MovementActionWrapper,
    "LastAction": LastActionWrapper,
}