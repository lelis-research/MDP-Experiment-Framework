import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActionWrapper, ObservationWrapper
from nle import nethack

ALLOWED_OBS_SPACES = (gym.spaces.Box, gym.spaces.Discrete, gym.spaces.Dict)


def _require_allowed_space(space, wrapper_name):
    if not isinstance(space, ALLOWED_OBS_SPACES):
        raise TypeError(
            f"{wrapper_name} requires observation_space to be Box, Discrete, or Dict, "
            f"but got {type(space).__name__}"
        )


class OneHotCharsWrapper(ObservationWrapper):
    """One-hot encode the ASCII character grid into an extra channel dimension."""

    def __init__(self, env, char_vocab=(" ", "-", "|", "#", ".", "<", ">", "@", "+")):
        super().__init__(env)
        _require_allowed_space(env.observation_space, self.__class__.__name__)
        if "chars" not in env.observation_space:
            raise KeyError("Observation must contain 'chars'")

        self.char_vocab = tuple(char_vocab)
        self.unknown_idx = len(self.char_vocab)
        self.char_to_idx = {c: i for i, c in enumerate(self.char_vocab)}

        w, h = env.observation_space["chars"].shape
        new_spaces = dict(env.observation_space.spaces)
        new_spaces["chars"] = spaces.Box(
            low=0,
            high=1,
            shape=(w, h, len(self.char_vocab) + 1),
            dtype=np.int8,
        )
        self.observation_space = spaces.Dict(new_spaces)

    def observation(self, obs):
        chars = np.vectorize(chr)(obs["chars"])
        idx = np.vectorize(self.char_to_idx.get)(chars, self.unknown_idx)
        obs = dict(obs)
        obs["chars"] = np.eye(len(self.char_vocab) + 1, dtype=np.int8)[idx]
        return obs


class MovementActionWrapper(ActionWrapper):
    """Restrict actions to the set of compass directions used in MiniHack."""

    def __init__(self, env):
        super().__init__(env)
        self.movement_actions = list(nethack.CompassDirection)
        self.action_space = spaces.Discrete(len(self.movement_actions))

    def action(self, action):
        return self.movement_actions[action]



WRAPPING_TO_WRAPPER = {
    "OneHotChars": OneHotCharsWrapper,
    "MovementAction": MovementActionWrapper,
}
