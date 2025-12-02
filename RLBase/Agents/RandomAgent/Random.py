import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete

from ..Base import BaseAgent, BasePolicy
from ...registry import register_agent, register_policy


@register_policy
class RandomPolicy(BasePolicy):
    """
    Stateless policy that samples uniformly using the internal RNG (RandomGenerator).
    """
    def select_action(self, state=None, greedy: bool = False):
        space = self.action_space
        if isinstance(space, Discrete):
            return int(self._rand_int(0, space.n))
        if isinstance(space, MultiDiscrete):
            return self.np_random.integers(low=0, high=space.nvec)
        if isinstance(space, Box):
            if np.issubdtype(space.dtype, np.integer):
                return self.np_random.integers(low=space.low, high=space.high + 1, size=space.shape, dtype=space.dtype)
            else:
                return self.np_random.uniform(low=space.low, high=space.high, size=space.shape)
            
        raise NotImplementedError(f"RandomPolicy does not support action space of type {type(space)}")


@register_agent
class RandomAgent(BaseAgent):
    name = "Random"
    SUPPORTED_ACTION_SPACES = (Discrete, MultiDiscrete, Box)

    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device="cpu"):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device=device)
        
        self.policy = RandomPolicy(action_space, hyper_params, device=device)

    def act(self, observation, greedy=False):
        # Vectorized Observation
        action = []
        for _ in range(self.num_envs):
            action.append(self.policy.select_action(None, greedy=greedy))
        return action
