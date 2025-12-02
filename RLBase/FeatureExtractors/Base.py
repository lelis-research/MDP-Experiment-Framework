

from gymnasium.spaces import Box, Discrete, Dict, MultiDiscrete
import numpy as np
import torch

from ..utils import RandomGenerator

ALLOWED_SPACES = (Box, Discrete, Dict, MultiDiscrete)

class BaseFeature(RandomGenerator):
    # NOTE: Feature Extractor always assumes the input has a batch dimension (even if it's 1)
    def __init__(self, observation_space, device='cpu', allowed_spaces=ALLOWED_SPACES):
        self.allowed_spaces = allowed_spaces
        self._validate_space(observation_space)
        
        self.observation_space = observation_space
        self.device = device
        
    
    def _validate_space(self, space):
        if not isinstance(space, self.allowed_spaces):
            raise TypeError(f"{self.__class__.__name__} only supports {self.allowed_spaces} spaces (got {type(space).__name__})")

        if isinstance(space, Dict):
            for key, subspace in space.spaces.items():
                if not isinstance(subspace, self.allowed_spaces):
                    raise TypeError(
                        f"{self.__class__.__name__} Dict subspace '{key}' must be {self.allowed_spaces} "
                        f"(got {type(subspace).__name__})"
                    )
    
    @property
    def features_dict(self):
        raise NotImplementedError("features_dict property must be implemented by the child class")
    
    def __call__(self, observation):
        raise NotImplementedError("Must be implemented by the child class")
    
    def update(self):
        pass

    def reset(self, seed):
        self.set_seed(seed)
    
    def save(self, file_path=None):
        checkpoint = {
            'observation_space': self.observation_space,
            'feature_extractor_class': self.__class__.__name__,
            'rng_state': self.get_rng_state(),
        }
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_feature_extractor.t")
        return checkpoint
    
    @classmethod
    def load(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        instance = cls(checkpoint['observation_space'])
        instance.set_rng_state(checkpoint['rng_state'])
        return instance
