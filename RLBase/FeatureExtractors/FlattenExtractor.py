
from gymnasium.spaces import Box, Discrete, Dict, MultiDiscrete
import numpy as np
import gymnasium

from ..registry import register_feature_extractor
from .Base import BaseFeature

ALLOWED_SPACES = (Box, Discrete, Dict, MultiDiscrete)

@register_feature_extractor
class FlattenFeature(BaseFeature):
    """
    Convert supported Gymnasium observations into hashable tuples.
    """

    def __init__(self, observation_space, device="cpu"):
        super().__init__(observation_space, device=device, allowed_spaces=ALLOWED_SPACES)
        self._features_dict = {"x": self._flat_dim(observation_space)}
        
    def _flat_dim(self, space):
        if isinstance(space, Discrete):
            return 1
        
        elif isinstance(space, MultiDiscrete):
            # shape is (len(nvec),), so flattened dim is the number of elements
            return int(np.prod(space.shape))  # or len(space.nvec)
        
        elif isinstance(space, Box):
            return int(np.prod(space.shape))
        
        elif isinstance(space, Dict):
            return sum(self._flat_dim(subspace) for subspace in space.spaces.values())
        
        else:
            raise TypeError(f"Unsupported space {type(space)}")
        

    def __call__(self, observation):
        if isinstance(self.observation_space, Discrete):
            flat = self._encode_discrete(observation)
        elif isinstance(self.observation_space, MultiDiscrete):
            flat = self._encode_multi_discrete(observation)
        elif isinstance(self.observation_space, Box):
            flat = self._encode_box(observation)
        elif isinstance(self.observation_space, Dict):
            flat = self._encode_dict(observation)
        else:
            raise ValueError(f"Observation Space {self.observation_space} is not supported by {self.__class__.__name__}")
        
        return flat

    @property
    def features_dict(self):
        return self._features_dict


    
    def _encode_discrete(self, observation, observation_space=None):
        observation_space = self.observation_space if observation_space is None else observation_space
        assert isinstance(observation, np.ndarray), f"Expected observation to be np.ndarray, got {type(observation)}"
        assert len(observation.shape) == 1, f"Expected observation to have shape (batch_size, ), got {observation.shape}"
            
        flat = observation.reshape(-1, 1)

        return flat

    def _encode_multi_discrete(self, observation, observation_space=None):
        observation_space = self.observation_space if observation_space is None else observation_space
        assert isinstance(observation, np.ndarray), f"Expected observation to be np.ndarray, got {type(observation)}"
        assert observation.shape[1:] == observation_space.shape, f"Expected observation to have shape {observation_space.shape}, got {observation.shape}"
        
        flat = observation.reshape(observation.shape[0], -1)
        return flat
        
    def _encode_box(self, observation, observation_space=None):
        observation_space = self.observation_space if observation_space is None else observation_space
        assert isinstance(observation, np.ndarray), f"Expected observation to be np.ndarray, got {type(observation)}"
        assert observation.shape[1:] == observation_space.shape, f"Expected observation to have shape {observation_space.shape}, got {observation.shape}"
        
        flat = observation.reshape(observation.shape[0], -1)
        return flat

    def _encode_dict(self, observation):
        assert isinstance(observation, dict), f"Expected observation to be dict, got {type(observation)}"
        
        encoded_items = []
        for key, subspace in self.observation_space.spaces.items():
            value = observation[key]

            if isinstance(subspace, Discrete):
                flat = self._encode_discrete(value, subspace)
                encoded_items.append(flat)
            elif isinstance(subspace, MultiDiscrete):
                flat = self._encode_multi_discrete(value, subspace)
                encoded_items.append(flat)
                
            elif isinstance(subspace, Box):
                flat = self._encode_box(value, subspace)
                encoded_items.append(flat)
            else:
                raise TypeError(f"Unsupported subspace {type(subspace).__name__} for key {key}")
        
        concated = np.concatenate(encoded_items, axis=1)
        return concated
       



def _sample_batch(space, batch_size):
    # ---- Dict space ----
    if isinstance(space, Dict):
        return {
            key: _sample_batch(subspace, batch_size)
            for key, subspace in space.spaces.items()
        }


    # ---- Discrete ----
    if isinstance(space, Discrete):
        return np.array([space.sample() for _ in range(batch_size)])
        
    
    # ---- Box ----
    if isinstance(space, Box):
        return np.array([space.sample() for _ in range(batch_size)])

    # ---- MultiDiscrete ----
    if isinstance(space, MultiDiscrete):
        return np.array([space.sample() for _ in range(batch_size)])

def print_sample(space, sample):
    if isinstance(space, Dict):
        for key, subspace in space.spaces.items():
            print(f"Key: {key}")
            print_sample(subspace, sample[key])
    else:
        if isinstance(sample, np.ndarray):
            print(f"Sample shape: {sample.shape}")
        else:
            print(f"Sample type: {type(sample)}, Sample value: {sample}")


if __name__ == "__main__":
    spaces = {
        "discrete": Discrete(5),
        "multi_discrete": MultiDiscrete([3, 2, 4, 5]),
        "float_box": Box(low=-1.0, high=1.0, shape=(4, 5), dtype=np.float32),
        "int_box": Box(low=0, high=3, shape=(3, 2), dtype=np.int32),
        "dict": Dict(
            {
                "grid": Box(low=0, high=2, shape=(6,), dtype=np.int32),
                "img": Box(low=0, high=2, shape=(3,7), dtype=np.int32),
                "state": Discrete(4),
                "multi": MultiDiscrete([2,3,4]),
            }
        ),
    }
    for space_name, space in spaces.items():
        print(f"\nSpace: {space_name}")
        batch_sample = _sample_batch(space, batch_size=2)
        print_sample(space, batch_sample)
        
        extractor = FlattenFeature(space)
        batch = extractor(batch_sample)
        print("Features: ", batch.shape, extractor.features_dict)
        
        

          
