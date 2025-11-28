
from gymnasium.spaces import Box, Discrete, Dict, MultiDiscrete
import numpy as np
import gymnasium

from ...registry import register_feature_extractor
from ..Utils.BaseClasses import BaseFeature

ALLOWED_SPACES = (Box, Discrete, Dict, MultiDiscrete)

@register_feature_extractor
class TabularFeature(BaseFeature):
    """
    Convert supported Gymnasium observations into hashable tuples.

    The implementation keeps the logic deliberately lightweight: each space
    type has a dedicated encoder and batching is handled through a single
    helper that reshapes observations when needed.
    """

    def __init__(self, observation_space, device="cpu"):
        super().__init__(observation_space, device=device, allowed_spaces=ALLOWED_SPACES)
        self._features_dict = {"tabular": 1}
        print(self.observation_space.shape)

    def __call__(self, observation):
        if isinstance(self.observation_space, Discrete):
            return self._encode_discrete(observation)
        elif isinstance(self.observation_space, MultiDiscrete):
            return self._encode_multi_discrete(observation)
        elif isinstance(self.observation_space, Box):
            return self._encode_box(observation)
        elif isinstance(self.observation_space, Dict):
            return self._encode_dict(observation)
        else:
            raise ValueError(f"Observation Space {self.observation_space} is not supported by {self.__class__.__name__}")
        

    @property
    def features_dict(self):
        return self._features_dict


    
    def _encode_discrete(self, observation, check_validity=True):
        arr = np.asarray(observation)
        flat = arr.reshape(-1)

        if check_validity:
            if not all(isinstance(obs, (int, np.integer)) for obs in flat):
                raise ValueError(
                    f"Observation type {type(observation)} is not supported for discrete spaces by {self.__class__.__name__}"
                )

        return flat.reshape(-1, 1)

    def _encode_multi_discrete(self, observation, check_validity=True):
        arr = np.asarray(observation)
        flat = arr.reshape(-1)
        expected_dim = len(self.observation_space.nvec)

        if check_validity:
            if not all(isinstance(obs, (int, np.integer)) for obs in flat):
                raise ValueError(
                    f"Observation type {type(observation)} is not supported for multi-discrete spaces by {self.__class__.__name__}"
                )
            if flat.size % expected_dim != 0:
                raise ValueError(
                    f"MultiDiscrete observation has size {flat.size}, which is not divisible by expected dimension {expected_dim}"
                )

        try:
            return flat.reshape(-1, expected_dim)
        except ValueError as exc:
            raise ValueError(
                f"Could not reshape MultiDiscrete observation of shape {arr.shape} to (-1, {expected_dim})"
            ) from exc
    
    def _encode_box(self, observation, check_validity=True):
        arr = np.asarray(observation, dtype=float)
        feature_dim = int(np.prod(self.observation_space.shape))

        if check_validity:
            if arr.size % feature_dim != 0:
                raise ValueError(
                    f"Box observation has size {arr.size}, which is not divisible by flattened dimension {feature_dim}"
                )

        try:
            return arr.reshape(-1, feature_dim)
        except ValueError as exc:
            raise ValueError(
                f"Could not reshape Box observation of shape {arr.shape} to (-1, {feature_dim})"
            ) from exc
    
    def _encode_dict(self, observation, check_validity=True):
        if isinstance(observation, (list, tuple)):
            return tuple(self._encode_dict(single) for single in observation)
        elif isinstance(observation, (dict, Dict)):
            encoded_items = []
            for key, subspace in self.observation_space.spaces.items():
                value = observation[key]
                if isinstance(subspace, Discrete):
                    encoded_items.append((key, self._encode_discrete(value)))
                elif isinstance(subspace, MultiDiscrete):
                    encoded_items.append((key, self._encode_multi_discrete(value)))
                elif isinstance(subspace, Box):
                    encoded_items.append((key, self._encode_box(value)))
                else:
                    raise TypeError(f"Unsupported subspace {type(subspace).__name__} for key {key}")
            return tuple(encoded_items)
        else:
            raise ValueError(f"Observation type {type(observation)} is not supported for dict spaces by {self.__class__.__name__}")



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
                "state": Discrete(4),
            }
        ),
    }
    for space_name, space in spaces.items():
        print(f"\nSpace: {space_name}")
        sample = space.sample()
        batch_sample = _sample_batch(space, batch_size=10)
        
        
        print("**" * 10)
        extractor = TabularFeature(space)

        single = extractor(sample)
        print_sample(space, sample)
        print(f"Single Feature Shape:{single.shape}")
        
        batch = extractor(batch_sample)
        print_sample(space, batch_sample)
        print(f"Batch Feature Shape:{batch.shape}")

          
