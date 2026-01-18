
from gymnasium.spaces import Box, Discrete, Dict, MultiDiscrete
import numpy as np
import torch

from ..registry import register_feature_extractor
from .Base import BaseFeature

ALLOWED_SPACES = (Box, Discrete, Dict, MultiDiscrete)

@register_feature_extractor
class MirrorFeature(BaseFeature):
    """
    Return the observation as-is (same structure), but converted to torch tensors.
    - Discrete -> torch.int64 (B,)
    - MultiDiscrete -> torch.int64 (B, *shape)
    - Box -> torch.float32 if float box else torch.int64 (B, *shape)
    - Dict -> dict of tensors with same keys (restricted to allowed_keys)
    """
    
    def __init__(self, observation_space, device="cpu"):
        super().__init__(observation_space, device=device, allowed_spaces=ALLOWED_SPACES)
        self._features_dict = self._compute_feature_shapes(observation_space)
        
    def _compute_feature_shapes(self, space):
        if isinstance(space, Discrete):
            return {"x": ()}  # means output is (B,)

        if isinstance(space, MultiDiscrete):
            return {"x": tuple(space.shape)}  # output is (B, *shape)

        if isinstance(space, Box):
            return {"x": tuple(space.shape)}  # output is (B, *shape)

        if isinstance(space, Dict):
            shapes = {}
            for key in self.allowed_keys:
                subspace = space[key]
                # skip unsupported (e.g. MissionSpace) safely
                if not isinstance(subspace, ALLOWED_SPACES):
                    continue
                shapes[key] = self._compute_feature_shapes(subspace)["x"]
            return shapes

        raise TypeError(f"Unsupported space {type(space)}")
        
      

    @property
    def features_dict(self):
        return self._features_dict

    def __call__(self, observation):
        space = self.observation_space
        return self._to_torch(observation, space)
    
    def _to_torch(self, obs, space):
        # Dict: preserve dict structure
        if isinstance(space, Dict):
            out = {}
            for key in self.allowed_keys:
                subspace = space[key]
                out[key] = self._to_torch(obs[key], subspace)["x"]
            return out

        # Non-dict: return under key "x" (consistent with your pipeline)
        return {"x": self._to_tensor(obs, space)}

    def _to_tensor(self, obs, space):
        # Convert obs -> np array if needed
        if not isinstance(obs, np.ndarray):
            obs = np.asarray(obs)

        # Dtype rules
        if isinstance(space, (Discrete, MultiDiscrete)):
            # return torch.as_tensor(obs, device=self.device, dtype=torch.int64)
            return torch.as_tensor(obs, device=self.device, dtype=torch.float32)

        if isinstance(space, Box):
            if np.issubdtype(space.dtype, np.floating):
                return torch.as_tensor(obs, device=self.device, dtype=torch.float32)
            elif np.issubdtype(space.dtype, np.integer):
                # return torch.as_tensor(obs, device=self.device, dtype=torch.int64)
                return torch.as_tensor(obs, device=self.device, dtype=torch.float32)
            else:
                # fallback: keep as float32
                return torch.as_tensor(obs, device=self.device, dtype=torch.float32)

        raise TypeError(f"Unsupported space {type(space)} in _to_tensor")

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
        batch_sample = _sample_batch(space, batch_size=1)
        print_sample(space, batch_sample)
        
        extractor = FlattenFeature(space)
        batch = extractor(batch_sample)
        print("Features: ", batch["x"].shape, extractor.features_dict)
        
        

          
