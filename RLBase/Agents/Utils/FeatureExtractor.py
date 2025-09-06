import numpy as np
import torch 
import gymnasium as gym
from collections.abc import Mapping

from ...registry import register_feature_extractor
from .BaseClasses import BaseFeature

@register_feature_extractor
class TabularFeature(BaseFeature):
    def __call__(self, observation):
        # Flatten and convert observation to a tuple.
        return tuple(observation.flatten().tolist())

@register_feature_extractor
class FLattenFeature(BaseFeature):
    @property
    def features_dim(self):
        # Total number of features.
        return int(np.prod(self.observation_space.shape))
    
    def __call__(self, observation):
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, device=self.device, dtype=torch.float32)

        # Ensure a batch dimension and flatten each observation.
        if observation.dim() == len(self.observation_space.shape):
            observation = observation.unsqueeze(0)
        batch_size = observation.shape[0]
        observation = observation.reshape(batch_size, -1)
        return observation
    
@register_feature_extractor
class ImageFeature(BaseFeature):
    def _image_space(self):
        """Return the Box space corresponding to the image."""
        obs_space = self.observation_space
        if isinstance(obs_space, gym.spaces.Dict):
            if "image" not in obs_space.spaces:
                raise KeyError("Dict observation_space must contain an 'image' key.")
            img_space = obs_space.spaces["image"]
            if not isinstance(img_space, gym.spaces.Box):
                raise TypeError("'image' in Dict observation_space must be a Box.")
            return img_space
        elif isinstance(obs_space, gym.spaces.Box):
            return obs_space
        else:
            raise TypeError("Observation space must be a Box or Dict(with 'image').")
        
    @property
    def features_dim(self):
        # Convert (W, H, C) to (C, W, H) as you had.
        shape = self._image_space().shape  # expected (W, H, C)
        if len(shape) != 3:
            raise ValueError(f"Expected 3D image shape (W,H,C), got {shape}.")
        return (shape[2], shape[0], shape[1])
    
    def _to_tensor(self, x):
        """Convert numpy → torch on the correct device/dtype; pass through tensors."""
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=torch.float32)
        # handle numpy / lists
        return torch.as_tensor(x, device=self.device, dtype=torch.float32)

    def _extract_image_from_obs(self, observation):
        """Pull image array/tensor from obs that may be dict-like or direct."""
        if isinstance(observation, Mapping):  # dict-like: get 'image'
            if "image" not in observation:
                raise KeyError("Observation dict must contain an 'image' key.")
            return observation["image"]
        return observation  # assume it's already the image
    
    def __call__(self, observation):
        # 1) If dict, pick observation['image']; otherwise use observation as-is
        img = self._extract_image_from_obs(observation)

        # 2) Ensure torch tensor on device (handles numpy/torch)
        img_t = self._to_tensor(img)

       # 3) Add batch dim if missing (expecting (W,H,C) without batch)
        # If already batched (N,W,H,C), leave it.
        if img_t.dim() == 3:
            img_t = img_t.unsqueeze(0)
        elif img_t.dim() != 4:
            raise ValueError(f"Expected image tensor with 3 or 4 dims, got {img_t.shape}.")
        
        
        # 4) Permute (N, W, H, C) -> (N, C, W, H)
        img_t = img_t.permute(0, 3, 1, 2).contiguous()

        return img_t
        