import numpy as np
import torch 
import gymnasium as gym
from collections.abc import Mapping
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX
import torch.nn.functional as F

from ...registry import register_feature_extractor
from .BaseClasses import BaseFeature


@register_feature_extractor
class MirrorFeature(BaseFeature):
    def __call__(self, observation):
        # Flatten and convert observation to a tuple.
        return observation
    
@register_feature_extractor
class TabularFeature(BaseFeature):
    def __call__(self, observation):
        # Flatten and convert observation to a tuple.
        return tuple(observation.flatten().tolist())

@register_feature_extractor
class TabularSymbolicFeature(BaseFeature):
    def __call__(self, observation):
        img = observation['image']
        dir = observation['direction']
        # Flatten image and append direction
        features = img.flatten().tolist()
        features.append(int(dir))  # ensure it's a plain int, not np.int64
        
        return tuple(features)
    
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
class MiniGridOneHotFlatWithDirCarryFeature(BaseFeature):
    """
    Output (always batched): (N, features_dim)
      features = [ onehot(image) , onehot(direction) , onehot(carry_type) , onehot(carry_color) ]

    Accepts single obs dict or batch dict:
      - image: (W,H,3) or (N,W,H,3)
      - direction: () or (N,)
      - carrying: (2,) or (N,2) with -1 meaning "none"
    """

    def __init__(self, observation_space, device="cpu"):
        super().__init__(observation_space, device=device)

        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError("Expected a Dict observation space.")

        # Validate image
        if "image" not in observation_space.spaces:
            raise KeyError("Observation space must contain 'image'.")
        img_space = observation_space.spaces["image"]
        if not isinstance(img_space, gym.spaces.Box) or len(img_space.shape) != 3:
            raise ValueError("obs['image'] must be a 3D Box (W,H,3).")
        W, H, C = img_space.shape
        if C != 3:
            raise ValueError(f"Expected image channels=3 (type,color,state); got {C}.")
        self.W, self.H = int(W), int(H)

        # Required keys
        if "direction" not in observation_space.spaces:
            raise KeyError("Observation space must contain 'direction' (Discrete(4)).")
        if "carrying" not in observation_space.spaces:
            raise KeyError("Observation space must contain 'carrying' (2-vector).")

        # Vocab sizes from MiniGrid
        self.num_obj   = len(OBJECT_TO_IDX)
        self.num_color = len(COLOR_TO_IDX)
        self.num_state = len(STATE_TO_IDX)
        self.num_bits  = self.num_obj + self.num_color + self.num_state

        # Direction and carry
        self.num_dir = 4
        self.carry_type_dim  = self.num_obj
        self.carry_color_dim = self.num_color

        # Precompute total features
        self._features_dim = (
            self.W * self.H * self.num_bits  # image one-hot flattened
            + self.num_dir                   # direction one-hot
            + self.carry_type_dim            # carry type one-hot
            + self.carry_color_dim           # carry color one-hot
        )

    @property
    def features_dim(self) -> int:
        return self._features_dim

    # --------- helpers ---------
    def _ensure_batch_img(self, img: torch.Tensor) -> torch.Tensor:
        # Accept (W,H,3) or (N,W,H,3), return (N,W,H,3)
        if img.dim() == 3:
            return img.unsqueeze(0)
        if img.dim() == 4:
            return img
        raise ValueError(f"Expected image rank 3 or 4, got {img.dim()}.")

    def _ensure_batch_dir(self, d: torch.Tensor) -> torch.Tensor:
        # Accept scalar () or (N,), return (N,)
        if d.dim() == 0:
            return d.unsqueeze(0)
        if d.dim() == 1:
            return d
        raise ValueError(f"Expected direction rank 0 or 1, got {d.dim()}.")

    def _ensure_batch_carry(self, c: torch.Tensor) -> torch.Tensor:
        # Accept (2,) or (N,2), return (N,2)
        if c.dim() == 1:
            if c.numel() != 2:
                raise ValueError("carrying must have length 2.")
            return c.unsqueeze(0)
        if c.dim() == 2 and c.shape[-1] == 2:
            return c
        raise ValueError(f"Expected carrying shape (2,) or (N,2), got {tuple(c.shape)}.")

    def _safe_one_hot(self, idx_t: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Robust one-hot that tolerates invalid indices (e.g., -1, 255).
        Invalid positions become all-zeros.
        """
        idx_t = idx_t.to(torch.long)
        valid = (idx_t >= 0) & (idx_t < num_classes)
        sanitized = idx_t.clamp(min=0, max=max(0, num_classes - 1))
        oh = F.one_hot(sanitized, num_classes=num_classes).to(torch.float32)
        return oh * valid.unsqueeze(-1).to(torch.float32)

    # --------- main ---------
    def __call__(self, observation):
        # ---- IMAGE ----
        img = observation["image"]
        img_t = torch.as_tensor(img, device=self.device)  # int dtype ok
        if img_t.shape[-1] != 3:
            raise ValueError(f"Expected image last dim=3, got {img_t.shape[-1]}.")
        img_t = self._ensure_batch_img(img_t).to(torch.long)           # (N,W,H,3)

        # Split channels
        type_idx  = img_t[..., 0]                                      # (N,W,H)
        color_idx = img_t[..., 1]                                      # (N,W,H)
        state_idx = img_t[..., 2]                                      # (N,W,H)

        # One-hot per cell, then concat channels
        type_oh  = self._safe_one_hot(type_idx,  self.num_obj)         # (N,W,H,num_obj)
        color_oh = self._safe_one_hot(color_idx, self.num_color)       # (N,W,H,num_color)
        state_oh = self._safe_one_hot(state_idx, self.num_state)       # (N,W,H,num_state)

        onehot_img = torch.cat([type_oh, color_oh, state_oh], dim=-1).to(torch.float32)  # (N,W,H,num_bits)
        N = onehot_img.shape[0]
        flat_img = onehot_img.view(N, -1)                              # (N, W*H*num_bits)

        # ---- DIRECTION ----
        dir_val = torch.as_tensor(observation["direction"], device=self.device)
        dir_val = self._ensure_batch_dir(dir_val).to(torch.long)       # (N,)
        dir_oh  = self._safe_one_hot(dir_val, self.num_dir).to(torch.float32)  # (N,4)

        # ---- CARRYING ----
        carry = torch.as_tensor(observation["carrying"], device=self.device)
        carry = self._ensure_batch_carry(carry).to(torch.long)         # (N,2)
        carry_type_idx  = carry[:, 0]
        carry_color_idx = carry[:, 1]
        carry_type_oh   = self._safe_one_hot(carry_type_idx,  self.carry_type_dim)   # (N, |OBJ|)
        carry_color_oh  = self._safe_one_hot(carry_color_idx, self.carry_color_dim)  # (N, |COLOR|)

        # ---- CONCAT ----
        feats = torch.cat([flat_img, dir_oh, carry_type_oh, carry_color_oh], dim=1)  # (N, features_dim)
        return feats
    
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
    
    def __call__(self, observation, normalization_factor=1):
        # 1) If dict, pick observation['image']; otherwise use observation as-is
        img = self._extract_image_from_obs(observation)

        # 2) Ensure torch tensor on device (handles numpy/torch)
        img_t = self._to_tensor(img) / normalization_factor 
        
       # 3) Add batch dim if missing (expecting (W,H,C) without batch)
        # If already batched (N,W,H,C), leave it.
        if img_t.dim() == 3:
            img_t = img_t.unsqueeze(0)
        elif img_t.dim() != 4:
            raise ValueError(f"Expected image tensor with 3 or 4 dims, got {img_t.shape}.")
        
        
        # 4) Permute (N, W, H, C) -> (N, C, W, H)
        img_t = img_t.permute(0, 3, 1, 2).contiguous()

        return img_t
        