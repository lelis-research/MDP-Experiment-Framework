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
class OneHotImageDirCarryFeature(BaseFeature):
    """
    One-hot encode obs['image'] with categorical indices (W,H,3).
    Vocab sizes and offsets are inferred from Box.low/high per channel.
    """

    def __init__(self, observation_space, device="cpu"):
        super().__init__(observation_space, device=device)
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError("Expected a Dict observation space.")

        # -------- IMAGE -------- (BOX())
        if "image" not in observation_space.spaces:
            raise KeyError("Observation space must contain 'image'.")
        img_space = observation_space.spaces["image"]

        if not isinstance(img_space, gym.spaces.Box) or len(img_space.shape) != 3:
            raise ValueError("obs['image'] must be a 3D Box (W,H,3).")
        W, H, C = img_space.shape
        if C != 3:
            raise ValueError(f"Expected image channels=3 (type,color,state); got {C}.")
        self.W, self.H = int(W), int(H)

        # Infer per-channel lows/highs (supports broadcasted bounds)
        low, high = img_space.low, img_space.high
        if low.shape  != (W, H, 3):  low  = np.broadcast_to(low,  (W, H, 3))
        if high.shape != (W, H, 3):  high = np.broadcast_to(high, (W, H, 3))

        chan_low  = low.min(axis=(0, 1))     # (3,)
        chan_high = high.max(axis=(0, 1))    # (3,)

        if np.any(low.max(axis=(0,1)) != chan_low):
            raise ValueError("Per-channel lows vary across space; expected uniform per-channel bounds.")
        if np.any(high.min(axis=(0,1)) != chan_high):
            raise ValueError("Per-channel highs vary across space; expected uniform per-channel bounds.")
        if np.any(chan_high < chan_low):
            raise ValueError("Invalid bounds: high < low for at least one channel.")

        chan_card = (chan_high.astype(np.int64) - chan_low.astype(np.int64) + 1)
        if np.any(chan_card <= 0):
            raise ValueError("Non-positive cardinality detected in image channels.")

        self.chan_low   = chan_low.astype(np.uint8)   # [obj_low, color_low, state_low]
        self.chan_high  = chan_high.astype(np.uint8)  # [obj_high, color_high, state_high]
        self.chan_card  = chan_card.astype(np.int64)  # [|obj|, |color|, |state|]

        self.num_obj, self.num_color, self.num_state = map(int, self.chan_card.tolist())
        self.num_bits = int(self.num_obj + self.num_color + self.num_state)

        self.out_W, self.out_H, self.out_C = self.W, self.H, self.num_bits
        
        # -------- DIRECTION -------- (Discrete())
        if "direction" not in observation_space.spaces:
            raise KeyError("Observation space must contain 'direction'.")
        
        dir_space = observation_space.spaces["direction"]
        if not isinstance(dir_space, gym.spaces.Discrete):
            raise TypeError("obs['direction'] must be gym.spaces.Discrete.")
        
        self.dir_low, self.dir_high = 0, int(dir_space.n) - 1
        self.dir_card = int(dir_space.n)   # e.g., 4
        
        # -------- CARRYING -------- (Box()

        if "carrying" not in observation_space.spaces:
            raise KeyError("Observation space must contain 'carrying' (pair: type,color).")
        
        carry_space = observation_space.spaces["carrying"]
        if not isinstance(carry_space, gym.spaces.Box) or carry_space.shape != (2,):
            raise TypeError("obs['carrying'] must be a Box with shape (2,) = (type_idx, color_idx).")
        
        # Per-component bounds and cardinalities
        carry_low  = np.asarray(carry_space.low,  dtype=np.int64)  # [-1, -1]
        carry_high = np.asarray(carry_space.high, dtype=np.int64)  # [|OBJ|-1, |COLOR|-1]
        if np.any(carry_high < carry_low):
            raise ValueError("Invalid 'carrying' bounds: high < low")
        carry_card = (carry_high - carry_low + 1)                  # [|OBJ|+1_if_none, |COLOR|+1_if_none]
        self.carry_low_vec  = carry_low
        self.carry_high_vec = carry_high
        self.carry_card_vec = carry_card
        # total length after concatenating per-component one-hots
        self.carry_dim = int(carry_card[0] + carry_card[1])
        

    @property
    def num_features(self):
        return 2
    
    @property
    def features_dim(self) -> tuple:
        return (self.out_C, self.out_W , self.out_H), self.carry_dim + int(self.dir_card)

    @staticmethod
    def one_hot_shifted(idxs: torch.Tensor, low: int, card: int) -> torch.Tensor:
        shifted = (idxs - low).clamp(min=0, max=card - 1)
        return F.one_hot(shifted, num_classes=card).to(torch.float32)

    def __call__(self, observation):
        # ---- IMAGE ----
        img = observation["image"]
        img_t = torch.as_tensor(img, device=self.device)

        # ensure batch dimension → (N,W,H,3)
        if img_t.dim() == 3:
            img_t = img_t.unsqueeze(0)
        elif img_t.dim() != 4:
            raise ValueError(f"Expected image tensor with 3 or 4 dims, got {tuple(img_t.shape)}.")

        img_t = img_t.to(torch.long)

        # Split channels (N,W,H)
        type_idx  = img_t[..., 0]
        color_idx = img_t[..., 1]
        state_idx = img_t[..., 2]

        # Scalars from stored bounds/cardinalities
        obj_low, color_low, state_low   = [int(x) for x in self.chan_low.tolist()]
        obj_high, color_high, state_high = [int(x) for x in self.chan_high.tolist()]
        obj_card, color_card, state_card = [int(x) for x in self.chan_card.tolist()]

        # Strict validity checks (fail fast during development)
        tmin, tmax = type_idx.min().item(),  type_idx.max().item()
        cmin, cmax = color_idx.min().item(), color_idx.max().item()
        smin, smax = state_idx.min().item(), state_idx.max().item()
        assert (tmin >= obj_low   and tmax <= obj_high),   f"object_type out of range [{tmin},{tmax}] vs [{obj_low},{obj_high}]"
        assert (cmin >= color_low and cmax <= color_high), f"color out of range       [{cmin},{cmax}] vs [{color_low},{color_high}]"
        assert (smin >= state_low and smax <= state_high), f"state out of range        [{smin},{smax}] vs [{state_low},{state_high}]"

        # One-hot per channel
        type_oh  = self.one_hot_shifted(type_idx,  obj_low,   obj_card)   # (N,W,H,|OBJ|)
        color_oh = self.one_hot_shifted(color_idx, color_low, color_card) # (N,W,H,|COLOR|)
        state_oh = self.one_hot_shifted(state_idx, state_low, state_card) # (N,W,H,|STATE|)

        # Concatenate planes → (N, W, H, num_bits)
        onehot_img = torch.cat([type_oh, color_oh, state_oh], dim=-1)  # float32
        onehot_img = onehot_img.permute(0, 3, 1, 2).contiguous()
        
        # ========= CARRYING =========  (type_idx, color_idx) or (-1, -1)
        carry = observation["carrying"]            # shape (2,) or (N, 2)
        carry_t = torch.as_tensor(carry, device=self.device)
        if carry_t.dim() == 1:
            carry_t = carry_t.unsqueeze(0)        # (N, 2)
        elif carry_t.dim() != 2 or carry_t.size(-1) != 2:
            raise ValueError(f"Expected 'carrying' shape (2,) or (N,2); got {tuple(carry_t.shape)}.")
        carry_t = carry_t.to(torch.long)

        carry_type  = carry_t[..., 0]              # (N,)
        carry_color = carry_t[..., 1]              # (N,)

        type_low, color_low = int(self.carry_low_vec[0]),  int(self.carry_low_vec[1])
        type_card, color_card = int(self.carry_card_vec[0]), int(self.carry_card_vec[1])

        type_oh  = self.one_hot_shifted(carry_type,  type_low,  type_card)   # (N, type_card)
        color_oh = self.one_hot_shifted(carry_color, color_low, color_card)  # (N, color_card)

        onehot_carry = torch.cat([type_oh, color_oh], dim=-1)                 # (N, type_card + color_card)

        # ========= DIRECTION =========  Discrete(4)
        direction = observation["direction"]       # scalar or (N,)
        dir_t = torch.as_tensor(direction, device=self.device)
        if dir_t.dim() == 0:
            dir_t = dir_t.unsqueeze(0)            # (N,)
        dir_t = dir_t.to(torch.long)

        onehot_dir = self.one_hot_shifted(dir_t, self.dir_low, self.dir_card)  # (N, dir_dim)

        # Return the tuple requested
        return onehot_img.to(self.device), torch.cat([onehot_carry, onehot_dir], dim=1).to(self.device)
            
    
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
        