"""Channel-first one-hot encoding for integer grid observations."""

from gymnasium.spaces import Box
import numpy as np
import torch

from ..registry import register_feature_extractor
from .Base import BaseFeature


@register_feature_extractor
class OneHotConvFeature(BaseFeature):
    """Encode an integer ``Box`` grid as a CNN input.

    A ``(H, W)`` grid with integer values in ``[low, high]`` becomes a
    channel-first tensor with shape ``(batch, channels, H, W)``. The returned
    feature key is ``img``, matching the convolutional network presets.
    """

    def __init__(self, observation_space, device="cpu"):
        if not isinstance(observation_space, Box):
            raise TypeError(
                "OneHotConvFeature requires a single integer Box observation space."
            )
        if not np.issubdtype(observation_space.dtype, np.integer):
            raise TypeError("OneHotConvFeature requires an integer Box dtype.")

        low = np.asarray(observation_space.low, dtype=np.int64)
        high = np.asarray(observation_space.high, dtype=np.int64)
        if not np.all(low == low.flat[0]) or not np.all(high == high.flat[0]):
            raise ValueError(
                "OneHotConvFeature requires uniform Box low and high bounds."
            )

        self.low = int(low.flat[0])
        self.high = int(high.flat[0])
        self.num_channels = self.high - self.low + 1
        if self.num_channels <= 0:
            raise ValueError("Box high/low must define at least one category.")

        super().__init__(observation_space, device=device)
        self._features_dict = {
            "img": (self.num_channels, *observation_space.shape),
        }

    @property
    def features_dict(self):
        return self._features_dict

    def __call__(self, observation):
        if not isinstance(observation, np.ndarray):
            raise TypeError(
                f"Expected observation to be np.ndarray, got {type(observation)}"
            )
        if observation.shape[1:] != self.observation_space.shape:
            raise ValueError(
                "Expected observation shape "
                f"(batch, {self.observation_space.shape}), got {observation.shape}"
            )

        shifted = observation.astype(np.int64) - self.low
        if np.any(shifted < 0) or np.any(shifted >= self.num_channels):
            raise ValueError("Observation contains values outside the Box bounds.")

        one_hot = np.eye(self.num_channels, dtype=np.float32)[shifted]
        channel_first = np.moveaxis(one_hot, -1, 1)
        return {
            "img": torch.from_numpy(channel_first).to(
                self.device,
                dtype=torch.float32,
            )
        }
