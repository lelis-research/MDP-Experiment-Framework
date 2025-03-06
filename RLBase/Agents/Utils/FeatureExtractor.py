import numpy as np
import torch 
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
        # Ensure a batch dimension and flatten each observation.
        if observation.ndim == len(self.observation_space.shape):
            observation = np.expand_dims(observation, axis=0)
        batch_size = observation.shape[0]
        observation = observation.reshape(batch_size, -1)
        return observation
    
@register_feature_extractor
class ImageFeature(BaseFeature):
    @property
    def features_dim(self):
        # Convert (W, H, C) to (C, W, H).
        shape = self.observation_space.shape
        return (shape[2], shape[0], shape[1])
    
    def __call__(self, observation):
        # Add batch dimension if missing and transpose to (batch, C, W, H).
        if observation.ndim == len(self.observation_space.shape):
            observation = observation[np.newaxis, ...]
        observation = np.transpose(observation, (0, 3, 1, 2))
        return observation
        