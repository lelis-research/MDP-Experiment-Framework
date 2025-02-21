import numpy as np
import torch 

class BaseFeature:
    def __init__(self, observation_space):
        self.observation_space = observation_space
    
    @property
    def features_dim(self):
        raise NotImplementedError("Must be implemented by the child class")
    
    def __call__(self, observation):
        raise NotImplementedError("Must be implemented by the child class")
    
    def update(self):
        pass

    def reset(self, seed):
        pass
    
class TabularFeature(BaseFeature):
    def __call__(self, observation):
        return tuple(observation.flatten().tolist())

class FLattenFeature(BaseFeature):
    @property
    def features_dim(self):
        return int(np.prod(self.observation_space.shape))
    
    def __call__(self, observation):        
        # If the observation does not have a batch dimension,
        # add one. (i.e. its number of dimensions equals the observation_space shape length)
        if observation.ndim == len(self.observation_space.shape):
            observation = np.expand_dims(observation, axis=0)
        
        # Now, flatten each observation in the batch while preserving the batch dimension.
        batch_size = observation.shape[0]
        observation = observation.reshape(batch_size, -1)
        return observation

class ImageFeature(BaseFeature):
    @property
    def features_dim(self):
        # Assuming self.observation_space.shape is (W, H, C),
        # after permutation, the effective feature dimension will be (C, W, H).
        shape = self.observation_space.shape
        return (shape[2], shape[0], shape[1])
    
    def __call__(self, observation):    
        # If observation doesn't have a batch dimension, add one.
        if observation.ndim == len(self.observation_space.shape):
            observation = observation[np.newaxis, ...]
        
        # Now, assume observation has shape (batch, W, H, C) and convert to (batch, C, W, H)
        observation = np.transpose(observation, (0, 3, 1, 2))
        return observation
    
