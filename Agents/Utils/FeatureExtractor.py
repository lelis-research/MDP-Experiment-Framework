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
        if len(observation.shape) == 1:
            observation = np.expand_dims(observation, axis=0) # add batch dimension

        flatten_observation = observation.reshape(observation.shape[0], -1) 
        return flatten_observation
    
