import numpy as np
import torch 

class FLattenFeature:
    def __init__(self, observation_space):
        self.observation_space = observation_space
    
    @property
    def features_dim(self):
        return int(np.prod(self.observation_space.shape))
    
    def __call__(self, observation):
        return observation.flatten()
    
    def update(self):
        pass

    def reset(self, seed):
        pass