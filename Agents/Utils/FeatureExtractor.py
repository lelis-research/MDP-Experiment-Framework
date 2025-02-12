import numpy as np
import torch 

class FLattenFeature:
    def __init__(self, observation_space):
        self.observation_space = observation_space
    
    @property
    def features_dim(self):
        return int(np.prod(self.observation_space.shape))
    
    def __call__(self, observation):
        if len(observation.shape) == 1:
            observation = np.expand_dims(observation, axis=0) # add batch dimension

        flatten_observation = observation.reshape(observation.shape[0], -1) 
        return flatten_observation
    
    def update(self):
        pass

    def reset(self, seed):
        pass