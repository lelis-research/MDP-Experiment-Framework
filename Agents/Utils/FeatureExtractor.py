import numpy as np
import torch 


class BaseFeature:
    def __init__(self, observation_space):
        self.observation_space = observation_space
    
    @property
    def features_dim(self):
        # Must be implemented by subclasses.
        raise NotImplementedError("Must be implemented by the child class")
    
    def __call__(self, observation):
        # Must be implemented by subclasses.
        raise NotImplementedError("Must be implemented by the child class")
    
    def update(self):
        pass

    def reset(self, seed):
        pass
    
    def save(self, file_path=None):
        checkpoint = {
            'observation_space': self.observation_space,
            'feature_extractor_class': self.__class__.__name__,
        }
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_feature_extractor.t")
        return checkpoint
    
    def load_from_checkpoint(self, checkpoint):
        self.observation_space = checkpoint['observation_space']
    
class TabularFeature(BaseFeature):
    def __call__(self, observation):
        # Flatten and convert observation to a tuple.
        return tuple(observation.flatten().tolist())

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
        