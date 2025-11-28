from ....loaders import load_policy, load_feature_extractor
import torch

class BaseOption:
    def __init__(self, option_len):
        self.option_len = option_len
 
        
    def select_action(self, observation):
        pass
    
    def is_terminated(self, observation):
        pass
    
    def save(self, file_path=None):

        checkpoint = {
            'option_len': self.option_len,
            'option_class': self.__class__.__name__,
        }
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_options.t")
        return checkpoint


    @classmethod
    def load(cls, file_path, checkpoint=None):
        """
        Load the Q-table and hyper-parameters.
        
        Args:
            file_path (str): File path from which to load the checkpoint.
        """
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)

        instance = cls(option_len=checkpoint['option_len'])

        return instance

    def __repr__(self): return f"{self.__class__.__name__}(len={self.option_len})"