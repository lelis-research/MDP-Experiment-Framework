import numpy as np
import gymnasium as gym

class BasePolicy:
    """Abstract base class for policies."""
    def __init__(self, seed=None):
        self.seed = seed
            
    def select_action(self, observation):
        """
        Given an observation, return an action.
        This must be implemented by subclasses.
        
        Args:
            observation: The state from the environment.
        
        Returns:
            An action to take.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    