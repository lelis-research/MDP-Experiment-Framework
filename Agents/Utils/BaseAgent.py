import numpy as np
import random

class BaseAgent:
    """Base class for an RL agent."""
    
    def __init__(self, policy, hyper_params=None, seed=None):
        """
        Initialize an agent with a policy.

        """
        self.hp = hyper_params
        self.seed = seed
        self.policy = policy
        
    
    def act(self, observation):
        """
        Select an action using the agent's policy.
        
        Returns:
            An action chosen by the policy.
        """
        return self.policy.select_action(observation)
    
    def update(self, observation, reward, terminated, truncated):
        """
        Update the agent (e.g., learning step).
        This should be implemented by learning agents.
        """
        pass  # Default: No learning

    def reset(self):
        """
        Reset the agent's learning state.
        """
        pass
    
    def set_hp(self, hp):
        """
        Update the set of Hyper-Params
        """
        self.hp = hp

    def __repr__(self):
        return f"{self.__class__.__name__}({self.hp})"