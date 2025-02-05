import numpy as np
import random

class BasePolicy:
    """Abstract base class for policies."""
    def __init__(self, action_space, seed=None):
        self.action_space = action_space
        self.seed = seed

        if seed is not None:
            self.action_space.seed(seed)
            self._py_rng = random.Random(seed)
        else:
            self._py_rng = random

            
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
    
    def reset(self, seed=None):
        if seed is not None:
            self.action_space.seed(seed)
            self._py_rng = random.Random(seed)
        else:
            self._py_rng = random

class BaseAgent:
    """Base class for an RL agent."""
    
    def __init__(self, action_space, hyper_params=None, seed=None):
        """
        Initialize an agent with a policy.

        """
        self.hp = hyper_params
        self.seed = seed
        self.policy = BasePolicy(action_space)
        
    
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

    def reset(self, seed=None):
        """
        Reset the agent's learning state.
        """
        if seed is not None:
            self.seed = seed
        self.policy.reset(seed)
    
    def set_hp(self, hp):
        """
        Update the set of Hyper-Params
        """
        self.hp = hp

    def __repr__(self):
        return f"{self.__class__.__name__}({self.hp})"