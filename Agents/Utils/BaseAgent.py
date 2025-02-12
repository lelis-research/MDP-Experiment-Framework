import numpy as np
import random
import torch

class BasePolicy:
    """Abstract base class for policies."""
    def __init__(self, action_space, hyper_params=None):
        self.action_space = action_space
        self.set_hp(hyper_params)
            
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
    
    def select_parallel_actions(self, observations):
        """
        Given numpy array of parallel observations, return a numpy array of actions
        This must be implemented by subclasses.
        
        Args:
            observations: The states from the environment.
        
        Returns:
            Actions to take.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def reset(self, seed):
        self.action_space.seed(seed)
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def set_hp(self, hp):
        self.hp = hp
        
class BaseAgent:
    """Base class for an RL agent."""
    
    def __init__(self, action_space, observation_space=None, hyper_params=None, num_envs=None):
        """
        Initialize an agent with a policy.
        """
        self.hp = hyper_params
        self.action_space = action_space
        self.observation_space = observation_space
        self.num_envs = num_envs

        self.policy = BasePolicy(action_space)
        
    
    def act(self, observation):
        """
        Select an action using the agent's policy.
        
        Returns:
            An action chosen by the policy.
        """
        return self.policy.select_action(observation)
    
    def parallel_act(self, observations):
        """
        Select actions in parallel environments
        
        Returns:
            An action chosen by the policy.
        """
        return self.policy.select_parallel_actions(observations)
    
    def update(self, observation, reward, terminated, truncated):
        """
        Update the agent (e.g., learning step).
        This should be implemented by learning agents.
        """
        pass  # Default: No learning

    def parallel_update(self, observations, rewards, terminateds, truncateds):
        """
        Update the agent (e.g., learning step).
        This should be implemented by learning agents.
        """
        pass  # Default: No learning

    def reset(self, seed):
        """
        Reset the agent's learning state.
        """
        self.policy.reset(seed)
    
    def set_hp(self, hp):
        """
        Update the set of Hyper-Params
        """
        self.hp = hp
        self.policy.set_hp(hp)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.hp})"