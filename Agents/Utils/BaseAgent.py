import numpy as np
import random
import torch
import pickle

class BasePolicy:
    """Abstract base class for policies."""
    def __init__(self, action_space, hyper_params=None):
        self.action_space = action_space
        self.action_dim = int(action_space.n)
        self.set_hp(hyper_params)
            
    def select_action(self, observation):
        # Must be implemented by subclasses.
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def select_parallel_actions(self, observations):
        # Must be implemented by subclasses.
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def reset(self, seed):
        # Set seeds for reproducibility.
        self.action_space.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def set_hp(self, hp):
        self.hp = hp

    def save(self, file_path):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def load(self, file_path):
        raise NotImplementedError("This method should be implemented by subclasses.")
        
class BaseAgent:
    """Base class for an RL agent using a policy."""
    def __init__(self, action_space, observation_space=None, hyper_params=None, num_envs=None):
        self.hp = hyper_params
        self.action_space = action_space
        self.observation_space = observation_space
        self.num_envs = num_envs
        self.policy = BasePolicy(action_space)
        
    def act(self, observation):
        return self.policy.select_action(observation)
    
    def parallel_act(self, observations):
        return self.policy.select_parallel_actions(observations)
    
    def update(self, observation, reward, terminated, truncated, call_back=None):
        # For training updates; override in subclasses.
        pass
    
    def parallel_update(self, observations, rewards, terminateds, truncateds, call_back=None):
        # For parallel training updates; override in subclasses.
        pass
    
    def reset(self, seed):
        self.policy.reset(seed)
    
    def set_hp(self, hp):
        self.hp = hp
        self.policy.set_hp(hp)
    
    def save(self, file_path):
        self.policy.save(file_path)
    
    def load(self, file_path):
        self.policy.load(file_path)
        self.set_hp(self.policy.hp)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.hp})"