import numpy as np
import random
import torch
from copy import copy

class BaseFeature:
    def __init__(self, observation_space, device='cpu'):
        self.observation_space = observation_space
        self.device = device
    
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
    
    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        instance = cls(checkpoint['observation_space'])
        return instance
    
    def load_from_checkpoint(self, checkpoint):
        self.observation_space = checkpoint['observation_space']

class BasePolicy:
    """Abstract base class for policies."""
    def __init__(self, action_space, hyper_params=None, device='cpu'):
        self.action_space = action_space
        self.device = device
        self.set_hp(hyper_params)
    
    @property
    def action_dim(self):
        """Number of discrete actions available to the policy."""
        if hasattr(self.action_space, 'n'):
            return self.action_space.n # discrete action space
        elif hasattr(self.action_space, 'shape'):
            return self.action_space.shape[0] # continuous action space
        else:
            raise ValueError("Action dim is not defined in this action space")
            
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

    @classmethod
    def load_from_file(cls, file_path, seed=0):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def load_from_checkpoint(self, checkpoint):
        raise NotImplementedError("This method should be implemented by subclasses.")
        
class BaseAgent:
    """Base class for an RL agent using a policy."""
    def __init__(self, action_space, observation_space=None, hyper_params=None, num_envs=None, feature_extractor_class=None, device='cpu'):
        self.hp = hyper_params
        self.action_space = action_space
        self.observation_space = observation_space
        self.num_envs = num_envs
        self.device = device

        self.feature_extractor = None if feature_extractor_class is None else feature_extractor_class(observation_space, device=device)
        self.policy = BasePolicy(action_space, device=device)
        
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
       
    def save(self, file_path=None):
        policy_checkpoint = self.policy.save(file_path=None)
        feature_extractor_checkpoint = None if self.feature_extractor is None else self.feature_extractor.save()

        checkpoint = {
            'action_space': self.action_space,
            'observation_space': self.observation_space,
            'hyper_params': self.hp,
            'num_envs': self.num_envs,
            'feature_extractor_class': self.feature_extractor.__class__,

            'policy': policy_checkpoint,
            'feature_extractor': feature_extractor_checkpoint,

            'agent_class': self.__class__.__name__,
        }
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_agent.t")
        return checkpoint

    @classmethod
    def load_from_file(cls, file_path, seed=0, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        instance = cls(checkpoint['action_space'], checkpoint['observation_space'], 
                       checkpoint['hyper_params'], checkpoint['num_envs'],
                       checkpoint['feature_extractor_class'])
        instance.reset(seed)

        instance.feature_extractor.load_from_checkpoint(checkpoint['feature_extractor'])
        instance.policy.load_from_checkpoint(checkpoint['policy'])
        
        return instance

    def __repr__(self):
        return f"{self.__class__.__name__}({self.hp})"
    
class BaseContiualPolicy(BasePolicy):
    def trigger_option_learner(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def init_options(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

# class BaseContinualAgent(BaseAgent):
#     def __init__(self, action_space, observation_space=None, hyper_params=None, num_envs=None, feature_extractor_class=None, 
#                  option_learner_class=None, initial_options_lst=[], device='cpu'):
#         super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device)
#         self.option_learner = option_learner_class()
#         self.options_lst = initial_options_lst
#         self.initial_options_lst = copy(initial_options_lst)
    
#     def reset(self, seed):
#         self.policy.reset(seed)
#         self.option_learner.reset(seed)
        
#     def save(self, file_path=None):
#         policy_checkpoint = self.policy.save(file_path=None)
#         feature_extractor_checkpoint = None if self.feature_extractor is None else self.feature_extractor.save()
#         option_learner_checkpoint = None if self.option_learner is None else self.option_learner.save()
#         checkpoint = {
#             'action_space': self.action_space,
#             'observation_space': self.observation_space,
#             'hyper_params': self.hp,
#             'num_envs': self.num_envs,
#             'feature_extractor_class': self.feature_extractor.__class__,
#             'option_learner_class': self.option_learner.__class__,
#             'initial_options_lst': self.initial_options_lst,

#             'policy': policy_checkpoint,
#             'feature_extractor': feature_extractor_checkpoint,
#             'option_learner': option_learner_checkpoint,
            

#             'agent_class': self.__class__.__name__,
#         }
#         if file_path is not None:
#             torch.save(checkpoint, f"{file_path}_agent.t")
#         return checkpoint

#     @classmethod
#     def load_from_file(cls, file_path, seed=0, checkpoint=None):
#         if checkpoint is None:
#             checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
#         instance = cls(checkpoint['action_space'], checkpoint['observation_space'], 
#                        checkpoint['hyper_params'], checkpoint['num_envs'],
#                        checkpoint['feature_extractor_class'], checkpoint['option_learner_class'],
#                        checkpoint['initial_options_lst'])
#         instance.reset(seed)

#         instance.feature_extractor.load_from_checkpoint(checkpoint['feature_extractor'])
#         instance.policy.load_from_checkpoint(checkpoint['policy'])
#         instance.option_learner.load_from_checkpoint(checkpoint['option_learner'])
        
#         return instance

    
    
