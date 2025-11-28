import numpy as np
import random
import torch
from copy import copy
from typing import Any, Iterable, SupportsFloat, TypeVar
from gymnasium.utils import seeding
from gymnasium.spaces import Box, Discrete, Dict, MultiDiscrete

T = TypeVar("T")
ALLOWED_SPACES = (Box, Discrete, Dict, MultiDiscrete)

class RandomGenerator:
    _np_random = None
    
    def _rand_int(self, low: int, high: int) -> int:
        """
        Generate random integer in [low,high[
        """

        return self.np_random.integers(low, high)

    def _rand_float(self, low: float, high: float) -> float:
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self) -> bool:
        """
        Generate random boolean value
        """

        return self.np_random.integers(0, 2) == 0

    def _rand_elem(self, iterable: Iterable[T]) -> T:
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable: Iterable[T], num_elems: int) -> list[T]:
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        n = len(lst)
        assert num_elems <= n

        # Sample indices without replacement using our RNG
        idxs = self.np_random.choice(n, size=num_elems, replace=False)

        # Gather elements
        return [lst[i] for i in idxs]
    
    
    def set_seed(self, seed: int | None = None):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        self._np_random, self._np_random_seed = seeding.np_random(seed)

    @property
    def np_random(self) -> np.random.Generator:
        """Return internal RNG, creating one if necessary."""
        if self._np_random is None:
            self._np_random, self._np_random_seed = seeding.np_random()
        return self._np_random

    
    def get_rng_state(self):
        """Return RNG state dict (can be pickled or saved)."""
        return self.np_random.bit_generator.state

    def set_rng_state(self, state_dict):
        """Restore RNG from a saved state dict."""
        self.np_random.bit_generator.state = state_dict

class BaseFeature(RandomGenerator):
    def __init__(self, observation_space, device='cpu', allowed_spaces=ALLOWED_SPACES):
        self.allowed_spaces = allowed_spaces
        self._validate_space(observation_space)
        
        self.observation_space = observation_space
        self.device = device
        
    
    def _validate_space(self, space):
        if not isinstance(space, self.allowed_spaces):
            raise TypeError(f"{self.__class__.__name__} only supports {self.allowed_spaces} spaces (got {type(space).__name__})")

        if isinstance(space, Dict):
            for key, subspace in space.spaces.items():
                if not isinstance(subspace, self.allowed_spaces):
                    raise TypeError(
                        f"{self.__class__.__name__} Dict subspace '{key}' must be {self.allowed_spaces} "
                        f"(got {type(subspace).__name__})"
                    )
    
    @property
    def features_dict(self):
        return 1
    
    def __call__(self, observation):
        # Must be implemented by subclasses.
        raise NotImplementedError("Must be implemented by the child class")
    
    def update(self):
        pass

    def reset(self, seed):
        self.set_seed(seed)
    
    def save(self, file_path=None):
        checkpoint = {
            'observation_space': self.observation_space,
            'feature_extractor_class': self.__class__.__name__,
            'rng_state': self.get_rng_state(),
        }
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_feature_extractor.t")
        return checkpoint
    
    @classmethod
    def load(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        instance = cls(checkpoint['observation_space'])
        instance.set_rng_state(checkpoint['rng_state'])
        return instance

class BasePolicy(RandomGenerator):
    """Abstract base class for policies"""
    def __init__(self, action_space, hyper_params=None, device='cpu'):
        self.action_space = action_space
        self.device = device
        self.set_hp(hyper_params)
    
    @property
    def action_dim(self):
        """Number of actions available to the policy."""
        if hasattr(self.action_space, 'n'):
            return int(self.action_space.n) # discrete action space
        elif hasattr(self.action_space, 'shape'):
            return self.action_space.shape[0] # continuous action space
        else:
            raise ValueError("Action dim is not defined in this action space")
            
    def select_action(self, state):
        # Must be implemented by subclasses. 
        # State is a batch of data
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def reset(self, seed):
        # Set seeds for reproducibility.
        self.set_seed(seed)
    
    def set_hp(self, hp):
        self.hp = hp

    def save(self, file_path=None):
        checkpoint = {
            'hyper_params': self.hp,
            'action_space': self.action_space,
            'policy_class': self.__class__.__name__,
            'rng_state': self.get_rng_state()
        }
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_policy.t")
        return checkpoint

    @classmethod
    def load(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        instance = cls(checkpoint['action_space'], checkpoint['hyper_params'])
        instance.set_rng_state(checkpoint['rng_state'])
        return instance
        
class BaseAgent(RandomGenerator):
    """Base class for an RL agent using a policy."""
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device='cpu'):
        self.hp = hyper_params
        self.action_space = action_space
        self.observation_space = observation_space
        self.num_envs = num_envs
        self.device = device
        self.feature_extractor_class = feature_extractor_class
        
        self.feature_extractor = feature_extractor_class(observation_space, device=device)
        self.policy = BasePolicy(action_space, device=device)
        
    def act(self, observation, greedy=False):
        # Vectorized Observation
        state = self.feature_extractor(observation)
        return self.policy.select_action(state)
    
    def update(self, observation, reward, terminated, truncated, call_back=None):
        # For training updates; override in subclasses.
        pass
    
    def reset(self, seed):
        self.set_seed(seed)
        self.policy.reset(seed)
        self.feature_extractor.reset(seed)
    
    def set_hp(self, hp):
        self.hp = hp
        self.policy.set_hp(hp)
       
    def save(self, file_path=None):
        policy_checkpoint = self.policy.save(file_path=None)
        feature_extractor_checkpoint = self.feature_extractor.save(file_path=None) 

        checkpoint = {
            'action_space': self.action_space,
            'observation_space': self.observation_space,
            'hyper_params': self.hp,
            'num_envs': self.num_envs,
            'feature_extractor_class': self.feature_extractor_class,

            'policy': policy_checkpoint,
            'feature_extractor': feature_extractor_checkpoint,
            
            'rng_state': self.get_rng_state(),

            'agent_class': self.__class__.__name__,
        }
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_agent.t")
        return checkpoint

    def set_num_env(self, num_envs):
        self.num_envs = num_envs

    @classmethod
    def load(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        instance = cls(checkpoint['action_space'], checkpoint['observation_space'], 
                       checkpoint['hyper_params'], checkpoint['num_envs'],
                       checkpoint['feature_extractor_class'])
        instance.set_rng_state(checkpoint['rng_state'])
        instance.feature_extractor = instance.feature_extractor.load(file_path=None, checkpoint=checkpoint['feature_extractor'])
        instance.policy = instance.policy.load(file_path=None, checkpoint=checkpoint['policy'])
        
        return instance

    def __repr__(self):
        return f"{self.__class__.__name__}({self.hp})"
    

    
class BaseContiualPolicy(BasePolicy):
    def trigger_option_learner(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def init_options(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    

 



    
    
