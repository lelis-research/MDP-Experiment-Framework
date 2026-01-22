import torch
import numpy as np

from ..utils import RandomGenerator

class BasePolicy(RandomGenerator):
    """
    Abstract policy interface.
    Subclasses implement `select_action(state)` where `state` is a
    batched feature dict/tensor produced by the agent's feature extractor.
    """
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
        # `state` should include a batch dimension, even if size 1.
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
            'rng_state': self.get_rng_state(),
            'device': self.device,
        }
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_policy.t")
        return checkpoint

    @classmethod
    def load(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        
        instance = cls(checkpoint['action_space'], checkpoint['hyper_params'], checkpoint['device'])
        instance.set_rng_state(checkpoint['rng_state'])
        
        return instance
        
class BaseAgent(RandomGenerator):
    """Base class for an RL agent using a policy and feature extractor."""
    SUPPORTED_ACTION_SPACES = ()
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device='cpu'):
        if not isinstance(action_space, self.SUPPORTED_ACTION_SPACES):
            raise TypeError(f"{self.__class__.__name__} only supports action spaces {self.SUPPORTED_ACTION_SPACES}, got {type(action_space)}")
        
        self.hp = hyper_params
        self.action_space = action_space
        self.observation_space = observation_space
        self.num_envs = num_envs
        self.device = device
        self.feature_extractor_class = feature_extractor_class
        
        self.feature_extractor = feature_extractor_class(observation_space, device=device)
        self.policy = BasePolicy(action_space, device=device)
        
        self._mode = "train"
        self._init_log_buf()

        
    @property
    def training(self):
        if self._mode =="train":
            return True
        elif self._mode == "eval":
            return False
        raise ValueError(f"Training mode {self._mode} is not known")
    
    def train(self):
        self._mode = "train"
    def eval(self):
        self._mode = "eval"
           
    def act(self, observation):
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
        self._init_log_buf()
        
    def log(self):
        # returns list length num_envs, each item either None or dict-of-arrays
        return self._flush_log_buf()
    
    def _init_log_buf(self):
        # one buffer per env slot, to avoid mixing logs between envs
        self.log_buf = []
        for _ in range(self.num_envs):
            self.log_buf.append({}) # append the necessary data keys

    def _flush_log_buf(self):
        """
        Return a list of per-env logs, each as dict-of-numpy-arrays.
        Also clears buffers.
        """
        out = []
        for env_i, buf in enumerate(self.log_buf):
            tmp_dict = {}
            for key in buf:
                data = buf[key]
                if len(data) == 0:
                    out.append(None) # no logs for this env since last flush
                    continue
                tmp_dict[key] = np.stack(data) # should be list of numpy arrays
                buf[key].clear()
            out.append(tmp_dict)
        return out

    
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
            'device': self.device,
            
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
                       checkpoint['feature_extractor_class'], checkpoint['device'])
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
    

 



    
    
