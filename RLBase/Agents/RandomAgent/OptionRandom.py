import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete

from ..Base import BaseAgent, BasePolicy
from ...registry import register_agent, register_policy
from ..Utils import get_single_observation_nobatch
from .Random import RandomAgent, RandomPolicy

@register_policy
class OptionRandomPolicy(RandomPolicy):
    pass


@register_agent
class OptionRandomAgent(RandomAgent):
    name = "OptionRandom"
    SUPPORTED_ACTION_SPACES = (Discrete, )

    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, init_option_lst=None):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class)
        
        # Keep atomic action space; extend for options.
        self.atomic_action_space = action_space
        self.options_lst = [] if init_option_lst is None else init_option_lst
        
        # Replace action_space with extended (primitive + options)
        action_option_space = Discrete(self.atomic_action_space.n + len(self.options_lst))
        
        # Swap policy with OptionQLearningPolicy (shares same interface)
        self.policy = OptionRandomPolicy(action_option_space, hyper_params)
        
        # Option execution bookkeeping
        self.running_option_index = [None for _ in range(self.num_envs)]       # index into options_lst (or None)


    def act(self, observation, greedy=False):
        action = []
        for i in range(self.num_envs):
            # If an option is currently running, either continue it or end it here.
            obs_option = get_single_observation_nobatch(observation, i)
            curr_option_idx = self.running_option_index[i]
            if curr_option_idx is not None:
                a = self.options_lst[curr_option_idx].select_action(obs_option)
            else:
                # Choose an extended action (might be a primitive or an option)
                a = self.policy.select_action(None, greedy=greedy)
                if a >= self.atomic_action_space.n:
                    # Start an option
                    curr_option_idx = a - self.atomic_action_space.n
                    a = self.options_lst[curr_option_idx].select_action(obs_option)
                else:
                    curr_option_idx = None
            
            self.running_option_index[i] = curr_option_idx
            action.append(a)
        return action

    def update(self, observation, reward, terminated, truncated, call_back=None):
        
        for i in range(self.num_envs):
            if call_back is not None:
                call_back({
                    f"train/option_usage_env_{i}": 1 if self.running_option_index[i] is not None else 0,
                })
            
            obs_option = get_single_observation_nobatch(observation, i)
            curr_option_idx = self.running_option_index[i]
            
            if curr_option_idx is not None:
                if self.options_lst[curr_option_idx].is_terminated(obs_option) or terminated[i] or truncated[i]: 
                    self.running_option_index[i] = None
                    self.options_lst[curr_option_idx].reset()


                
       
      
    
    def reset(self, seed):
        super().reset(seed)
        
        # Option execution bookkeeping
        self.running_option_index = [None for _ in range(self.num_envs)]       # index into options_lst (or None)
