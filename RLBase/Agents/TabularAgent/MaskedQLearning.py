import numpy as np
import random
import torch
from gymnasium.spaces import Discrete

from .QLearning import QLearningAgent, QLearningPolicy
from ...registry import register_agent, register_policy
from ...Options.MaskedOptions import LevinLossMaskedOptions, LevinLossMaskedOptionLearner


@register_policy
class MaskedQLearningPolicy(QLearningPolicy):
    pass

@register_agent        
class MaskedQLearningAgent(QLearningAgent):
    """
    Tabular Q-Learning agent using a options.
    
    Assumes that act is called before update, so last_state and last_action are available.
    
    Args:
        action_space (gym.spaces.Discrete): Action space.
        observation_space: Environment observation space.
        hyper_params: Hyper-parameters (epsilon, gamma, step_size).
        num_envs (int): Number of parallel environments.
        feature_extractor_class: Class for feature extraction.
    """
    name = "MaskedQLearning"
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, initial_options):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class)
        self.atomic_action_space = action_space
        self.options = initial_options
        
        # action space includes actions and options
        self.action_space = Discrete(self.atomic_action_space.n + self.options.n) 

        # policy doesn't know if it is choosing options or actions
        self.policy = MaskedQLearningPolicy(self.action_space, hyper_params)

        self.running_option_index = None
        
    def act(self, observation):
        state = self.feature_extractor(observation)

        if self.options.is_terminated(observation):
            self.running_option_index = None

        # either get the next action from an option or from the main policy
        if self.running_option_index is not None:
            # continue running previous option
            action = self.options.select_action(observation, self.running_option_index)
        else:
            # get a new action/option from policy
            action = self.policy.select_action(state)
            
            if action >= self.atomic_action_space.n:
                #if it is a new option
                self.running_option_index = action - self.atomic_action_space.n
                action = self.options.select_action(observation, self.running_option_index)
            
        
        self.last_action = action
        self.last_state = state
        return action
    
    def update(self, observation, reward, terminated, truncated, call_back=None):
        state = self.feature_extractor(observation)
        if terminated or truncated:
            self.running_option_index = None
            
        if self.running_option_index is not None:
            self.policy.update(self.last_state, self.running_option_index + self.atomic_action_space.n, state, reward, terminated, truncated, call_back=call_back)
        else:
            self.policy.update(self.last_state, self.last_action, state, reward, terminated, truncated, call_back=call_back)
        
        

    def reset(self, seed):
        super().reset(seed)
        self.running_option_index = None