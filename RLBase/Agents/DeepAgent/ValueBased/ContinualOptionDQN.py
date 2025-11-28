import numpy as np
import random
import torch
from copy import copy
from gymnasium.spaces import Discrete

from .OptionDQN import OptionDQNAgent, OptionDQNPolicy
from ...Utils import BaseContiualPolicy
from ....registry import register_agent, register_policy

@register_policy
class ContinualOptionDQNPolicy(OptionDQNPolicy, BaseContiualPolicy):
    pass

@register_agent
class ContinualOptionDQNAgent(OptionDQNAgent):
    name = "ContinualOptionDQN"
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, option_learner_class, options_lst = [], device="cpu"):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class, options_lst, device=device)
        
        # Replace action_space with extended (primitive + options)
        action_option_space = Discrete(self.atomic_action_space.n + len(self.options_lst))
        
        self.initial_options_lst = copy(options_lst)
        self.option_learner = option_learner_class()

        # Swap policy with ContinualOptionDQNPolicy (shares same interface)
        self.policy = ContinualOptionDQNPolicy(
            action_option_space, 
            self.feature_extractor.features_dim,
            self.feature_extractor.num_features, 
            hyper_params,
            device=device
        )

    
    def act(self, observation, greedy=False):
        """
        Returns an atomic (primitive) action to execute.
        If an option is running, returns its current primitive action.
        Otherwise, samples from the extended action space (primitives + options).
        """
        self.last_observation = observation
        action = super().act(observation, greedy)
        return action
    
    def update(self, observation, reward, terminated, truncated, call_back=None):
        """
        After env.step, update Q-values.
        - If inside an option, accumulate discounted reward and do a single SMDP update on option termination.
        - If primitive, do the usual 1-step Q-learning update (parent policy already supports it).
        """
        if self.policy.trigger_option_learner():
            learned_options = self.option_learner.learn(self.options_lst)
            self.options_lst += learned_options
            self.policy.init_options(learned_options)
            
        super().update(observation, reward, terminated, truncated, call_back)
        
    def reset(self, seed):
        super().reset(seed)
        self.option_learner.reset()
        self.options_lst = self.initial_options_lst
        