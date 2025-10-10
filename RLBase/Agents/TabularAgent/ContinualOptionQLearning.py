import numpy as np
import random
import torch
from gymnasium.spaces import Discrete

from ..Utils import BaseAgent, BasePolicy
from .OptionQLearning import OptionQLearningAgent, OptionQLearningPolicy
from ...registry import register_agent, register_policy
from ...Options.Utils import load_options_list, save_options_list

@register_policy
class ContinualOptionQLearningPolicy(OptionQLearningPolicy):
    pass

@register_agent
class ContinualOptionQLearningAgent(OptionQLearningAgent):
    name = "ContinualOptionQLearning"
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, option_learner_class):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class, options_lst=[])
        self.option_learner = option_learner_class()
        
        # Keep atomic action space; extend for options.
        self.atomic_action_space = action_space
        self.options_lst = []
        
        # Replace action_space with extended (primitive + options)
        action_option_space = Discrete(self.atomic_action_space.n + len(self.options_lst))

        # Swap policy with OptionQLearningPolicy (shares same interface)
        self.policy = ContinualOptionQLearningPolicy(action_option_space, hyper_params)

        # Option execution bookkeeping
        self.running_option_index = None       # index into options_lst (or None)
        self.option_start_state = None         # encoded state where option began
        self.option_cumulative_reward = 0.0    # discounted return accumulator R_{t:t+k}
        self.option_multiplier = 1.0           # current gamma^t during option

        # Parent keeps last_state/last_action for primitive updates
        self.last_state = None
        self.last_action = None
    
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
        if self.option_learner.evaluate_option_trigger(self.last_observation, self.last_action, observation, reward, self.options_lst):
            self.option_learner.extract_options(self.options_lst)
            self.option_learner.init_options(self.policy)
            
        super().update(observation, reward, terminated, truncated, call_back)
        
    
    def reset(self, seed):
        super().reset(seed)
        self.option_learner.reset()
        
    def log(self):
        if self.running_option_index is None:
            return {"OptionUsageLog": False, "NumOptions":len(self.options_lst)}
        else:
            return {"OptionUsageLog": True, "NumOptions":len(self.options_lst)}