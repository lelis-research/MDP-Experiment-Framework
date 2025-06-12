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

        # For handling options
        self.running_option_index = None
        self.option_start_state = None
        self.option_cumulative_reward = None
        self.option_multiplier = None  # will hold the running discount multiplier
        
    def act(self, observation):
        state = self.feature_extractor(observation)

        # Check if the currently running option has terminated.
        if self.running_option_index is not None and self.options.is_terminated(observation):
            # Option termination detected.
            self.running_option_index = None

        if self.running_option_index is not None:
            # Continue executing the previously chosen option.
            action = self.options.select_action(observation, self.running_option_index)
        else:
            # No option is running; select a new action or option from the policy.
            action = self.policy.select_action(state)
            
            if action >= self.atomic_action_space.n:
                # The policy chose an option.
                self.running_option_index = action - self.atomic_action_space.n
                # Record the starting state of the option.
                self.option_start_state = state
                # Initialize the accumulator for the cumulative discounted reward.
                self.option_cumulative_reward = 0.0
                # Initialize the discount multiplier.
                self.option_multiplier = 1.0
                # Get the first primitive action from the option.
                action = self.options.select_action(observation, self.running_option_index)
            # For primitive actions, record the last state and action for 1-step update.
            self.last_state = state
            self.last_action = action
        
        return action
    
    def update(self, observation, reward, terminated, truncated, call_back=None):
        state = self.feature_extractor(observation)
        
        # Check if an option is running.
        if self.running_option_index is not None:
            # We are executing an option. Instead of updating at every step,
            # we accumulate the discounted reward.
            self.option_cumulative_reward += self.option_multiplier * reward
            self.option_multiplier *= self.hp.gamma

            # If the option terminates (or the episode ends), perform the update.
            if terminated or truncated or self.options.is_terminated(observation):
                # Perform a multi-step update at the starting state of the option.
                self.policy.update(
                    self.option_start_state, 
                    self.running_option_index + self.atomic_action_space.n, 
                    state, 
                    self.option_cumulative_reward, 
                    terminated, 
                    truncated, 
                    call_back=call_back
                )
                # Reset option-related variables.
                self.running_option_index = None
                self.option_start_state = None
                self.option_cumulative_reward = None
                self.option_multiplier = None
        else:
            # No option is running, so perform a standard one-step update for primitive actions.
            self.policy.update(
                self.last_state, 
                self.last_action, 
                state, 
                reward, 
                terminated, 
                truncated, 
                call_back=call_back
            )
        
        

    def reset(self, seed):
        super().reset(seed)
        self.running_option_index = None