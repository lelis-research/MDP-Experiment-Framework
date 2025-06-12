import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import Discrete

from ...Utils import (
    BasicBuffer,
    NetworkGen,
    prepare_network_config,
)
from .DQN import DQNAgent, DQNPolicy
from ....registry import register_agent, register_policy
from ....loaders import load_option

@register_policy
class MaskedDQNPolicy(DQNPolicy):
    pass

@register_agent
class MaskedDQNAgent(DQNAgent):
    """
    Deep Q-Network (DQN) agent that uses experience replay and target networks.
    
    Args:
        action_space (gym.spaces.Discrete): The environment's action space.
        observation_space: The environment's observation space.
        hyper_params: Hyper-parameters container (see DQNPolicy).
        num_envs (int): Number of parallel environments.
        feature_extractor_class: Class to extract features from observations.
    """
    name = "MaskedDQN"
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, initial_options, device="cpu"):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device=device)
        self.atomic_action_space = action_space
        self.options = initial_options
        print(f"Number of options: {self.options.n}")
        # action space includes actions and options
        self.action_space = Discrete(self.atomic_action_space.n + self.options.n) 

        # Experience Replay Buffer
        self.replay_buffer = BasicBuffer(hyper_params.replay_buffer_cap)  

        # Create DQNPolicy using the feature extractor's feature dimension.
        self.policy = MaskedDQNPolicy(
            self.action_space, 
            self.feature_extractor.features_dim, 
            hyper_params,
            device=device
        )
        # For handling options.
        self.running_option_index = None
        # Record the state where the option was initiated.
        self.option_start_state = None
        # Accumulate the discounted reward during option execution.
        self.option_cumulative_reward = None
        # Running discount multiplier.
        self.option_multiplier = None
        
    def act(self, observation):
        """
        Select an action based on the current observation.
        
        Args:
            observation (np.array or similar): Raw observation from the environment.
        
        Returns:
            int: Selected action.
        """
        state = self.feature_extractor(observation)
        
        # Check if the option should terminate.
        if self.options.is_terminated(observation):
            self.running_option_index = None

        if self.running_option_index is not None:
            # Continue executing the currently running option.
            action = self.options.select_action(observation, self.running_option_index)
        else:
            # No option is running; select a new action (or option) from the policy.
            action = self.policy.select_action(state)
            if action >= self.atomic_action_space.n:
                # An option was selected. Record its initiation state and initialize accumulators.
                self.running_option_index = action - self.atomic_action_space.n
                self.option_start_state = state
                self.option_cumulative_reward = 0.0
                self.option_multiplier = 1.0
                action = self.options.select_action(observation, self.running_option_index)

        self.last_state = state
        self.last_action = action
        return action
    
    def update(self, observation, reward, terminated, truncated, call_back):
        """
        Store the transition and, if enough samples are available, perform a learning step.
        
        Args:
            observation (np.array or similar): New observation after action.
            reward (float): Reward received.
            terminated (bool): True if the episode has terminated.
            truncated (bool): True if the episode was truncated.
            call_back (function): Callback function to track training progress.
        """
        state = self.feature_extractor(observation)
        
        if self.running_option_index is not None:
            # We're executing an option, so accumulate the discounted reward.
            self.option_cumulative_reward += self.option_multiplier * reward
            self.option_multiplier *= self.hp.gamma  # Use discount factor from hyper-parameters
            
            # Check if the option terminates due to environment termination, truncation, or internal option termination.
            if terminated or truncated or self.options.is_terminated(observation):
                # Create a single transition representing the entire option execution.
                transition = (
                    self.option_start_state, 
                    self.running_option_index + self.atomic_action_space.n, 
                    self.option_cumulative_reward, 
                    state, 
                    terminated
                )
                self.replay_buffer.add_single_item(transition)
                
                # Clear option-related variables.
                self.running_option_index = None
                self.option_start_state = None
                self.option_cumulative_reward = None
                self.option_multiplier = None
        else:
            # For primitive actions, store the transition as usual.
            transition = (self.last_state, self.last_action, reward, state, terminated)
            self.replay_buffer.add_single_item(transition)
        
        # Perform learning step if there are enough samples.
        if self.replay_buffer.size >= self.hp.batch_size:
            batch = self.replay_buffer.get_random_batch(self.hp.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            self.policy.update(states, actions, rewards, next_states, dones, call_back=call_back)
          
    def reset(self, seed):
        """
        Reset the agent's learning state, including feature extractor and replay buffer.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.replay_buffer.reset()

    def save(self, file_path=None):
        checkpoint = super().save(file_path=None)
        options_checkpoint = self.options.save(file_path=None)

        checkpoint['options'] = options_checkpoint
        checkpoint['atomic_action_space'] = self.atomic_action_space

        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_agent.t")
        return checkpoint

    @classmethod
    def load_from_file(cls, file_path, seed=0, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        options = load_option(file_path=None, checkpoint=checkpoint['options'])

        instance = cls(checkpoint['atomic_action_space'], checkpoint['observation_space'], 
                       checkpoint['hyper_params'], checkpoint['num_envs'],
                       checkpoint['feature_extractor_class'], options)
        instance.reset(seed)

        instance.feature_extractor.load_from_checkpoint(checkpoint['feature_extractor'])
        instance.policy.load_from_checkpoint(checkpoint['policy'])
        
        return instance