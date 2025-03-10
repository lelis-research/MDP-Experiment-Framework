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
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, initial_options):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class)
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
            hyper_params
        )
        self.running_option_index = None
        
    def act(self, observation):
        """
        Select an action based on the current observation.
        
        Args:
            observation (np.array or similar): Raw observation from the environment.
        
        Returns:
            int: Selected action.
        """
        state = self.feature_extractor(observation)
        if self.options.is_terminated(observation):
            self.running_option_index = None
        
        if self.running_option_index is not None:
            action = self.options.select_action(observation, self.running_option_index)
        else:
            action = self.policy.select_action(state)
            if action >= self.atomic_action_space.n:
                self.running_option_index = action - self.atomic_action_space.n
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
            transition = (self.last_state[0], self.running_option_index + self.atomic_action_space.n, reward, state[0], terminated)
        else:
            transition = (self.last_state[0], self.last_action, reward, state[0], terminated)
        self.replay_buffer.add_single_item(transition)
        
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