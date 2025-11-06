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
    calculate_n_step_returns_with_discounts,
)
from .DQN import DQNAgent, DQNPolicy
from ....registry import register_agent, register_policy
from ....loaders import load_option
from ....Options import load_options_list, save_options_list
#NOTE: DOESN'T INCLUDE THE N-STEP

@register_policy
class OptionDQNPolicy(DQNPolicy):
    pass

@register_agent
class OptionDQNAgent(DQNAgent):
    """
    Deep Q-Network (DQN) agent that uses experience replay and target networks.
    
    Args:
        action_space (gym.spaces.Discrete): The environment's action space.
        observation_space: The environment's observation space.
        hyper_params: Hyper-parameters container (see DQNPolicy).
        num_envs (int): Number of parallel environments.
        feature_extractor_class: Class to extract features from observations.
    """
    name = "OptionDQN"
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, options_lst, device="cpu"):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device=device)
        
        self.atomic_action_space = action_space
        self.options_lst = options_lst
        print(f"Number of options: {len(options_lst)}")
        
        # action space includes actions and options
        action_option_space = Discrete(self.atomic_action_space.n + len(self.options_lst)) 

        # Experience Replay Buffer
        self.replay_buffer = BasicBuffer(hyper_params.replay_buffer_cap)  

        # Create DQNPolicy using the feature extractor's feature dimension.
        self.policy = OptionDQNPolicy(
            action_option_space, 
            self.feature_extractor.features_dim, 
            hyper_params,
            device=device
        )
        # Buffer to accumulate n-step transitions.
        self.n_step_buffer = BasicBuffer(hyper_params.n_steps)
        
        # Option execution bookkeeping
        self.running_option_index = None       # index into options_lst (or None)
        self.option_start_state = None         # encoded state where option began
        self.option_cumulative_reward = 0.0    # discounted return accumulator R_{t:t+k}
        self.option_multiplier = 1.0           # current gamma^t during option
        self.option_steps = 0
        
        # Parent keeps last_state/last_action for primitive updates
        self.last_state = None
        self.last_action = None
          
    def act(self, observation, greedy=False):
        """
        Select an action based on the current observation.
        
        Args:
            observation (np.array or similar): Raw observation from the environment.
        
        Returns:
            int: Selected action.
        """
        state = self.feature_extractor(observation)
        
        # If an options is running currently
        if self.running_option_index is not None:
            # Check if the option should terminate.
            if self.options_lst[self.running_option_index].is_terminated(observation):
                self.running_option_index = None
            else:
                # Continue executing the currently running option.
                action = self.options_lst[self.running_option_index].select_action(observation)
        
        if self.running_option_index is None:
            # No option is running; select a new action (or option) from the policy.
            action = self.policy.select_action(state, greedy=greedy)
            if action >= self.atomic_action_space.n:
                # An option was selected. Record its initiation state and initialize accumulators.
                self.running_option_index = action - self.atomic_action_space.n
                self.option_start_state = state
                self.option_cumulative_reward = 0.0
                self.option_multiplier = 1.0
                self.option_steps = 0
                action = self.options_lst[self.running_option_index].select_action(observation)

        self.last_state = state
        self.last_action = action
        return action
    
    def update(self, observation, reward, terminated, truncated, call_back=None):
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
        
        # add option or action to the n_step_buffer
        if self.running_option_index is not None:
            # We're executing an option, so accumulate the discounted reward.
            self.option_cumulative_reward += self.option_multiplier * reward
            self.option_multiplier *= self.hp.gamma  # Use discount factor from hyper-parameters
            self.option_steps += 1
            
            # Check if the option terminates due to environment termination, truncation, or internal option termination.
            if terminated or truncated or self.options_lst[self.running_option_index].is_terminated(observation):
                # Create a single transition representing the entire option execution.
                transition = (
                    self.option_start_state, 
                    self.running_option_index + self.atomic_action_space.n, 
                    self.option_cumulative_reward, 
                    state, 
                    terminated,
                    self.option_multiplier, 
                    self.option_steps
                )
                self.n_step_buffer.add_single_item(transition)
                
                # Clear option-related variables.
                self.running_option_index = None
                self.option_start_state = None
                self.option_cumulative_reward = 0.0
                self.option_multiplier = 1.0
                self.option_steps = 0
        else:
            # For primitive actions, store the transition as usual.
            transition = (self.last_state, self.last_action, reward, state, terminated, self.hp.gamma, 1)
            self.n_step_buffer.add_single_item(transition)
            
        
        # If enough transitions are accumulated or if episode ends:
        if self.n_step_buffer.size >= self.hp.n_steps or terminated or truncated:
            rollout = self.n_step_buffer.get_all()
            states, actions, rewards, next_states, dones, discounts, steps = zip(*rollout)
            
            # Compute n-step returns using the accumulated rewards.
            returns = calculate_n_step_returns_with_discounts(rewards, 0.0, discounts)
            all_steps = sum(steps)
            if terminated or truncated:
                # For episode end, flush all transitions.
                for i in range(self.n_step_buffer.size):
                    trans = (states[i], actions[i], returns[i], next_states[-1], dones[-1], all_steps - steps[i])
                    self.replay_buffer.add_single_item(trans)
                self.n_step_buffer.reset()
            else:
                # Otherwise, add only the oldest transition.
                trans = (states[0], actions[0], returns[0], next_states[-1], dones[-1], all_steps)
                self.replay_buffer.add_single_item(trans)
                self.n_step_buffer.remove_oldest()
        
        # Perform learning step if there are enough samples.
        if self.replay_buffer.size >= self.hp.batch_size:
            batch = self.replay_buffer.get_random_batch(self.hp.batch_size)
            states, actions, rewards, next_states, dones, n_steps = zip(*batch)
            self.policy.update(states, actions, rewards, next_states, dones, n_steps, call_back=call_back)
          
    def reset(self, seed):
        """
        Reset the agent's learning state, including feature extractor and replay buffer.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.replay_buffer.reset()
        
        # Clear any running option state
        self.running_option_index = None       # index into options_lst (or None)
        self.option_start_state = None         # encoded state where option began
        self.option_cumulative_reward = 0.0    # discounted return accumulator R_{t:t+k}
        self.option_multiplier = 1.0           # current gamma^t during option
        self.option_steps = 0
        
        # Parent keeps last_state/last_action for primitive updates
        self.last_state = None
        self.last_action = None

    def save(self, file_path=None):
        checkpoint = super().save(file_path=None)
        options_checkpoint = save_options_list(self.options_lst, file_path=None)

        checkpoint['options_lst'] = options_checkpoint
        checkpoint['atomic_action_space'] = self.atomic_action_space

        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_agent.t")
        return checkpoint

    @classmethod
    def load_from_file(cls, file_path, seed=0, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        options_lst = load_options_list(file_path=None, checkpoint=checkpoint['options_lst'])

        instance = cls(
            checkpoint['atomic_action_space'], 
            checkpoint['observation_space'], 
            checkpoint['hyper_params'], 
            checkpoint['num_envs'],
            checkpoint['feature_extractor_class'], 
            options_lst
        )
        instance.reset(seed)

        instance.feature_extractor.load_from_checkpoint(checkpoint['feature_extractor'])
        instance.policy.load_from_checkpoint(checkpoint['policy'])
        
        return instance