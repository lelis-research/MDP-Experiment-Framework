import numpy as np
import random

from Agents.Utils import (
    BaseAgent,
    BasePolicy,
    BasicBuffer,
    calculate_n_step_returns,
    register_agent,
    register_policy,
)
from .QLearning import QLearningAgent, QLearningPolicy

@register_policy
class NStepQLearningPolicy(QLearningPolicy):
    """
    n-step Q-Learning policy that maintains a buffer of the last n transitions.
    When the buffer has at least n transitions, it computes an n‑step return G and updates
    the Q-value for the oldest state–action pair.
    
    Hyper-parameters (in hp) must include:
        - epsilon (float)
        - gamma (float)
        - step_size (float)
        - n_steps (int)
    
    Args:
        action_space (gym.spaces.Discrete): The environment's action space.
        hyper_params: Hyper-parameters container.
    """
    
    def reset(self, seed):
        """
        Reset the policy for a new episode.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        self.rollout_buffer = BasicBuffer(np.inf)  # Buffer with infinite capacity

    def update(self, last_state, last_action, state, reward, terminated, truncated, call_back=None):
        """
        Update the Q-table using an n-step return.
        
        Args:
            last_state (hashable): The previous state (encoded) where the last action was taken.
            last_action (int): The action taken in the previous state.
            state (hashable): The current state (encoded) after taking the action.
            reward (float): The reward received after taking the action.
            terminated (bool): True if the episode has terminated.
            truncated (bool): True if the episode was truncated.
            call_back (function, optional): A callback function for tracking training progress.
        """
        # Append the latest transition (last_state, last_action, reward) to the buffer.
        transition = (last_state, last_action, reward)
        self.rollout_buffer.add_single_item(transition)
        
        # Ensure the current state is initialized in the Q-table.
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_dim)
        
        # If we have collected at least n_steps transitions, update Q-value for the oldest transition.
        if self.rollout_buffer.size >= self.hp.n_steps:
            rollout = self.rollout_buffer.get_all()  # All transitions in the buffer
            # Unpack transitions (states and actions not used directly here).
            rollout_states, rollout_actions, rollout_rewards = zip(*rollout)
            
            # Use the max Q-value from the current state as bootstrap, unless terminated.
            bootstrap_value = np.max(self.q_table[state]) if not terminated else 0.0

            # Compute n-step return (only the first value is used for the oldest transition).
            n_step_return = calculate_n_step_returns(rollout_rewards, bootstrap_value, self.hp.gamma)[0]

            # Remove the oldest transition from the buffer and update its Q-value.
            s, a, _ = self.rollout_buffer.remove_oldest()
            td_error = n_step_return - self.q_table[s][a]
            self.q_table[s][a] += self.hp.step_size * td_error

            if call_back is not None:
                call_back({"value_loss": td_error})
        
        # At episode end, flush any remaining transitions in the buffer.
        if terminated or truncated:
            while self.rollout_buffer.size > 0:
                rollout = self.rollout_buffer.get_all()
                rollout_states, rollout_actions, rollout_rewards = zip(*rollout)
                bootstrap_value = np.max(self.q_table[state]) if not terminated else 0.0
                n_step_return = calculate_n_step_returns(rollout_rewards, bootstrap_value, self.hp.gamma)[0]
                s, a, _ = self.rollout_buffer.remove_oldest()
                td_error = n_step_return - self.q_table[s][a]
                self.q_table[s][a] += self.hp.step_size * td_error
                if call_back is not None:
                    call_back({"value_loss": td_error})
                
@register_agent
class NStepQLearningAgent(QLearningAgent):
    """
    n-step Q-Learning agent that uses a feature extractor and performs multi-step updates.
    
    Hyper-parameters (in hp) must include:
        - epsilon (float)
        - gamma (float)
        - step_size (float)
        - n_steps (int)
    
    Args:
        action_space (gym.spaces.Discrete): The environment's action space.
        observation_space: The environment's observation space.
        hyper_params: Hyper-parameters container.
        num_envs (int): Number of parallel environments.
        feature_extractor_class (class): Class for extracting features from observations.
    """
    name = "NStepQLearning"
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class)
        self.policy = NStepQLearningPolicy(action_space, hyper_params)