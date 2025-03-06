import numpy as np
import random
import torch

from ..Utils import BaseAgent, BasePolicy
from ...registry import register_agent, register_policy


@register_policy
class DoubleQLearningPolicy(BasePolicy):
    """
    Double Q-Learning policy with two Q-tables (q1_table and q2_table).
    Action selection is epsilon-greedy based on Q1 + Q2.
    Hyper-parameters (hp) should include:
      - epsilon (float): Exploration probability.
      - gamma (float): Discount factor.
      - step_size (float): Learning rate.
    """

    def select_action(self, state):
        """
        Select an action using epsilon-greedy strategy.
        
        Args:
            state (hashable): Encoded state (e.g., tuple) used as key in Q-tables.
        
        Returns:
            int: Selected action.
        """
        if state not in self.q1_table:
            self.q1_table[state] = np.zeros(self.action_dim)
        if state not in self.q2_table:
            self.q2_table[state] = np.zeros(self.action_dim)

        if random.random() < self.hp.epsilon:
            return self.action_space.sample()
        else:
            q_sum = self.q1_table[state] + self.q2_table[state]
            return int(np.argmax(q_sum))

    def update(self, last_state, last_action, state, reward, terminated, truncated, call_back=None):
        """
        Perform a Double Q-Learning update on one of the Q-tables.
        
        Args:
            last_state (hashable): Previous state (encoded) from which action was taken.
            last_action (int): Action taken in the last state.
            state (hashable): New state after action.
            reward (float): Reward received.
            terminated (bool): Whether the episode has terminated.
            truncated (bool): Whether the episode was truncated.
            call_back (function, optional): Callback to track training progress.
        """
        if state not in self.q1_table:
            self.q1_table[state] = np.zeros(self.action_dim)
        if state not in self.q2_table:
            self.q2_table[state] = np.zeros(self.action_dim)

        # Randomly update Q1 or Q2.
        if random.random() < 0.5:
            target = reward if terminated else reward + self.hp.gamma * np.max(self.q2_table[state])
            td_error = target - self.q1_table[last_state][last_action]
            self.q1_table[last_state][last_action] += self.hp.step_size * td_error
        else:
            target = reward if terminated else reward + self.hp.gamma * np.max(self.q1_table[state])
            td_error = target - self.q2_table[last_state][last_action]
            self.q2_table[last_state][last_action] += self.hp.step_size * td_error

        if call_back is not None:
            call_back({"value_loss": td_error})

    def reset(self, seed):
        """
        Reset the policy and clear Q-tables.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        self.q1_table = {}
        self.q2_table = {}

    def save(self, file_path=None):
        """
        Save the current Q-tables and hyper-parameters.
        
        Args:
            file_path (str): Path to save the checkpoint.
        """
        checkpoint = {
            'q1_table': self.q1_table,
            'q2_table': self.q2_table,
            'hyper_params': self.hp,

            'action_space': self.action_space,
            'hyper_params': self.hp,

            'action_dim': self.action_dim,  
            'policy_class': self.__class__.__name__,

        }
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_policy.t")
        return checkpoint

    @classmethod         
    def load_from_file(cls, file_path, seed=0, checkpoint=None):
        """
        Load Q-tables and hyper-parameters from a checkpoint.
        
        Args:
            file_path (str): Path to the checkpoint file.
        """
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        instance = cls(checkpoint['action_space'], checkpoint['hyper_params'])
        
        instance.reset(seed)

        instance.q1_table = checkpoint.get('q1_table')
        instance.q2_table = checkpoint.get('q2_table')
        return instance
    
    def load_from_checkpoint(self, checkpoint):
        """
        Load the Q-table and hyper-parameters.
        
        Args:
            file_path (str): File path from which to load the checkpoint.
        """
        self.q1_table = checkpoint.get('q1_table')
        self.q2_table = checkpoint.get('q2_table')

        self.action_space = checkpoint.get('action_space')
        self.hp = checkpoint.get('hyper_params')

        self.action_dim = checkpoint.get('action_dim')
    


@register_agent
class DoubleQLearningAgent(BaseAgent):
    """
    Agent that uses Double Q-Learning with a feature extractor.
    """
    name = "DoubleQLearning"
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class):
        """
        Initialize the agent.
        
        Args:
            action_space (gym.spaces.Discrete): Action space.
            observation_space: Observation space from the environment.
            hyper_params: Hyper-parameters (must include epsilon, gamma, step_size).
            num_envs (int): Number of parallel environments.
            feature_extractor_class (class): Class to extract features from observations.
        """
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class)
        self.policy = DoubleQLearningPolicy(action_space, hyper_params)

    def act(self, observation):
        """
        Select an action using the feature extractor and policy.
        
        Args:
            observation (np.array or similar): Raw observation from the environment.
            
        Returns:
            int: Action selected.
        """
        state = self.feature_extractor(observation)
        action = self.policy.select_action(state)
        self.last_state = state
        self.last_action = action
        return action

    def update(self, observation, reward, terminated, truncated, call_back=None):
        """
        Update the agent using Double Q-Learning.
        
        Args:
            observation (np.array or similar): New observation after action.
            reward (float): Reward received.
            terminated (bool): Whether the episode terminated.
            truncated (bool): Whether the episode was truncated.
            call_back (function, optional): Callback for tracking training progress.
        """
        state = self.feature_extractor(observation)
        self.policy.update(self.last_state, self.last_action, state, reward, terminated, truncated, call_back=call_back)