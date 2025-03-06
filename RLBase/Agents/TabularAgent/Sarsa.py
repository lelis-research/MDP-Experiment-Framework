import numpy as np
import random
import torch

from ..Utils import BaseAgent, BasePolicy
from ...registry import register_agent, register_policy


@register_policy
class SarsaPolicy(BasePolicy):
    """
    Epsilon-greedy SARSA policy that updates Q(s,a) using Q(s',a') from the selected action.
    
    Hyper-parameters (in hp) must include:
        - epsilon (float)
        - gamma (float)
        - step_size (float)
    
    Args:
        action_space (gym.spaces.Discrete): Environment's action space.
        hyper_params: Container for hyper-parameters.
    """

    def select_action(self, state):
        """
        Select an action using epsilon-greedy exploration.
        
        Args:
            state (hashable): Encoded state (e.g., tuple) used as key in Q-table.
            
        Returns:
            int: Selected action.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_dim)

        if random.random() < self.hp.epsilon:
            return self.action_space.sample()
        else:
            return int(np.argmax(self.q_table[state]))
    
    def update(self, last_state, last_action, state, reward, terminated, truncated, call_back=None):
        """
        Update Q-value using SARSA update.
        
        Args:
            last_state (hashable): Previous state (encoded) where action was taken.
            last_action (int): Action taken in last_state.
            state (hashable): Current state (encoded).
            reward (float): Reward received.
            terminated (bool): True if episode terminated.
            truncated (bool): True if episode truncated.
            call_back (function, optional): Callback for tracking training progress.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_dim)
        
        if terminated:
            target = reward
        else:
            next_action = self.select_action(state)
            target = reward + self.hp.gamma * self.q_table[state][next_action]

        td_error = target - self.q_table[last_state][last_action]
        self.q_table[last_state][last_action] += self.hp.step_size * td_error
        
        if call_back is not None:
            call_back({"value_loss": td_error})

    def reset(self, seed):
        """
        Reset the policy: clear Q-table and seed RNGs.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        self.q_table = {}
    
    def save(self, file_path=None):
        """
        Save the Q-table and hyper-parameters.
        
        Args:
            file_path (str): File path for saving the checkpoint.
        """
        checkpoint = {
            'q_table': self.q_table,
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
        Load the Q-table and hyper-parameters.
        
        Args:
            file_path (str): File path from which to load the checkpoint.
        """
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        instance = cls(checkpoint['action_space'], checkpoint['hyper_params'])
        
        instance.reset(seed)

        instance.q_table = checkpoint.get('q_table')
        return instance
    
    def load_from_checkpoint(self, checkpoint):
        """
        Load the Q-table and hyper-parameters.
        
        Args:
            file_path (str): File path from which to load the checkpoint.
        """
        self.q_table = checkpoint.get('q_table')

        self.action_space = checkpoint.get('action_space')
        self.hp = checkpoint.get('hyper_params')

        self.action_dim = checkpoint.get('action_dim')

@register_agent
class SarsaAgent(BaseAgent):
    """
    SARSA agent that uses a feature extractor and SARSA policy.
    
    Args:
        action_space (gym.spaces.Discrete): Environment's action space.
        observation_space: Environment's observation space.
        hyper_params: Hyper-parameters (epsilon, gamma, step_size).
        num_envs (int): Number of parallel environments.
        feature_extractor_class: Class to extract features from observations.
    """
    name = "Sarsa"
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class)
        self.policy = SarsaPolicy(action_space, hyper_params)

    def act(self, observation):
        """
        Select an action based on the current observation.
        
        Args:
            observation (np.array or similar): Raw observation.
        
        Returns:
            int: Selected action.
        """
        state = self.feature_extractor(observation)
        action = self.policy.select_action(state)
        self.last_state = state
        self.last_action = action
        return action
    
    def update(self, observation, reward, terminated, truncated, call_back=None):
        """
        Update the Q-table using SARSA update.
        
        Args:
            observation (np.array or similar): New observation after taking action.
            reward (float): Reward received.
            terminated (bool): True if episode terminated.
            truncated (bool): True if episode truncated.
            call_back (function, optional): Callback for tracking training progress.
        """
        state = self.feature_extractor(observation)
        self.policy.update(self.last_state, self.last_action, state, reward, terminated, truncated, call_back=call_back)