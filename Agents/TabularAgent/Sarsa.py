import numpy as np
import random
import pickle

from Agents.Utils import BaseAgent, BasePolicy

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
    
    def save(self, file_path):
        """
        Save the Q-table and hyper-parameters.
        
        Args:
            file_path (str): File path for saving the checkpoint.
        """
        checkpoint = {
            'q_table': self.q_table,
            'hyper_params': self.hp,
        }
        with open(file_path, 'wb') as f:
            pickle.dump(checkpoint, f)

    def load(self, file_path):
        """
        Load the Q-table and hyper-parameters.
        
        Args:
            file_path (str): File path from which to load the checkpoint.
        """
        with open(file_path, 'rb') as f:
            checkpoint = pickle.load(f)
        self.q_table = checkpoint.get('q_table', {})
        self.hp = checkpoint.get('hyper_params', self.hp)

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
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class):
        super().__init__(action_space, observation_space, hyper_params, num_envs)
        self.feature_extractor = feature_extractor_class(observation_space)
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