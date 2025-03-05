import numpy as np
import random
import pickle

from Agents.TabularAgent.Basics import QLearningAgent, QLearningPolicy

class MaskedQLearningPolicy(QLearningPolicy):
    """
    Epsilon-greedy policy using a Q-table.
    
    Hyper-parameters in hp must include:
        - epsilon (float)
        - gamma (float)
        - step_size (float)
    
    Args:
        action_space (gym.spaces.Discrete): Action space.
        hyper_params: Hyper-parameters container.
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
        Update the Q-table using the Q-Learning update rule.
        
        Args:
            last_state (hashable): Previous state (encoded) where the last action was taken.
            last_action (int): Action taken in the previous state.
            state (hashable): Current state (encoded) after action.
            reward (float): Reward received.
            terminated (bool): True if the episode has terminated.
            truncated (bool): True if the episode was truncated.
            call_back (function, optional): Callback for tracking training progress.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_dim)

        target = reward if terminated else reward + self.hp.gamma * np.max(self.q_table[state])
        td_error = target - self.q_table[last_state][last_action]
        self.q_table[last_state][last_action] += self.hp.step_size * td_error
        
        if call_back is not None:
            call_back({"value_loss": td_error})

    def reset(self, seed):
        """
        Reset the Q-table and seed random generators.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        self.q_table = {}

    def save(self, file_path):
        """
        Save the Q-table and hyper-parameters.
        
        Args:
            file_path (str): File path for saving.
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
            file_path (str): File path to load from.
        """
        with open(file_path, 'rb') as f:
            checkpoint = pickle.load(f)        
        self.q_table = checkpoint.get('q_table', {})
        self.hp = checkpoint.get('hyper_params', self.hp)
        
class MaskedQLearningAgent(QLearningAgent):
    """
    Tabular Q-Learning agent using a feature extractor and QLearningPolicy.
    
    Assumes that act is called before update, so last_state and last_action are available.
    
    Args:
        action_space (gym.spaces.Discrete): Action space.
        observation_space: Environment observation space.
        hyper_params: Hyper-parameters (epsilon, gamma, step_size).
        num_envs (int): Number of parallel environments.
        feature_extractor_class: Class for feature extraction.
    """
    name = "MaskedQLearning"
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class):
        super().__init__(action_space, observation_space, hyper_params, num_envs)
        self.feature_extractor = feature_extractor_class(observation_space)
        self.policy = QLearningPolicy(action_space, hyper_params)
        
    def act(self, observation):
        """
        Select an action based on the observation.
        
        Args:
            observation (np.array or similar): Raw observation from the environment.
        
        Returns:
            int: Selected action.
        """
        state = self.feature_extractor(observation)
        action = self.policy.select_action(state)
        self.last_action = action
        self.last_state = state
        return action
    
    def update(self, observation, reward, terminated, truncated, call_back=None):
        """
        Update the Q-table based on the new observation and received reward.
        
        Args:
            observation (np.array or similar): New observation after action.
            reward (float): Reward received.
            terminated (bool): True if the episode has terminated.
            truncated (bool): True if the episode was truncated.
            call_back (function, optional): Callback for tracking training progress.
        """
        state = self.feature_extractor(observation)
        self.policy.update(self.last_state, self.last_action, state, reward, terminated, truncated, call_back=call_back)