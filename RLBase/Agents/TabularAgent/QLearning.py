import numpy as np
import random
import torch

from ..Utils import BaseAgent, BasePolicy
from ...registry import register_agent, register_policy


@register_policy
class QLearningPolicy(BasePolicy):
    """
    Epsilon-greedy policy using a Q-table.
    
    Hyper-parameters in hp must include:
        - epsilon_start (float)
        - epsilon_end (float)
        - epilon_decay_steps (int)
        - gamma (float)
        - step_size (float)
    
    Args:
        action_space (gym.spaces.Discrete): Action space.
        hyper_params: Hyper-parameters container.
    """       
    def __init__(self, action_space, hyper_params=None, device='cpu'):
        super().__init__(action_space, hyper_params, device)
        self.action_dim = int(action_space.n)
        self.epsilon = self.hp.epsilon_start
        self.step_counter = 0
        
    def select_action(self, state, greedy=False):
        """
        Select an action using epsilon-greedy exploration.
        
        Args:
            state (hashable): Encoded state (e.g., tuple) used as key in Q-table.
        
        Returns:
            int: Selected action.
        """ 
        self.step_counter += 1       
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_dim)
        
        if random.random() < self.epsilon and not greedy:
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

        #Update Value Function
        target = reward if terminated else reward + self.hp.gamma * np.max(self.q_table[state])
        td_error = target - self.q_table[last_state][last_action]
        self.q_table[last_state][last_action] += self.hp.step_size * td_error
        
        #Update Epsilon
        frac = 1.0 - (self.step_counter / self.hp.epilon_decay_steps)
        self.epsilon = self.hp.epsilon_end + (self.hp.epsilon_start - self.hp.epsilon_end) * frac
            
        if call_back is not None:
            call_back({"value_loss": td_error,
                       "epsilon": self.epsilon,
                       })

    def reset(self, seed):
        """
        Reset the Q-table and seed random generators.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        self.q_table = {}
        self.epsilon = self.hp.epsilon_start
        self.step_counter = 0

    def save(self, file_path=None):
        """
        Save the Q-table and hyper-parameters.
        
        Args:
            file_path (str): File path for saving.
        """
        checkpoint = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
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
        instance.epsilon = checkpoint.get('epsilon')
        return instance
    
    def load_from_checkpoint(self, checkpoint):
        """
        Load the Q-table and hyper-parameters.
        
        Args:
            file_path (str): File path from which to load the checkpoint.
        """
        self.q_table = checkpoint.get('q_table')
        self.epsilon = checkpoint.get('epsilon')

        self.action_space = checkpoint.get('action_space')
        self.hp = checkpoint.get('hyper_params')

@register_agent        
class QLearningAgent(BaseAgent):
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
    name = "QLearning"
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class)
        self.policy = QLearningPolicy(action_space, hyper_params)
        
    def act(self, observation, greedy=False):
        """
        Select an action based on the observation.
        
        Args:
            observation (np.array or similar): Raw observation from the environment.
        
        Returns:
            int: Selected action.
        """
        state = self.feature_extractor(observation)
        action = self.policy.select_action(state, greedy=greedy)
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
    
    def reset(self, seed):
        super().reset(seed)

        self.last_state = None
        self.last_action = None
        
