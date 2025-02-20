import numpy as np
import random
import pickle

from Agents.Utils.BaseAgent import BaseAgent, BasePolicy
from Agents.Utils.FeatureExtractor import TabularFeature

class QLearningPolicy(BasePolicy):
    """
    Epsilon-greedy policy that selects actions based on the Q-table.

        Init Args:
            action_space: The environment's action space (assumed gym.spaces.Discrete)
            hyper-parameters:
                - epsilon
                - gamma
                - step_size
    """       
 
    def select_action(self, state):
        """
        Select an action using epsilon-greedy exploration.
        """        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)

        # With probability epsilon choose a random action...
        if random.random() < self.hp.epsilon:
            return self.action_space.sample()
        else:
            # ...otherwise, choose the action with the highest Q-value.
            return int(np.argmax(self.q_table[state]))
    
            
    def update(self, last_state, last_action, state, reward, terminated, truncated):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)

        # Compute the target.
        if terminated:
            target = reward
        else:
            target = reward + self.hp.gamma * np.max(self.q_table[state])
        
        # Perform the Q-Learning update.
        self.q_table[last_state][last_action] += self.hp.step_size * \
            (target - self.q_table[last_state][last_action])

    def reset(self, seed):
        super().reset(seed)
        self.q_table = {}

    def save(self, file_path):
        checkpoint = {
            'q_table': self.q_table,
            'hyper_params': self.hp,  # Ensure hp is pickle-serializable
        }
        with open(file_path, 'wb') as f:
            pickle.dump(checkpoint, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            checkpoint = pickle.load(f)        
        self.q_table = checkpoint.get('q_table', {})
        self.hp = checkpoint.get('hyper_params', self.hp)
        
class QLearningAgent(BaseAgent):
    """
    Tabular Q-Learning agent.
    
    It uses a TabularQLearningPolicy for action selection and implements
    
    ASSUMPTION: either of the act methods will be called before 
                an update call so no need to reset the 
                last_action and last_observation values
    """
    def __init__(self, action_space, observation_space, hyper_params, num_envs):
        """
        Args:
            action_space: The environment's action space (assumed gym.spaces.Discrete)
            hyper-parameters:
                - epsilon
                - gamma
                - step_size        
        """
        
        super().__init__(action_space, observation_space, hyper_params, num_envs)
        self.feature_extractor = TabularFeature(observation_space)
        self.policy = QLearningPolicy(action_space, hyper_params)
        
    
    def act(self, observation):
        state = self.feature_extractor(observation)
        action = self.policy.select_action(state)

        self.last_action = action
        self.last_state = state
        return action
    
    def update(self, observation, reward, terminated, truncated):
        state = self.feature_extractor(observation)
        self.policy.update(self.last_state, self.last_action, state, reward, terminated, truncated)
    
    
    