import numpy as np
import random
from Agents.Utils.BaseAgent import BaseAgent
from Agents.Utils.BasePolicy import BasePolicy

def get_state(observation):
    """
    Helper function to convert an observation (assumed to be a NumPy array)
    into a hashable (discrete) state representation.
    """
    # Flatten the observation and convert to tuple.
    # You can customize this function if you need a different discretization.
    return tuple(observation.flatten().tolist())

class QLearningPolicy(BasePolicy):
    """
    Epsilon-greedy policy that selects actions based on the Q-table.
    """
    def __init__(self, action_space, epsilon, seed=None):
        """
        Args:
            action_space: The environment's action space (assumed gym.spaces.Discrete)
            epsilon: The exploration rate.
        """
        super().__init__(seed)
        self.action_space = action_space
        self.epsilon = epsilon

        if seed is not None:
            self.action_space.seed(seed)
            self._py_rng = random.Random(seed)
        else:
            self._py_rng = random

    def select_action(self, observation, q_table):
        """
        Select an action using epsilon-greedy exploration.
        """
        
        state = get_state(observation)
        
        # With probability epsilon choose a random action...
        if self._py_rng.random() < self.epsilon:
            return self.action_space.sample()
        else:
            # ...otherwise, choose the action with the highest Q-value.
            return int(np.argmax(q_table[state]))

    def reset(self, seed=None):
        if seed is not None:
            self.action_space.seed(seed)
            self._py_rng = random.Random(seed)
        else:
            self._py_rng = random


class QLearningAgent(BaseAgent):
    """
    Tabular Q-Learning agent.
    
    It uses a TabularQLearningPolicy for action selection and implements
    the Q-Learning update rule in its update() method.
    """
    def __init__(self, action_space, hyper_params, seed=None):
        """
        Args:
            action_space: The environment's action space.
            alpha: Learning rate.
            gamma: Discount factor.
            epsilon: Exploration rate for epsilon-greedy policy.
        """
        
        # Initialize the Q-table as an empty dictionary.
        self.q_table = {}
        self.hp = hyper_params

        # These will store the state and action taken at the previous time step.
        self.last_state = None
        self.last_action = None
        
        # Create the Q-Learning policy, providing it with the Q-table.
        policy = QLearningPolicy(action_space, hyper_params.epsilon, seed)
        super().__init__(policy, hyper_params, seed)
    
    def act(self, observation):
        """
        Overrides the BaseAgent.act() to store the current state and action.
        This allows the agent to later update the Q-value for the state-action pair.
        """
        # Convert the observation to a discrete state.
        state = get_state(observation)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.policy.action_space.n)
        
        # Select an action using the Q-Learning policy.
        action = self.policy.select_action(observation, self.q_table)

        self.last_action = action
        self.last_state = state
        return action
    
    def update(self, observation, reward, terminated, truncated):
        """
        Update the Q-table using the Q-Learning update rule.
        
        Args:
            observation: The new observation after the action.
            reward: The reward received after taking the action.
            terminated: Whether the episode terminated.
            truncated: Whether the episode was truncated.
        """
        # If for some reason no previous state/action is stored.
        if self.last_state is None or self.last_action is None:
            raise ValueError("Last State or Last Action are None")
        
        # Convert the new observation to a discrete state.
        state = get_state(observation)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.policy.action_space.n)
        
        # Compute the target.
        if terminated:
            target = reward
        else:
            target = reward + self.hp.gamma * np.max(self.q_table[state])
        
        # Perform the Q-Learning update.
        self.q_table[self.last_state][self.last_action] += self.hp.alpha * (target - self.q_table[self.last_state][self.last_action])
        
    def reset(self, seed=None):
        self.q_table = {}
        self.last_state = None
        self.last_action = None
        if seed is not None:
            self.seed = seed
        self.policy.reset(seed)
    