import numpy as np
import random
from Agents.Utils.BaseAgent import BaseAgent
from Agents.Utils.BasePolicy import BasePolicy

def get_state(observation):
    """
    Helper function to convert an observation (assumed to be a NumPy array)
    into a hashable (discrete) state representation.
    """
    return tuple(observation.flatten().tolist())

class QLearningPolicy(BasePolicy):
    """
    Epsilon-greedy policy that selects actions based on the Q-table.
    (Same as before; the policy only reads from the Q-table.)
    """
    def __init__(self, action_space, q_table, epsilon, seed=None):
        """
        Args:
            action_space: The environment's action space (assumed gym.spaces.Discrete).
            q_table: A dictionary mapping states to Q-value arrays.
            epsilon: The exploration rate.
            seed: Optional seed for reproducibility.
        """
        super().__init__(seed)
        self.action_space = action_space
        self.q_table = q_table
        self.epsilon = epsilon

        if seed is not None:
            self.action_space.seed(seed)
            self._py_rng = random.Random(seed)
        else:
            self._py_rng = random

    def select_action(self, observation):
        """
        Select an action using epsilon-greedy exploration.
        Assumes the state is already initialized in the Q-table.
        """
        state = get_state(observation)
        if self._py_rng.random() < self.epsilon:
            return self.action_space.sample()
        else:
            return int(np.argmax(self.q_table[state]))

    def reset(self, q_table):
        self.q_table = q_table

class NStepQLearningAgent(BaseAgent):
    """
    n‑Step Tabular Q‑Learning agent.
    
    The agent uses an n‑step update rule. It maintains a buffer of the last n transitions.
    When the buffer is full, it computes an n‑step return G and updates the Q-value for the oldest
    state–action pair. When an episode terminates, any remaining transitions in the buffer are flushed.
    """
    def __init__(self, action_space, n_step, alpha=0.1, gamma=0.99, epsilon=0.1, seed=None):
        """
        Args:
            action_space: The environment's action space.
            n_step (int): The number of steps to look ahead for the update.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Exploration rate for epsilon-greedy policy.
            seed (int): Optional seed for reproducibility.
        """
        self.n_step = n_step
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize the Q-table as an empty dictionary.
        self.q_table = {}
        # Buffer to store transitions: each element is a tuple (state, action, reward).
        self.buffer = []

        # Variables to store the last state and action (used when receiving a new reward).
        self.last_state = None
        self.last_action = None

        # Create the Q-Learning policy using the shared Q-table.
        policy = QLearningPolicy(action_space, self.q_table, epsilon, seed)
        super().__init__(policy, seed)
    
    def act(self, observation):
        """
        Select an action while ensuring the current state is initialized in the Q-table.
        Stores the current state and chosen action for use in update().
        """
        state = get_state(observation)
        if state not in self.q_table:
            # Initialize Q-values for unseen states.
            self.q_table[state] = np.zeros(self.policy.action_space.n)
        action = self.policy.select_action(observation)
        self.last_state = state
        self.last_action = action
        return action
    
    def update(self, observation, reward, terminated, truncated):
        """
        Update the Q-table using the n‑step Q‑Learning update rule.
        
        The agent appends the transition (last_state, last_action, reward) to its buffer.
        If the buffer length reaches n_step, the agent computes the n‑step return G:
        
            G = sum_{i=0}^{n-1} gamma^i * r_{t+i+1}   [ + gamma^n * max_a Q(s_{t+n}, a) if not terminated ]
        
        and updates the Q-value for the oldest state–action pair. If the episode terminates,
        the remaining transitions in the buffer are flushed.
        """
        # Check that we have a valid transition.
        if self.last_state is None or self.last_action is None:
            raise ValueError("Last state or last action is None")
        
        # Append the most recent transition (from the previous action) to the buffer.
        self.buffer.append((self.last_state, self.last_action, reward))
        
        # Convert the new observation into a state.
        next_state = get_state(observation)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.policy.action_space.n)
        
        # If the buffer has at least n_step transitions, perform an update.
        if len(self.buffer) >= self.n_step:
            G = 0.0
            # Sum over the first n_step rewards.
            for i in range(self.n_step):
                G += (self.gamma ** i) * self.buffer[i][2]
            # If the episode is not terminated, bootstrap using the estimated value of next_state.
            if not terminated:
                G += (self.gamma ** self.n_step) * np.max(self.q_table[next_state])
            # Update the Q-value for the oldest transition.
            s, a, _ = self.buffer[0]
            self.q_table[s][a] += self.alpha * (G - self.q_table[s][a])
            # Remove the oldest transition from the buffer.
            self.buffer.pop(0)
        
        # If the episode terminated, flush any remaining transitions in the buffer.
        if terminated:
            while len(self.buffer) > 0:
                G = 0.0
                k = len(self.buffer)
                for i in range(k):
                    G += (self.gamma ** i) * self.buffer[i][2]
                s, a, _ = self.buffer[0]
                self.q_table[s][a] += self.alpha * (G - self.q_table[s][a])
                self.buffer.pop(0)
    
    def reset(self):
        """
        Reset the agent's learning state.
        """
        self.q_table = {}
        self.last_state = None
        self.last_action = None
        self.buffer = []
        self.policy.reset(self.q_table)