import numpy as np
import random
from Agents.Utils.BaseAgent import BaseAgent, BasePolicy
from Agents.TabularAgent.QLearningAgent import QLearningAgent, QLearningPolicy

def get_state(observation):
    """
    Helper function to convert an observation (assumed to be a NumPy array)
    into a hashable (discrete) state representation.
    """
    # Flatten the observation and convert to tuple.
    # You can customize this function if you need a different discretization.
    return tuple(observation.flatten().tolist())

class NStepQLearningPolicy(QLearningPolicy):
    """
    Epsilon-greedy policy that selects actions based on the Q-table (QLearning). 
    """
    pass

class NStepQLearningAgent(QLearningAgent):
    """
    n‑Step Tabular Q‑Learning agent.
    
    The agent uses an n‑step update rule. It maintains a buffer of the last n transitions.
    When the buffer is full, it computes an n‑step return G and updates the Q-value for the oldest
    state–action pair. When an episode terminates, any remaining transitions in the buffer are flushed.
    """
    def __init__(self, action_space, hyper_params, seed=None):
        """
        n-step Q-Learning agent that uses a buffer of transitions to perform
        multi-step updates.
        
        Hyper-parameters expected in `hyper_params`:
        - alpha: Learning rate.
        - gamma: Discount factor.
        - epsilon: Exploration rate.
        - n_steps: Number of steps to look ahead in the update.
        """
        super().__init__(action_space, hyper_params, seed)
        
        # Buffer to store transitions for n step: each entry is (state, action, reward)
        self.buffer = []

        # Create the n-step Q-Learning policy.
        self.policy = NStepQLearningPolicy(action_space, hyper_params.epsilon, seed)
   
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
        state = get_state(observation)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.policy.action_space.n)
        
        # If the buffer has at least n_step transitions, perform an update.
        if len(self.buffer) >= self.hp.n_steps:
            bootstrap_val = np.max(self.q_table[state]) if not terminated else 0.0
            G = self._compute_n_step_return(n=self.hp.n_steps, 
                                            bootstrap_value=bootstrap_val, 
                                            bootstrap=not terminated)

            # Update the Q-value for the oldest transition.
            s, a, _ = self.buffer[0]
            self.q_table[s][a] += self.hp.alpha * (G - self.q_table[s][a])
            # Remove the oldest transition from the buffer.
            self.buffer.pop(0)
        
        # Now handle the end-of-episode scenario.
        # If the episode is terminated (truly ended), flush the buffer with no bootstrapping.
        if terminated:
            while len(self.buffer) > 0:
                G = self._compute_n_step_return(bootstrap_value=0.0, bootstrap=False)
                s, a, _ = self.buffer[0]
                self.q_table[s][a] += self.hp.alpha * (G - self.q_table[s][a])
                self.buffer.pop(0)

            self.last_state = None
            self.last_action = None
        
        # If the episode was truncated, flush the buffer but include bootstrapping.
        elif truncated:
            while len(self.buffer) > 0:
                bootstrap_val = np.max(self.q_table[state])
                G = self._compute_n_step_return(bootstrap_value=bootstrap_val, bootstrap=True)
                s, a, _ = self.buffer[0]
                self.q_table[s][a] += self.hp.alpha * (G - self.q_table[s][a])
                self.buffer.pop(0)
            
            self.last_state = None
            self.last_action = None
    
    def _compute_n_step_return(self, n=None, bootstrap_value=0.0, bootstrap=True):
        """
        Compute the n-step return
        
        Args:
            n (int, optional): Number of transitions from the buffer to use.
                            If None, use the entire buffer.
            bootstrap_value (float): The value to bootstrap from if applicable.
            bootstrap (bool): Whether to include bootstrapping.
        
        Returns:
            G (float): The computed n-step return.
        """
        if n is None:
            n = len(self.buffer)
        G = 0.0
        for i in range(n):
            G += (self.hp.gamma ** i) * self.buffer[i][2] # index 2 is for reward
        if bootstrap:
            G += (self.hp.gamma ** n) * bootstrap_value
        return G
    
    def reset(self, seed=None):
        """
        Reset the agent's learning state.
        """
        super().reset(seed)
        self.buffer = []