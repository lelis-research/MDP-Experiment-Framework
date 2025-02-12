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
    It maintains a buffer of the last n transitions.
    When the buffer is full, it computes an n‑step return G and updates the Q-value for the oldest
    state–action pair.
    When an episode terminates, any remaining transitions in the buffer are flushed.
    
    Init Args:
            action_space: The environment's action space (assumed gym.spaces.Discrete)
            hyper-parameters:
                - epsilon
                - gamma
                - step_size
                - n_steps
    """

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
    
    def reset(self, seed):
        super().reset(seed)
        self.buffer = []
        
    def update(self, last_observation, last_action, observation, reward, terminated, truncated):
        last_state = get_state(last_observation)

        # Append the most recent transition (from the previous action) to the buffer.
        self.buffer.append((last_state, last_action, reward))
        
        # Convert the new observation into a state.
        state = get_state(observation)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)
        
        # If the buffer has at least n_step transitions, perform an update.
        if len(self.buffer) >= self.hp.n_steps:
            bootstrap_val = np.max(self.q_table[state]) if not terminated else 0.0
            G = self._compute_n_step_return(n=self.hp.n_steps, 
                                            bootstrap_value=bootstrap_val, 
                                            bootstrap=not terminated)

            # Update the Q-value for the oldest transition.
            s, a, _ = self.buffer[0]
            self.q_table[s][a] += self.hp.step_size * (G - self.q_table[s][a])
            # Remove the oldest transition from the buffer.
            self.buffer.pop(0)
        
        # Now handle the end-of-episode scenario.
        # If the episode is terminated (truly ended), flush the buffer with no bootstrapping.
        if terminated:
            while len(self.buffer) > 0:
                G = self._compute_n_step_return(bootstrap_value=0.0, bootstrap=False)
                s, a, _ = self.buffer[0]
                self.q_table[s][a] += self.hp.step_size * (G - self.q_table[s][a])
                self.buffer.pop(0)
        
        # If the episode was truncated, flush the buffer but include bootstrapping.
        elif truncated:
            while len(self.buffer) > 0:
                bootstrap_val = np.max(self.q_table[state])
                G = self._compute_n_step_return(bootstrap_value=bootstrap_val, bootstrap=True)
                s, a, _ = self.buffer[0]
                self.q_table[s][a] += self.hp.step_size * (G - self.q_table[s][a])
                self.buffer.pop(0)

class NStepQLearningAgent(QLearningAgent):
    """
    n‑Step Tabular Q‑Learning agent.
    
    The agent uses an n‑step update rule. 
    """
    def __init__(self, action_space, hyper_params):
        """
        n-step Q-Learning agent that uses a buffer of transitions to perform
        multi-step updates.
        
        Hyper-parameters:
                - epsilon
                - gamma
                - step_size
                - n_steps
        """
        super().__init__(action_space, hyper_params)
        
        # Create the n-step Q-Learning policy.
        self.policy = NStepQLearningPolicy(action_space, hyper_params)