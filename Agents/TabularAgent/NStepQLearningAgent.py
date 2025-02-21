import numpy as np
import random
from Agents.Utils.BaseAgent import BaseAgent, BasePolicy
from Agents.TabularAgent.QLearningAgent import QLearningAgent, QLearningPolicy
from Agents.Utils.Buffer import BasicBuffer
from Agents.Utils.HelperFunction import *

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
    
    def reset(self, seed):
        super().reset(seed)
        self.rollout_buffer = BasicBuffer(np.inf)
        
    def update(self, last_state, last_action, state, reward, terminated, truncated):

        # Append the most recent transition (from the previous action) to the buffer.
        transition = (last_state, last_action, reward)
        self.rollout_buffer.add_single_item(transition)
        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_dim)
        
        # If the buffer has at least n_step transitions, perform an update.
        if self.rollout_buffer.size >= self.hp.n_steps:
            rollout = self.rollout_buffer.get_all()
            rollout_states, rollout_actions, rollout_rewards = zip(*rollout)
            
            bootstrap_value = np.max(self.q_table[state]) if not terminated else 0.0

            # This will give the return from each state in the rollout, we only want the 1st hence index 0
            n_step_return = calculate_n_step_returns(rollout_rewards, bootstrap_value, self.hp.gamma)[0]

            # Update the Q-value for the oldest transition.
            s, a, _ = self.rollout_buffer.remove_oldest()
            self.q_table[s][a] += self.hp.step_size * (n_step_return - self.q_table[s][a])            
        
        # Now handle the end-of-episode scenario.
        if terminated or truncated:
            while self.rollout_buffer.size > 0:
                rollout = self.rollout_buffer.get_all()
                rollout_states, rollout_actions, rollout_rewards = zip(*rollout)

                bootstrap_value = np.max(self.q_table[state]) if not terminated else 0.0
                n_step_return = calculate_n_step_returns(rollout_rewards, bootstrap_value, self.hp.gamma)[0]
                
                s, a, _ = self.rollout_buffer.remove_oldest()
                self.q_table[s][a] += self.hp.step_size * (n_step_return - self.q_table[s][a])
                

class NStepQLearningAgent(QLearningAgent):
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class):
        """
        n-step Q-Learning agent that uses a buffer of transitions to perform
        multi-step updates.
        
        Hyper-parameters:
                - epsilon
                - gamma
                - step_size
                - n_steps
        """
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class)        
        self.policy = NStepQLearningPolicy(action_space, hyper_params)