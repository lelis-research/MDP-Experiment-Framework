import numpy as np
import random
from Agents.Utils.BaseAgent import BaseAgent, BasePolicy
from Agents.Utils.FeatureExtractor import TabularFeature

class SarsaPolicy(BasePolicy):
    """
    An epsilon-greedy SARSA policy that updates Q(s,a)
    using Q(s', a') from the *action actually selected* a'.

    Init Args:
        action_space: The environment's action space (assumed gym.spaces.Discrete)
        hyper-parameters:
            - epsilon
            - gamma
            - step_size
    """

    def select_action(self, state):
        """
        Epsilon-greedy action selection based on Q-values.
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
        
        if terminated:
            target = reward
        else:
            next_action = self.select_action(state)            
            target = reward + self.hp.gamma * self.q_table[state][next_action]

        # Update Q-value
        self.q_table[last_state][last_action] += self.hp.step_size * \
            (target - self.q_table[last_state][last_action])

    def reset(self, seed):
        """
        Resets the policy for a fresh run, clearing the Q-table and seeds.
        """
        super().reset(seed)
        self.q_table = {}

class SarsaAgent(BaseAgent):
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
        self.policy = SarsaPolicy(action_space, hyper_params)

    def act(self, observation):
        state = self.feature_extractor(observation)
        action = self.policy.select_action(state)

        self.last_state = state
        self.last_action = action
        return action
    
    def update(self, observation, reward, terminated, truncated):
        state = self.feature_extractor(observation)
        self.policy.update(self.last_state, self.last_action, state, reward, terminated, truncated)
