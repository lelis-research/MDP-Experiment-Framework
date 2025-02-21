import numpy as np
import random
import pickle

from Agents.Utils.BaseAgent import BaseAgent, BasePolicy


class DoubleQLearningPolicy(BasePolicy):
    """
    Double Q-Learning with two Q-tables: Q1 and Q2.
    Action selection is epsilon-greedy w.r.t. Q = (Q1 + Q2).
    Updates randomly pick one of the two Qs to update.
    
    Update rule:
      - With 50% chance, update Q1(s, a):
            Q1(s,a) ← Q1(s,a) + α * [r + γ * Q2(s', argmax_a' Q1(s',a')) - Q1(s,a)]
      - Otherwise, update Q2(s, a):
            Q2(s,a) ← Q2(s,a) + α * [r + γ * Q1(s', argmax_a' Q2(s',a')) - Q2(s,a)]

    Init Args:
        action_space: The environment's action space (assumed gym.spaces.Discrete)
        hyper-parameters:
            - epsilon
            - gamma
            - step_size
    """

    def select_action(self, state):
        """
        Epsilon-greedy action selection using Q = Q1 + Q2.
        """
        if state not in self.q1_table:
            self.q1_table[state] = np.zeros(self.action_dim)
        if state not in self.q2_table:
            self.q2_table[state] = np.zeros(self.action_dim)

        # With probability epsilon choose a random action...
        if random.random() < self.hp.epsilon:
            return self.action_space.sample()
        else:
            # Otherwise choose action that maximizes Q1 + Q2
            q_sum = self.q1_table[state] + self.q2_table[state]
            return int(np.argmax(q_sum))


    def update(self, last_state, last_action, state, reward, terminated, truncated):
        """
        Double Q-learning update. We randomly pick which Q-table to update.
        """
        if state not in self.q1_table:
            self.q1_table[state] = np.zeros(self.action_dim)
        if state not in self.q2_table:
            self.q2_table[state] = np.zeros(self.action_dim)


        # With probability 0.5, update Q1 using Q2
        # or update Q2 using Q1
        if random.random() < 0.5:
            # Q1 update
            if terminated:
                target = reward
            else:
                target = reward + self.hp.gamma * np.max(self.q2_table[state])
            self.q1_table[last_state][last_action] += self.hp.step_size * \
                (target - self.q1_table[last_state][last_action])
        else:
            # Q2 update
            if terminated:
                target = reward
            else:
                target = reward + self.hp.gamma * np.max(self.q1_table[state])
            self.q2_table[last_state][last_action] += self.hp.step_size * \
                (target - self.q2_table[last_state][last_action])


    def reset(self, seed):
        super().reset(seed)
        # Clear Q-tables
        self.q1_table = {}
        self.q2_table = {}

    def save(self, file_path):
        checkpoint = {
            'q1_table': self.q1_table,
            'q2_table': self.q2_table,
            'hyper_params': self.hp,
        }
        with open(file_path, 'wb') as f:
            pickle.dump(checkpoint, f)
            
    def load(self, file_path):
        with open(file_path, 'rb') as f:
            checkpoint = pickle.load(f)
        self.q1_table = checkpoint.get('q1_table', {})
        self.q2_table = checkpoint.get('q2_table', {})
        self.hp = checkpoint.get('hyper_params', self.hp)


class DoubleQLearningAgent(BaseAgent):
    """
    A Tabular Double Q-Learning agent that uses two Q-tables to reduce overestimation bias.
    """
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class):
        """
        Args:
            action_space: The environment's action space (assumed gym.spaces.Discrete)
            hyper-parameters:
                - epsilon
                - gamma
                - step_size        
        """
        super().__init__(action_space, observation_space, hyper_params, num_envs)
        self.feature_extractor = feature_extractor_class(observation_space)
        self.policy = DoubleQLearningPolicy(action_space, hyper_params)

    def act(self, observation):
        """
        Select an action via epsilon-greedy w.r.t. Q1+Q2.
        """
        state = self.feature_extractor(observation)
        action = self.policy.select_action(state)
        self.last_state = state
        self.last_action = action
        return action

    def update(self, observation, reward, terminated, truncated):
        """
        Perform the Double Q-Learning update.
        """
        state = self.feature_extractor(observation)
        self.policy.update(self.last_state, self.last_action, state, reward, terminated, truncated)
