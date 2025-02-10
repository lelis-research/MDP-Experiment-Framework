import numpy as np
import random
from Agents.Utils.BaseAgent import BaseAgent, BasePolicy

def get_state(observation):
    """
    Helper function to convert an observation (assumed to be a NumPy array)
    into a hashable (discrete) state representation.
    """
    return tuple(observation.flatten().tolist())


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

    def select_action(self, observation):
        """
        Epsilon-greedy action selection using Q = Q1 + Q2.
        """
        state = get_state(observation)

        if state not in self.q1_table:
            self.q1_table[state] = np.zeros(self.action_space.n)
        if state not in self.q2_table:
            self.q2_table[state] = np.zeros(self.action_space.n)

        # With probability epsilon choose a random action...
        if random.random() < self.hp.epsilon:
            return self.action_space.sample()
        else:
            # Otherwise choose action that maximizes Q1 + Q2
            q_sum = self.q1_table[state] + self.q2_table[state]
            return int(np.argmax(q_sum))

    def update(self, last_observation, last_action, observation, reward, terminated, truncated):
        """
        Double Q-learning update. We randomly pick which Q-table to update.
        """
        # Convert the new observation to a discrete state.
        state = get_state(observation)
        if state not in self.q1_table:
            self.q1_table[state] = np.zeros(self.action_space.n)
        if state not in self.q2_table:
            self.q2_table[state] = np.zeros(self.action_space.n)

        last_state = get_state(last_observation)


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


class DoubleQLearningAgent(BaseAgent):
    """
    A Tabular Double Q-Learning agent that uses two Q-tables to reduce overestimation bias.
    """
    def __init__(self, action_space, hyper_params):
        """
        Args:
            action_space: The environment's action space (assumed gym.spaces.Discrete)
            hyper-parameters:
                - epsilon
                - gamma
                - step_size        
        """
        super().__init__(action_space, hyper_params)

        self.last_observation = None
        self.last_action = None

        # Create the policy that implements Double Q-learning logic
        self.policy = DoubleQLearningPolicy(action_space, hyper_params)

    def act(self, observation):
        """
        Select an action via epsilon-greedy w.r.t. Q1+Q2.
        """
        action = self.policy.select_action(observation)
        self.last_observation = observation
        self.last_action = action
        return action

    def update(self, observation, reward, terminated, truncated):
        """
        Perform the Double Q-Learning update.
        """
        self.policy.update(self.last_observation, self.last_action, observation, reward, terminated, truncated)

        if terminated or truncated:
            self.last_observation = None
            self.last_action = None

    def reset(self, seed):
        super().reset(seed)
        self.last_observation = None
        self.last_action = None