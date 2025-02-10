import numpy as np
import random
from Agents.Utils.BaseAgent import BaseAgent, BasePolicy

def get_state(observation):
    """
    Helper function to convert an observation (assumed to be a NumPy array)
    into a hashable (discrete) state representation.
    """
    # Flatten the observation and convert to tuple.
    # You can customize this if you need a different discretization.
    return tuple(observation.flatten().tolist())

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

    def select_action(self, observation):
        """
        Epsilon-greedy action selection based on Q-values.
        """
        state = get_state(observation)
        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)

        # With probability epsilon choose a random action...
        if random.random() < self.hp.epsilon:
            return self.action_space.sample()
        else:
            # ...otherwise, choose the action with the highest Q-value.
            return int(np.argmax(self.q_table[state]))

    def update(self, last_observation, last_action, observation, reward, terminated, truncated):
        # Convert the new observation to a discrete state.
        state = get_state(observation)
        if state not in self.q_table:
                self.q_table[state] = np.zeros(self.action_space.n)
        
        # If episode ended, the target is just the reward
        if terminated:
            target = reward
        else:
            # Next action is the one the policy *would* pick in the new state
            next_action = self.select_action(observation)            
            target = reward + self.hp.gamma * self.q_table[state][next_action]

        last_state = get_state(last_observation)
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
    """
    A Tabular SARSA agent. Uses the SarsaPolicy for selecting and updating Q-values.
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

        # Store the last state and action so we can do the SARSA update
        self.last_observation = None
        self.last_action = None

        # Create the SARSA policy
        self.policy = SarsaPolicy(action_space, hyper_params)

    def act(self, observation):
        """
        Select an action via the SARSA policy's epsilon-greedy strategy.
        """
        action = self.policy.select_action(observation)

        self.last_observation = observation
        self.last_action = action
        return action

    def update(self, observation, reward, terminated, truncated):
        """
        Update the policy.
        
        Args:
            observation: The new observation after the action.
            reward: The reward received after taking the action.
            terminated: Whether the episode terminated.
            truncated: Whether the episode was truncated.
        """
        self.policy.update(self.last_observation, self.last_action, observation, reward, terminated, truncated)

        if terminated or truncated:
            self.last_observation = None
            self.last_action = None

    def reset(self, seed):
        """
        Reset the agent and policy for a new experimental run.
        """
        super().reset(seed)
        self.last_observation = None
        self.last_action = None