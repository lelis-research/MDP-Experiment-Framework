import numpy as np
import random
from Agents.Utils.BaseAgent import BaseAgent, BasePolicy

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

        Init Args:
            action_space: The environment's action space (assumed gym.spaces.Discrete)
            hyper-parameters:
                - epsilon
                - gamma
                - step_size
    """       
 
    def select_action(self, observation):
        """
        Select an action using epsilon-greedy exploration.
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
    
    def select_parallel_actions(self, observations):
        actions = []
        for observation in observations:
            state = get_state(observation)
            if state not in self.q_table:
                self.q_table[state] = np.zeros(self.action_space.n)

            # With probability epsilon choose a random action...
            if random.random() < self.hp.epsilon:
                actions.append(self.action_space.sample())
            else:
                # ...otherwise, choose the action with the highest Q-value.
                actions.append(int(np.argmax(self.q_table[state])))
        return np.asarray(actions)
            
    def update(self, last_observation, last_action, observation, reward, terminated, truncated):
        # Convert the new observation to a discrete state.
        state = get_state(observation)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)

        # Compute the target.
        if terminated:
            target = reward
        else:
            target = reward + self.hp.gamma * np.max(self.q_table[state])

        last_state = get_state(last_observation)
        
        # Perform the Q-Learning update.
        self.q_table[last_state][last_action] += self.hp.step_size * \
            (target - self.q_table[last_state][last_action])

    def reset(self, seed):
        super().reset(seed)
        self.q_table = {}

class QLearningAgent(BaseAgent):
    """
    Tabular Q-Learning agent.
    
    It uses a TabularQLearningPolicy for action selection and implements
    
    ASSUMPTION: either of the act methods will be called before 
                an update call so no need to reset the 
                last_action and last_observation values
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
        
        # Create the Q-Learning policy
        self.policy = QLearningPolicy(action_space, hyper_params)
        
    
    def act(self, observation):
        action = self.policy.select_action(observation)
        self.last_action = action
        self.last_observation = observation
        return action
    
    def parallel_act(self, observations):
        actions = self.policy.select_parallel_actions(observations)
        self.last_actions = actions
        self.last_observations = observations
        return actions
    
    def update(self, observation, reward, terminated, truncated):
        self.policy.update(self.last_observation, self.last_action, observation, reward, terminated, truncated)
    
    def parallel_update(self, observations, rewards, terminateds, truncateds):
        num_envs = len(rewards)  # or observations.shape[0]
        for i in range(num_envs):
            # Do the same tabular Q-Learning update as the single-env version
            self.policy.update(
                last_observation=self.last_observations[i],
                last_action=self.last_actions[i],
                observation=observations[i],
                reward=rewards[i],
                terminated=terminateds[i],
                truncated=truncateds[i]  
            )
    
    