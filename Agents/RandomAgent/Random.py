from Agents.Utils import BaseAgent, BasePolicy

class RandomPolicy(BasePolicy):
    """Policy that selects actions randomly."""
    
    def select_action(self, observation):
        # Return a random action.
        return self.action_space.sample()
    
    def select_parallel_actions(self, observations):
        # Return a random action (for parallel environments).
        return self.action_space.sample()
    
    def save(self, file_path):
        pass  # No saving mechanism implemented.

class RandomAgent(BaseAgent):
    """Agent that uses RandomPolicy."""
    
    def __init__(self, action_space, observation_space, hyper_params, num_envs):
        super().__init__(action_space, observation_space, hyper_params, num_envs)
        self.policy = RandomPolicy(action_space)