from Agents.Utils import BaseAgent, BasePolicy

class RandomPolicy(BasePolicy):
    """A policy that selects actions randomly."""
    
    def select_action(self, observation):
        """Select a random action."""
        return self.action_space.sample()
    
    def select_parallel_actions(self, observations):
        """Select a list random action."""
        return self.action_space.sample()
    
    def save(self, file_path):
        pass
    
class RandomAgent(BaseAgent):
    """An agent that follows a RandomPolicy."""
    
    def __init__(self, action_space, observation_space, hyper_params, num_envs):
        super().__init__(action_space, observation_space, hyper_params, num_envs)
        self.policy = RandomPolicy(action_space)
        
        