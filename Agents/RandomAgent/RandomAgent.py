from Agents.Utils.BaseAgent import BaseAgent, BasePolicy

class RandomPolicy(BasePolicy):
    """A policy that selects actions randomly."""
    
    def select_action(self, observation):
        """Select a random action."""
        return self.action_space.sample()
    
class RandomAgent(BaseAgent):
    """An agent that follows a RandomPolicy."""
    
    def __init__(self, action_space, hyper_params=None, seed=None):
        super().__init__(action_space, hyper_params, seed)
        self.policy = RandomPolicy(action_space, seed)
        