from Agents.Utils.BaseAgent import BaseAgent
from Agents.Utils.BasePolicy import BasePolicy

class RandomPolicy(BasePolicy):
    """A policy that selects actions randomly."""
    
    def __init__(self, action_space, seed=None):
        super().__init__(seed)
        self.action_space = action_space  # Store the environment's action space
        if seed is not None:
            self.action_space.seed(seed)
    
    def select_action(self, observation):
        """Select a random action."""
        return self.action_space.sample()
    
class RandomAgent(BaseAgent):
    """An agent that follows a RandomPolicy."""
    
    def __init__(self, action_space, seed):
        policy = RandomPolicy(action_space, seed)
        super().__init__(policy)