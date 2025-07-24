from gymnasium.spaces import Discrete


from ..Utils import BaseAgent, BasePolicy
from ...registry import register_agent, register_policy
from .Random import RandomPolicy

@register_policy
class OptionRandomPolicy(RandomPolicy):
    """Policy that selects actions randomly."""
    pass

@register_agent
class OptionRandomAgent(BaseAgent):
    """Agent that uses RandomPolicy."""
    
    name = "OptionRandom"
    
    def __init__(self, action_space, observation_space, hyper_params, num_envs, options_lst):
        super().__init__(action_space, observation_space, hyper_params, num_envs)
        self.atomic_action_space = action_space
        self.options_lst = options_lst
        
        self.action_space = Discrete(self.atomic_action_space.n + len(self.options_lst)) 
        
        self.policy = OptionRandomPolicy(self.action_space)
        self.running_option_index = None
        
    def act(self, observation, greedy=False):
        # If an options is running currently
        if self.running_option_index is not None:
            # Check if the option should terminate.
            if self.options_lst[self.running_option_index].is_terminated(observation):
                self.running_option_index = None
            else:
                # Continue executing the currently running option.
                action = self.options_lst[self.running_option_index].select_action(observation)
        
        if self.running_option_index is None:
            # No option is running; select a new action (or option) from the policy.
            action = self.policy.select_action(observation)
            if action >= self.atomic_action_space.n:
                # An option was selected. Record its initiation state and initialize accumulators.
                self.running_option_index = action - self.atomic_action_space.n
                action = self.options_lst[self.running_option_index].select_action(observation)
        
        return action
