import numpy as np
import random
import torch
from gymnasium.spaces import Discrete

from ..Utils import BaseAgent, BasePolicy
from .QLearning import QLearningAgent, QLearningPolicy
from ...registry import register_agent, register_policy
from ...Options.Utils import load_options_list, save_options_list

@register_policy
class OptionQLearningPolicy(QLearningPolicy):
    pass

@register_agent
class OptionQLearningAgent(QLearningAgent):
    name = "OptionQLearning"
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, options_lst):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class)

        # Keep atomic action space; extend for options.
        self.atomic_action_space = action_space
        self.options_lst = options_lst
        print(f"Number of options: {len(options_lst)}")
        
        # Replace action_space with extended (primitive + options)
        self.action_space = Discrete(self.atomic_action_space.n + len(self.options_lst))

        # Swap policy with OptionQLearningPolicy (shares same interface)
        self.policy = OptionQLearningPolicy(self.action_space, hyper_params)

        # Option execution bookkeeping
        self.running_option_index = None       # index into options_lst (or None)
        self.option_start_state = None         # encoded state where option began
        self.option_cumulative_reward = 0.0    # discounted return accumulator R_{t:t+k}
        self.option_multiplier = 1.0           # current gamma^t during option

        # Parent keeps last_state/last_action for primitive updates
        self.last_state = None
        self.last_action = None
    
    def act(self, observation, greedy=False):
        """
        Returns an atomic (primitive) action to execute.
        If an option is running, returns its current primitive action.
        Otherwise, samples from the extended action space (primitives + options).
        """
        state = self.feature_extractor(observation)

        # If an option is currently running, either continue it or end it here.
        if self.running_option_index is not None:
            if self.options_lst[self.running_option_index].is_terminated(observation):
                # Option ends; choose anew below
                self.running_option_index = None
            else:
                action = self.options_lst[self.running_option_index].select_action(observation)

        if self.running_option_index is None:
            # Choose an extended action (might be a primitive or an option)
            action = self.policy.select_action(state, greedy=greedy)

            if action >= self.atomic_action_space.n:
                # Start an option
                self.running_option_index = action - self.atomic_action_space.n
                self.option_start_state = state
                self.option_cumulative_reward = 0.0
                self.option_multiplier = 1.0
                action = self.options_lst[self.running_option_index].select_action(observation)


        self.last_state = state
        self.last_action = action
        return action
    
    def update(self, observation, reward, terminated, truncated, call_back=None):
        """
        After env.step, update Q-values.
        - If inside an option, accumulate discounted reward and do a single SMDP update on option termination.
        - If primitive, do the usual 1-step Q-learning update (parent policy already supports it).
        """
        state = self.feature_extractor(observation)

        if self.running_option_index is not None:
            # Accumulate SMDP return while option runs
            self.option_cumulative_reward += self.option_multiplier * float(reward)
            self.option_multiplier *= self.hp.gamma

            if terminated or truncated or self.options_lst[self.running_option_index].is_terminated(observation):
                self.policy.update(
                    last_state=self.option_start_state,
                    last_action=self.atomic_action_space.n + self.running_option_index,
                    state=state,
                    reward=self.option_cumulative_reward,
                    terminated=terminated,
                    truncated=truncated,
                    call_back=call_back
                )
                # Clear option state
                self.running_option_index = None
                self.option_start_state = None
                self.option_cumulative_reward = 0.0
                self.option_multiplier = 1.0

        else:
            self.policy.update(
                last_state=self.last_state,
                last_action=self.last_action,
                state=state,
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                call_back=call_back
            )
    
    def reset(self, seed):
        super().reset(seed)

        self.running_option_index = None
        self.option_start_state = None
        self.option_cumulative_reward = 0.0
        self.option_multiplier = 1.0

        self.last_state = None
        self.last_action = None
        
    
    def save(self, file_path=None):
        """
        Save extended agent (including options list).
        """
        checkpoint = super().save(file_path=None)  # parent saves feature_extractor, policy, hp, etc.


        # Save options list payload
        checkpoint['options_lst'] = save_options_list(self.options_lst, file_path=None)
        checkpoint['atomic_action_space'] = self.atomic_action_space

        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_agent.t")
        return checkpoint

    @classmethod
    def load_from_file(cls, file_path, seed=0, checkpoint=None):
        """
        Load OptionQLearningAgent including options list.
        """
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)

        options_lst = load_options_list(file_path=None, checkpoint=checkpoint['options_lst'])

        instance = cls(
            action_space=checkpoint['atomic_action_space'],
            observation_space=checkpoint['observation_space'],
            hyper_params=checkpoint['hyper_params'],
            num_envs=checkpoint['num_envs'],
            feature_extractor_class=checkpoint['feature_extractor_class'],
            options_lst=options_lst
        )
        instance.reset(seed)
        instance.feature_extractor.load_from_checkpoint(checkpoint['feature_extractor'])
        instance.policy.load_from_checkpoint(checkpoint['policy'])
        return instance