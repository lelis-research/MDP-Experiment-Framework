import numpy as np
import random
import torch
from gymnasium.spaces import Discrete

from ..Utils import calculate_n_step_returns_with_discounts, get_single_observation
from .QLearning import QLearningAgent, QLearningPolicy
from ...registry import register_agent, register_policy
from ...Options import load_options_list, save_options_list

@register_policy
class OptionQLearningPolicy(QLearningPolicy):
    pass

@register_agent
class OptionQLearningAgent(QLearningAgent):
    name = "OptionQLearning"
    SUPPORTED_ACTION_SPACES = (Discrete, )
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, init_option_lst=[]):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class)

        # Keep atomic action space; extend for options.
        self.atomic_action_space = action_space
        self.options_lst = init_option_lst
        
        # Replace action_space with extended (primitive + options)
        action_option_space = Discrete(self.atomic_action_space.n + len(self.options_lst))

        # Swap policy with OptionQLearningPolicy (shares same interface)
        self.policy = OptionQLearningPolicy(action_option_space, hyper_params)

        # Option execution bookkeeping
        self.running_option_index = [None for _ in range(self.num_envs)]       # index into options_lst (or None)
        self.option_start_state = [None for _ in range(self.num_envs)]         # encoded state where option began
        self.option_cumulative_reward = [0.0 for _ in range(self.num_envs)]    # discounted return accumulator R_{t:t+k}
        self.option_multiplier = [1.0 for _ in range(self.num_envs)]           # current gamma^t during option
        self.option_num_steps = [0 for _ in range(self.num_envs)]

    def act(self, observation, greedy=False):
        """
        Returns an atomic (primitive) action to execute.
        If an option is running, returns its current primitive action.
        Otherwise, samples from the extended action space (primitives + options).
        """
        state = self.feature_extractor(observation)
        action = []
        for i in range(self.num_envs):
            # If an option is currently running, either continue it or end it here.
            st = state[i]
            obs = get_single_observation(observation, i)
            curr_option_idx = self.running_option_index[i]
            if curr_option_idx is not None:
                if self.options_lst[curr_option_idx].is_terminated(obs):
                    # Option ends; choose anew below
                    curr_option_idx = None
                else:
                    a = self.options_lst[curr_option_idx].select_action(obs)

            if curr_option_idx is None:
                # Choose an extended action (might be a primitive or an option)
                a = self.policy.select_action(st, greedy=greedy)

                if a >= self.atomic_action_space.n:
                    # Start an option
                    curr_option_idx = a - self.atomic_action_space.n
                    self.option_start_state[i] = st
                    self.option_cumulative_reward[i] = 0.0
                    self.option_multiplier[i] = 1.0
                    self.option_num_steps[i] = 0
                    a = self.options_lst[curr_option_idx].select_action(obs)
            
            self.running_option_index[i] = curr_option_idx
            action.append(a)
            
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
        for i in range(self.num_envs):
            if call_back is not None:
                call_back({
                    f"train/option_usage_env_{i}": 1 if self.running_option_index[i] is not None else 0,
                })
            
            st = state[i]
            obs = get_single_observation(observation, i)
            curr_option_idx = self.running_option_index[i]
            
            if curr_option_idx is not None:
                # if an option is running
                # Accumulate SMDP return while option runs 
                self.option_cumulative_reward[i] += self.option_multiplier[i] * float(reward[i])
                self.option_multiplier[i] *= self.hp.gamma
                self.option_num_steps[i] += 1

                if self.options_lst[curr_option_idx].is_terminated(obs) or terminated[i] or truncated[i]:
                    transition = self.option_start_state[i], \
                                self.atomic_action_space.n + curr_option_idx, \
                                self.option_cumulative_reward[i], \
                                self.option_multiplier[i], \
                                self.option_num_steps[i]
                    self.rollout_buffer[i].add(transition)
                    self.running_option_index[i] = None

            else:
                transition = self.last_state[i], \
                            self.last_action[i], \
                            reward[i], \
                            self.hp.gamma, \
                            1
                self.rollout_buffer[i].add(transition)
                
            if terminated[i] or truncated[i]:
                rollout = self.rollout_buffer[i].all()
                rollout_states, rollout_actions, rollout_rewards, rollout_discounts, rollout_num_steps = zip(*rollout)
                n_step_return = calculate_n_step_returns_with_discounts(rollout_rewards, 0, rollout_discounts)
                all_steps = sum(s for s in rollout_num_steps)
                for j in range(len(self.rollout_buffer[i])):
                    self.policy.update(rollout_states[j], rollout_actions[j], st, n_step_return[j], terminated[i], 
                                       truncated[i], self.hp.gamma**(all_steps-rollout_num_steps[j]), call_back=call_back)
                self.rollout_buffer[i].clear() 
                
            elif len(self.rollout_buffer[i]) >= self.hp.n_steps:
                rollout = self.rollout_buffer[i].all() 
                rollout_states, rollout_actions, rollout_rewards, rollout_discounts, rollout_num_steps = zip(*rollout)
                n_step_return = calculate_n_step_returns_with_discounts(rollout_rewards, 0, rollout_discounts)
                all_steps = sum(s for s in rollout_num_steps)
                self.policy.update(rollout_states[0], rollout_actions[0], st, n_step_return[0], terminated[i], 
                                   truncated[i], self.hp.gamma**all_steps, call_back=call_back) 
                    
    def reset(self, seed):
        super().reset(seed)

        # Option execution bookkeeping
        self.running_option_index = [None for _ in range(self.num_envs)]       # index into options_lst (or None)
        self.option_start_state = [None for _ in range(self.num_envs)]         # encoded state where option began
        self.option_cumulative_reward = [0.0 for _ in range(self.num_envs)]    # discounted return accumulator R_{t:t+k}
        self.option_multiplier = [1.0 for _ in range(self.num_envs)]           # current gamma^t during option
        self.option_num_steps = [0 for _ in range(self.num_envs)]
        
    
    def log(self):
        pass
        # if self.running_option_index is None:
        #     return {"OptionUsageLog": False, "OptionIndex": self.running_option_index}
        # else:
        #     return {"OptionUsageLog": True, "OptionIndex": self.running_option_index}
        
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
    def load(cls, file_path, checkpoint=None):
        """
        Load OptionQLearningAgent including options list.
        """
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)

        instance = super().load(file_path, checkpoint)
        instance.options_lst = load_options_list(file_path=None, checkpoint=checkpoint['options_lst'])
        instance.atomic_action_space = checkpoint['atomic_action_space']
        
        action_option_space = Discrete(instance.atomic_action_space.n + len(instance.options_lst))
        instance.policy = OptionQLearningPolicy(action_option_space, instance.hp)
        return instance