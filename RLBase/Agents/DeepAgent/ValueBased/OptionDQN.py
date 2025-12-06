import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import Discrete

from .DQN import DQNAgent, DQNPolicy
from ...Utils import (
    calculate_n_step_returns_with_discounts, 
    get_single_observation, 
    stack_observations,
    get_single_state,
    stack_states
)
from ....registry import register_agent, register_policy
from ....Options import load_options_list, save_options_list

@register_policy
class OptionDQNPolicy(DQNPolicy):
    pass

@register_agent
class OptionDQNAgent(DQNAgent):
    name = "OptionDQN"
    SUPPORTED_ACTION_SPACES = (Discrete, )
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, init_option_lst=None, device="cpu"):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device=device)
        
        # Keep atomic action space; extend for options.
        self.atomic_action_space = action_space
        self.options_lst = [] if init_option_lst is None else init_option_lst
        
        # Replace action_space with extended (primitive + options)
        action_option_space = Discrete(self.atomic_action_space.n + len(self.options_lst))


        # Create DQNPolicy using the feature extractor's feature dimension.
        self.policy = OptionDQNPolicy(
            action_option_space, 
            self.feature_extractor.features_dict,
            hyper_params,
            device=device
        )
        
        # Option execution bookkeeping
        self.running_option_index = [None for _ in range(self.num_envs)]       # index into options_lst (or None)
        self.option_start_obs = [None for _ in range(self.num_envs)]         # encoded state where option began
        self.option_cumulative_reward = [0.0 for _ in range(self.num_envs)]    # discounted return accumulator R_{t:t+k}
        self.option_multiplier = [1.0 for _ in range(self.num_envs)]           # current gamma^t during option
        self.option_num_steps = [0 for _ in range(self.num_envs)]
        
          
    def act(self, observation, greedy=False):
        state = self.feature_extractor(observation) 
        # action = self.policy.select_action(state, greedy=greedy)
        
        action = []
        for i in range(self.num_envs):
            # If an option is currently running, either continue it or end it here.
            st = get_single_state(state, i)
            obs = get_single_observation(observation, i)
            curr_option_idx = self.running_option_index[i]
            if curr_option_idx is not None:
                a = self.options_lst[curr_option_idx].select_action(obs)
            else:
                # Choose an extended action (might be a primitive or an option)
                a = self.policy.select_action(st, greedy=greedy)[0] # because 'a' would be a batch of size 1
                if a >= self.atomic_action_space.n:
                    # Start an option
                    curr_option_idx = a - self.atomic_action_space.n
                    self.option_start_obs[i] = obs
                    self.option_cumulative_reward[i] = 0.0
                    self.option_multiplier[i] = 1.0
                    self.option_num_steps[i] = 0
                    a = self.options_lst[curr_option_idx].select_action(obs)
            
            self.running_option_index[i] = curr_option_idx
            action.append(a) 
            
        self.last_observation = observation
        self.last_action = action
        return action
    
    def update(self, observation, reward, terminated, truncated, call_back=None):
        """
        all arguments are batches
        """

        # reward = np.clip(reward, -1, 1) # for stability
        for i in range(self.num_envs):
            if call_back is not None:
                call_back({
                    f"train/option_usage_env_{i}": 1 if self.running_option_index[i] is not None else 0,
                })
                
            obs = get_single_observation(observation, i)
            curr_option_idx = self.running_option_index[i]
            
            if curr_option_idx is not None:
                # if an option is running
                # Accumulate SMDP return while option runs 
                self.option_cumulative_reward[i] += self.option_multiplier[i] * float(reward[i])
                self.option_multiplier[i] *= self.hp.gamma
                self.option_num_steps[i] += 1
                
                if self.options_lst[curr_option_idx].is_terminated(obs) or terminated[i] or truncated[i]:
                    transition = self.option_start_obs[i], \
                                self.atomic_action_space.n + curr_option_idx, \
                                self.option_cumulative_reward[i], \
                                self.option_multiplier[i], \
                                self.option_num_steps[i]
                    self.rollout_buffer[i].add(transition)
                    self.running_option_index[i] = None
            else:
                transition = get_single_observation(self.last_observation, i), \
                            self.last_action[i], \
                            reward[i], \
                            self.hp.gamma, \
                            1
                self.rollout_buffer[i].add(transition)
            

            
            if terminated[i] or truncated[i]:
                rollout = self.rollout_buffer[i].all()
                rollout_observations, rollout_actions, rollout_rewards, rollout_discounts, rollout_num_steps = zip(*rollout)     
                n_step_return = calculate_n_step_returns_with_discounts(rollout_rewards, 0, rollout_discounts)
                all_steps = sum(s for s in rollout_num_steps)
                for j in range(len(self.rollout_buffer[i])):                    
                    trans = (rollout_observations[j], 
                            rollout_actions[j], 
                            obs, 
                            n_step_return[j], 
                            terminated[i], 
                            truncated[i], 
                            self.hp.gamma**(all_steps-rollout_num_steps[j])
                            )
                    self.replay_buffer.add(trans)
                self.rollout_buffer[i].clear() 
            
            elif self.rollout_buffer[i].is_full():
                rollout = self.rollout_buffer[i].all() 
                rollout_observations, rollout_actions, rollout_rewards, rollout_discounts, rollout_num_steps  = zip(*rollout)
                n_step_return = calculate_n_step_returns_with_discounts(rollout_rewards, 0, rollout_discounts)
                all_steps = sum(s for s in rollout_num_steps)
                trans = (rollout_observations[0], rollout_actions[0], obs, \
                         n_step_return[0], terminated[i], \
                         truncated[i], self.hp.gamma**all_steps)
                self.replay_buffer.add(trans)
                
            
        if len(self.replay_buffer) >= self.hp.warmup_buffer_size:
            batch = self.replay_buffer.sample(self.hp.batch_size)

            observations, actions, next_observations, n_step_return, terminated, truncated, effective_discount = zip(*batch)
            observations, next_observations = stack_observations(observations), stack_observations(next_observations)
            states, next_states = self.feature_extractor(observations), self.feature_extractor(next_observations)
            self.policy.update(states, actions, next_states, n_step_return, terminated, truncated, effective_discount, call_back=call_back)
        
        if call_back is not None:            
            call_back({
                "train/buffer_size": len(self.replay_buffer),
                })
          
    def reset(self, seed):
        """
        Reset the agent's learning state, including feature extractor and replay buffer.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        
        # Option execution bookkeeping
        self.running_option_index = [None for _ in range(self.num_envs)]       # index into options_lst (or None)
        self.option_start_obs = [None for _ in range(self.num_envs)]         # encoded state where option began
        self.option_cumulative_reward = [0.0 for _ in range(self.num_envs)]    # discounted return accumulator R_{t:t+k}
        self.option_multiplier = [1.0 for _ in range(self.num_envs)]           # current gamma^t during option
        self.option_num_steps = [0 for _ in range(self.num_envs)]
        
    def log(self):
        pass
        # if self.running_option_index is None:
        #     return {
        #         "OptionUsageLog": False,
        #         "NumOptions": len(self.options_lst),
        #         "OptionIndex": None,
        #         "OptionClass": None,
        #     }
        # else:
        #     return {
        #         "OptionUsageLog": True,
        #         "NumOptions": len(self.options_lst),
        #         "OptionIndex": self.running_option_index,
        #         "OptionClass": self.options_lst[self.running_option_index].__class__,
        #     }
            
    def save(self, file_path=None):
        checkpoint = super().save(file_path=None)  # parent saves feature_extractor, policy, hp, etc.
        
        # Save options list payload
        checkpoint['options_lst'] = save_options_list(self.options_lst, file_path=None)
        checkpoint['atomic_action_space'] = self.atomic_action_space
        
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_agent.t")
        return checkpoint

    @classmethod
    def load(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        
        instance = super().load(file_path, checkpoint)
        instance.options_lst = load_options_list(file_path=None, checkpoint=checkpoint['options_lst'])
        instance.atomic_action_space = checkpoint['atomic_action_space']
        
        return instance