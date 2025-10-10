import numpy as np
import random
import torch
from gymnasium.spaces import Discrete

from ..Utils import BaseAgent, BasePolicy, calculate_n_step_returns_with_discounts
from .QLearning import QLearningAgent, QLearningPolicy
from ...registry import register_agent, register_policy
from ...Options.Utils import load_options_list, save_options_list

@register_policy
class OptionQLearningPolicy(QLearningPolicy):
    def update(self, last_state, last_action, state, reward, terminated, truncated, effective_discount, call_back=None):
        """
        Update the Q-table using the Q-Learning update rule.
        
        Args:
            last_state (hashable): Previous state (encoded) where the last action was taken.
            last_action (int): Action taken in the previous state.
            state (hashable): Current state (encoded) after action.
            reward (float): Reward received.
            terminated (bool): True if the episode has terminated.
            truncated (bool): True if the episode was truncated.
            call_back (function, optional): Callback for tracking training progress.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_dim)
        if last_state not in self.q_table:
            self.q_table[last_state] = np.zeros(self.action_dim)
            
        transition = (last_state, last_action, reward, effective_discount)
        self.rollout_buffer.add_single_item(transition)

        
        if self.rollout_buffer.size >= self.hp.n_steps:
            rollout = self.rollout_buffer.get_all()  # All transitions in the buffer
            # Unpack transitions (states and actions not used directly here).
            rollout_states, rollout_actions, rollout_rewards, rollout_discounts = zip(*rollout)
            
            # Use the max Q-value from the current state as bootstrap, unless terminated.
            bootstrap_value = np.max(self.q_table[state]) if not terminated else 0.0
            
            # Compute n-step return (only the first value is used for the oldest transition).
            target = calculate_n_step_returns_with_discounts(rollout_rewards, bootstrap_value, rollout_discounts)[0]
            
            
            #Update Value Function
            s, a, _ = self.rollout_buffer.remove_oldest()
            td_error = target - self.q_table[s][a]
            self.q_table[s][a] += self.hp.step_size * td_error
            
            #Update Epsilon
            frac = 1.0 - (self.step_counter / self.hp.epilon_decay_steps)
            self.epsilon = self.hp.epsilon_end + (self.hp.epsilon_start - self.hp.epsilon_end) * frac
                
            if call_back is not None:
                call_back({"value_loss": td_error,
                        "epsilon": self.epsilon,
                        })
        # At episode end, flush any remaining transitions in the buffer.
        if terminated or truncated:
            while self.rollout_buffer.size > 0:
                rollout = self.rollout_buffer.get_all()
                rollout_states, rollout_actions, rollout_rewards, rollout_discounts = zip(*rollout)
                
                bootstrap_value = np.max(self.q_table[state]) if not terminated else 0.0
                
                target = calculate_n_step_returns_with_discounts(rollout_rewards, bootstrap_value, rollout_discounts)[0]
                
                s, a, _ = self.rollout_buffer.remove_oldest()
                td_error = target - self.q_table[s][a]
                self.q_table[s][a] += self.hp.step_size * td_error
                
                #Update Epsilon
                frac = 1.0 - (self.step_counter / self.hp.epilon_decay_steps)
                self.epsilon = self.hp.epsilon_end + (self.hp.epsilon_start - self.hp.epsilon_end) * frac
                
                if call_back is not None:
                    call_back({"value_loss": td_error,
                        "epsilon": self.epsilon,
                        })

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
        action_option_space = Discrete(self.atomic_action_space.n + len(self.options_lst))

        # Swap policy with OptionQLearningPolicy (shares same interface)
        self.policy = OptionQLearningPolicy(action_option_space, hyper_params)

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

        # print(f"Agent -> Option: {self.running_option_index}, Action: {action}")

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
                    effective_discount=self.option_multiplier if self.hp.discount_option_flag else self.hp.gamma,
                    call_back=call_back
                )
                # Clear option state
                self.running_option_index = None
                self.option_start_state = None
                self.option_cumulative_reward = 0.0
                self.option_multiplier = 1.0
            
            
            if self.hp.update_action_within_option_flag:
                self.policy.update(
                last_state=self.last_state,
                last_action=self.last_action,
                state=state,
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                effective_discount=self.hp.gamma,
                call_back=call_back
            )

        else:
            self.policy.update(
                last_state=self.last_state,
                last_action=self.last_action,
                state=state,
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                effective_discount=self.hp.gamma,
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
    
    def log(self):
        if self.running_option_index is None:
            return {"OptionUsageLog": False}
        else:
            return {"OptionUsageLog": True}
        
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