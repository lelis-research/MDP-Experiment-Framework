import numpy as np
import random
import torch
from gymnasium.spaces import Discrete
from copy import copy

from ..Utils import BaseContiualPolicy
from .OptionQLearning import OptionQLearningAgent, OptionQLearningPolicy
from ...registry import register_agent, register_policy
from ...Options.Utils import load_options_list, save_options_list

@register_policy
class ContinualOptionQLearningPolicy(OptionQLearningPolicy, BaseContiualPolicy):
        
    def trigger_option_learner(self):
        pass
    
    def init_options(self, new_options):
        num_new_options = len(new_options)
        if num_new_options == 0:
            return 
        
        old_n = self.action_space.n
        new_n = old_n + num_new_options
        self.action_space = Discrete(new_n)
        
        self.epsilon = self.hp.epsilon_start
        self.epsilon_step_counter = 0
        
        mode = self.hp.option_init_mode  # "reset", "init_zero", "init_avg", "init_max", "init_uncertainty"
        
        states = list(self.q_table.keys())
        
        if mode == "reset":
            for s in states:
                self.q_table[s] = np.zeros(new_n, dtype=np.float32)
            return
        
        for s in states:
            q_values = np.asarray(self.q_table[s], dtype=np.float32)
            assert len(q_values) == old_n, f"State {s} Q length {len(q_values)} != old_n {old_n}"
            
            if mode == "init_zero":
                new_vals = np.zeros(num_new_options, dtype=np.float32)
                    
                
            elif mode == "init_avg":
                avg_values = float(np.mean(q_values)) if old_n > 0 else 0.0
                new_vals = np.full(num_new_options, avg_values, dtype=np.float32)

            elif mode == "init_max":
                max_values = float(np.max(q_values)) if old_n > 0 else 0.0
                new_vals = np.full(num_new_options, max_values, dtype=np.float32)
                
            elif mode == "init_uncertainty":
                # median-to-max interpolation scaled by U(s) and beta
                median_values = float(np.median(q_values)) if old_n > 0 else 0.0
                max_values  = float(np.max(q_values))    if old_n > 0 else median_values
                uncertainty = self._uncertainty(q_values)  # in [0,1]

                interpolation = median_values + self.hp.beta * uncertainty * (max_values - median_values)
                new_vals = np.full(num_new_options, interpolation, dtype=np.float32)

            else:
                raise NotImplementedError(f"Unknown option_init_mode: {mode}")
            
            self.q_table[s] = np.concatenate([q_values, new_vals]).astype(np.float32)
            
            
            
    
    def _softmax(self, x, tau: float):
        # stable softmax with temperature
        x = np.asarray(x, dtype=np.float64)
        z = (x - np.max(x)) / max(tau, 1e-8)
        ez = np.exp(z)
        return ez / np.sum(ez)

    def _uncertainty(self, q_values: np.ndarray) -> float:
        """
        Compute U(s) in [0,1] from current q_values over old action set.
        Uses hp.U_mode: 'entropy' or 'margin'.
        Entropy: U = H(p)/log|A| with p = softmax(q/tau)
        Margin:  U = 1 - sigmoid((q1-q2)/kappa)
        """
        A = len(q_values)
        
        if self.hp.uncertainty_mode == "entropy":
            p = self._softmax(q_values, self.hp.uncertainty_tau)
            # handle |A|=1 => log|A|=0 ; define U=0 (certain) in that trivial case
            denom = np.log(A) if A > 1 else 1.0
            H = -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)))
            return float(H / denom)

        elif self.hp.uncertainty_mode == "margin":
            if A == 1:
                return 0.0
            # top-2
            idx = np.argsort(q_values)[::-1]
            q1, q2 = q_values[idx[0]], q_values[idx[1]]
            gap = (q1 - q2) / max(self.uncertainty_kappa, 1e-8)
            sigma = 1.0 / (1.0 + np.exp(-gap))
            return float(1.0 - sigma)

        else:
            raise ValueError(f"Unknown uncertainty_mode: {self.hp.uncertainty_mode}")
        


@register_agent
class ContinualOptionQLearningAgent(OptionQLearningAgent):
    name = "ContinualOptionQLearning"
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, option_learner_class, initial_options_lst=[]):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class, options_lst=initial_options_lst)
        self.option_learner = option_learner_class()
        self.initial_options_lst = copy(initial_options_lst)
        
        # Replace action_space with extended (primitive + options)
        action_option_space = Discrete(self.atomic_action_space.n + len(self.options_lst))

        self.policy = ContinualOptionQLearningPolicy(action_option_space, hyper_params)

    
    def act(self, observation, greedy=False):
        """
        Returns an atomic (primitive) action to execute.
        If an option is running, returns its current primitive action.
        Otherwise, samples from the extended action space (primitives + options).
        """
       
        self.last_observation = observation
        action = super().act(observation, greedy)
        
        return action
    
    def update(self, observation, reward, terminated, truncated, call_back=None):
        """
        After env.step, update Q-values.
        - If inside an option, accumulate discounted reward and do a single SMDP update on option termination.
        - If primitive, do the usual 1-step Q-learning update (parent policy already supports it).
        """
        self.option_learner.update()
        if self.policy.trigger_option_learner():
            learned_options = self.option_learner.learn()
            self.options_lst += learned_options
            self.policy.init_options(learned_options)
            
        super().update(observation, reward, terminated, truncated, call_back)
        
    
    def reset(self, seed):
        super().reset(seed)
        self.option_learner.reset()
        self.options_lst = self.initial_options_lst
        
    def log(self):
        if self.running_option_index is None:
            return {"OptionUsageLog": False, "NumOptions":len(self.options_lst), "OptionIndex": self.running_option_index}
        else:
            return {"OptionUsageLog": True, "NumOptions":len(self.options_lst), "OptionIndex": self.running_option_index}
        
    def save(self, file_path=None):
        """
        Save extended agent (including options list).
        """
        checkpoint = super().save(file_path=None)  # parent saves feature_extractor, policy, hp, etc.


        # Save options list payload
        checkpoint['options_lst'] = save_options_list(self.options_lst, file_path=None)
        checkpoint['atomic_action_space'] = self.atomic_action_space
        checkpoint['option_learner'] = self.option_learner.save()
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
            initial_options_lst=options_lst
        )
        instance.reset(seed)
        instance.feature_extractor.load_from_checkpoint(checkpoint['feature_extractor'])
        instance.policy.load_from_checkpoint(checkpoint['policy'])
        instance.option_learner.load_from_checkpoint(checkpoint['option_learner'])
        return instance