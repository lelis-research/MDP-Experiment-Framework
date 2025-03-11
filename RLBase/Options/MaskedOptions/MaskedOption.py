from .PolicyMaskers import POLICY_TO_MASKER
from .LevinLoss import levin_loss_on_trajectory
from ..Utils import BaseOption
from ...loaders import load_policy, load_feature_extractor
from ...registry import register_option

import random
import torch
import numpy as np
from tqdm import tqdm
import copy
import nevergrad as ng

class LevinLossMaskedOptionLearner():
    def __init__(self, action_space, observation_space, policy=None, trajectories=None, feature_extractor=None, num_options=None, masked_layers=None):
        self.action_space = action_space
        self.observation_space = observation_space
        self.policy = None
        
        self.set_params(policy, trajectories, feature_extractor, num_options, masked_layers)
        

    def set_params(self, policy=None, trajectories=None, feature_extractor=None, num_options=None, masked_layers=None):
        if policy is not None:
            self.policy = copy.deepcopy(policy)
            if not (hasattr(self.policy, 'select_action_masked') and callable(getattr(self.policy, 'select_action_masked'))):
                self.policy.__class__ = POLICY_TO_MASKER[self.policy.__class__]  # Injects select_action_masked into the policy.
            self.maskable_layers = self.policy.maskable_layers

        if trajectories is not None:
            self.trajectories = trajectories
            self.org_loss = sum([levin_loss_on_trajectory(trajectory, None, self.action_space.n) for trajectory in trajectories]) / len(trajectories)

        if num_options is not None:
            self.num_options = num_options
        
        if masked_layers is not None:
            self.masked_layers = masked_layers 
        
        if feature_extractor is not None:
            self.feature_extractor = copy.deepcopy(feature_extractor)
        
        if not hasattr(self, "masked_layers") and hasattr(self, "policy"):
            self.masked_layers = self.maskable_layers
        
        self.mask_dict_size = self._get_mask_dict_size()
        
        
    def learn(self, policy=None, trajectories=None, feature_extractor=None, num_options=None, masked_layers=None, search_budget=10, verbose=True):
        self.set_params(policy, trajectories, feature_extractor, num_options, masked_layers)
        
        # Compute total number of parameters for the list of mask_dicts
        total_params = self.num_options * sum(sum(spec.values()) for spec in self.mask_dict_size)       
        instrum = ng.p.Array(shape=(total_params,), lower=-1, upper=1).set_integer_casting()
        optimizer = ng.optimizers.RandomSearch(parametrization=instrum, budget=search_budget, num_workers=10)

        if verbose:
            min_loss = self.org_loss
            pbar = tqdm(range(1, search_budget + 1), desc="Optimizing")

            for _ in pbar:
                candidate = optimizer.ask()
                # Evaluate the objective function on the candidate's value
                loss = self.objective(candidate.value)
                # Report the result to the optimizer
                optimizer.tell(candidate, loss)

                if loss < min_loss:
                    min_loss = loss

                pbar.set_postfix({
                    "loss": loss,
                    "min loss": min_loss,
                    "org loss": self.org_loss
                })
        else:
            optimizer.minimize(self.objective)

        recommendation = optimizer.provide_recommendation()
        # print("Best solution:", recommendation.value, self.objective(recommendation.value))    

        mask_dict_list = self._decode_candidate(recommendation.value)
        options = LevinLossMaskedOptions(self.feature_extractor, self.num_options, self.policy, mask_dict_list)
        return options

    def objective(self, candidate):
        mask_dict_list = self._decode_candidate(candidate)
        options = LevinLossMaskedOptions(self.feature_extractor, self.num_options, self.policy, mask_dict_list)
        loss = sum([levin_loss_on_trajectory(trajectory, options, self.action_space.n) for trajectory in self.trajectories]) / len(self.trajectories)
        return loss
    
    def _decode_candidate(self, candidate):
        """
        Convert a flat candidate list into a list of mask dictionaries.
        For example, if candidate is a list of 256 and 
        mask_dict_size = [{'1': 128}, {'3': 128}],
        then one mask dictionary will be constructed as:
            {'1': candidate_slice_of_length_128, '3': candidate_slice_of_length_128}
        """
        decoded = []
        pos = 0
        
        for _ in range(self.num_options):
            mask_dict = {}
            for spec in self.mask_dict_size:
                for key, size in spec.items():
                    mask_dict[key] = candidate[pos: pos + size]
                    pos += size
            decoded.append(mask_dict)
        return decoded
    
    def _get_mask_dict_size(self):
        '''
        Returns a list of mask_dicts with their size
        e.g. [{'1': 128}, {'3': 128}] means first and third layer and both have sizes 128
        '''
        maskable_layers = self.policy.maskable_layers
        space = []
        for layer_name in self.masked_layers:
            if layer_name not in maskable_layers:
                raise ValueError(f"Layer '{layer_name}' not in maskable layers {maskable_layers}")
            size = maskable_layers[layer_name]['size']

            space.append({layer_name: size})
        return space

  


@register_option
class LevinLossMaskedOptions(BaseOption):
    def __init__(self, feature_extractor, num_options, policy, mask_dict_list):
        '''Defines multiple different options (masking) for a given policy'''
        assert num_options == len(mask_dict_list)
        
        super().__init__(feature_extractor, num_options)
        self.policy = policy
        self.mask_dict_list = mask_dict_list
        self.step_counter = 0

    def select_action(self, observation, index):
        assert index < self.n
        state = self.feature_extractor(observation)
        action = self.policy.select_action_masked(state, self.mask_dict_list[index])
        self.step_counter += 1
        return action

    def is_terminated(self, observation):
        if self.step_counter > 10:
            self.step_counter = 0
            return True
        return False
    
    def save(self, file_path=None):
        policy_checkpoint = self.policy.save()
        feature_extractor_checkpoint = self.feature_extractor.save()
        checkpoint = {
            'feature_extractor':feature_extractor_checkpoint,
            'num_options': self.n,
            'policy': policy_checkpoint,
            'mask_dict_list': self.mask_dict_list,

            'option_class': self.__class__.__name__,
        }
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_options.t")
        return checkpoint


    @classmethod
    def load_from_file(cls, file_path, seed=0, checkpoint=None):
        """
        Load the Q-table and hyper-parameters.
        
        Args:
            file_path (str): File path from which to load the checkpoint.
        """
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        policy = load_policy("", checkpoint=checkpoint['policy'])
        feature_extractor = load_feature_extractor("", checkpoint=checkpoint['feature_extractor'])
        instance = cls(feature_extractor, checkpoint['num_options'], policy, checkpoint['mask_dict_list'])

        return instance
    
