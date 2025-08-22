from ..Utils import BaseOption
from ..Utils import discrete_levin_loss_on_trajectory
from ..Utils import save_options_list, load_options_list
from ...registry import register_option
from ...loaders import load_policy, load_feature_extractor
from .PolicyMaskers import POLICY_TO_MASKER

import random
import torch
import numpy as np
from tqdm import tqdm
import copy
from multiprocessing import Pool
import torch.nn as nn
import torch.optim as optim
import os

class MaskedOptionLearner():
    name="MaskedOptionLearner"
    def __init__(self, agent_lst=None, trajectories_lst=None, hyper_params=None):
        self.set_params(agent_lst, trajectories_lst, hyper_params)
        

    def set_params(self, agent_lst=None, trajectories_lst=None, hyper_params=None):
        if agent_lst is not None:
            self.agent_lst = copy.deepcopy(agent_lst)
            self.action_space = self.agent_lst[0].action_space
            self.maskable_layers = []
            for agent in self.agent_lst:
                if not (hasattr(agent.policy, 'select_action_masked') and callable(getattr(agent.policy, 'select_action_masked'))):
                    agent.policy.__class__ = POLICY_TO_MASKER[agent.policy.__class__]  # Injects select_action_masked into the policy.
                self.maskable_layers.append(agent.policy.maskable_layers)
                

        if trajectories_lst is not None:
            self.trajectories_lst = trajectories_lst
            self.org_loss = sum([discrete_levin_loss_on_trajectory(trajectory, None, self.action_space.n) for trajectory in self.trajectories_lst]) / len(self.trajectories_lst)
        
        if hyper_params is not None:
            self.hyper_params = hyper_params
    
    def learn(self, agent_lst=None, trajectories_lst=None, hyper_params=None, verbose=True, seed=None, num_workers=1, exp_dir=None):
        self.num_workers = num_workers
        self.exp_dir = exp_dir
        self.set_params(agent_lst, trajectories_lst, hyper_params)
        
        if self.exp_dir is not None and os.path.exists(os.path.join(self.exp_dir, "sub_trajectories.t")):
            print("Loading sub-trajectories")
            sub_trajectory_lst = torch.load(os.path.join(self.exp_dir, "sub_trajectories.t"), weights_only=False)
            print(f"Number of loaded sub-trajectories: {len(sub_trajectory_lst)}")
        else:
            print("Extracting sub-trajectories")
            sub_trajectory_lst = self._get_all_sub_trajectories()
            torch.save(sub_trajectory_lst, os.path.join(self.exp_dir, "sub_trajectories.t"))
        
        
        if self.exp_dir is not None and os.path.exists(os.path.join(self.exp_dir, "all_options.t")):
            print("Loading all options")
            self.options_lst = load_options_list(os.path.join(self.exp_dir, "all_options.t"))
            print(f"Number of loaded options: {len(self.options_lst)}")
        else:
            print("Training all options")
            self.options_lst = self._get_all_options(sub_trajectory_lst, verbose)
            save_options_list(self.options_lst, os.path.join(self.exp_dir, "all_options.t"))
        
        
        if self.exp_dir is not None and os.path.exists(os.path.join(self.exp_dir, f"selected_options_{self.hyper_params.max_num_options}.t")):
            print("Loading selected options")
            self.selected_options_lst = load_options_list(os.path.join(self.exp_dir, f"selected_options_{self.hyper_params.max_num_options}.t"))
            print(f"Number of loaded options: {len(self.selected_options_lst)}")
        else:
            print("Selecting from all options")
            self.selected_options_lst, best_loss = self._select_options_hc(seed=seed)
            save_options_list(self.selected_options_lst, os.path.join(self.exp_dir, f"selected_options_{self.hyper_params.max_num_options}.t"))
            print(f"Number of selected options: {len(self.selected_options_lst)}")

            
        if verbose:
            print(f"Total options after selected: {len(self.selected_options_lst)}")
            best_loss = sum([discrete_levin_loss_on_trajectory(trajectory, self.selected_options_lst, self.action_space.n) for trajectory in self.trajectories_lst]) / len(self.trajectories_lst)
            print(f"Selected Levin Loss: {best_loss}")
        return self.selected_options_lst
        
    def _get_all_sub_trajectories(self):
        sub_trajectory_lst = []
        for trajectory in self.trajectories_lst:
            sub_traj = []
            max_len = min(self.hyper_params.max_option_len , len(trajectory))
            for l in range(2, max_len + 1):
                for start in range(len(trajectory) - l + 1):
                    sub_traj.append(trajectory[start : start + l])
            sub_trajectory_lst.append(sub_traj)
        return sub_trajectory_lst    
    
    def _one_mask_train(self, task):
        # task includes policy, feature_extractor, and the trajectory
        policy, feature_extractor, traj, maskable_layers = task
        masked_layers = {k: maskable_layers[k]['size'] for k in maskable_layers if k in self.hyper_params.masked_layers}
        mask_dict = nn.ParameterDict({name: nn.Parameter(torch.zeros(3, size), requires_grad=True) for name, size in masked_layers.items()})
        
        optimizer = optim.Adam(mask_dict.parameters(), lr=self.hyper_params.mask_lr)
        loss_fn   = nn.CrossEntropyLoss()
        
        for _ in range(self.hyper_params.n_epochs):
            total_loss = 0.0
            for observation, action in traj:
                state_t = feature_extractor(observation)      # (1, State_Dim)
                action_t = torch.from_numpy(np.array(action)).unsqueeze(0) # (1, )    
                pred_a, pred_logits_t = policy.select_action_masked(state_t, mask_dict) # (1, num_actions)
                # regularization_term = self._calculate_regularization(mask_dict)
                # print(regularization_term)
                # exit(0)
                loss = loss_fn(pred_logits_t, action_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        # Check if the Fine Tuned policy learned the sub trajectory
        # If not don't add it to the options
        for observation, action in traj:
            state_t = feature_extractor(observation) 
            pred_a, pred_logits_t = policy.select_action_masked(state_t, mask_dict)
            if pred_a != action:
                return None
            
        for p in mask_dict.values():
            p.requires_grad_(False)
        return MaskedOption(feature_extractor, policy, len(traj), mask_dict)
    
    def _calculate_regularization(self, mask_dict):
        reg = 0
        for key in mask_dict:
            mask = mask_dict[key]
            probs = torch.softmax(mask, dim=0)
            reg += probs[-1, :] ** 2
        return reg
            
    
    def _get_all_options(self, sub_trajectory_lst, verbose):
        options_lst = []
        num_useless_masks = 0
        
        if verbose:
            print(f"Original Levin Loss: {self.org_loss}")
        
        tasks = []
        for i, sub_trajectory in enumerate(sub_trajectory_lst):
            for j, agent in enumerate(self.agent_lst):
                if i == j:
                    continue
                for traj in sub_trajectory:
                    policy = copy.deepcopy(agent.policy)
                    feature_extractor = copy.deepcopy(agent.feature_extractor)
                    tasks.append((policy, feature_extractor, traj, self.maskable_layers[j]))
        
        # parallel map
        with Pool(processes=self.num_workers) as pool:
            for opt in tqdm(
                pool.imap_unordered(self._one_mask_train, tasks),
                total=len(tasks),
                desc="Masking policies"
            ):
                if opt is not None:
                    options_lst.append(opt)
                else:
                    num_useless_masks += 1
          
        
        new_loss = sum([discrete_levin_loss_on_trajectory(trajectory, options_lst, self.action_space.n) for trajectory in self.trajectories_lst]) / len(self.trajectories_lst)       
        if verbose:
            print(f"Total number of raw options: {len(options_lst)}")
            print(f"Number of useless masks: {num_useless_masks}")
            print(f"New Levin Loss: {new_loss}")
                        
        return options_lst
         
    def _one_restart(self, seed):
        """
        Perform one stochastic hill‐climb restart.
        Returns (best_subset, best_loss).
        """
        random.seed(seed)

        # start from empty (or random size if you like)
        subset = set()
        curr_loss = self.org_loss
        best_subset = list(subset)
        best_loss = curr_loss
        for _ in range(self.hyper_params.n_iteration):
            candidates = random.sample(self.options_lst, k=min(self.hyper_params.n_neighbours, len(self.options_lst)))
            improved = False
            for opt in candidates:
                if opt in subset:
                    cand = list(subset - {opt})
                else:
                    if self.hyper_params.max_num_options is not None and len(subset) >= self.hyper_params.max_num_options:
                        continue
                    cand = list(subset | {opt})
                cand_loss = sum([discrete_levin_loss_on_trajectory(trajectory, cand, self.action_space.n) for trajectory in self.trajectories_lst]) / len(self.trajectories_lst)

                if cand_loss < curr_loss:
                    subset, curr_loss = set(cand), cand_loss
                    improved = True
                    break

            if curr_loss < best_loss:
                best_subset, best_loss = list(subset), curr_loss

            if not improved:
                break

        return best_subset, best_loss
    
    def _select_options_hc(self, seed):
        """
        Parallelized over restarts.
        """
        # prepare args for each restart (use different seeds if you want)
        tasks = [seed + r for r in range(self.hyper_params.n_restarts)]

        best_subset, best_loss = [], float('inf')

        with Pool(processes=self.num_workers) as pool:
            for subset_r, loss_r in tqdm(
                pool.imap_unordered(self._one_restart, tasks),
                total=self.hyper_params.n_restarts,
                desc="Restarts",
                unit="run"
            ):
                # track global best
                if loss_r < best_loss:
                    best_loss = loss_r
                    best_subset = subset_r

        return best_subset, best_loss
    
  


@register_option
class MaskedOption(BaseOption):
    def __init__(self, feature_extractor, policy, option_len, mask_dict): 
        super().__init__(feature_extractor, policy)
        self.step_counter = 0
        self.option_len = option_len
        self.mask_dict = mask_dict

    def select_action(self, observation):
        state = self.feature_extractor(observation)
        action, logits = self.policy.select_action_masked(state, self.mask_dict)
        self.step_counter += 1
        return action

    def is_terminated(self, observation):
        if self.step_counter >= self.option_len:
            self.step_counter = 0
            return True
        return False
    
    def save(self, file_path=None):
        checkpoint = super().save()
        checkpoint["option_len"] = self.option_len
        checkpoint["mask_dict"] = self.mask_dict
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_options.t")
        return checkpoint
    
    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        """
        Load the Q-table and hyper-parameters.
        
        Args:
            file_path (str): File path from which to load the checkpoint.
        """
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        policy = load_policy("", checkpoint=checkpoint['policy'])
        feature_extractor = load_feature_extractor("", checkpoint=checkpoint['feature_extractor'])
        instance = cls(feature_extractor, policy, checkpoint["option_len"], checkpoint["mask_dict"])

        return instance
    
    
