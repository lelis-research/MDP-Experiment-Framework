from ..Utils import BaseOption
from ..Utils import discrete_levin_loss_on_trajectory
from ...registry import register_option
from ...loaders import load_policy, load_feature_extractor
from ..Utils import save_options_list, load_options_list

import random
import torch
import numpy as np
from tqdm import tqdm
import copy
from multiprocessing import Pool
import os

class DecWholeOptionLearner():
    name="DecWholeOptionLearner"
    def __init__(self, agent_lst=None, trajectories_lst=None, hyper_params=None):
        self.set_params(agent_lst, trajectories_lst, hyper_params)

    def set_params(self, agent_lst=None, trajectories_lst=None, hyper_params=None):
        if agent_lst is not None:
            self.agent_lst = copy.deepcopy(agent_lst)
            self.action_space = self.agent_lst[0].action_space

        if trajectories_lst is not None:
            self.trajectories_lst = trajectories_lst
            self.org_loss = sum([discrete_levin_loss_on_trajectory(trajectory, None, self.action_space.n) for trajectory in trajectories_lst]) / len(trajectories_lst)
        
        if hyper_params is not None:
            self.hyper_params = hyper_params
    
    def learn(self, agent_lst=None, trajectories_lst=None, hyper_params=None, verbose=True, seed=None, num_workers=1, exp_dir=None):
        self.num_workers = num_workers
        self.exp_dir = exp_dir
        self.set_params(agent_lst, trajectories_lst, hyper_params)
        
        if verbose:
            print(f"Original Levin Loss: {self.org_loss}")
                
        if self.exp_dir is not None and os.path.exists(os.path.join(self.exp_dir, "all_options.t")):
            print("Loading all options")
            self.options_lst = load_options_list(os.path.join(self.exp_dir, "all_options.t"))
            print(f"Number of loaded options: {len(self.options_lst)}")
            if verbose:
                new_loss = sum([discrete_levin_loss_on_trajectory(trajectory, self.options_lst, self.action_space.n) for trajectory in self.trajectories_lst]) / len(self.trajectories_lst)
                print(f"Total options before selection: {len(self.options_lst)}")
                print(f"New Levin Loss: {new_loss}")
        else:
            print("Training all options")
            self.options_lst = self._get_all_options(verbose)
            save_options_list(self.options_lst, os.path.join(self.exp_dir, "all_options.t"))
            print(f"Number of trained options: {len(self.options_lst)}")
            if verbose:
                new_loss = sum([discrete_levin_loss_on_trajectory(trajectory, self.options_lst, self.action_space.n) for trajectory in self.trajectories_lst]) / len(self.trajectories_lst)
                print(f"Total options before selection: {len(self.options_lst)}")
                print(f"New Levin Loss: {new_loss}")
            
        if self.exp_dir is not None and os.path.exists(os.path.join(self.exp_dir, f"selected_options_{self.hyper_params.max_num_options}.t")):
            print("Loading selected options")
            self.selected_options_lst = load_options_list(os.path.join(self.exp_dir, f"selected_options_{self.hyper_params.max_num_options}.t"))
            print(f"Number of loaded selected options: {len(self.selected_options_lst)}")
        else:
            print("Selecting from all options")
            self.selected_options_lst, best_loss = self._select_options_hc(seed=seed)
            save_options_list(self.selected_options_lst, os.path.join(self.exp_dir, f"selected_options_{self.hyper_params.max_num_options}.t"))
            print(f"Number of selected options: {len(self.selected_options_lst)}")
            
        print(" ******** ")
        if verbose:
            print(f"Total options after selected: {len(self.selected_options_lst)}")
            best_loss = sum([discrete_levin_loss_on_trajectory(trajectory, self.selected_options_lst, self.action_space.n) for trajectory in self.trajectories_lst]) / len(self.trajectories_lst)
            print(f"Selected Levin Loss: {best_loss}")
        return self.selected_options_lst
        
        
    def _get_all_options(self, verbose):
        options_lst = []
            
        for agent in self.agent_lst:
            for option_len in range(1, self.hyper_params.max_option_len):
                option = DecWholeOption(agent.feature_extractor, agent.policy, option_len)
                options_lst.append(option)
                        
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
    
    def _select_options_hc(self, seed, num_workers=16):
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
class DecWholeOption(BaseOption):
    def __init__(self, feature_extractor, policy, option_len): 
        super().__init__(feature_extractor, policy)
        self.step_counter = 0
        self.option_len = option_len

    def select_action(self, observation):
        state = self.feature_extractor(observation)
        action = self.policy.select_action(state)
        
        if isinstance(action, list) or isinstance(action, tuple):
            # some policies return more than just action e.g. log_prob
            # We assume the action is always the first index 
            # NOTE: Make sure the action is always the first index!
            action = action[0] 
            
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
        instance = cls(feature_extractor, policy, checkpoint["option_len"])

        return instance
    
    
