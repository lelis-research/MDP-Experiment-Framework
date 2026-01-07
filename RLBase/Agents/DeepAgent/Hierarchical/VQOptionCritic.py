import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical, Normal, Independent, TransformedDistribution
import gymnasium
from gymnasium.spaces import Discrete, Box
from typing import Optional
import copy

from ....utils import RandomGenerator
from ...Base import BaseAgent, BasePolicy
from ..PolicyGradient import OptionPPOPolicy
from ....Buffers import BaseBuffer, ReplayBuffer
from ...Utils import (
    calculate_gae_with_discounts, 
    get_single_observation, 
    stack_observations, 
    grad_norm,
    explained_variance,
    get_single_state,
    get_single_observation_nobatch,
    get_batch_observation,
    get_batch_state,
    HyperParameters,
)
from ....registry import register_agent, register_policy
from ....Networks.NetworkFactory import NetworkGen, prepare_network_config
from ....FeatureExtractors import get_batch_features


from ....Options.SymbolicOptions.PreDesigned import GoToBlueGoalOption

class Encoder(RandomGenerator):
    def __init__(self, hyper_params, features_dict, device):
        self.features_dict = features_dict
        self.hp = hyper_params
        self.device = device
        
        encoder_discription = prepare_network_config(
            self.hp.encoder_network,
            input_dims=self.features_dict,
            output_dim=self.hp.encoder_dim,
        )
        self.encoder = NetworkGen(layer_descriptions=encoder_discription).to(self.device)
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=self.hp.encoder_step_size, eps=self.hp.encoder_eps)
    
    def __call__(self, state):
        x = self.encoder(**state)
        return x
    
    def reset(self, seed):
        self.set_seed(seed)
        
        encoder_discription = prepare_network_config(
            self.hp.encoder_network,
            input_dims=self.features_dict,
            output_dim=self.hp.encoder_dim,
        )
        self.encoder = NetworkGen(layer_descriptions=encoder_discription).to(self.device)
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=self.hp.encoder_step_size, eps=self.hp.encoder_eps)
    

class HighLevelPolicy(OptionPPOPolicy):
    pass

class LowLevelPolicy(BasePolicy):
    def __init__(self, action_space, hyper_params, device):
        super().__init__(action_space, hyper_params, device)
    
    def select_action(self, x, emb, greedy):
        return [0] * len(x)

class CodeBook(RandomGenerator):
    def __init__(self, hyper_params, num_initial_codes: int, device):
        self.num_codes = num_initial_codes
        self.device = device
        self.hp = hyper_params
        
        self.emb = nn.Embedding(self.num_codes, self.hp.embedding_dim).to(self.device)
        self._init_weights()
    
    def _init_weights(self):
        with torch.no_grad():
            eps = 1.0 / max(1, self.num_codes)
            nn.init.uniform_(self.emb.weight, -eps, eps)
            
        # with torch.no_grad():
        #     self.emb.weight.zero_()
        #     for i in range(self.num_codes):
        #         self.emb.weight[i, i] = 1.0
            
    def reset(self, seed):
        self.set_seed(seed)
        
        self.emb = nn.Embedding(self.num_codes, self.hp.embedding_dim).to(self.device)
        self._init_weights()


    def get_closest_ind(self, proto: torch.Tensor) -> torch.Tensor:
        """
        proto: (B, d)
        returns idx: (B,)
        """
        if proto.ndim != 2:
            raise ValueError(f"proto must be (B, d), got {tuple(proto.shape)}")
        if proto.shape[1] != self.hp.embedding_dim:
            raise ValueError(f"proto dim mismatch: expected d={self.hp.embedding_dim}, got {proto.shape[1]}")

        code = self.emb.weight  # (K, d)
        proto2 = (proto ** 2).sum(dim=1, keepdim=True)          # (B, 1)
        code2  = (code ** 2).sum(dim=1).unsqueeze(0)            # (1, K)
        dist = proto2 + code2 - 2.0 * (proto @ code.t())        # (B, K)
        idx = dist.argmin(dim=1)
        return idx

    def get_closest_emb(self, proto: torch.Tensor) -> torch.Tensor:
        idx = self.get_closest_ind(proto)
        return self.emb(idx)  # (B, d)

    def __call__(self, proto: torch.Tensor):
        """
        Quantize proto -> (idx, e, e_st)
          idx : (B,)
          e   : (B, d) selected embedding
          e_st: (B, d) straight-through version
        """
        idx = self.get_closest_ind(proto)
        e = self.emb(idx)  # (B, d)
        e_st = proto + (e - proto).detach()  # STE
        return idx, e, e_st

    def add_row(self, new_emb: Optional[torch.Tensor] = None) -> int:
        """
        Add one new codebook row.
        - If new_emb is provided: uses it (shape (d,) or (1,d))
        - Else: random init using torch RNG (NO reseeding)

        Returns: index of the newly added code.
        """
        d = self.hp.embedding_dim
        old_weight = self.emb.weight.data  # (K, d)
        K_old = old_weight.size(0)
        K_new = K_old + 1

        new_weight = torch.empty((K_new, d), device=self.device, dtype=old_weight.dtype)
        new_weight[:K_old].copy_(old_weight)

        if new_emb is None:
            eps = 1.0 / max(1, K_new)
            new_vec = torch.empty((d,), device=self.device, dtype=old_weight.dtype)
            nn.init.uniform_(new_vec, -eps, eps)  # uses global torch RNG state
        else:
            if new_emb.dim() == 2 and new_emb.size(0) == 1:
                new_emb = new_emb.squeeze(0)
            if new_emb.dim() != 1 or new_emb.numel() != d:
                raise ValueError(f"new_emb must be shape (d,) or (1,d) with d={d}, got {tuple(new_emb.shape)}")
            new_vec = new_emb.to(device=self.device, dtype=old_weight.dtype)

        new_weight[K_old].copy_(new_vec)

        # Replace embedding module (note: optimizer must be updated if you optimize codebook params)
        self.emb = nn.Embedding(K_new, d).to(self.device)
        with torch.no_grad():
            self.emb.weight.copy_(new_weight)

        self.num_codes = K_new
        return K_old





@register_agent
class VQOptionCriticAgent(BaseAgent):
    name = "VQOptionCritic"
    SUPPORTED_ACTION_SPACES = (Discrete, Box)
    
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, init_option_lst=None, device='cpu'):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device=device)
        self.options_lst = [] if init_option_lst is None else init_option_lst
        
        self.code_book = CodeBook(hyper_params, len(self.options_lst), device)
        hl_action_space = Box(
            low=self.hp.embedding_low,
            high=self.hp.embedding_high,
            shape=(self.hp.embedding_dim,),
            dtype=np.float32
        )
        self.hl_policy = HighLevelPolicy(hl_action_space, self.feature_extractor.features_dict, hyper_params.hl, device)
        self.rollout_buffer = [BaseBuffer(self.hp.hl.rollout_steps) for _ in range(self.num_envs)]
        
        self.running_option_index = [None for _ in range(self.num_envs)]      
        self.running_option_emb = [None for _ in range(self.num_envs)]      
        self.running_option_proto_emb = [None for _ in range(self.num_envs)]
 
        self.option_start_obs = [None for _ in range(self.num_envs)]        
        self.option_cumulative_reward = [0.0 for _ in range(self.num_envs)]    
        self.option_multiplier = [1.0 for _ in range(self.num_envs)]           
        self.option_num_steps = [0 for _ in range(self.num_envs)]
        self.option_log_prob = [None for _ in range(self.num_envs)]
        
        
        self.tmp_counter = 0
        self.tmp_flag = False
        self.log_lst = []
    
    def log(self):
        logs = self.log_lst
        self.log_lst = []
        return logs
    
    def act(self, observation, greedy=False):
        state = self.feature_extractor(observation)
        
        # 1) Determine which envs need a new option
        need_new = torch.zeros(self.num_envs, dtype=torch.bool)
        for i in range(self.num_envs):
            # start if none
            if self.running_option_index[i] is None:
                need_new[i] = True
        
        # 2) Batch high-level policy only for those envs
        if need_new.any():
            need_new = need_new.nonzero(as_tuple=False).squeeze(-1)
            with torch.no_grad():
                needed_state = get_batch_state(state, need_new)
                proto_e, proto_log_prob = self.hl_policy.select_action(needed_state, greedy=greedy)
                idx, e, _ = self.code_book(torch.from_numpy(proto_e))
                
                # add the new ones to the lists
                for j, env_i in enumerate(need_new.tolist()):                    
                    self.running_option_index[env_i] = int(idx[j].item())
                    self.running_option_emb[env_i] = e[j].detach()
                    self.running_option_proto_emb[env_i] = proto_e[j]
                    
                    self.option_start_obs[env_i] = get_single_observation(observation, env_i)
                    self.option_log_prob[env_i] = proto_log_prob[j]
                    self.option_cumulative_reward[env_i] = 0.0
                    self.option_multiplier[env_i] = 1.0
                    self.option_num_steps[env_i] = 0
                    
         
        # 3) Batch low-level policy for all envs
        with torch.no_grad():
            action = []
            for i in range(self.num_envs):
                obs_option = get_single_observation_nobatch(observation, i)
                curr_option_idx = self.running_option_index[i]                
                a = self.options_lst[curr_option_idx].select_action(obs_option)        
                action.append(a)
        
        return action

    def update(self, observation, reward, terminated, truncated, call_back=None):
        call_back({"tmp_counter": self.tmp_counter})
        if terminated[0]:
            # reached goal
            self.tmp_counter += 1
        # if not self.tmp_flag and self.tmp_counter >= 3000:
        #     self.tmp_flag = True
        #     self.code_book.add_row()
        #     self.options_lst.append(GoToBlueGoalOption())
        #     print("[Info] VQOptionCriticAgent: Added new codebook entry. Total codes:", self.code_book.num_codes)
        call_back({"num_codes": self.code_book.num_codes})
        
        self.update_hl(observation, reward, terminated, truncated, call_back=call_back)
        
    def update_hl(self, observation, reward, terminated, truncated, call_back=None):
        for i in range(self.num_envs):
            # add to the rollouts
            obs = get_single_observation(observation, i)
            obs_option = get_single_observation_nobatch(observation, i)
            curr_option_idx = self.running_option_index[i]
            
            self.option_cumulative_reward[i] += self.option_multiplier[i] * float(reward[i])
            self.option_multiplier[i] *= self.hp.hl.gamma
            self.option_num_steps[i] += 1
            if self.options_lst[curr_option_idx].is_terminated(obs_option) or terminated[i] or truncated[i]:
                self.log_lst.append({
                    "proto_e": self.running_option_proto_emb[i],
                    "e": self.running_option_emb[i],
                    "ind": self.running_option_index[i],
                })
                call_back({"option": curr_option_idx})
                
                transition = (
                    self.option_start_obs[i], 
                    self.option_log_prob[i],
                    curr_option_idx,
                    self.running_option_emb[i], 
                    self.running_option_proto_emb[i],
                    self.option_cumulative_reward[i], 
                    self.option_multiplier[i], 
                    obs, 
                    terminated[i],
                    truncated[i]
                )    
                self.options_lst[curr_option_idx].reset()
                self.running_option_index[i] = None
                self.rollout_buffer[i].add(transition)
            
            # update with the rollouts
            if self.rollout_buffer[i].is_full():
                rollout = self.rollout_buffer[i].all() 
                (
                    rollout_observations,
                    rollout_log_probs,
                    rollout_option_idx,
                    rollout_option_emb,
                    rollout_option_proto_emb,
                    rollout_rewards,
                    rollout_discounts,
                    rollout_next_observations,
                    rollout_terminated,
                    rollout_truncated,
                ) = zip(*rollout)
                
                rollout_observations, rollout_next_observations = (
                    stack_observations(rollout_observations), 
                    stack_observations(rollout_next_observations)
                )
                rollout_states, rollout_next_states = (
                    self.feature_extractor(rollout_observations), 
                    self.feature_extractor(rollout_next_observations)
                )
                
                self.hl_policy.update(rollout_states, 
                                   rollout_option_proto_emb, 
                                   rollout_log_probs, 
                                   rollout_next_states, 
                                   rollout_rewards,
                                   rollout_discounts, 
                                   rollout_terminated, 
                                   rollout_truncated, 
                                   call_back=call_back)
                
                self.rollout_buffer[i].clear()
            
    def reset(self, seed):
        super().reset(seed)
        self.code_book.reset(seed)
        self.hl_policy.reset(seed)
        
        self.running_option_index = [None for _ in range(self.num_envs)]      
        self.running_option_emb = [None for _ in range(self.num_envs)]      
        self.running_option_proto_emb = [None for _ in range(self.num_envs)]
 
        self.option_start_obs = [None for _ in range(self.num_envs)]        
        self.option_cumulative_reward = [0.0 for _ in range(self.num_envs)]    
        self.option_multiplier = [1.0 for _ in range(self.num_envs)]           
        self.option_num_steps = [0 for _ in range(self.num_envs)]
        self.option_log_prob = [None for _ in range(self.num_envs)]

         