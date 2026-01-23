import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical, Normal, Independent, TransformedDistribution
import gymnasium
from gymnasium.spaces import Discrete, Box
from typing import Optional, Union
import copy

from ....utils import RandomGenerator
from ...Base import BaseAgent, BasePolicy
from ..PolicyGradient import OptionPPOPolicy
from ....Buffers import BaseBuffer, ReplayBuffer
from ....Options import load_options_list, save_options_list
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

def _to_torch(x, device, dtype=torch.float32):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, device=device, dtype=dtype)
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.array(x), device=device, dtype=dtype)

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
    
@register_policy
class HighLevelPolicy(OptionPPOPolicy):
     def update(self, states, proto_emb, emb, old_log_probs, next_states, rewards, discounts, terminated, truncated, call_back=None):
        self.update_counter += 1
            
        # LR annealing (optional)
        if self.hp.enable_stepsize_anneal:
            frac = 1.0 - (self.update_counter - 1.0) / float(self.hp.total_updates)  # linear from initial->0
            for param_groups in self.actor_optimizer.param_groups:
                param_groups["lr"] = frac * self.hp.actor_step_size
            for param_groups in self.critic_optimizer.param_groups:
                param_groups["lr"] = frac * self.hp.critic_step_size    
                
        if self.hp.anneal_clip_range_actor:
            frac = 1.0 - (self.update_counter - 1.0) / float(self.hp.total_updates) 
            self.hp.update(clip_range_actor = float(frac * self.hp.clip_range_actor_init))
        else:
            self.hp.update(clip_range_actor = self.hp.clip_range_actor_init)
        
        if self.hp.anneal_clip_range_critic:
            frac = 1.0 - (self.update_counter - 1.0) / float(self.hp.total_updates) 
            self.hp.update(clip_range_critic = float(frac * self.hp.clip_range_critic_init))
        else:
            self.hp.update(clip_range_critic = self.hp.clip_range_critic_init)
            
        if isinstance(self.action_space, Discrete):
            proto_emb_t = torch.tensor(np.array(proto_emb), dtype=torch.int64, device=self.device)  # (T,)
            emb_t = torch.tensor(np.array(emb), dtype=torch.int64, device=self.device)  # (T,)
        elif isinstance(self.action_space, Box):
            proto_emb_t = torch.tensor(np.array(proto_emb), dtype=torch.float32, device=self.device)  # (T, A)
            emb_t = torch.tensor(np.array(emb), dtype=torch.float32, device=self.device)  # (T, A)
        
        log_probs_old_t = torch.tensor(np.array(old_log_probs), dtype=torch.float32, device=self.device)  # (T,)

        
        with torch.no_grad():
            values = self.critic(**states).squeeze(-1)               # (T, )
            next_values = self.critic(**next_states).squeeze(-1) # (T, )
                    
        returns, advantages = calculate_gae_with_discounts(
            rewards,
            values.detach(),
            next_values.detach(),
            terminated,
            truncated,
            discounts,
            lamda=self.hp.lamda,
        )
        
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device) # (T, )
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device) # (T, )

        if self.hp.enable_advantage_normalization and advantages_t.numel() > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        assert advantages_t.shape[0] == returns_t.shape[0] == log_probs_old_t.shape[0] == \
               proto_emb_t.shape[0] == values.shape[0] == next_values.shape[0] == emb_t.shape[0]

        datasize = advantages_t.shape[0] #self.hp.rollout_steps
        indices = np.arange(datasize)
        continue_training = True
        
        # for logging
        entropy_losses, actor_losses, critic_losses, \
        commit_losses, clip_fractions, losses, approx_kl_divs = [], [], [], [], [], [], []
        actor_grad_norms, critic_grad_norms = [], []

        for epoch in range(self.hp.num_epochs):
            if not continue_training:
                break
            
            indices = self._rand_permutation(datasize)
            for start in range(0, datasize, self.hp.mini_batch_size):
                batch_indices = indices[start:start + self.hp.mini_batch_size]
                
                batch_states     = get_batch_features(states, batch_indices) # shape (B, obs_dim)
                batch_proto_emb_t  = proto_emb_t[batch_indices] # shape (B, action_dim)
                batch_emb_t        = emb_t[batch_indices] # shape (B, action_dim)
                batch_log_probs_t  = log_probs_old_t[batch_indices] # shape (B, )
                batch_advantages_t = advantages_t[batch_indices]     # shape (B, )
                batch_returns_t    = returns_t[batch_indices]        # shape (B, )
                batch_values_t     = values[batch_indices].squeeze() # shape (B, )
                    
                # new log-probs under current policy
                _, batch_log_probs_new_t, entropy, batch_new_proto_emb_mean = self.get_logprob_entropy(batch_states, batch_proto_emb_t)
            
                log_ratio = batch_log_probs_new_t - batch_log_probs_t
                ratios = torch.exp(log_ratio)  # [B, ]
                surr1 = - batch_advantages_t * ratios
                surr2 = - batch_advantages_t * torch.clamp(ratios, 1 - self.hp.clip_range_actor, 1 + self.hp.clip_range_actor)
                actor_loss = torch.max(surr1, surr2).mean()
                actor_losses.append(actor_loss.item())
                
                if call_back is not None:
                    #logging
                    clip_fraction = torch.mean((torch.abs(ratios - 1) > self.hp.clip_range_actor).float()).item()
                    clip_fractions.append(clip_fraction)
                
                # critic loss 
                batch_new_values_t = self.critic(**batch_states).squeeze()
                
                if self.hp.clip_range_critic is None:
                    values_pred = batch_new_values_t
                else:
                    values_pred = batch_values_t + torch.clamp(batch_new_values_t - batch_values_t,
                                                           -self.hp.clip_range_critic, self.hp.clip_range_critic)
                
                critic_loss = F.mse_loss(batch_returns_t, values_pred)
                critic_losses.append(critic_loss.item())
                
                entropy_bonus = entropy.mean()    
                entropy_losses.append(entropy_bonus.item())
                
                commit_loss = F.mse_loss(batch_emb_t, batch_new_proto_emb_mean)
                commit_losses.append(commit_loss.item())
                
                loss = actor_loss + self.hp.critic_coef * critic_loss - self.hp.entropy_coef * entropy_bonus + self.hp.commit_coef * commit_loss
                losses.append(loss.item())
                
                # early stopping
                with torch.no_grad():
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item()
                    approx_kl_divs.append(approx_kl)
                
                if self.hp.target_kl is not None and approx_kl > 1.5 * self.hp.target_kl:
                    continue_training = False
                    break
                    
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                if call_back is not None:
                    actor_grad_norms.append(grad_norm(self.actor.parameters()))
                    critic_grad_norms.append(grad_norm(self.critic.parameters()))
        
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.hp.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.hp.max_grad_norm)
                if self.actor_logstd is not None:
                    nn.utils.clip_grad_norm_([self.actor_logstd], self.hp.max_grad_norm)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
        if call_back is not None:
            payload = {
                "hl_critic_loss": float(np.mean(critic_losses)),
                "hl_actor_loss": float(np.mean(actor_losses)),
                "hl_entropy_loss": float(np.mean(entropy_losses)),
                "hl_commit_loss":float(np.mean(commit_losses)),
                "hl_loss": float(np.mean(losses)),
                        
                "hl_clip_fraction": float(np.mean(clip_fractions)),
                "hl_approx_kl": float(np.mean(approx_kl_divs)),
                "hl_explained_variance": explained_variance(values, returns_t),
                
                "hl_advantage_mean": float(advantages_t.mean().item()),
                "hl_advantage_std": float(advantages_t.std().item()),
                
                "hl_values_mean": float(values.mean().item()),
                "hl_values_std": float(values.std().item()),
                "hl_returns_mean": float(returns_t.mean().item()),
                "hl_returns_std": float(returns_t.std().item()),
                
                "hl_lr_actor": float(self.actor_optimizer.param_groups[0]["lr"]),
                "hl_lr_critic": float(self.critic_optimizer.param_groups[0]["lr"]), 
                
                "hl_actor_grad_norms": float(np.mean(actor_grad_norms)),
                "hl_critic_grad_norms": float(np.mean(critic_grad_norms)),
            }
            if isinstance(self.action_space, Box) and self.actor_logstd is not None:
                payload["actor_logstd_mean"] = float(self.actor_logstd.mean().item())
                payload["actor_logstd_std"] = float(self.actor_logstd.std().item())
            
            call_back(payload)

@register_policy
class LowLevelPolicy(BasePolicy):
    def __init__(self, action_space, hyper_params, device):
        super().__init__(action_space, hyper_params, device)
    
    def select_action(self, x, emb, greedy):
        return [0] * len(x)

class CodeBook(RandomGenerator):
    def __init__(self, hyper_params, num_initial_codes, device, init_embs=None):
        self.num_codes = num_initial_codes
        self.init_embs = init_embs
        
        self.device = device
        self.hp = hyper_params
        
        self.emb = nn.Embedding(self.num_codes, self.hp.embedding_dim).to(self.device)
        self.optimizer = optim.Adam(self.emb.parameters(), lr=self.hp.step_size, eps=self.hp.eps)
    
    def _init_weights(self):
        if self.init_embs is None:
            with torch.no_grad():
                nn.init.uniform_(self.emb.weight, -self.hp.init_emb_range, self.hp.init_emb_range)
        else:
            init = torch.as_tensor(self.init_embs, 
                                   dtype=self.emb.weight.dtype, 
                                   device=self.emb.weight.device)

            assert init.shape == self.emb.weight.shape, \
                f"init_embs shape {init.shape} != emb.weight {self.emb.weight.shape}"
            
            with torch.no_grad():
                self.emb.weight.copy_(init)
            
    def reset(self, seed):
        self.set_seed(seed)
        
        self.emb = nn.Embedding(self.num_codes, self.hp.embedding_dim).to(self.device)
        self._init_weights()
        self.optimizer = optim.Adam(self.emb.parameters(), lr=self.hp.step_size, eps=self.hp.eps)
    
    def update(self, proto_e, call_back=None) -> float:
        proto_e = torch.as_tensor(np.array(proto_e), device=self.device, dtype=torch.float32)  # (T,d)
        if proto_e.numel() == 0:
            if call_back is not None:
                call_back({"cb_loss": 0.0, "cb_batch_T": 0})
            return 0.0

        # Assign nearest codes (non-diff)
        with torch.no_grad():
            idx = self.get_closest_ind(proto_e)  # (T,)

        # Optional: snapshot weights (debug only)
        w_before = self.emb.weight.detach().clone() if call_back is not None else None

        e = self.emb(idx)  # (T,d)
        loss_cb = F.mse_loss(e, proto_e.detach())

        self.optimizer.zero_grad(set_to_none=True)
        loss_cb.backward()

        if call_back is not None:
            grad_n = float(nn.utils.clip_grad_norm_(self.emb.parameters(), self.hp.max_grad_norm).item())
        else:
            nn.utils.clip_grad_norm_(self.emb.parameters(), self.hp.max_grad_norm)
            grad_n = None

        self.optimizer.step()

        if call_back is not None:
            with torch.no_grad():
                idx_int = idx.to(torch.int64)
                K = self.emb.num_embeddings
                counts = torch.bincount(idx_int, minlength=K).float()
                used = int((counts > 0).sum().item())
                frac_used = float(used / max(1, K))

                p = counts / counts.sum().clamp_min(1.0)
                entropy = float(-(p[p > 0] * torch.log(p[p > 0])).sum().item())

                dist = (e.detach() - proto_e.detach()).norm(dim=-1)  # (T,)
                mean_dist = float(dist.mean().item())
                max_dist = float(dist.max().item())

                step_mag = float((self.emb.weight.detach() - w_before).norm(dim=-1).mean().item()) if w_before is not None else None

            call_back({
                "cb_loss": float(loss_cb.item()),
                "cb_grad_norm": grad_n,
                "cb_used_codes": used,
                "cb_frac_used": frac_used,
                "cb_usage_entropy": entropy,
                "cb_assign_mean_l2": mean_dist,
                "cb_assign_max_l2": max_dist,
                "cb_mean_code_step": step_mag,
                "cb_num_codes": int(K),
                "cb_batch_T": int(proto_e.shape[0]),
            })

        return float(loss_cb.item())
    
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
            new_vec = torch.empty((d,), device=self.device, dtype=old_weight.dtype)
            nn.init.uniform_(new_vec, -self.hp.init_emb_range, self.hp.init_emb_range)  # uses global torch RNG state
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
        
        self.optimizer = optim.Adam(self.emb.parameters(), lr=self.hp.step_size, eps=self.hp.eps)
        return K_old


    def save(self, file_path: str | None = None):
        """
        Save codebook weights + optimizer + RNG state + hp.
        """
        checkpoint = {
            "class": self.__class__.__name__,
            "hyper_params": self.hp,
            "num_codes": int(self.num_codes),
            "device": self.device,
            "rng_state": self.get_rng_state(),
            "emb_state_dict": self.emb.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_codebook.t")
        return checkpoint

    @classmethod
    def load(cls, file_path: str | None = None, checkpoint=None, map_location="cpu"):
        """
        Reconstruct CodeBook from checkpoint (file or dict).
        """
        if checkpoint is None:
            assert file_path is not None, "Need file_path or checkpoint"
            checkpoint = torch.load(file_path, map_location=map_location, weights_only=False)

        instance = cls(
            hyper_params=checkpoint["hyper_params"],
            num_initial_codes=int(checkpoint["num_codes"]),
            device=checkpoint["device"],
        )

        # restore RNG
        instance.set_rng_state(checkpoint["rng_state"])

        # restore params
        instance.emb.load_state_dict(checkpoint["emb_state_dict"])
        instance.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # keep num_codes in sync
        instance.num_codes = int(checkpoint["num_codes"])
        return instance


@register_agent
class VQOptionCriticAgent(BaseAgent):
    """
    Hyper-parmas:
        "gamma": 0.99,
        "hl_lamda": 0.95,
        "hl_rollout_steps": 2048,
        "hl_mini_batch_size": 64,
        "hl_num_epochs": 10,
        "hl_target_kl": None,

        "hl_actor_network": "MiniGrid/PPO/conv_imgdir_actor",
        "hl_actor_step_size": 3e-4,
        "hl_actor_eps": 1e-8,
        "hl_clip_range_actor_init": 0.2,
        "hl_anneal_clip_range_actor": False,

        "hl_critic_network": "MiniGrid/PPO/conv_imgdir_critic",
        "hl_critic_step_size": 3e-4,
        "hl_critic_eps": 1e-8,
        "hl_clip_range_critic_init": 0.2,
        "hl_anneal_clip_range_critic": False,

        "hl_critic_coef": 0.5,
        "hl_entropy_coef": 0.0,
        "hl_max_grad_norm": 0.5,

        "hl_min_logstd": None,
        "hl_max_logstd": None,

        "hl_enable_stepsize_anneal": False,
        "hl_total_steps": 200_000,
        "hl_enable_advantage_normalization": True,
        "hl_enable_transform_action": True,

        "commit_coef": 0.2,
        ----
        "codebook_embedding_dim": 2,
        "codebook_embedding_low": -1,
        "codebook_embedding_high": 1,

        "codebook_step_size": 3e-4,
        "codebook_eps": 1e-5,
        "codebook_max_grad_norm": 1.0,
        
    """
    name = "VQOptionCritic"
    SUPPORTED_ACTION_SPACES = (Discrete, Box)
    
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, 
                 init_option_lst=None, init_option_embs=None, device='cpu'):
        print("[Info] VQOptionCriticAgent: Total Initial Options: ", len(init_option_lst) if init_option_lst is not None else 0)
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device=device)
        self.options_lst = [] if init_option_lst is None else init_option_lst
                
        self.code_book = CodeBook(hyper_params.codebook, len(self.options_lst), device, init_embs=init_option_embs)
        hl_action_space = Box(
            low=self.hp.codebook.embedding_low,
            high=self.hp.codebook.embedding_high,
            shape=(self.hp.codebook.embedding_dim,),
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
        
        self.option_learner_tmp = {}
        
    
    def _init_log_buf(self):
        # one buffer per env slot, to avoid mixing logs between envs
        self.log_buf = []
        for _ in range(self.num_envs):
            self.log_buf.append({
                "proto_e": [],       # list of (d,) arrays
                "e": [],             # list of (d,) arrays
                "num_options": [],   # list of ints
                "option_index": [],  # list of ints
            })

    def act(self, observation):
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
                proto_e, proto_log_prob = self.hl_policy.select_action(needed_state, greedy=not self.training)
                proto_e_t = torch.as_tensor(proto_e, device=self.device, dtype=torch.float32)
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
        if self.training:
            self.update_hl(observation, reward, terminated, truncated, call_back=call_back)
            for i in range(self.num_envs):
                if terminated[i] or truncated[i]:
                    obs_option = get_single_observation_nobatch(observation, i)
                    for c, opt in enumerate(self.hp.all_options):
                        if hasattr(opt, "should_initiate") and \
                        opt.should_initiate(obs_option) and \
                        opt not in self.options_lst:
                            if opt.option_id not in self.option_learner_tmp:
                                self.option_learner_tmp[opt.option_id] = 0
                            self.option_learner_tmp[opt.option_id] += 1

                            if self.option_learner_tmp[opt.option_id] >= self.hp.count_to_add:
                                print(f"Added option with id: {opt.option_id}")
                                self.options_lst.append(opt)
                                self.code_book.add_row(torch.from_numpy(self.hp.all_embeddings[c]))
                    
        else:
            for i in range(self.num_envs):
                # add to the rollouts
                obs = get_single_observation(observation, i)
                obs_option = get_single_observation_nobatch(observation, i)
                curr_option_idx = self.running_option_index[i]
                
                self.option_cumulative_reward[i] += self.option_multiplier[i] * float(reward[i])
                self.option_multiplier[i] *= self.hp.hl.gamma
                self.option_num_steps[i] += 1
                if self.options_lst[curr_option_idx].is_terminated(obs_option) or terminated[i] or truncated[i]:
                    self.log_buf[i]["proto_e"].append(np.asarray(self.running_option_proto_emb[i], dtype=np.float32))
                    self.log_buf[i]["e"].append(self.running_option_emb[i].cpu().numpy())
                    self.log_buf[i]["num_options"].append(np.array([len(self.options_lst)]))
                    self.log_buf[i]["option_index"].append(np.array([curr_option_idx]))
                    if call_back is not None:
                        call_back({"curr_hl_option_idx": curr_option_idx})
                    
                    self.options_lst[curr_option_idx].reset()
                    self.running_option_index[i] = None
            

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
                
                self.log_buf[i]["proto_e"].append(np.asarray(self.running_option_proto_emb[i], dtype=np.float32))
                self.log_buf[i]["e"].append(self.running_option_emb[i].cpu().numpy())
                self.log_buf[i]["num_options"].append(np.array([len(self.options_lst)]))
                self.log_buf[i]["option_index"].append(np.array([curr_option_idx]))

                call_back({"curr_hl_option_idx": curr_option_idx,
                           "num_options": len(self.options_lst)})
                
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
                                   rollout_option_emb,
                                   rollout_log_probs, 
                                   rollout_next_states, 
                                   rollout_rewards,
                                   rollout_discounts, 
                                   rollout_terminated, 
                                   rollout_truncated, 
                                   call_back=call_back)
                # update codebook
                self.code_book.update(rollout_option_proto_emb,
                                      call_back=call_back)
                
                # self.code_book.fake_update(rollout_option_proto_emb,
                #                       direction=None,
                #                       step_size=0.1,
                #                       call_back=call_back)
                
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
        
        self.option_learner_tmp = {}

    
    def save(self, file_path: str | None = None):
        """
        Save EVERYTHING needed to resume:
        - feature_extractor (already has save)
        - hl_policy (PPOPolicy.save exists)
        - code_book (new CodeBook.save)
        - options list
        - rng states
        - key runtime bookkeeping (optional)
        """
        checkpoint = super().save(file_path=None)


        checkpoint['options_lst'] = save_options_list(self.options_lst, file_path=None)
        checkpoint['hl_policy'] = self.hl_policy.save(file_path=None)
        checkpoint['code_book'] = self.code_book.save(file_path=None)

        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_agent.t")
        return checkpoint

    @classmethod
    def load(cls, file_path: str, checkpoint=None):
        """
        Load full VQ agent checkpoint.
        NOTE: we reconstruct with init_option_lst loaded from checkpoint, so
        codebook size matches options length.
        """
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)


        # 1) load options first (so we can size codebook consistently if needed)
        options_lst = load_options_list(file_path=None, checkpoint=checkpoint["options_lst"])

        # 2) construct instance
        instance = cls(
            action_space=checkpoint["action_space"],
            observation_space=checkpoint["observation_space"],
            hyper_params=checkpoint["hyper_params"],
            num_envs=int(checkpoint["num_envs"]),
            feature_extractor_class=checkpoint["feature_extractor_class"],
            init_option_lst=options_lst,
            device=checkpoint["device"],
        )

        # 3) restore agent RNG
        instance.set_rng_state(checkpoint["rng_state"])

        # 4) restore feature extractor
        instance.feature_extractor = instance.feature_extractor.load(
            file_path=None,
            checkpoint=checkpoint["feature_extractor"],
        )

        # 5) restore high-level policy
        instance.hl_policy = instance.hl_policy.load(
            file_path=None,
            checkpoint=checkpoint["hl_policy"],
        )

        # 6) restore codebook (weights + optimizer + rng)
        instance.code_book = instance.code_book.load(
            file_path=None,
            checkpoint=checkpoint["code_book"],
        )

        # Ensure options list is set (already used in __init__, but keep explicit)
        instance.options_lst = options_lst

        return instance