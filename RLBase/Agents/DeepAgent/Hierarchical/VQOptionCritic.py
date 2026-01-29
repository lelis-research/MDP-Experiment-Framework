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

@register_policy
class LowLevelPolicy(BasePolicy):
    def __init__(self, action_space, hyper_params, device):
        super().__init__(action_space, hyper_params, device)
    
    def select_action(self, x, emb, greedy):
        return [0] * len(x)


class Encoder(RandomGenerator):
    def __init__(self, hyper_params, features_dict, device):
        self.features_dict = features_dict
        self.hp = hyper_params
        self.device = device
        
        encoder_discription = prepare_network_config(
            self.hp.enc_network,
            input_dims=self.features_dict,
            output_dim=self.hp.enc_dim,
        )
        self.encoder = NetworkGen(layer_descriptions=encoder_discription).to(self.device)
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=self.hp.step_size, eps=self.hp.eps)
    
    def __call__(self, state):
        x = self.encoder(**state)
        return {"x": x}
    
    @property
    def encoder_features_dict(self):
        return {"x": self.hp.enc_dim}
    
    def reset(self, seed):
        self.set_seed(seed)
        
        encoder_discription = prepare_network_config(
            self.hp.enc_network,
            input_dims=self.features_dict,
            output_dim=self.hp.enc_dim,
        )
        self.encoder = NetworkGen(layer_descriptions=encoder_discription).to(self.device)
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=self.hp.step_size, eps=self.hp.eps)
    
    def save(self, file_path: str | None = None):
        checkpoint = {
            "class": self.__class__.__name__,
            "hyper_params": self.hp,
            "features_dict": self.features_dict,
            "rng_state": self.get_rng_state(),
            "encoder_state_dict": self.encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "device": self.device,
        }
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_encoder.t")
        return checkpoint

    @classmethod
    def load(cls, file_path: str | None = None, checkpoint=None, map_location="cpu"):
        if checkpoint is None:
            assert file_path is not None
            checkpoint = torch.load(file_path, map_location=map_location, weights_only=False)

        inst = cls(
            hyper_params=checkpoint["hyper_params"],
            features_dict=checkpoint["features_dict"],
            device=checkpoint["device"],
        )
        inst.set_rng_state(checkpoint["rng_state"])
        inst.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        inst.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return inst
    
@register_policy
class HighLevelPolicy(OptionPPOPolicy):
    def __init__(self, action_space, features_dict, hyper_params, device="cpu"):
        super().__init__(action_space, features_dict, hyper_params, device)
        self.reheat()

    def reset(self, seed):
        self.reheat()
        return super().reset(seed)

    def reheat(self):
        self.tau_counter = 0
        self.hp.update(tau=self.hp.tau_init)
        
    def get_logprob_entropy(self, state, action_t=None):
        """
        Returns:
            action_t: sampled or given action
              - Discrete: shape [B]  (dtype long)
              - Continuous: shape [B, action_dim]
            log_prob: shape [B]
            entropy: shape [B]
        """
        
        if self.hp.distribution_type == "categorical":
            # actor outputs mu (B,d)
            mu = self.actor(**state)
            
            logits = self.code_book.get_logits(mu, tau=self.hp.tau, detach_code=True)

            dist = Categorical(logits=logits)

            # action is now an index k
            if action_t is None:
                action_t = dist.sample()          # (B,) long
            else:
                action_t = action_t.to(device=logits.device, dtype=torch.long)

            log_prob = dist.log_prob(action_t)    # (B,)
            entropy  = dist.entropy()             # (B,)
            greedy_action = logits.argmax(dim=-1) # (B,)
            
            raw_action = mu  # for future commitment losses
            
            
        elif self.hp.distribution_type == "continuous":
            action_t, log_prob, entropy, greedy_action = super().get_logprob_entropy(state, action_t)
            raw_action = greedy_action # for future commitment losses
        
         
        return action_t, log_prob, entropy, greedy_action, raw_action
    
    def select_action(self, state, greedy=False):
        """
        Returns a numpy action of shape [action_dim].
        """
        
        with torch.no_grad():
            encoded_state = self.encoder(state)
            proto_e, proto_log_prob, _, greedy_proto_e, proto_raw = self.get_logprob_entropy(encoded_state)
            if greedy:
                proto_e = greedy_proto_e
                
            if self.hp.distribution_type == "continuous":
                idx, e, _ = self.code_book(proto_e)
            elif self.hp.distribution_type == "categorical":
                idx = proto_e                          # (B,) long
                e = self.code_book.emb(idx)             # (B,d)
            
        return idx.cpu().numpy(), e.cpu().numpy(), proto_e.cpu().numpy(), proto_log_prob.cpu().numpy(), proto_raw.cpu().numpy()
                
    def update(self, states, proto_emb, emb, old_log_probs, next_states, rewards, discounts, terminated, truncated, call_back=None):
        self.update_counter += 1
        self.tau_counter += 1
        self.hp.update(
            tau=max(self.hp.tau_min, 
                    self.hp.tau_init * (self.hp.tau_decay ** self.tau_counter))
        )

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
            if self.hp.distribution_type == "continuous":
                proto_emb_t = torch.tensor(np.array(proto_emb), dtype=torch.float32, device=self.device)  # (T, A)
            elif self.hp.distribution_type == "categorical":
                proto_emb_t = torch.tensor(np.array(proto_emb), dtype=torch.int64, device=self.device)  # (T,)
                
            emb_t = torch.tensor(np.array(emb), dtype=torch.float32, device=self.device)  # (T, A)
        
        log_probs_old_t = torch.tensor(np.array(old_log_probs), dtype=torch.float32, device=self.device)  # (T,)
        
        with torch.no_grad():
            values = self.critic(**self.encoder(states)).squeeze(-1)               # (T, )
            next_values = self.critic(**self.encoder(next_states)).squeeze(-1) # (T, )
                    
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
        actor_grad_norms, critic_grad_norms, encoder_grad_norms = [], [], []

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
                batch_encoded_states = self.encoder(batch_states) # forward pass to Encoder
                _, batch_log_probs_new_t, entropy, _, raw_action = self.get_logprob_entropy(batch_encoded_states, batch_proto_emb_t)
            
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
                if self.hp.block_critic_to_encoder:
                    batch_encoded_states_detached = {k: v.detach() for k, v in batch_encoded_states.items()}
                    batch_new_values_t = self.critic(**batch_encoded_states_detached).squeeze()
                else:
                    batch_new_values_t = self.critic(**batch_encoded_states).squeeze()
                
                if self.hp.clip_range_critic is None:
                    values_pred = batch_new_values_t
                else:
                    values_pred = batch_values_t + torch.clamp(batch_new_values_t - batch_values_t,
                                                           -self.hp.clip_range_critic, self.hp.clip_range_critic)
                
                critic_loss = F.mse_loss(batch_returns_t, values_pred)
                critic_losses.append(critic_loss.item())
                
                entropy_bonus = entropy.mean()    
                entropy_losses.append(entropy_bonus.item())
                
                commit_loss = self.code_book.commit_loss(batch_emb_t, raw_action)
                    
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
                self.encoder.optimizer.zero_grad()
                loss.backward()
                if call_back is not None:
                    actor_grad_norms.append(grad_norm(self.actor.parameters()))
                    critic_grad_norms.append(grad_norm(self.critic.parameters()))
                    encoder_grad_norms.append(grad_norm(self.encoder.encoder.parameters()))
        
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.hp.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.hp.max_grad_norm)
                nn.utils.clip_grad_norm_(self.encoder.encoder.parameters(), self.hp.max_grad_norm)
                if self.actor_logstd is not None:
                    nn.utils.clip_grad_norm_([self.actor_logstd], self.hp.max_grad_norm)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                self.encoder.optimizer.step()
                
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
                "encoder_grad_norms": float(np.mean(encoder_grad_norms)),
                
                "hl_tau": float(self.hp.tau),
                "hl_tau_counter": int(self.tau_counter),
            }
            if isinstance(self.action_space, Box) and self.actor_logstd is not None:
                payload["actor_logstd_mean"] = float(self.actor_logstd.mean().item())
                payload["actor_logstd_std"] = float(self.actor_logstd.std().item())
            
            call_back(payload)

    def set_encoder(self, encoder):
        self.encoder = encoder
    
    def set_codebook(self, code_book):
        self.code_book = code_book
        
class CodeBook(RandomGenerator):
    def __init__(self, hyper_params, num_initial_codes, device, init_embs=None):
        self.num_codes = num_initial_codes
        self.init_embs = init_embs
        
        self.device = device
        self.hp = hyper_params
        self.emb = nn.Embedding(self.num_codes, self.hp.embedding_dim).to(self.device)
        
        if self.hp.update_type == "grad":
            self.optimizer = optim.Adam(self.emb.parameters(), lr=self.hp.step_size, eps=self.hp.eps)
        elif self.hp.update_type == "ema":
            self.ema_counts = torch.zeros(self.num_codes, device=self.device, dtype=torch.float32)
            self.ema_sum = torch.zeros(self.num_codes, self.hp.embedding_dim, device=self.device, dtype=torch.float32)
        else:
            raise ValueError(f"[CodeBook] Unknown update_type={self.hp.update_type}")
            
    
    def _init_weights(self):
        if self.init_embs is None:
            with torch.no_grad():
                if "uniform" in self.hp.init_type:
                    nn.init.uniform_(self.emb.weight, -self.hp.init_emb_range, self.hp.init_emb_range)
                elif "onehot" in self.hp.init_type:
                    if self.hp.embedding_dim >= self.num_codes:
                        eye = torch.eye(self.num_codes, self.hp.embedding_dim, 
                                        dtype=self.emb.weight.dtype, 
                                        device=self.device)
                    else:
                        raise ValueError(f"[CodeBook] Cannot onehot init with Number of Codes ={self.num_codes} > embedding dim={self.hp.embedding_dim}")
                    self.emb.weight.copy_(eye)
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
        
        if self.hp.update_type == "grad":
            self.optimizer = optim.Adam(self.emb.parameters(), lr=self.hp.step_size, eps=self.hp.eps)
        elif self.hp.update_type == "ema":
            self.ema_counts = torch.zeros(self.num_codes, device=self.device, dtype=torch.float32)
            self.ema_sum = torch.zeros(self.num_codes, self.hp.embedding_dim, device=self.device, dtype=torch.float32)
    
    def commit_loss(self, e, proto_e):
        if self.hp.similarity_metric == "l2":
            loss = F.mse_loss(e, proto_e)
        elif self.hp.similarity_metric == "cosine":
            e_n = F.normalize(e, dim=-1, eps=1e-8)
            p_n = F.normalize(proto_e, dim=-1, eps=1e-8)
            # 1 - cosine similarity
            loss = (1.0 - (e_n * p_n).sum(dim=-1)).mean()
        elif self.hp.similarity_metric == "dot":
            loss = - (e * proto_e).sum(dim=-1).mean()
        else:
            raise ValueError(f"[CodeBook] similarity metric {self.hp.similarity_metric} not defined for commit_loss")
        return loss
    
    def update(self, proto_e, call_back=None) -> float:
        proto_e = torch.as_tensor(np.array(proto_e), device=self.device, dtype=torch.float32)  # (T,d)
        if proto_e.numel() == 0:
            if call_back is not None:
                call_back({"cb_loss": 0.0, "cb_batch_T": 0})
            return 0.0

        # Assign nearest codes (non-diff)
        with torch.no_grad():
            idx = self.get_closest_ind(proto_e)  # (T,)


        e = self.emb(idx)  # (T,d)
        loss_cb = self.commit_loss(e, proto_e.detach())
        
        if self.hp.update_type == "grad":
            self.optimizer.zero_grad(set_to_none=True)
            loss_cb.backward()

            if call_back is not None:
                grad_n = float(nn.utils.clip_grad_norm_(self.emb.parameters(), self.hp.max_grad_norm).item())
            else:
                nn.utils.clip_grad_norm_(self.emb.parameters(), self.hp.max_grad_norm)
                grad_n = None

            self.optimizer.step()
        
        elif self.hp.update_type == "ema":
            with torch.no_grad():
                K = self.emb.num_embeddings
                idx_int = idx.to(torch.int64)

                # counts per code in this batch
                batch_counts = torch.bincount(idx_int, minlength=K).float()  # (K,)

                # sum of proto_e per code in this batch
                batch_sum = torch.zeros((K, self.hp.embedding_dim), device=self.device, dtype=torch.float32)
                if self.hp.similarity_metric == "cosine":
                    proto_use = F.normalize(proto_e, dim=-1, eps=1e-8)
                elif self.hp.similarity_metric in ("l2", "dot"):
                    proto_use = proto_e
                else:
                    raise ValueError(f"[CodeBook] similarity metric {self.hp.similarity_metric} not defined for ema")
                    
                batch_sum.index_add_(0, idx_int, proto_use)  # (K,d)

                # EMA stats
                self.ema_counts.mul_(self.hp.ema_decay).add_(batch_counts, alpha=(1.0 - self.hp.ema_decay))
                self.ema_sum.mul_(self.hp.ema_decay).add_(batch_sum, alpha=(1.0 - self.hp.ema_decay))

                # Laplace smoothing to avoid division by 0 for dead codes
                n = self.ema_counts.sum()
                smoothed_counts = (self.ema_counts + self.hp.ema_eps) / (n + K * self.hp.ema_eps) * n  # (K,)
                # new_weight = self.ema_sum / self.ema_counts.unsqueeze(1).clamp_min(self.hp.ema_eps)
                
                if self.hp.similarity_metric == "cosine":
                    new_weight = self.ema_sum / smoothed_counts.unsqueeze(1).clamp_min(self.hp.ema_eps)  # (K,d)
                    new_weight = F.normalize(new_weight, dim=-1, eps=1e-8)
                elif self.hp.similarity_metric in ("l2", "dot"):
                    new_weight = self.ema_sum / smoothed_counts.unsqueeze(1).clamp_min(self.hp.ema_eps)  # (K,d)
                
                

                self.emb.weight.data.copy_(new_weight)

            grad_n = 0.0            

            
        with torch.no_grad():
            if self.hp.similarity_metric == "cosine":
                self.emb.weight.copy_(F.normalize(self.emb.weight, dim=-1, eps=1e-8))
            elif self.hp.similarity_metric in ("l2", "dot"):
                self.emb.weight.clamp_(self.hp.embedding_low, self.hp.embedding_high)
    
        if call_back is not None:
            with torch.no_grad():
                idx_int = idx.to(torch.int64)
                K = self.emb.num_embeddings
                counts = torch.bincount(idx_int, minlength=K).float()
                used = int((counts > 0).sum().item())
                frac_used = float(used / max(1, K))

                p = counts / counts.sum().clamp_min(1.0)
                entropy = float(-(p[p > 0] * torch.log(p[p > 0])).sum().item())

                # unified "distance" metric consistent with your training objective
                cb_commit = float(self.commit_loss(e.detach(), proto_e.detach()).item())

            payload = {
                "cb_loss": float(loss_cb.item()),
                "cb_commit": cb_commit,
                "cb_grad_norm": grad_n,
                "cb_used_codes": used,
                "cb_frac_used": frac_used,
                "cb_usage_entropy": entropy,
                "cb_num_codes": int(K),
                "cb_batch_T": int(proto_e.shape[0]),
            }

            call_back(payload)

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
        
        if self.hp.similarity_metric == "l2":
            proto2 = (proto ** 2).sum(dim=1, keepdim=True)          # (B, 1)
            code2  = (code ** 2).sum(dim=1).unsqueeze(0)            # (1, K)
            dist = proto2 + code2 - 2.0 * (proto @ code.t())        # (B, K)
            idx = dist.argmin(dim=-1)
            
        elif self.hp.similarity_metric == "cosine":
            # Normalize both sides
            proto_n = F.normalize(proto, dim=1, eps=1e-8)      # (B, d)
            code_n  = F.normalize(code, dim=1, eps=1e-8)       # (K, d)

            # cosine similarity in [-1, 1]
            sim = proto_n @ code_n.t()                          # (B, K)
            idx = sim.argmax(dim=-1)
        
        elif self.hp.similarity_metric == "dot":
            # maximize dot product
            sim = proto @ code.t()                                   # (B, K)
            idx = sim.argmax(dim=-1)
        
        else:
            raise ValueError(f"[CodeBook] similarity metric {self.hp.similarity_metric} is not defined")
        
        return idx

    def get_closest_emb(self, proto: torch.Tensor) -> torch.Tensor:
        idx = self.get_closest_ind(proto)
        return self.emb(idx)  # (B, d)

    def get_logits(self, proto: torch.Tensor, tau: float = 1.0, detach_code: bool = True) -> torch.Tensor:
        """
        Build categorical logits over K codes given proto vectors.

        proto: (B, d)
        returns logits: (B, K) where larger => more likely

        tau: temperature (>0). smaller => sharper distribution
        detach_code: if True, treat codebook weights as constant for PPO (recommended)
        """
        if proto.ndim != 2:
            raise ValueError(f"proto must be (B, d), got {tuple(proto.shape)}")
        if proto.shape[1] != self.hp.embedding_dim:
            raise ValueError(f"proto dim mismatch: expected d={self.hp.embedding_dim}, got {proto.shape[1]}")

        tau = max(float(tau), 1e-8)

        code = self.emb.weight
        if detach_code:
            code = code.detach()

        if self.hp.similarity_metric == "cosine":
            # logits = cosine_similarity / tau
            proto_n = F.normalize(proto, dim=1, eps=1e-8)   # (B,d)
            code_n  = F.normalize(code,  dim=1, eps=1e-8)   # (K,d)
            sim = proto_n @ code_n.t()                      # (B,K) in [-1,1]
            logits = sim / tau

        elif self.hp.similarity_metric == "l2":
            # logits = -squared_l2_distance / tau
            # dist2 = ||p||^2 + ||c||^2 - 2 p·c
            proto2 = (proto ** 2).sum(dim=1, keepdim=True)      # (B,1)
            code2  = (code  ** 2).sum(dim=1).unsqueeze(0)       # (1,K)
            dist2 = proto2 + code2 - 2.0 * (proto @ code.t())   # (B,K)
            logits = (-dist2) / tau
        
        elif self.hp.similarity_metric == "dot":
            sim = proto @ code.t()                             # (B,K)
            logits = sim / tau

        else:
            raise ValueError(f"[CodeBook] similarity_metric={self.hp.similarity_metric} not defined")

        return logits
    
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
            if "uniform" in self.hp.init_type:
                nn.init.uniform_(new_vec, -self.hp.init_emb_range, self.hp.init_emb_range)  # uses global torch RNG state
            elif "onehot" in self.hp.init_type:
                new_vec.zero_()
                if d >= K_new:
                    new_vec[K_old] = 1.0
                else:
                    raise ValueError(f"[CodeBook] Cannot onehot init new code with embedding dim={d} < Number of Codes ={K_new}")
            else:
                raise ValueError(f"[CodeBook] Unknown init_type={self.hp.init_type} for new code addition")
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
        
        if self.hp.update_type == "grad":
            self.optimizer = optim.Adam(self.emb.parameters(), lr=self.hp.step_size, eps=self.hp.eps)
            
        elif self.hp.update_type == "ema":
            old_counts = self.ema_counts
            old_sum = self.ema_sum

            self.ema_counts = torch.zeros(K_new, device=self.device, dtype=torch.float32)
            self.ema_sum = torch.zeros(K_new, d, device=self.device, dtype=torch.float32)

            self.ema_counts[:K_old].copy_(old_counts)
            self.ema_sum[:K_old].copy_(old_sum)

            # optional: seed new code to avoid tiny-denominator weirdness
            self.ema_counts[K_old] = 1.0
            self.ema_sum[K_old].copy_(self.emb.weight.data[K_old].to(torch.float32))
            
            
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
        }
        if self.hp.update_type == "grad":
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        else:
            checkpoint["ema_counts"] = self.ema_counts.detach().cpu()
            checkpoint["ema_sum"] = self.ema_sum.detach().cpu()
        
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
        instance.num_codes = int(checkpoint["num_codes"])
        
        if instance.hp.update_type == "grad":        
            instance.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        elif instance.hp.update_type == "ema":
            instance.ema_counts.copy_(checkpoint["ema_counts"].to(instance.device))
            instance.ema_sum.copy_(checkpoint["ema_sum"].to(instance.device))
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
        
        self.encoder = Encoder(self.hp.enc, self.feature_extractor.features_dict, device)
        self.code_book = CodeBook(self.hp.codebook, len(self.options_lst), device, init_embs=init_option_embs)
        hl_action_space = Box(
            low=self.hp.codebook.embedding_low,
            high=self.hp.codebook.embedding_high,
            shape=(self.hp.codebook.embedding_dim,),
            dtype=np.float32
        )
        # self.hl_policy = HighLevelPolicy(hl_action_space, self.feature_extractor.features_dict, hyper_params.hl, device)
        self.hl_policy = HighLevelPolicy(hl_action_space, self.encoder.encoder_features_dict, hyper_params.hl, device)
        self.hl_policy.set_encoder(self.encoder)
        self.hl_policy.set_codebook(self.code_book)
        
        self.rollout_buffer = [BaseBuffer(self.hp.hl.rollout_steps) for _ in range(self.num_envs)]
        
        self.running_option_index = [None for _ in range(self.num_envs)]      
        self.running_option_emb = [None for _ in range(self.num_envs)]      
        self.running_option_proto_emb = [None for _ in range(self.num_envs)]
        self.running_option_proto_raw = [None for _ in range(self.num_envs)]
 
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
                idx, e, proto_e, proto_log_prob, proto_raw = self.hl_policy.select_action(needed_state, greedy=not self.training)
                
                # add the new ones to the lists
                for j, env_i in enumerate(need_new.tolist()):                    
                    self.running_option_index[env_i] = int(idx[j])
                    self.running_option_emb[env_i] = e[j]
                    self.running_option_proto_emb[env_i] = proto_e[j]
                    self.running_option_proto_raw[env_i] = proto_raw[j]
                    
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
            
            # add new options
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
                                new_embs = torch.from_numpy(self.hp.all_embeddings[c]) if self.hp.all_embeddings is not None else None
                                self.code_book.add_row(new_embs)
                                self.hl_policy.reheat()
                                if self.hp.option_learner_reset_at_add:
                                    self.option_learner_tmp = {}
                    
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
                    self.log_buf[i]["e"].append(np.asarray(self.running_option_emb[i], dtype=np.float32))
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
                self.log_buf[i]["e"].append(np.asarray(self.running_option_emb[i], dtype=np.float32))
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
                    self.running_option_proto_raw[i],
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
                    rollout_option_proto_raw,
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
                if not ("fixed" in self.code_book.hp.init_type):
                    self.code_book.update(rollout_option_proto_raw,
                                        call_back=call_back)
            
                
                self.rollout_buffer[i].clear()
            
    def reset(self, seed):
        super().reset(seed)
        
        self.code_book.reset(seed)
        self.encoder.reset(seed)
        self.hl_policy.reset(seed)
        self.hl_policy.set_encoder(self.encoder)
        self.hl_policy.set_codebook(self.code_book)
        
        self.running_option_index = [None for _ in range(self.num_envs)]      
        self.running_option_emb = [None for _ in range(self.num_envs)]      
        self.running_option_proto_emb = [None for _ in range(self.num_envs)]
        self.running_option_proto_raw = [None for _ in range(self.num_envs)]
 
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
        checkpoint['encoder'] = self.encoder.save(file_path=None)

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

        # 5) restore encoder
        instance.encoder = instance.encoder.load(
            file_path=None,
            checkpoint=checkpoint["encoder"]
        )
        
        # 5) restore high-level policy
        instance.hl_policy = instance.hl_policy.load(
            file_path=None,
            checkpoint=checkpoint["hl_policy"],
        )
        instance.hl_policy.set_encoder(instance.encoder)
        
        # 6) restore codebook (weights + optimizer + rng)
        instance.code_book = instance.code_book.load(
            file_path=None,
            checkpoint=checkpoint["code_book"],
        )
        instance.hl_policy.set_codebook(instance.code_book)

        # Ensure options list is set (already used in __init__, but keep explicit)
        instance.options_lst = options_lst

        return instance