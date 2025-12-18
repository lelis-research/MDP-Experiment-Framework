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
from ....Buffers import BaseBuffer, ReplayBuffer
from ...Utils import calculate_gae, get_single_observation, stack_observations, grad_norm, explained_variance
from ....registry import register_agent, register_policy
from ....Networks.NetworkFactory import NetworkGen, prepare_network_config
from ....FeatureExtractors import get_batch_features

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
    

class HighLevelPolicy(BasePolicy):
    def  __init__(self, hyper_params, device):
        self.device = device
        self.set_hp(hyper_params)
        
        actor_description = prepare_network_config(
            self.hp.hl_actor_network,
            input_dims={"x": self.hp.encoder_dim},
            output_dim=self.hp.embedding_dim,
        )
        critic_description = prepare_network_config(
            self.hp.hl_critic_network,
            input_dims={"x": self.hp.encoder_dim, "o": self.hp.embedding_dim},
            output_dim=1,
        )
        
        self.actor = NetworkGen(layer_descriptions=actor_description).to(self.device)
        self.actor_target = NetworkGen(layer_descriptions=actor_description).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.hp.hl_actor_step_size, eps=self.hp.hl_actor_eps)
        self.actor_target.load_state_dict(self.actor.state_dict())
        for p in self.actor_target.parameters():
            p.requires_grad_(False)
        
        self.critic1 = NetworkGen(layer_descriptions=critic_description).to(self.device)
        self.critic2 = NetworkGen(layer_descriptions=critic_description).to(self.device)
        self.critic1_target = NetworkGen(layer_descriptions=critic_description).to(self.device)
        self.critic2_target = NetworkGen(layer_descriptions=critic_description).to(self.device)

        # one optimizer for both (simple)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.hp.hl_critic_step_size,
            eps=self.hp.hl_critic_eps,
        )
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        for p in self.critic1_target.parameters(): p.requires_grad_(False)
        for p in self.critic2_target.parameters(): p.requires_grad_(False)
        
        self.num_critic_updates = 0
    
    def reset(self, seed):
        super().reset(seed)
        
        actor_description = prepare_network_config(
            self.hp.hl_actor_network,
            input_dims={"x": self.hp.encoder_dim},
            output_dim=self.hp.embedding_dim,
        )
        critic_description = prepare_network_config(
            self.hp.hl_critic_network,
            input_dims={"x": self.hp.encoder_dim, "o": self.hp.embedding_dim},
            output_dim=1,
        )
        
        self.actor = NetworkGen(layer_descriptions=actor_description).to(self.device)
        self.actor_target = NetworkGen(layer_descriptions=actor_description).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.hp.hl_actor_step_size, eps=self.hp.hl_actor_eps)
        self.actor_target.load_state_dict(self.actor.state_dict())
        for p in self.actor_target.parameters():
            p.requires_grad_(False)
        
        self.critic1 = NetworkGen(layer_descriptions=critic_description).to(self.device)
        self.critic2 = NetworkGen(layer_descriptions=critic_description).to(self.device)
        self.critic1_target = NetworkGen(layer_descriptions=critic_description).to(self.device)
        self.critic2_target = NetworkGen(layer_descriptions=critic_description).to(self.device)

        # one optimizer for both (simple)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.hp.hl_critic_step_size,
            eps=self.hp.hl_critic_eps,
        )
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        for p in self.critic1_target.parameters(): p.requires_grad_(False)
        for p in self.critic2_target.parameters(): p.requires_grad_(False)
        
        self.num_critic_updates = 0
    
    def select_action(self, x, greedy: bool):
        with torch.no_grad():
            proto_e = self.actor(x)

        if greedy:
            return proto_e

        noise = torch.randn_like(proto_e) * self.hp.hl_exploration_noise_sigma
        proto_e_noisy = proto_e + noise

        return proto_e_noisy

    def update(self, state, proto_option_emb, returns, discounts, option_len, next_state, terminated, encoder, call_back=None):
        x = encoder(state)
        with torch.no_grad():
            next_x = encoder(next_state)
        
        proto_option_emb = torch.stack(proto_option_emb).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        discounts = torch.tensor(discounts, dtype=torch.float32).to(self.device)
        option_len = torch.tensor(option_len, dtype=torch.float32).to(self.device)
        terminated = torch.tensor(terminated, dtype=torch.float32).to(self.device)
        
        
        # ---------- TD3 critic target ----------
        with torch.no_grad():
            a2 = self.actor_target(next_x)
            
            noise = torch.randn_like(a2) * float(self.hp.hl_target_policy_noise)
            noise = noise.clamp(-float(self.hp.hl_target_noise_clip), float(self.hp.hl_target_noise_clip))
            a2 = a2 + noise
            
            q1_t = self.critic1_target(x=next_x, o=a2).squeeze(-1)  # (B,)
            q2_t = self.critic2_target(x=next_x, o=a2).squeeze(-1)  # (B,)
            q_t = torch.minimum(q1_t, q2_t)                         # (B,)
            
            y = returns + (1.0 - terminated) * discounts * q_t                       # (B,)
            
        
        # ---------- critic update (both critics) ----------
        q1 = self.critic1(x=x, o=proto_option_emb).squeeze(-1)  # (B,)
        q2 = self.critic2(x=x, o=proto_option_emb).squeeze(-1)  # (B,)

        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        encoder.optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        encoder.optimizer.step()
        
        self.num_critic_updates += 1

        if call_back is not None:
            call_back({
            "hl/critic_loss": float(critic_loss.item()),
            "hl/q1_mean": float(q1.mean().item()),
            "hl/q2_mean": float(q2.mean().item()),
            "hl/target_mean": float(y.mean().item()),
            })
        
        # ---------- delayed actor + target updates ----------
        if self.num_critic_updates >= self.hp.hl_policy_delay:
            self.num_critic_updates = 0
            # freeze critics during actor update (prevents wasting grad compute)
            for p in list(self.critic1.parameters()) + list(self.critic2.parameters()):
                p.requires_grad_(False)
            
            # IMPORTANT: detach x so actor update doesn't backprop into encoder
            x_actor = x.detach()
            
            a_pi = self.actor(x_actor)  # (B, de)
            actor_loss = -self.critic1(x=x_actor, o=a_pi).squeeze(-1).mean()

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()

            for p in list(self.critic1.parameters()) + list(self.critic2.parameters()):
                p.requires_grad_(True)

            if call_back is not None:
                call_back({
                    "hl/actor_loss": float(actor_loss.item())
                })

            # soft update targets (TD3 typically does this on actor update)
            with torch.no_grad():
                for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
                    pt.data.mul_(1.0 - self.hp.hl_tau).add_(self.hp.hl_tau * p.data)

                for p, pt in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                    pt.data.mul_(1.0 - self.hp.hl_tau).add_(self.hp.hl_tau * p.data)

                for p, pt in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                    pt.data.mul_(1.0 - self.hp.hl_tau).add_(self.hp.hl_tau * p.data)


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
        """Init using torch RNG (assumes you seeded torch once at experiment start)."""
        with torch.no_grad():
            eps = 1.0 / max(1, self.num_codes)
            nn.init.uniform_(self.emb.weight, -eps, eps)
            
    def reset(self, seed):
        self.set_seed(seed)
        
        self.emb = nn.Embedding(self.num_codes, self.hp.embedding_dim).to(self.device)
        self._init_weights()


    def get_closest_ind(self, proto: torch.Tensor) -> torch.Tensor:
        """
        proto: (B, d)
        returns idx: (B,)
        """
        if proto.dim() != 2:
            raise ValueError(f"proto must be (B, d), got {tuple(proto.shape)}")
        if proto.size(1) != self.hp.embedding_dim:
            raise ValueError(f"proto dim mismatch: expected d={self.hp.embedding_dim}, got {proto.size(1)}")

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
        
        self.encoder = Encoder(hyper_params, self.feature_extractor.features_dict, device)
        self.code_book = CodeBook(hyper_params, len(self.options_lst) + 1, device)
        self.hl_policy = HighLevelPolicy(self.encoder, self.code_book, hyper_params, device)
        self.ll_policy = LowLevelPolicy(self.encoder, self.code_book, action_space, hyper_params, device)
        
        self.running_option_index = [None for _ in range(self.num_envs)]      
        self.running_option_emb = [None for _ in range(self.num_envs)]      
        self.running_option_proto_emb = [None for _ in range(self.num_envs)]
 
        self.option_start_obs = [None for _ in range(self.num_envs)]        
        self.option_cumulative_reward = [0.0 for _ in range(self.num_envs)]    
        self.option_multiplier = [1.0 for _ in range(self.num_envs)]           
        self.option_num_steps = [0 for _ in range(self.num_envs)]
        
        self.hl_buffer = ReplayBuffer(self.hp.hl_buffer_capacity)

        
    def act(self, observation, greedy=False):
        state = self.feature_extractor(observation)

        with torch.no_grad():
            x_all = self.encoder(state)  # (N, dx)

            # 1) Determine which envs need a new option
            need_new = torch.zeros(self.num_envs, dtype=torch.bool)
            for i in range(self.num_envs):
                # start if none
                if self.running_option_index[i] is None:
                    need_new[i] = True

            # 2) Batch high-level policy only for those envs
            if need_new.any():
                idxs = need_new.nonzero(as_tuple=False).squeeze(-1)  # (K,)
                x_new = x_all[idxs]                                  # (K, dx)

                # proto with exploration noise (behavior proto)
                proto_e_new = self.hl_policy.select_action(x_new, greedy)  # (K, de)

                # quantize to codebook
                option_idx_new, e_new, _ = self.code_book(proto_e_new)     # option_idx:(K,), e_sel:(K,de)

                # write results back to per-env slots
                idxs_list = idxs.tolist()
                for j, env_i in enumerate(idxs_list):
                    self.running_option_index[env_i] = int(option_idx_new[j].item())

                    # store behavior proto (useful if you want critic to condition on proto)
                    self.running_option_proto_emb[env_i] = proto_e_new[j].detach()

                    # store executed embedding (this is what LL conditions on)
                    self.running_option_emb[env_i] = e_new[j].detach()

                    # store raw obs for encoder gradients in HL update later
                    self.option_start_obs[env_i] = get_single_observation(observation, env_i)

                    # reset SMDP accumulators (HL)
                    self.option_cumulative_reward[env_i] = 0.0
                    self.option_multiplier[env_i] = 1.0
                    self.option_num_steps[env_i] = 0

            # 3) Low-level action selection for ALL envs (batched)
            # stack current option embeddings into (N, de)
            e_all = torch.stack(
                [self.running_option_emb[i] for i in range(self.num_envs)],
                dim=0
            ).to(self.device)
            
            # actions for all envs
            actions = self.ll_policy.select_action(x_all, e_all, greedy=greedy)

            # 4) Increment option step counters (bookkeeping)
            for i in range(self.num_envs):
                self.option_num_steps[i] += 1
        
        # cache for update
        self.last_observation = observation
        self.last_action = actions
        return actions
                
            
    def update(self, observation, reward, terminated, truncated, call_back=None):
        # 1) accumulate option returns for each env
        for i in range(self.num_envs):
            if self.running_option_index[i] is None:
                raise ValueError("running_option_index cannot be None")
                # continue  # no option running, nothing to accumulate

            self.option_cumulative_reward[i] += float(self.option_multiplier[i]) * float(reward[i])
            self.option_multiplier[i] *= float(self.hp.gamma)
            
        # 2) decide which env options ended
        ended = np.zeros(self.num_envs, dtype=np.bool_)
        for i in range(self.num_envs):
            if terminated[i] or truncated[i]:
                ended[i] = True
            elif self.options_lst[self.running_option_index[i]].is_terminated(get_single_observation(observation, i)):
                ended[i] = True
                
        # 3) push HIGH-LEVEL transitions to buffer
        for i in range(self.num_envs):
            obs = get_single_observation(observation, i)
            
            if not (self.options_lst[self.running_option_index[i]].is_terminated(obs) or terminated[i] or truncated[i]):
                continue

            # --- add to high-level buffer ---
            transition = (
                self.option_start_obs[i],
                self.running_option_index[i],
                self.running_option_emb[i],
                self.running_option_proto_emb[i],
                self.option_cumulative_reward[i],
                self.option_num_steps[i],
                obs,
                self.option_multiplier[i],
                terminated[i],
            )
            self.hl_buffer.add(transition)

            # 4) reset per-env option bookkeeping
            self.running_option_index[i] = None
            self.running_option_emb[i] = None
            self.running_option_proto_emb[i] = None

            self.option_start_obs[i] = None
            self.option_cumulative_reward[i] = 0.0
            self.option_multiplier[i] = 1.0
            self.option_num_steps[i] = 0

        
        if len(self.hl_buffer) >= self.hp.hl_warmup_size:
            batch = self.hl_buffer.sample(self.hp.hl_batch_size)
            (
                batch_start_obs,
                batch_option_idx,
                batch_option_emb,
                batch_option_proto_emb,
                batch_option_return,
                batch_option_steps,
                batch_next_obs,
                batch_discount,
                batch_terminated
            ) = zip(*batch)
            batch_obs, batch_next_obs = (
                stack_observations(batch_start_obs),
                stack_observations(batch_next_obs)
            )
            batch_state, batch_next_state = (
                self.feature_extractor(batch_obs),
                self.feature_extractor(batch_next_obs)
            )
            
            self.hl_policy.update(batch_state, batch_option_proto_emb, batch_option_return,
                                  batch_discount, batch_option_steps, batch_next_state, 
                                  batch_terminated, self.encoder, call_back)
            
            
            
            
            
    
    def reset(self, seed):
        super().reset(seed)
        
        self.running_option_index = [None for _ in range(self.num_envs)]      
        self.running_option_emb = [None for _ in range(self.num_envs)]      
        self.running_option_proto_emb = [None for _ in range(self.num_envs)]
 
        self.option_start_obs = [None for _ in range(self.num_envs)]        
        self.option_cumulative_reward = [0.0 for _ in range(self.num_envs)]    
        self.option_multiplier = [1.0 for _ in range(self.num_envs)]           
        self.option_num_steps = [0 for _ in range(self.num_envs)]
            
         