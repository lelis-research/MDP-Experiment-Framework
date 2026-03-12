import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from gymnasium.spaces import Discrete
from typing import Optional
import os, pickle

from ...utils import RandomGenerator
from ..Base import BaseAgent, BasePolicy
from ...Buffers import BaseBuffer
from ...Options import load_options_list, save_options_list
from ..Utils import (
    get_single_observation_nobatch,
    get_single_observation,
    stack_observations,
    HyperParameters,
    grad_norm,
)
from ...registry import register_agent, register_policy
FEAT_KEYS= ["onehot_image"]
H, W, C_img_onehot, C_dir_onthot = 11, 11, 22, 4


class FiLM2d(nn.Module):
    """
    Feature-wise Linear Modulation for (B, C, H, W) using option embedding e: (B, E).
    """
    def __init__(self, n_channels: int, emb_dim: int, hidden: int = 128):
        super().__init__()
        self.to_gamma_beta = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * n_channels),
        )
        # init so FiLM starts near identity: gamma≈1, beta≈0
        nn.init.zeros_(self.to_gamma_beta[-1].weight)
        nn.init.zeros_(self.to_gamma_beta[-1].bias)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        gb = self.to_gamma_beta(e)                  # (B, 2C)
        gamma, beta = gb.chunk(2, dim=-1)           # (B, C), (B, C)
        gamma = 1.0 + gamma                         # start near 1
        return gamma[:, :, None, None] * x + beta[:, :, None, None]
    
    def gamma_beta(self, e: torch.Tensor):
        gb = self.to_gamma_beta(e)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = 1.0 + gamma
        return gamma, beta
    
class ConvFiLMEncoder(nn.Module):
    def __init__(self, in_channels: int, emb_dim: int, out_dim: int = 256):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.f1 = FiLM2d(64, emb_dim)
        self.c2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.f2 = FiLM2d(64, emb_dim)
        self.c3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.f3 = FiLM2d(64, emb_dim)

        self.head = nn.Conv2d(64, in_channels, kernel_size=1)


    def forward(self, obs: torch.Tensor, e: torch.Tensor) -> torch.Tensor:

        x = torch.relu(self.c1(obs))
        x = self.f1(x, e)

        x = torch.relu(self.c2(x))
        x = self.f2(x, e)

        x = torch.relu(self.c3(x))
        x = self.f3(x, e)

        pred = self.head(x)
        return pred, x

class OptionCondPolicyHead(nn.Module):
    def __init__(self, C, emb_dim, num_actions):
        super().__init__()
        self.c1 = nn.Conv2d(C, 64, 3, 1, 1)
        self.f1 = FiLM2d(64, emb_dim)
        self.c2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.f2 = FiLM2d(64, emb_dim)
        self.c3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.f3 = FiLM2d(64, emb_dim)

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Linear(64, num_actions)

    def forward(self, obs, e):
        # obs: (B,C,H,W)
        x = F.relu(self.c1(obs)); x = self.f1(x, e)
        x = F.relu(self.c2(x));   x = self.f2(x, e)
        x = F.relu(self.c3(x));   x = self.f3(x, e)
        x = self.pool(x).flatten(1)     # (B,64)
        return self.head(x)             # (B,A)
 
 
class OptionClassifierCNN2(nn.Module):
    """
    Predict option id from SF image (B,C,H,W).
    """
    def __init__(self, in_channels: int, num_options: int, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2 * in_channels, width, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                     # downsample once (11->5)
            nn.Conv2d(width, width, 3, 1, 1), nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(width, num_options),
        )

    def forward(self, sf_img: torch.Tensor) -> torch.Tensor:
        return self.net(sf_img)
    
class OptionClassifierCNN(nn.Module):
    def __init__(self, in_channels: int, num_options: int, width: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1 * in_channels, width, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 11->5
            nn.Conv2d(width, width, 3, 1, 1), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),                 # <-- keep spatial info
            nn.Linear(width * 5 * 5, num_options),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.head(x)
    
class OptionClassifierFC(nn.Module):
    """
    Predict option id from vector features (B, D).
    Suitable for:
        - channel_changed
        - channel_change_magnitude
        - GAP(start) + GAP(delta)
    """
    def __init__(self, in_dim: int, num_options: int, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, num_options),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
    
        return self.net(x)



   
@register_policy
class OptionRandomSFPolicy(BasePolicy):
    """
    Randomly selects an option index (0..num_options-1).
    """
    def __init__(self, num_options: int, hyper_params, device="cpu"):
        super().__init__(action_space=Discrete(num_options), hyper_params=hyper_params, device=device)
        self.num_options = int(num_options)

    def reset(self, seed):
        self.set_seed(seed)

    def select_option_index(self) -> int:
        # uses RandomGenerator RNG (via BasePolicy -> RandomGenerator in your codebase)
        return int(self._rand_int(0, self.num_options))


class SFCodeBook(RandomGenerator):
    """
    Codebook where each code corresponds to one option index (same K as options_lst length).
    Trained ONLY from delta-SF regression:
        pred_delta = sf_head([e_k, start_feat])  ->  target_delta_sf
    Gradients update BOTH sf_head and the selected embeddings e_k.
    """
    def __init__(self, hyper_params, num_codes: int, device="cpu", init_embs=None, num_actions=None):
        self.hp = hyper_params
        self.device = device
        self.num_codes = int(num_codes)
        self.init_embs = init_embs
        self.num_actions = num_actions
        
        self.emb = nn.Embedding(self.num_codes, self.hp.embedding_dim).to(self.device)
        
        
        if "opt_classifier" in self.hp.test_type:
            if self.hp.net_type == "cnn":
                self.net = OptionClassifierCNN(C_img_onehot, self.num_codes)
            elif self.hp.net_type == "fc":
                self.net = OptionClassifierFC(C_img_onehot, self.num_codes)
            self.sf_params = list(self.net.parameters()) + \
                            list(self.emb.parameters()) 
            
        self.optimizer = optim.Adam(
            params=self.sf_params,
            lr=self.hp.step_size,
            eps=self.hp.eps,
        )

        self._init_weights()
        
    def _init_weights(self):
        with torch.no_grad():
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

    def reset(self, seed):
        self.set_seed(seed)
        self.emb = nn.Embedding(self.num_codes, self.hp.embedding_dim).to(self.device)
        self._init_weights()
        
        if "opt_classifier" in self.hp.test_type:
            if self.hp.net_type == "cnn":
                self.net = OptionClassifierCNN(C_img_onehot, self.num_codes)
            elif self.hp.net_type == "fc":
                self.net = OptionClassifierFC(C_img_onehot, self.num_codes)
            self.sf_params = list(self.net.parameters()) + \
                            list(self.emb.parameters()) 
            
        self.optimizer = optim.Adam(
            params=self.sf_params,
            lr=self.hp.step_size,
            eps=self.hp.eps,
        )

    def option_sequence_features(self, x_thwc: torch.Tensor):
        if x_thwc.ndim != 4:
            raise ValueError("Expected (T,H,W,C) tensor")

        T = x_thwc.shape[0]
        device = x_thwc.device
        dtype = x_thwc.dtype

        first_obs = x_thwc[0]
        last_obs  = x_thwc[-1]

        # 1) Last Observation
        out_last = last_obs

        # 2) Delta Last and First
        out_delta_last_first = last_obs - first_obs

        # discounts
        t = torch.arange(T, device=device, dtype=dtype)

        w_fwd = (self.hp.gamma ** t).view(T,1,1,1)
        w_rev = (self.hp.gamma ** t.flip(0)).view(T,1,1,1)
        Z = w_fwd.sum().clamp_min(1e-8)
        
        
        # 3) Forward SF
        sf_forward = (w_fwd * x_thwc).sum(dim=0) / Z

        # 4) Delta Forward SF (correct)
        delta_forward = x_thwc - first_obs      # subtract at every step
        delta_sf_forward = (w_fwd * delta_forward).sum(dim=0) / Z

        # 5) Reverse SF
        sf_reverse = (w_rev * x_thwc).sum(dim=0) / Z

        # 6) Delta Reverse SF (correct)
        delta_sf_reverse = (w_rev * delta_forward).sum(dim=0) / Z

        return first_obs, out_last, out_delta_last_first, sf_forward, delta_sf_forward, sf_reverse, delta_sf_reverse
            
    def update(self, option_id_seq, action_seq, obs_seq, call_back=None):
        """
        option_idx: (T,) int (option index per transition)
        delta_sf  : (T, feat_dim) float32
        start_feat: (T, feat_dim) float32
        """ 
        # ---- data ----
        # option_idx: list/array length N (N rollouts)
        # obs_seq:    list length N, obs_seq[i] = (L_i, H, W, C)
        # action_seq: list length N, action_seq[i] = (L_i,)
        option_id_t = torch.as_tensor(np.array(option_id_seq), device=self.device, dtype=torch.int64)
        
        # ---- early exit ----
        if option_id_t.numel() == 0:
            if call_back is not None:
                call_back({"cb_total_loss": 0.0, "cb_batch_T": 0})
            return

        last_lst, d_last_lst, sf_lst, d_sf_lst, sf_reverse_lst, d_sf_reverse_lst = [], [], [], [], [], []
        # act_list = []
        start_lst = []
        opt_lst = []

        for o, obs_i, act_i in zip(option_id_seq, obs_seq, action_seq):
            first, last, d_last, sf, d_sf, sf_reverse, d_sf_reverse = self.option_sequence_features(obs_i)
            start_lst.append(first.unsqueeze(0))
            last_lst.append(last.unsqueeze(0))
            d_last_lst.append(d_last.unsqueeze(0))
            sf_lst.append(sf.unsqueeze(0))
            d_sf_lst.append(d_sf.unsqueeze(0))
            sf_reverse_lst.append(sf_reverse.unsqueeze(0))
            d_sf_reverse_lst.append(d_sf_reverse.unsqueeze(0))
            opt_lst.append(torch.tensor([o]))
            

        opt_t = torch.cat(opt_lst, dim=0).to(self.device).long()   # (Ttot,)
        start_t = torch.cat(start_lst, dim=0).to(self.device).permute(0, 3, 1, 2).contiguous()
        if self.hp.out_feature == "last":
            last_t = torch.cat(last_lst, dim=0).to(self.device).permute(0, 3, 1, 2).contiguous()          # (Ttot,H,W,C)
            out = last_t
        elif self.hp.out_feature == "d-last":
            d_last_t = torch.cat(d_last_lst, dim=0).to(self.device).permute(0, 3, 1, 2).contiguous()            # (Ttot,H,W,C)
            out = d_last_t
        elif self.hp.out_feature == "sf":
            sf_t = torch.cat(sf_lst, dim=0).to(self.device).permute(0, 3, 1, 2).contiguous()            # (Ttot,H,W,C)
            out = sf_t
        elif self.hp.out_feature == "d-sf":
            d_sf_t = torch.cat(d_sf_lst, dim=0).to(self.device).permute(0, 3, 1, 2).contiguous()            # (Ttot,H,W,C)
            out = d_sf_t
        elif self.hp.out_feature == "sf-reverse":
            sf_reverse_t = torch.cat(sf_reverse_lst, dim=0).to(self.device).permute(0, 3, 1, 2).contiguous()            # (Ttot,H,W,C)
            out = sf_reverse_t
        elif self.hp.out_feature == "d-sf-reverse":
            d_sf_reverse_t= torch.cat(d_sf_reverse_lst, dim=0).to(self.device).permute(0, 3, 1, 2).contiguous()            # (Ttot,H,W,C)
            out = d_sf_reverse_t
        elif self.hp.out_feature == "abs-d-last":
            d_last_t = torch.cat(d_last_lst, dim=0).to(self.device).permute(0, 3, 1, 2).contiguous()  # (N,C,H,W)
            out = d_last_t.abs()
        elif self.hp.out_feature == "abs-d-last-changed":
            d_last_t = torch.cat(d_last_lst, dim=0).to(self.device).permute(0, 3, 1, 2).contiguous()            # (Ttot,H,W,C)
            eps = 0.0  # for exact one-hot deltas; use 1e-6 if floats
            out = (d_last_t.abs() > eps).to(d_last_t.dtype)
        elif self.hp.out_feature == "cell-changed":
            d_last_t = torch.cat(d_last_lst, dim=0).to(self.device).permute(0, 3, 1, 2).contiguous()  # (N,C,H,W)
            eps = 0.0  # 1e-6 if floats
            abs_d = d_last_t.abs()
            cell_changed = (abs_d.sum(dim=1, keepdim=True) > eps).to(d_last_t.dtype)
            out = cell_changed
        elif self.hp.out_feature == "channel-changed":
            d_last_t = torch.cat(d_last_lst, dim=0).to(self.device).permute(0, 3, 1, 2).contiguous()  # (N,C,H,W)
            eps = 0.0  # 1e-6 if floats
            abs_d = d_last_t.abs()
            channel_changed = (abs_d.sum(dim=(2,3), keepdim=True) > eps).to(d_last_t.dtype)
            out = channel_changed
        else:
            raise ValueError(f"Unknown out_feature: {self.hp.out_feature}")
        
        if "start" in self.hp.out_feature2:
            out = torch.cat([start_t, out], dim=1)
        
        epochs = int(getattr(self.hp, "opt_epochs", 10))
        mb_size = int(getattr(self.hp, "opt_minibatch", 64))

        B = out.size(0)

        # shuffle once per call (like offline)
        perm = torch.randperm(B, device=out.device)
        out = out[perm]
        opt_t = opt_t[perm]

        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for ep in range(epochs):
            # reshuffle each epoch (offline-style)
            perm = torch.randperm(B, device=out.device)
            out_ep = out[perm]
            y_ep = opt_t[perm]

            for s in range(0, B, mb_size):
                xb = out_ep[s:s+mb_size]
                yb = y_ep[s:s+mb_size]

                logits = self.net(xb)
                loss = F.cross_entropy(logits, yb)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.sf_params, self.hp.max_grad_norm)
                self.optimizer.step()

                total_loss += float(loss.item()) * yb.size(0)
                total_correct += (logits.argmax(dim=1) == yb).sum().item()
                total_seen += yb.size(0)

        # Optional: averaged metrics over all SGD steps in this update call
        avg_loss = total_loss / max(1, total_seen)
        avg_acc = total_correct / max(1, total_seen)

        
        
        if call_back is not None:
            call_back({
                "cb_total_loss": float(avg_loss),
                "cb_accuracy": float(avg_acc),
                "cb_batch_T": int(B),
                "cb_epochs": int(epochs),
                "cb_minibatch": int(mb_size),
            })

    def encode_img(self, img_bhwc, e_b):
        """
        img_bhwc: (B, H, W, C) float32 on self.device
        returns:  (B, 32)
        """
        B= img_bhwc.shape[0]
        
        x = img_bhwc.view(B, H * W, C_img_onehot)   # (B, HW, 21)
        h = self.token_mlp(x)                       # (B, HW, 64)
        
        # Query from option embedding
        q = self.q_proj(e_b).unsqueeze(1)           # (B, 1, 64)
        
        # Keys from tokens
        k = self.k_proj(h)                          # (B, HW, 64)
        
        # Dot-product attention scores
        scores = (q * k).sum(dim=-1, keepdim=True)  # (B, HW, 1)

        w = torch.softmax(scores, dim=1)                # (B, HW, 1)
        pooled = (w * h).sum(dim=1)                     # (B, 64)
        phi = self.out(pooled)                          # (B, 32)
        return phi
    
    def add_row(self, new_emb: Optional[torch.Tensor] = None) -> int:
        """
        Add one new code (for a newly-added option).
        Returns the new code index.
        """
        d = int(self.hp.embedding_dim)
        old_weight = self.emb.weight.data
        K_old = old_weight.size(0)
        K_new = K_old + 1

        new_weight = torch.empty((K_new, d), device=self.device, dtype=old_weight.dtype)
        new_weight[:K_old].copy_(old_weight)

        if new_emb is None:
            new_vec = torch.empty((d,), device=self.device, dtype=old_weight.dtype)
            if "onehot" in getattr(self.hp, "init_type", "uniform"):
                new_vec.zero_()
                if d < K_new:
                    raise ValueError(f"[SFCodeBook] Cannot onehot init new code: embedding_dim={d} < num_codes={K_new}")
                new_vec[K_old] = 1.0
            else:
                r = float(getattr(self.hp, "init_emb_range", 0.01))
                nn.init.uniform_(new_vec, -r, r)
        else:
            if new_emb.dim() == 2 and new_emb.size(0) == 1:
                new_emb = new_emb.squeeze(0)
            if new_emb.dim() != 1 or new_emb.numel() != d:
                raise ValueError(f"new_emb must be shape (d,) with d={d}, got {tuple(new_emb.shape)}")
            new_vec = new_emb.to(device=self.device, dtype=old_weight.dtype)

        new_weight[K_old].copy_(new_vec)

        self.emb = nn.Embedding(K_new, d).to(self.device)
        with torch.no_grad():
            self.emb.weight.copy_(new_weight)

        self.num_codes = K_new

        # refresh optimizer params (keep it simple; you can also do param_group surgery)
        self.optimizer = optim.Adam(
            list(self.emb.parameters()) + list(self.sf_head.parameters()),
            lr=self.hp.step_size,
            eps=self.hp.eps,
        )
        return K_old

    def save(self, file_path: str | None = None):
        ckpt = {
            "class": self.__class__.__name__,
            "hyper_params": self.hp,
            "num_codes": int(self.num_codes),
            "device": self.device,
            "rng_state": self.get_rng_state(),
            "emb_state_dict": self.emb.state_dict(),
            # "conv_opt_class_state_dict": self.conv_opt_class.state_dict(),
            # "conv_film_state_dict": self.con.state_dict(),
            # "sf_net_state_dict": self.sf_net.state_dict(),
            # "img_net_state_dict": self.img_net.state_dict(),
            # "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if file_path is not None:
            torch.save(ckpt, f"{file_path}_sfcodebook.t")
        return ckpt

    @classmethod
    def load(cls, file_path: str | None = None, checkpoint=None, map_location="cpu"):
        if checkpoint is None:
            assert file_path is not None
            checkpoint = torch.load(file_path, map_location=map_location, weights_only=False)

        inst = cls(
            hyper_params=checkpoint["hyper_params"],
            num_codes=int(checkpoint["num_codes"]),
            device=checkpoint["device"],
        )
        inst.set_rng_state(checkpoint["rng_state"])
        inst.emb.load_state_dict(checkpoint["emb_state_dict"])
        # inst.conv_opt_class.load_state_dict(checkpoint["conv_opt_class_state_dict"])
        # inst.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return inst

    def analysis(self):
        if False:
            device = self.device
            K = self.num_codes

            G = []
            B = []
            for o in range(K):
                with torch.no_grad():
                    e = self.emb(torch.tensor([o], device=device))  # make sure device!
                    gamma, beta = self.conv_film.f3.gamma_beta(e)
                G.append(gamma.squeeze(0).detach())
                B.append(beta.squeeze(0).detach())

            G = torch.stack(G)  # (K, 64)
            B = torch.stack(B)  # (K, 64)

            print("gamma std over options (mean over channels):", G.std(dim=0).mean().item())
            print("beta  std over options (mean over channels):", B.std(dim=0).mean().item())

            G_n = F.normalize(G, dim=1)
            sim = G_n @ G_n.T
            print("gamma cosine sim min/mean/max:", sim.min().item(), sim.mean().item(), sim.max().item())

@register_agent
class OptionRandomSFCodebookAgent(BaseAgent):
    """
    Random option-selection agent + SF-trained codebook (one code per option).

    - Chooses a random option index when no option is running.
    - Executes that option until option termination (or env termination/truncation).
    - Computes delta-SF using *flattened observations* as features:
        feat_t = flatten(obs_t)
        start_feat = feat at option start
        cumulative_feat = sum_{t=0..L-1} gamma^t feat_{start+t}
        cumulative_discounts = sum_{t=0..L-1} gamma^t
        delta_sf = cumulative_feat - cumulative_discounts * start_feat
        delta_sf /= (cumulative_discounts + 1e-8)   # normalization (recommended)
    - Trains codebook embeddings (and sf_head) to predict delta_sf from (e_k, start_feat).
    """
    name = "OptionRandomSFCodebook"
    SUPPORTED_ACTION_SPACES = (Discrete,)

    def __init__(
        self,
        action_space,
        observation_space,
        hyper_params,
        num_envs,
        feature_extractor_class,
        init_option_lst=None,
        init_option_embs=None,
        device="cpu",
    ):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device=device)

        self.atomic_action_space = action_space
        self.options_lst = [] if init_option_lst is None else init_option_lst

        if len(self.options_lst) == 0:
            raise ValueError("[OptionRandomSFCodebookAgent] init_option_lst is empty; need at least 1 option.")

        # Expect Dict obs space with keys "onehot_direction" and "onehot_image"
        if not hasattr(self.observation_space, "spaces"):
            raise ValueError("[OptionRandomSFCodebookAgent] observation_space must be a Dict space.")

        # Policy chooses option indices uniformly
        self.policy = OptionRandomSFPolicy(num_options=len(self.options_lst), hyper_params=self.hp, device=self.device)

        self.hp.codebook.update(gamma=self.hp.gamma)
        # Codebook aligns to options (K == len(options_lst))
        self.code_book = SFCodeBook(
            hyper_params=self.hp.codebook,
            num_codes=len(self.options_lst),
            device=self.device,
            init_embs=init_option_embs,
            num_actions=action_space.n
        )

        # Per-env execution bookkeeping
        self.running_option_index = [None for _ in range(self.num_envs)]
        self.option_start_obs = [None for _ in range(self.num_envs)]
        self.option_num_steps = [0 for _ in range(self.num_envs)]

        # SF accumulators (per env)
        self.sf_actions = [[] for _ in range(self.num_envs)]
        self.sf_observations = [[] for _ in range(self.num_envs)]
        

        # Small buffer to batch SF updates (optional but usually nicer)
        self.sf_buffer = [BaseBuffer(int(self.hp.sf_rollout_steps)) for _ in range(self.num_envs)]

        self._init_log_buf()
        
        self.dump_path = "Random_Data/options_fixedseed-100_dup_50K.pkl"
        

    def _init_log_buf(self):
        self.ep_counter = 0

        self.log_buf = []
        for _ in range(self.num_envs):
            self.log_buf.append({
                "num_options": [],
                "option_index": [],
                "code_book": [],
            })

    def sf_bookkeeping(self, obs_option, env_id, action, mode=None):
        assert mode in ["s", "m", "f"]
        img_t = torch.as_tensor(obs_option['onehot_image'], device=self.device, dtype=torch.float32)
        if mode == "s":            
            self.sf_actions[env_id] = []
            self.sf_observations[env_id] = [img_t]
        
        elif mode == "m":
            self.sf_actions[env_id].append(action)
            self.sf_observations[env_id].append(img_t)
                
        elif mode == "f":
            return torch.tensor(self.sf_actions[env_id]), torch.stack(self.sf_observations[env_id])
        
        else:
            raise ValueError(f"Mode {mode} is not defined")
          
    def act(self, observation):       
        action = []

        for i in range(self.num_envs):
            obs_option = get_single_observation_nobatch(observation, i)

            # start option if none running
            if self.running_option_index[i] is None:
                while True:
                    opt_idx = int(self.policy.select_option_index())
                    valid_option = self.options_lst[opt_idx].can_initiate(obs_option)
                    if valid_option:
                        break
                    
                
                self.running_option_index[i] = opt_idx
                self.option_start_obs[i] = get_single_observation(observation, i)
                self.option_num_steps[i] = 0

                # initialize SF accumulators from start obs
                self.sf_bookkeeping(obs_option, i, action=None, mode="s")
                

            # execute current option
            curr_idx = self.running_option_index[i]
            a = self.options_lst[curr_idx].select_action(obs_option)
            action.append(a)
        
        self.last_action = action
        return action
        
    def update_buffers(self, observation, reward, terminated, truncated, call_back=None):
        # add SF step contribution (current obs)
        for i in range(self.num_envs):
            obs_option = get_single_observation_nobatch(observation, i)
            curr_option_idx = self.running_option_index[i]
            
            if curr_option_idx is None:
                continue
            
            curr_option_terminated = self.options_lst[curr_option_idx].is_terminated(obs_option)
                
            self.option_num_steps[i] += 1
            self.sf_bookkeeping(obs_option, i, self.last_action[i], mode="m")
            
            if curr_option_terminated or terminated[i] or truncated[i]:
                action_seq, obs_seq = self.sf_bookkeeping(obs_option, i, action=None, mode="f")
                self.log_buf[i]["num_options"].append(np.array([len(self.options_lst)]))
                self.log_buf[i]["option_index"].append(np.array([curr_option_idx]))
                
                if call_back is not None:
                    call_back({"curr_hl_option_idx": curr_option_idx,
                                "num_options": len(self.options_lst)})
                
                self.dump_option_rollout(curr_option_idx, action_seq, obs_seq)    
                transition = (
                    int(curr_option_idx),
                    action_seq,
                    obs_seq
                )
        
                self.options_lst[curr_option_idx].reset()
                self.running_option_index[i] = None
                
                self.sf_buffer[i].add(transition)
        
    def update_codebook(self, observation, reward, terminated, truncated, call_back=None):
        for i in range(self.num_envs):
            if self.sf_buffer[i].is_full():
                batch = self.sf_buffer[i].all()
                option_id_seq, action_seq, obs_seq = zip(*batch)
                
                self.code_book.update(option_id_seq, action_seq, obs_seq, call_back=call_back)
                self.sf_buffer[i].clear()

    def update(self, observation, reward, terminated, truncated, call_back=None):
        if self.training:
            self.update_buffers(observation, reward, terminated, truncated, call_back=call_back)
            self.update_codebook(observation, reward, terminated, truncated, call_back=call_back)
            for i in range(self.num_envs):
                if terminated[i] or truncated[i]:
                    self.log_buf[i]["code_book"].append(self.code_book.emb.weight.detach().cpu().numpy())
                    
                    if self.ep_counter % 500 == 0:
                        self.code_book.analysis()
                        
                    self.ep_counter += 1
                    
        else:
            for i in range(self.num_envs):
                obs_option = get_single_observation_nobatch(observation, i)
                curr_option_idx = self.running_option_index[i]
                
                if curr_option_idx is None:
                    continue

                if self.options_lst[curr_option_idx].is_terminated(obs_option) or terminated[i] or truncated[i]:
                    self.log_buf[i]["num_options"].append(np.array([len(self.options_lst)]))
                    self.log_buf[i]["option_index"].append(np.array([curr_option_idx]))
                    if call_back is not None:
                        call_back({"curr_hl_option_idx": curr_option_idx})
                    
                    self.options_lst[curr_option_idx].reset()
                    self.running_option_index[i] = None
                
    def dump_option_rollout(self, option_id: int, action_seq: torch.Tensor, obs_seq: torch.Tensor):
        """
        Dump one completed option rollout to self.dump_path.

        Args:
            option_id: int, the option index (curr_option_idx)
            action_seq: torch.Tensor shape (L,), actions taken during the option
            obs_seq: torch.Tensor shape (L+1, H, W, C), observations collected for the option
            env_id: optional int, which vector-env slot produced it
        Output:
            Appends a dict to a pickle stream at self.dump_path.
            Each dict contains option_id, actions, observations, length, env_id.
        """
        record = {
            "option_id": int(option_id),
            "len": int(action_seq.shape[0]),
            "actions": action_seq.detach().cpu().numpy().astype(np.int16, copy=False),
            "observations": obs_seq.detach().cpu().numpy().astype(np.float16, copy=False),
        }

        os.makedirs(os.path.dirname(self.dump_path) or ".", exist_ok=True)
        with open(self.dump_path, "ab") as f:
            pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)

    def reset(self, seed):
        super().reset(seed)
        self.policy.reset(seed)
        self.code_book.reset(seed)

        self.running_option_index = [None for _ in range(self.num_envs)]
        self.option_start_obs = [None for _ in range(self.num_envs)]
        self.option_num_steps = [0 for _ in range(self.num_envs)]

        # SF accumulators (per env)
        self.sf_start_img = [torch.zeros(size=(H, W, C_img_onehot), device=self.device, dtype=torch.float32) for _ in range(self.num_envs)]
        self.sf_cumulative_img = [torch.zeros(size=(H, W, C_img_onehot), device=self.device, dtype=torch.float32) for _ in range(self.num_envs)]
        self.sf_finish_img = [torch.zeros(size=(H, W, C_img_onehot), device=self.device, dtype=torch.float32) for _ in range(self.num_envs)]
        self.sf_cumulative_discounts = [0.0 for _ in range(self.num_envs)]
        self.sf_actions = [[] for _ in range(self.num_envs)]

        self.sf_buffer = [BaseBuffer(int(getattr(self.hp, "sf_rollout_steps", 256))) for _ in range(self.num_envs)]
        self._init_log_buf()

    
    def save(self, file_path: str | None = None):
        checkpoint = super().save(file_path=None)
        checkpoint["options_lst"] = save_options_list(self.options_lst, file_path=None)
        checkpoint["sf_code_book"] = self.code_book.save(file_path=None)

        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_agent.t")
        return checkpoint

    @classmethod
    def load(cls, file_path: str, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)

        options_lst = load_options_list(file_path=None, checkpoint=checkpoint["options_lst"])

        inst = cls(
            action_space=checkpoint["action_space"],
            observation_space=checkpoint["observation_space"],
            hyper_params=checkpoint["hyper_params"],
            num_envs=int(checkpoint["num_envs"]),
            feature_extractor_class=checkpoint["feature_extractor_class"],
            init_option_lst=options_lst,
            device=checkpoint["device"],
        )
        inst.set_rng_state(checkpoint["rng_state"])
        inst.feature_extractor = inst.feature_extractor.load(file_path=None, checkpoint=checkpoint["feature_extractor"])
        inst.code_book = inst.code_book.load(file_path=None, checkpoint=checkpoint["sf_code_book"])
        inst.options_lst = options_lst
        return inst