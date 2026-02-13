import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from gymnasium.spaces import Discrete
from typing import Optional

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


def _dir_img_feat(obs) -> np.ndarray:
    """
    Feature = concat(flat(onehot_direction), flat(onehot_image)) as float32.
    Both keys are expected to be Box spaces.
    """
    d = np.asarray(obs["onehot_direction"], dtype=np.float32).reshape(-1)
    im = np.asarray(obs["onehot_image"], dtype=np.float32).reshape(-1)
    return np.concatenate([d, im], axis=0)

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
    def __init__(self, hyper_params, num_codes: int, feat_dim: int, device="cpu", init_embs=None):
        self.hp = hyper_params
        self.device = device
        self.num_codes = int(num_codes)
        self.feat_dim = int(feat_dim)
        self.init_embs = init_embs
        
        self.emb = nn.Embedding(self.num_codes, self.hp.embedding_dim).to(self.device)
        
        if len(self.hp.obs_proj_dims) > 0:
            self.obs_proj = nn.Sequential()
            in_dim = self.feat_dim
            for i, hdim in enumerate(self.hp.obs_proj_dims):
                self.obs_proj.add_module(f"obs_proj_fc{i}", nn.Linear(in_dim, hdim))
                # self.obs_proj.add_module(f"obs_proj_ln{i}", nn.LayerNorm(hdim))
                self.obs_proj.add_module(f"obs_proj_tanh{i}", nn.Tanh())
                in_dim = hdim
            self.proj_dim = in_dim
        else:
            self.obs_proj = nn.Identity()
            self.proj_dim = self.feat_dim
        
        
        self.sf_head = nn.Sequential()
        
        in_dim = 0
        if "obs" in self.hp.pred_input:
            in_dim += self.proj_dim
        if "emb" in self.hp.pred_input:
            in_dim += self.hp.embedding_dim
            
        for i, hdim in enumerate(self.hp.sf_hidden_dims):
            self.sf_head.add_module(f"sf_head_fc{i}", nn.Linear(in_dim, hdim))
            self.sf_head.add_module(f"sf_head_tanh{i}", nn.Tanh())
            in_dim = hdim
        # last layer to feat_dim
        self.sf_head.add_module(f"sf_head_out", nn.Linear(in_dim, self.feat_dim))
        
        self.sf_params = list(self.emb.parameters()) + \
                            list(self.obs_proj.parameters()) + \
                            list(self.sf_head.parameters())
                            
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
        
        if len(self.hp.obs_proj_dims) > 0:
            self.obs_proj = nn.Sequential()
            in_dim = self.feat_dim
            for i, hdim in enumerate(self.hp.obs_proj_dims):
                self.obs_proj.add_module(f"obs_proj_fc{i}", nn.Linear(in_dim, hdim))
                # self.obs_proj.add_module(f"obs_proj_ln{i}", nn.LayerNorm(hdim))
                self.obs_proj.add_module(f"obs_proj_tanh{i}", nn.Tanh())
                in_dim = hdim
            self.proj_dim = in_dim
        else:
            self.obs_proj = nn.Identity()
            self.proj_dim = self.feat_dim
        
        self.sf_head = nn.Sequential()
        
        in_dim = 0
        if "obs" in self.hp.pred_input:
            in_dim += self.proj_dim
        if "emb" in self.hp.pred_input:
            in_dim += self.hp.embedding_dim
        
        for i, hdim in enumerate(self.hp.sf_hidden_dims):
            self.sf_head.add_module(f"sf_head_fc{i}", nn.Linear(in_dim, hdim))
            self.sf_head.add_module(f"sf_head_tanh{i}", nn.Tanh())
            in_dim = hdim
        # last layer to feat_dim
        self.sf_head.add_module(f"sf_head_out", nn.Linear(in_dim, self.feat_dim))
        
        self.sf_params = list(self.emb.parameters()) + \
                            list(self.obs_proj.parameters()) + \
                            list(self.sf_head.parameters())
        
        self.optimizer = optim.Adam(
            params=self.sf_params,
            lr=self.hp.step_size,
            eps=self.hp.eps,
        )

    def update(self, option_idx, delta_sf, start_feat, call_back=None):
        """
        option_idx: (T,) int (option index per transition)
        delta_sf  : (T, feat_dim) float32
        start_feat: (T, feat_dim) float32
        """
        idx_t = torch.as_tensor(np.array(option_idx), device=self.device, dtype=torch.int64)
        delta_t = torch.as_tensor(np.array(delta_sf), device=self.device, dtype=torch.float32)
        start_t = torch.as_tensor(np.array(start_feat), device=self.device, dtype=torch.float32)
        
        if idx_t.numel() == 0:
            if call_back is not None:
                call_back({"cb_sf_loss": 0.0, "cb_batch_T": 0})
            return

        e = self.emb(idx_t)  # (T, emb_dim)
        proj_feat = self.obs_proj(start_t)  # (T, proj_dim)
        proj_feat = F.dropout(proj_feat, p=self.hp.obs_dropout, training=True)
        
        if self.hp.pred_input == "obs-emb":
            inp = torch.cat([e, proj_feat], dim=-1)  # (T, emb_dim + proj_dim)    
        elif self.hp.pred_input == "obs":
            inp = proj_feat               # (T, proj_dim)
        elif self.hp.pred_input == "emb":
            inp = e                       # (T, emb_dim)
        
        pred = self.sf_head(inp)               # (T, feat_dim)
        loss_sf = F.mse_loss(pred, delta_t)
        
        loss_nce = torch.tensor(0.0, device=self.device)
        loss_kl = torch.tensor(0.0, device=self.device)
        if "emb" in self.hp.pred_input:
            with torch.no_grad():
                # behavior distances (delta-SF)
                dist_b = torch.cdist(delta_t, delta_t, p=2)  # (T,T)
                
            # pairwise squared L2 distances in embedding space
            dist_e2 = torch.cdist(e, e, p=2).pow(2) # (T,T)
            
            if self.hp.nce_coef > 0.0:
                with torch.no_grad():
                    dist_b_clone = dist_b.clone()
                    dist_b_clone.fill_diagonal_(float("inf"))
                    pos_idx = dist_b_clone.argmin(dim=1)  # (T,)
                    
                logits = -dist_e2 / max(self.hp.nce_tau, 1e-8)
                logits.fill_diagonal_(float("-inf"))
                loss_nce = F.cross_entropy(logits, pos_idx)
            
            if self.hp.kl_coef > 0.0:
                with torch.no_grad():
                    teacher_logits = -dist_b / max(self.hp.kl_b_tau, 1e-8)
                    teacher_logits.fill_diagonal_(float("-inf"))  # exclude self
                
                student_logits = -dist_e2 / max(self.hp.kl_e_tau, 1e-8)
                student_logits.fill_diagonal_(float("-inf"))  # exclude self
        
                # Convert to distributions
                teacher_probs = torch.softmax(teacher_logits, dim=1)      # (T,T)  (no grad)
                student_logp  = torch.log_softmax(student_logits, dim=1)  # (T,T)
                
                loss_kl = F.kl_div(student_logp, teacher_probs, reduction="batchmean")        
            

        loss = loss_sf + self.hp.nce_coef * loss_nce + self.hp.kl_coef * loss_kl

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        g = float(nn.utils.clip_grad_norm_(self.sf_params, float(self.hp.max_grad_norm)).item())
        self.optimizer.step()
        
        with torch.no_grad():
            self.emb.weight.clamp_(self.hp.embedding_low, self.hp.embedding_high)

        if call_back is not None:
            with torch.no_grad():
                K = self.emb.num_embeddings
                counts = torch.bincount(idx_t, minlength=K).float()
                used = int((counts > 0).sum().item())
                frac_used = float(used / max(1, K))

            call_back({
                "cb_sf_loss": float(loss_sf.item()),
                "cb_nce_loss": float(loss_nce.item()),
                "cb_kl_loss": float(loss_kl.item()),
                "cb_total_loss": float(loss.item()),
                
                "cb_grad_norm": g,
                "cb_used_codes": used,
                "cb_frac_used": frac_used,
                "cb_num_codes": int(K),
                "cb_batch_T": int(idx_t.shape[0]),
            })

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
            "feat_dim": int(self.feat_dim),
            "device": self.device,
            "rng_state": self.get_rng_state(),
            "emb_state_dict": self.emb.state_dict(),
            "sf_head_state_dict": self.sf_head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
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
            feat_dim=int(checkpoint["feat_dim"]),
            device=checkpoint["device"],
        )
        inst.set_rng_state(checkpoint["rng_state"])
        inst.emb.load_state_dict(checkpoint["emb_state_dict"])
        inst.sf_head.load_state_dict(checkpoint["sf_head_state_dict"])
        inst.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return inst


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

        spaces = self.observation_space.spaces
        for k in ["onehot_direction", "onehot_image"]:
            if k not in spaces:
                raise ValueError(f"[OptionRandomSFCodebookAgent] missing key '{k}' in observation_space.spaces.")

        dir_space = spaces["onehot_direction"]
        img_space = spaces["onehot_image"]

        # Both should be Box; compute dim via shape product
        if not (hasattr(dir_space, "shape") and dir_space.shape is not None):
            raise ValueError("[OptionRandomSFCodebookAgent] onehot_direction must have a shape.")
        if not (hasattr(img_space, "shape") and img_space.shape is not None):
            raise ValueError("[OptionRandomSFCodebookAgent] onehot_image must have a shape.")

        self.feat_dim = int(np.prod(dir_space.shape) + np.prod(img_space.shape))

        # Policy chooses option indices uniformly
        self.policy = OptionRandomSFPolicy(num_options=len(self.options_lst), hyper_params=self.hp, device=self.device)

        # Codebook aligns to options (K == len(options_lst))
        self.code_book = SFCodeBook(
            hyper_params=self.hp.codebook,
            num_codes=len(self.options_lst),
            feat_dim=self.feat_dim,
            device=self.device,
            init_embs=init_option_embs,
        )

        # Per-env execution bookkeeping
        self.running_option_index = [None for _ in range(self.num_envs)]
        self.option_start_obs = [None for _ in range(self.num_envs)]
        self.option_num_steps = [0 for _ in range(self.num_envs)]

        # SF accumulators (per env)
        self.sf_start_feat = [torch.zeros(self.feat_dim, device=self.device, dtype=torch.float32) for _ in range(self.num_envs)]
        self.sf_cumulative_feat = [torch.zeros(self.feat_dim, device=self.device, dtype=torch.float32) for _ in range(self.num_envs)]
        self.sf_cumulative_discounts = [0.0 for _ in range(self.num_envs)]

        # Small buffer to batch SF updates (optional but usually nicer)
        self.sf_buffer = [BaseBuffer(int(self.hp.sf_rollout_steps)) for _ in range(self.num_envs)]

        self._init_log_buf()

    def _init_log_buf(self):
        self.log_buf = []
        for _ in range(self.num_envs):
            self.log_buf.append({
                "num_options": [],
                "option_index": [],
                "code_book": [],
            })

    def act(self, observation):
        action = []

        for i in range(self.num_envs):
            obs_i = get_single_observation_nobatch(observation, i)

            # start option if none running
            if self.running_option_index[i] is None:
                opt_idx = int(self.policy.select_option_index())
                self.running_option_index[i] = opt_idx
                self.option_start_obs[i] = get_single_observation(observation, i)
                self.option_num_steps[i] = 0

                # initialize SF accumulators from start obs
                start_feat_np = _dir_img_feat(obs_i)
                start_feat_t = torch.as_tensor(start_feat_np, device=self.device, dtype=torch.float32)

                self.sf_start_feat[i].copy_(start_feat_t)
                self.sf_cumulative_feat[i].zero_()
                self.sf_cumulative_discounts[i] = 0.0

            # execute current option
            curr_idx = self.running_option_index[i]
            a = self.options_lst[curr_idx].select_action(obs_i)
            action.append(a)

        return action
        
    def update_buffers(self, observation, reward, terminated, truncated, call_back=None):
        # add SF step contribution (current obs)
        for i in range(self.num_envs):
            obs_option = get_single_observation_nobatch(observation, i)
            curr_option_idx = self.running_option_index[i]
            
            if curr_option_idx is None:
                continue
            
            feat_np = _dir_img_feat(obs_option)
            feat_t = torch.as_tensor(feat_np, device=self.device, dtype=torch.float32)
            
            self.option_num_steps[i] += 1
            
            w = (self.hp.gamma ** (self.option_num_steps[i] - 1))
            self.sf_cumulative_feat[i] += w * feat_t
            self.sf_cumulative_discounts[i] += w
            
            
            if self.options_lst[curr_option_idx].is_terminated(obs_option) or terminated[i] or truncated[i]:
                delta_sf = self.sf_cumulative_feat[i] - (self.sf_cumulative_discounts[i] * self.sf_start_feat[i])
                delta_sf = delta_sf / (self.sf_cumulative_discounts[i] + 1e-8)
               
                self.log_buf[i]["num_options"].append(np.array([len(self.options_lst)]))
                self.log_buf[i]["option_index"].append(np.array([curr_option_idx]))
                
                if call_back is not None:
                    call_back({"curr_hl_option_idx": curr_option_idx,
                                "num_options": len(self.options_lst)})
                    
                transition = (
                    int(curr_option_idx),
                    delta_sf.detach().cpu().numpy(),
                    self.sf_start_feat[i],
                )
                self.options_lst[curr_option_idx].reset()
                self.running_option_index[i] = None
                
                self.sf_buffer[i].add(transition)
        
    def update_codebook(self, observation, reward, terminated, truncated, call_back=None):
        for i in range(self.num_envs):
            if self.sf_buffer[i].is_full():
                batch = self.sf_buffer[i].all()
                opt_idx, delta_sf, start_feat = zip(*batch)
                
                self.code_book.update(opt_idx, delta_sf, start_feat, call_back=call_back)
                self.sf_buffer[i].clear()

    def update(self, observation, reward, terminated, truncated, call_back=None):
        if self.training:
            self.update_buffers(observation, reward, terminated, truncated, call_back=call_back)
            self.update_codebook(observation, reward, terminated, truncated, call_back=call_back)
            for i in range(self.num_envs):
                if terminated[i] or truncated[i]:
                    self.log_buf[i]["code_book"].append(self.code_book.emb.weight.detach().cpu().numpy())
                    
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
                

    def reset(self, seed):
        super().reset(seed)
        self.policy.reset(seed)
        self.code_book.reset(seed)

        self.running_option_index = [None for _ in range(self.num_envs)]
        self.option_start_obs = [None for _ in range(self.num_envs)]
        self.option_num_steps = [0 for _ in range(self.num_envs)]

        self.sf_start_feat = [torch.zeros(self.feat_dim, device=self.device, dtype=torch.float32) for _ in range(self.num_envs)]
        self.sf_cumulative_feat = [torch.zeros(self.feat_dim, device=self.device, dtype=torch.float32) for _ in range(self.num_envs)]
        self.sf_cumulative_discounts = [0.0 for _ in range(self.num_envs)]

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