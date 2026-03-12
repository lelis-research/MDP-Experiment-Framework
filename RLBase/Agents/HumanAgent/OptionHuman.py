import numpy as np
from gymnasium.spaces import Discrete
import torch

from .Human import HumanAgent
from ..Utils.HelperFunctions import get_single_observation_nobatch
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX

IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}
IDX_TO_COLOR  = {v: k for k, v in COLOR_TO_IDX.items()}
IDX_TO_STATE  = {v: k for k, v in STATE_TO_IDX.items()}

O = len(OBJECT_TO_IDX)
C = len(COLOR_TO_IDX) + 1
S = len(STATE_TO_IDX) + 1

def channel_to_name(ch):
    """
    Given a channel index, return what it represents.
    """

    if ch < O:
        return f"object:{IDX_TO_OBJECT[ch]}"

    elif ch < O + C:
        if ch - O == len(COLOR_TO_IDX):
            return "color:none"
        return f"color:{IDX_TO_COLOR[ch - O]}"

    elif ch < O + C + S:
        if ch - O - C == len(STATE_TO_IDX):
            return "state:none"
        return f"state:{IDX_TO_STATE[ch - O - C]}"

    else:
        raise ValueError(f"Invalid channel index: {ch}")

def decode_tensor(vec):
    idxs = vec.nonzero(as_tuple=True)[0].tolist()
    return [channel_to_name(i) for i in idxs]
    
class OptionHumanAgent(HumanAgent):
    """
    Simple human-in-the-loop agent.
    - No options
    - No assumptions about observation structure
    - Prints obs['text'] if available
    """
    name = "OptionHuman"
    SUPPORTED_ACTION_SPACES = (Discrete, )

    def __init__(self, action_space, observation_space, hyper_params,
                 num_envs, feature_extractor_class, init_option_lst, device="cpu"):
        super().__init__(action_space, observation_space, hyper_params,
                         num_envs, feature_extractor_class, device=device)

        self.options_lst = init_option_lst
        print(f"Num Options: {len(self.options_lst)}")
        
        self.running_option_index = None
        self.option_cumulative_reward = 0.0
        self.option_multiplier = 1.0
        
        self.sf_actions = [[] for _ in range(self.num_envs)]
        self.sf_observations = [[] for _ in range(self.num_envs)]

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
        
    # ----------------------------------------------------------
    # ACTION SELECTION
    # ----------------------------------------------------------
    def act(self, observation, greedy=False):
        if self.num_envs != 1:
            raise ValueError("OptionHumanAgent is intended for num_envs == 1.")
        
        obs_option = get_single_observation_nobatch(observation, 0)
        
        if self.running_option_index is not None:
            action = self.options_lst[self.running_option_index].select_action(obs_option)
        else:
            self._analyze_obs(observation)
            self.print_action_menu()
            
            while True:  # option selected
                action = self._read_user_action()
                if action < self.action_space.n:
                    break
                opt_idx = action - self.action_space.n
                valid_option = self.options_lst[opt_idx].can_initiate(obs_option)
                if valid_option:
                    break
                        
            if action >= self.action_space.n:
                self.running_option_index = action - self.action_space.n
                
                action = self.options_lst[self.running_option_index].select_action(obs_option)
                self.option_cumulative_reward = 0.0
                self.option_multiplier = 1.0
                
                self.sf_bookkeeping(obs_option, 0, action=None, mode="s")
        action = np.array([action], dtype=np.int64)
        self.last_action = action
        return action

    # ----------------------------------------------------------
    # UPDATE / RESET
    # ----------------------------------------------------------
    def update(self, observation, reward, terminated, truncated, call_back=None):
        reward = reward[0]
        # observation = get_single_observation(observation, 0)
        obs_option = get_single_observation_nobatch(observation, 0)
        
        terminated, truncated = terminated[0], truncated[0]
        
        self.sf_bookkeeping(obs_option, 0, self.last_action[0], mode="m")
        
        if self.running_option_index is None:
            print("Reward:", reward)
        else:
            self.option_cumulative_reward += self.option_multiplier * reward
            self.option_multiplier *= self.hp.gamma
            
            if terminated or truncated or self.options_lst[self.running_option_index].is_terminated(obs_option):
                action_seq, obs_seq = self.sf_bookkeeping(obs_option, 0, action=None, mode="f")
                self.option_analysis(obs_seq)
                print("Option Reward:", self.option_cumulative_reward)
                self.options_lst[self.running_option_index].reset()
                self.running_option_index = None
                self.option_cumulative_reward = 0.0
                self.option_multiplier = 1.0
        
        if terminated or truncated:
            print("Episode finished.")
        
        

    def reset(self, seed):
        super().reset(seed)
        
        self.running_option_index = None
        self.option_cumulative_reward = 0.0
        self.option_multiplier = 1.0
        
        self.sf_actions = [[] for _ in range(self.num_envs)]
        self.sf_observations = [[] for _ in range(self.num_envs)]
    
    def option_analysis(self, obs_seq):
        first, last, d_last, sf, d_sf, sf_reverse, d_sf_reverse = self.option_sequence_features(obs_seq)
        d_last_t = d_last.permute(2, 0, 1).contiguous()  # (C,H,W)
        eps = 0.0  # 1e-6 if floats
        abs_d = d_last_t.abs()
        channel_changed = (abs_d.sum(dim=(1,2), keepdim=True) > eps).to(d_last_t.dtype).flatten(start_dim=0)  # (C,)
        
        print("*" * 50)
        print(decode_tensor(channel_changed))
        print("*" * 50)
    
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

    # ----------------------------------------------------------
    # MENU & INPUT
    # ----------------------------------------------------------
    def print_action_menu(self):
        print("\n" + "=" * 40)
        print("Available Actions")
        print("=" * 40)

        if hasattr(self.hp, "actions_enum"):
            actions_enum = [a.name for a in self.hp.actions_enum]
            print(actions_enum)
        else:
            print(f"Atomic actions: 0 .. {self.action_space.n - 1}")
        
        for i, option in enumerate(self.options_lst):
            if hasattr(option, 'option_id'):
                print(f"Option {i + self.action_space.n}: {option.option_id}")
            else:
                print(f"Option {i + self.action_space.n}: {option.__class__.__name__}")
        # print(f"Options: {self.action_space.n} .. {self.action_space.n + len(self.options_lst) - 1}")

        print("=" * 40)

    def _read_user_action(self):
        while True:
            a = input(f"Action [0..{self.action_space.n - 1}] Option [{self.action_space.n} .. {self.action_space.n + len(self.options_lst) - 1}] (or 'q' to quit): ")
            if a.strip().lower() == "q":
                exit(0)
            try:
                val = int(a)
                if 0 <= val < self.action_space.n + len(self.options_lst):
                    return val
            except:
                pass
            print("Invalid input.")

    # ----------------------------------------------------------
    # OBSERVATION ANALYSIS (your requested change)
    # ----------------------------------------------------------
    def _analyze_obs(self, observation):
        """
        Only print observation['text'] if it exists (and is a string).
        Otherwise print nothing.
        """
        if isinstance(observation, dict) and "text" in observation:
            txt = observation["text"]
            print("\n--- TEXT ---")
            print(txt[0])
            print("------------")