import numpy as np
from gymnasium.spaces import Discrete

from .Human import HumanAgent
from ..Utils.HelperFunctions import get_single_observation_nobatch

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
            action = self._read_user_action()
            if action >= self.action_space.n:
                self.running_option_index = action - self.action_space.n
                action = self.options_lst[self.running_option_index].select_action(obs_option)
                self.option_cumulative_reward = 0.0
                self.option_multiplier = 1.0


        return np.array([action], dtype=np.int64)

    # ----------------------------------------------------------
    # UPDATE / RESET
    # ----------------------------------------------------------
    def update(self, observation, reward, terminated, truncated, call_back=None):
        reward = reward[0]
        # observation = get_single_observation(observation, 0)
        obs_option = get_single_observation_nobatch(observation, 0)
        terminated, truncated = terminated[0], truncated[0]
        
        
        if self.running_option_index is None:
            print("Reward:", reward)
        else:
            self.option_cumulative_reward += self.option_multiplier * reward
            self.option_multiplier *= self.hp.gamma
            
            if terminated or truncated or self.options_lst[self.running_option_index].is_terminated(obs_option):
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
        
        print(f"Options: {self.action_space.n} .. {self.action_space.n + len(self.options_lst) - 1}")

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