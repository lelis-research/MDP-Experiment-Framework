import numpy as np
from gymnasium.spaces import Discrete
from ..Utils import BaseAgent
from minigrid.core.constants import OBJECT_TO_IDX, DIR_TO_VEC

A_LEFT, A_RIGHT, A_FORWARD, A_PICKUP, A_DROP, A_TOGGLE, A_DONE = 0, 1, 2, 3, 4, 5, 6

class HumanAgent(BaseAgent):
    """
    Human-in-the-loop agent that can run primitive actions or high-level options.
    Adds detailed logging so we can debug option behavior end-to-end.
    """
    name = "Human"

    def __init__(self, action_space, observation_space, hyper_params, num_envs,
                 feature_extractor_class, options_lst, device="cpu", verbose_obs=False):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device=device)
        self.atomic_action_space = action_space
        self.options_lst = options_lst
        self.action_space = Discrete(self.atomic_action_space.n + len(self.options_lst))
        self.running_option_index = None
        self.verbose_obs = verbose_obs

        print(f"Number of options: {len(self.options_lst)}")
        print(f"Observation space: {observation_space}")
        
        self.option_info = []

    def act(self, observation, greedy=False):
        """Select either an atomic action or continue a running option."""
        self._analyze_obs(observation)

        # You still run your feature extractor; no behavior change
        state = self.feature_extractor(observation)

        # 1) If we have a running option, continue it unless it has just terminated
        if self.running_option_index is not None:
            opt = self.options_lst[self.running_option_index]
            if opt.is_terminated(observation):
                
                print(f"Option {self.running_option_index} info: {self.option_info}")
                self.option_info = []
                
                self.running_option_index = None
            else:
                action = opt.select_action(observation)
                self.option_info.append(action)
                
                self.last_state, self.last_action = state, action
                return action

        # 2) No running option: show menu and take user choice
        self.print_action_menu()
        action = self._read_user_action()

        # Atomic action
        if action < self.atomic_action_space.n:
            self.last_state, self.last_action = state, action
            return action

        # Start an option
        self.running_option_index = action - self.atomic_action_space.n
        opt = self.options_lst[self.running_option_index]
        action = opt.select_action(observation)
        
        self.option_info = [action]

        self.last_state, self.last_action = state, action
        return action

    def update(self, observation, reward, terminated, truncated, call_back):
        """Keep your current behavior; just add a tiny log."""
        if truncated or terminated:
            self.running_option_index = None
        print("Reward:", reward)

    def reset(self, seed):
        super().reset(seed)
        self.feature_extractor.reset(seed)
        # If an option was mid-flight last episode, drop it cleanly
        self.running_option_index = None

    # ---------------- UI & logs ----------------

    def print_action_menu(self):
        print("\n" + "=" * 35)
        print("🌟  Available Actions  🌟")
        print("=" * 35)
        print("Atomic actions:", [i for i in self.hp.actions_enum])
        print("\nOptions:")
        base = self.atomic_action_space.n
        line_elems = []
        for i, opt in enumerate(self.options_lst):
            item = f"{base + i:02d}: {opt}"
            line_elems.append(item)
            if len(line_elems) == 3:
                print("   " + " | ".join(line_elems)); line_elems = []
        if line_elems:
            print("   " + " | ".join(line_elems))
        print("=" * 40 + "\n")

    def _read_user_action(self):
        while True:
            a = input("Action: ")
            try:
                a = int(a)
                if 0 <= a < self.action_space.n:
                    return a
            except Exception:
                if a == "q":
                    exit(0)
            print(f"Please enter an integer in [0, {self.action_space.n - 1}]")



    def _analyze_obs(self, observation):
        print("\n\n****** Observation Description ******")
        img = observation["image"]
        # if self.verbose_obs:
        #     print("Observation img:\n", img)
        # print(f"img shape: {img.shape}")
        agent_pos = np.argwhere(img[..., 0] == OBJECT_TO_IDX["agent"])
        print("Agent pos:", agent_pos)
        
        # Keep the rest commented unless you need them:
        # key_pos = np.argwhere(img[..., 0] == OBJECT_TO_IDX["key"])
        # door_pos = np.argwhere(img[..., 0] == OBJECT_TO_IDX["door"])
        # goal_pos = np.argwhere(img[..., 0] == OBJECT_TO_IDX["goal"])