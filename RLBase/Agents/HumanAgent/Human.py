import numpy as np
from gymnasium.spaces import Discrete

from ..Base import BaseAgent
from ..Utils.HelperFunctions import get_single_observation

class HumanAgent(BaseAgent):
    """
    Simple human-in-the-loop agent.
    - No options
    - No assumptions about observation structure
    - Prints obs['text'] if available
    """
    name = "Human"
    SUPPORTED_ACTION_SPACES = (Discrete, )

    def __init__(self, action_space, observation_space, hyper_params,
                 num_envs, feature_extractor_class, device="cpu"):
        super().__init__(action_space, observation_space, hyper_params,
                         num_envs, feature_extractor_class, device=device)

        print("HumanAgent initialized.")
        print("Action space:", action_space)
        print("Observation space:", observation_space)

    # ----------------------------------------------------------
    # ACTION SELECTION
    # ----------------------------------------------------------
    def act(self, observation, greedy=False):
        if self.num_envs != 1:
            raise ValueError("HumanAgent is intended for num_envs == 1.")

        # Only print text if exists
        self._analyze_obs(get_single_observation(observation, 0))

        # still run feature extractor
        state = self.feature_extractor(observation)

        # print menu & ask
        self.print_action_menu()
        action = self._read_user_action()

        # log
        self.last_state, self.last_action = state, action

        return np.array([action], dtype=np.int64)

    # ----------------------------------------------------------
    # UPDATE / RESET
    # ----------------------------------------------------------
    def update(self, observation, reward, terminated, truncated, call_back=None):
        print("Reward:", reward)
        if terminated or truncated:
            print("Episode finished.")

    def reset(self, seed):
        super().reset(seed)
        self.feature_extractor.reset(seed)

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

        print("=" * 40)

    def _read_user_action(self):
        while True:
            a = input(f"Action [0..{self.action_space.n - 1}] (or 'q' to quit): ")
            if a.strip().lower() == "q":
                exit(0)
            try:
                val = int(a)
                if 0 <= val < self.action_space.n:
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