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

        print("Action space:", action_space)
        print("Observation space:", observation_space)
        self._last_admissible_commands = []

    # ----------------------------------------------------------
    # ACTION SELECTION
    # ----------------------------------------------------------
    def act(self, observation, greedy=False):
        if self.num_envs != 1:
            raise ValueError("HumanAgent is intended for num_envs == 1.")

        self._analyze_obs(get_single_observation(observation, 0))


        # print menu & ask
        self.print_action_menu()
        action = self._read_user_action()

        return np.array([action], dtype=np.int64)

    # ----------------------------------------------------------
    # UPDATE / RESET
    # ----------------------------------------------------------
    def update(self, observation, reward, terminated, truncated, call_back=None):
        reward = reward[0]
        terminated, truncated = terminated[0], truncated[0]
        
        print("Reward:", reward)
        if terminated or truncated:
            print("Episode finished.")

    # ----------------------------------------------------------
    # MENU & INPUT
    # ----------------------------------------------------------
    def print_action_menu(self):
        print("\n" + "=" * 40)
        print("Available Actions")
        print("=" * 40)

        if hasattr(self.hp, "actions_enum") and self.hp.actions_enum is not None:
            actions_enum = [a.name for a in self.hp.actions_enum]
            print(actions_enum)
        elif self._last_admissible_commands:
            for i, cmd in enumerate(self._last_admissible_commands):
                print(f"  {i}: {cmd}")
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
        if isinstance(observation, dict):
            if "obs" in observation and isinstance(observation["obs"], str):
                print("\n" + observation["obs"])
            elif "text" in observation and isinstance(observation["text"], str):
                print("\n--- TEXT ---")
                print(observation["text"])
                print("------------")
            if "admissible_commands" in observation:
                cmds = observation["admissible_commands"]
                if isinstance(cmds, str) and cmds:
                    self._last_admissible_commands = cmds.split("\n")
                elif isinstance(cmds, (list, tuple)):
                    self._last_admissible_commands = [c for c in cmds if c]
                else:
                    self._last_admissible_commands = []