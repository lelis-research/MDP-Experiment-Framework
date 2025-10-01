import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import Discrete

from ..Utils import (
    BasicBuffer,
    BaseAgent,
)
from minigrid.core.constants import IDX_TO_OBJECT, DIR_TO_VEC, OBJECT_TO_IDX

class HumanAgent(BaseAgent):
    """
    Deep Q-Network (DQN) agent that uses experience replay and target networks.
    
    Args:
        action_space (gym.spaces.Discrete): The environment's action space.
        observation_space: The environment's observation space.
        hyper_params: Hyper-parameters container (see DQNPolicy).
        num_envs (int): Number of parallel environments.
        feature_extractor_class: Class to extract features from observations.
    """
    name = "Human"
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, options_lst, device="cpu"):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device=device)
        self.atomic_action_space = action_space
        self.options_lst = options_lst

        print(f"Number of options: {len(self.options_lst)}")
        # action space includes actions and options
        self.action_space = Discrete(self.atomic_action_space.n + len(self.options_lst)) 


        self.running_option_index = None
        
    def act(self, observation, greedy=False):
        """
        Select an action based on the current observation.
        
        Args:
            observation (np.array or similar): Raw observation from the environment.
        
        Returns:
            int: Selected action.
        """
        self.analyze_obs(observation)

        state = self.feature_extractor(observation)
         # If an options is running currently
        if self.running_option_index is not None:
            # Check if the option should terminate.
            if self.options_lst[self.running_option_index].is_terminated(observation):
                self.running_option_index = None
            else:
                # Continue executing the currently running option.
                action = self.options_lst[self.running_option_index].select_action(observation)
                print(action, end=",")
        if self.running_option_index is None:
            self.print_action_menu()
            action = int(input("Action:"))
            if action >= self.atomic_action_space.n:
                self.running_option_index = action - self.atomic_action_space.n
                action = self.options_lst[self.running_option_index].select_action(observation)
                print(action, end=",")

        self.last_state = state
        self.last_action = action
        return action
    
    def update(self, observation, reward, terminated, truncated, call_back):
        """
        Store the transition and, if enough samples are available, perform a learning step.
        
        Args:
            observation (np.array or similar): New observation after action.
            reward (float): Reward received.
            terminated (bool): True if the episode has terminated.
            truncated (bool): True if the episode was truncated.
            call_back (function): Callback function to track training progress.
        """
        if truncated or terminated:
            self.running_option_index = None

        
          
    def reset(self, seed):
        """
        Reset the agent's learning state, including feature extractor and replay buffer.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        self.feature_extractor.reset(seed)

    def print_action_menu(self):
        print("")
        print("****** Available Actions ******")
        print("Atomic actions:", [i for i in self.hp.actions_enum])
        print("Options:", list(range(self.atomic_action_space.n, self.atomic_action_space.n+len(self.options_lst))))

    def analyze_obs(self, observation):
        print("")
        print("")
        print("****** Observation Description ******")
        
        img = observation["image"]
        print(f"img shape: {img.shape}")
       
        agent_id = OBJECT_TO_IDX["agent"]
                
        agent_pos = np.argwhere(img[..., 0] == agent_id)
        agent_direction = DIR_TO_VEC[observation["direction"]]
        print(agent_pos)
        print(observation.keys())
        
        
        
    def save(self, file_path=None):
        pass

    @classmethod
    def load_from_file(cls, file_path, seed=0, checkpoint=None):
        pass
        

