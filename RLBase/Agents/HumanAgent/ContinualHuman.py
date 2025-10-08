import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import Discrete

from .Human import HumanAgent
from ..Utils import (
    BasicBuffer,
    BaseAgent,
)
from minigrid.core.constants import IDX_TO_OBJECT, DIR_TO_VEC, OBJECT_TO_IDX

class ContinualHumanAgent(HumanAgent):
    """
    Deep Q-Network (DQN) agent that uses experience replay and target networks.
    
    Args:
        action_space (gym.spaces.Discrete): The environment's action space.
        observation_space: The environment's observation space.
        hyper_params: Hyper-parameters container (see DQNPolicy).
        num_envs (int): Number of parallel environments.
        feature_extractor_class: Class to extract features from observations.
    """
    name = "ContinualHuman"
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, option_learner_class, device="cpu"):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device=device)
        self.atomic_action_space = action_space
        self.options_lst = []
        
        self.option_learner = option_learner_class()

        # action space includes actions and options
        self.action_space = Discrete(self.atomic_action_space.n + len(self.options_lst)) 


        self.running_option_index = None
        
    def act(self, observation, greedy=False):
        self.last_observation = observation
        
        action = super().act(observation, greedy)
        return action
    
    def update(self, observation, reward, terminated, truncated, call_back):
        if self.option_learner.evaluate_option_trigger(self.last_observation, self.last_action, observation, reward, self.options_lst):
            self.options_lst = self.option_learner.extract_options(self.options_lst)
            # self.option_learner.init_options(self.policy)
            
        if truncated or terminated:
            self.running_option_index = None
            
        super().update(observation, reward, terminated, truncated, call_back)

        
          
    def reset(self, seed):
        """
        Reset the agent's learning state, including feature extractor and replay buffer.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.option_learner.reset()

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
        

