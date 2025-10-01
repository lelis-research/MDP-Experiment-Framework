from ..Utils import BaseOption
from ..Utils import discrete_levin_loss_on_trajectory
from ...registry import register_option
from ...loaders import load_policy, load_feature_extractor
from ..Utils import save_options_list, load_options_list

import random
import torch
import numpy as np
from tqdm import tqdm
import copy
from multiprocessing import Pool
import os
import numpy as np
import shutil
from minigrid.core.constants import COLOR_NAMES, IDX_TO_OBJECT, DIR_TO_VEC, OBJECT_TO_IDX

@register_option
class FindGoalOption(BaseOption):
    """
    Specifically works for minigrid
    """
    def __init__(self, option_len):
        self.agent_id = OBJECT_TO_IDX["agent"]
        self.goal_id = OBJECT_TO_IDX["goal"]
        self.wall_id = OBJECT_TO_IDX["wall"]
        self.option_len = option_len
        self.step_counter = 0
        

    # ----------------------- Option API -----------------------

    def select_action(self, observation):
        self.step_counter += 1
        img = observation["image"]
        
        goal_pos = np.argwhere(img[..., 0] == self.goal_id) 
        agent_pos = np.argwhere(img[..., 0] == self.agent_id)[0] 
       
        if len(goal_pos) > 0:
            # more than 0 goal exists
            goal_pos = goal_pos[0]
        else:
            return 6 # action done (do nothing because there is no goal!)
        
        goal_direction = goal_pos - agent_pos
        agent_direction = DIR_TO_VEC[observation["direction"]]
        
        if np.array_equal(goal_direction, agent_direction):
            return 2 # forward get the goal
        
        goal_direction = np.sign(goal_direction)
        
        if agent_direction[0] == goal_direction[0] != 0 or agent_direction[1] == goal_direction[1] != 0:
            agent_forward = agent_pos + agent_direction # front of the agent
            if img[tuple(agent_forward)][2] == self.wall_id:
                # if there is a wall in front of the agent turn
                return 0 #left
            return 2 # forward
        else:
            return 0 #left
        

    def is_terminated(self, observation):
        img = observation["image"]
        goal_pos = np.argwhere(img[..., 0] == self.goal_id) 
        if len(goal_pos) == 0 or self.step_counter >= self.option_len:
            # no keys are in the observation
            self.step_counter = 0
            return True
        return False

    def save(self, file_path=None):
        checkpoint = {
            "option_class": self.__class__.__name__,
            "option_len": self.option_len
        }
        if file_path is not None:
            # Save checkpoint (.t)
            torch.save(checkpoint, f"{file_path}_options.t")

            # Also copy this source file as a .py snapshot
            src_file = os.path.abspath(__file__)
            dst_file = f"{file_path}_{self.__class__.__name__}.py"
            shutil.copyfile(src_file, dst_file)

        return checkpoint

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        
        instance = cls(checkpoint["option_len"])
        return instance
    
   