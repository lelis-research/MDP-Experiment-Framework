from . import BaseContinualOptionLearner
from ..ManualSymbolicOptions import FindKeyOption, FindGoalOption, OpenDoorOption

import numpy as np
from minigrid.core.constants import COLOR_NAMES, IDX_TO_OBJECT, DIR_TO_VEC, OBJECT_TO_IDX

class TabularContinualOptionLearner(BaseContinualOptionLearner):
    name = "TabularContinualOptionLearner"
    
    def __init__(self):
        self.buffer = []
        
        self.agent_id = OBJECT_TO_IDX["agent"]
        self.key_id = OBJECT_TO_IDX["key"]
        self.wall_id = OBJECT_TO_IDX["wall"]
        self.door_id = OBJECT_TO_IDX["door"]

        
        self.add_find_key_option = False
        self.add_open_door_option = False
    
    def evaluate_option_trigger(self, last_observation, last_action, observation, reward, options_lst):
        flag = False
        img = observation["image"]
        
        key_pos = np.argwhere(img[..., 0] == self.key_id)
        if len(key_pos) == 0:
            # There is no keys -> it has been picked up
            self.add_find_key_option = True
            flag = True
                    
        door_pos = np.argwhere(img[..., 0] == self.door_id)[0]
        door_opened = img[door_pos[0], door_pos[1], 2] == 0 
        if door_opened:
            self.add_open_door_option = True
            flag = True
        
        return flag

        
    
    def extract_options(self, options_lst):
        if not any(isinstance(opt, FindKeyOption) for opt in options_lst) and self.add_find_key_option:
            option = FindKeyOption(option_len=20)
            options_lst.append(option)
        
        if not any(isinstance(opt, OpenDoorOption) for opt in options_lst) and self.add_open_door_option:
            option = OpenDoorOption(option_len=20)
            options_lst.append(option)
    
    def init_options(self, policy):
        pass
    
    def reset(self):
        self.buffer = []
        
        self.agent_id = OBJECT_TO_IDX["agent"]
        self.key_id = OBJECT_TO_IDX["key"]
        self.wall_id = OBJECT_TO_IDX["wall"]
        
        self.add_find_key_option = False
        self.add_open_door_option = False