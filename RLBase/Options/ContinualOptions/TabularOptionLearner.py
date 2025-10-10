from . import BaseContinualOptionLearner
from ..ManualSymbolicOptions import FindKeyOption, FindGoalOption, OpenDoorOption

import numpy as np
from minigrid.core.constants import COLOR_NAMES, IDX_TO_OBJECT, DIR_TO_VEC, OBJECT_TO_IDX
from gymnasium.spaces import Discrete

class TabularContinualOptionLearner(BaseContinualOptionLearner):
    name = "TabularContinualOptionLearner"
    
    def __init__(self):
        
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
        if len(key_pos) == 0 and not self.add_find_key_option:
            # There is no keys -> it has been picked up
            self.add_find_key_option = True
            flag = True
                    
        door_pos = np.argwhere(img[..., 0] == self.door_id)
        if len(door_pos) > 0 and not self.add_open_door_option:
            door_pos = door_pos[0]
            door_opened = img[door_pos[0], door_pos[1], 2] == 0 
            if door_opened:
                self.add_open_door_option = True
                flag = True
        
        return flag

        
    
    def extract_options(self, options_lst):
        self.new_options = []
        if not any(isinstance(opt, FindKeyOption) for opt in options_lst) and self.add_find_key_option:
            option = FindKeyOption(option_len=20)
            options_lst.append(option)
            self.new_options.append(option)
        
        if not any(isinstance(opt, OpenDoorOption) for opt in options_lst) and self.add_open_door_option:
            option = OpenDoorOption(option_len=20)
            options_lst.append(option)
            self.new_options.append(option)
    
    def init_options(self, policy):
        policy.action_space = Discrete(policy.action_space.n + len(self.new_options))
        policy.action_dim = int(policy.action_space.n)
        
        for state in policy.q_table:
            old_q_values = policy.q_table[state]
            num_new = len(self.new_options)
            new_q_values = np.zeros(num_new, dtype=old_q_values.dtype)
            policy.q_table[state] = np.concatenate([old_q_values, new_q_values])
        
    
    def reset(self):
        
        self.agent_id = OBJECT_TO_IDX["agent"]
        self.key_id = OBJECT_TO_IDX["key"]
        self.wall_id = OBJECT_TO_IDX["wall"]
        
        self.add_find_key_option = False
        self.add_open_door_option = False
        self.new_options = []