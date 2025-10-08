from . import BaseContinualOptionLearner
from ..ManualSymbolicOptions import FindKeyOption, FindGoalOption, OpenDoorOption

import numpy as np
from minigrid.core.constants import COLOR_NAMES, IDX_TO_OBJECT, DIR_TO_VEC, OBJECT_TO_IDX

class TabularContinualOptionLearner(BaseContinualOptionLearner):
    name = "TabularContinualOptionLearner"
    
    def __init__(self):
        self.buffer = []
    
    def evaluate_option_trigger(self, last_observation, last_action, observation, reward, options_lst):
        img = observation["image"]
        key_pos = np.argwhere(img[..., 0] == self.key_id)
        if len(key_pos) == 0:
            # There is no keys -> it has been picked up
            options_lst.append(FindKeyOption)

        
    
    def extract_options(self, current_options):
        pass
    
    def init_options(self, policy):
        pass
    
    def reset(self):
        pass