from . import BaseContinualOptionLearner
from ..ManualSymbolicOptions import FindKeyOption, FindGoalOption, OpenDoorOption

import numpy as np
from minigrid.core.constants import COLOR_NAMES, IDX_TO_OBJECT, DIR_TO_VEC, OBJECT_TO_IDX
from gymnasium.spaces import Discrete
import torch

class ManualContinualOptionLearner(BaseContinualOptionLearner):
    name = "ManualContinualOptionLearner"
    
    def __init__(self):
        self.counter = 0
    
    def update(self):
        self.counter += 1

        
    
    def learn(self, options_lst):
        
        if not any(isinstance(opt, FindKeyOption) for opt in options_lst):
            option = FindKeyOption(option_len=20)
            return [option]
        
        elif not any(isinstance(opt, OpenDoorOption) for opt in options_lst):
            option = OpenDoorOption(option_len=20)
            return [option]
    
        
    
    def reset(self):
       pass
   
