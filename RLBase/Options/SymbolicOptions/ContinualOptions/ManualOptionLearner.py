from . import BaseContinualOptionLearner
from ..ManualSymbolicOptions import ALL_OPTIONS, FindKeyOption, OpenDoorOption

import numpy as np
from minigrid.core.constants import COLOR_NAMES, IDX_TO_OBJECT, DIR_TO_VEC, OBJECT_TO_IDX
from gymnasium.spaces import Discrete
import torch
import random

class ManualContinualOptionLearner(BaseContinualOptionLearner):
    name = "ManualContinualOptionLearner"
        
    def learn_old(self, options_lst):        
        if not any(isinstance(opt, FindKeyOption) for opt in options_lst):
            option = FindKeyOption(option_len=20)
            return [option]
        
        elif not any(isinstance(opt, OpenDoorOption) for opt in options_lst):
            option = OpenDoorOption(option_len=20)
            return [option]
        
        return []
    
    def learn(self, options_lst):
        """
        Randomly select a new option class from ALL_OPTIONS that isn't already
        in options_lst, instantiate it, and return it as a list.
        """
        # Get the classes of currently learned options
        existing_classes = {type(opt) for opt in options_lst}

        # Filter out the ones already in options_lst
        remaining = [opt_cls for opt_cls in ALL_OPTIONS if opt_cls not in existing_classes]

        if not remaining:
            # No new options available
            return []

        # Pick one at random
        new_opt_cls = random.choice(remaining)
        new_option = new_opt_cls(option_len=20)

        return [new_option]

   
