from ..Utils import BaseOption
from ..Utils import discrete_levin_loss_on_trajectory
from ...registry import register_option
from ...loaders import load_policy, load_feature_extractor
from ..Utils import save_options_list, load_options_list
from .FindKeyOption import FindKey

import random
import torch
import copy
import os
class ManualSymbolicOptionLearner():
    name="ManualSymbolicOptionLearner"
    def __init__(self, hyper_params=None):
        self.hyper_params = hyper_params
      
    def learn(self, exp_dir=None):
        self.exp_dir = exp_dir
        
        self.options_lst = [FindKey()]
        save_options_list(self.options_lst, os.path.join(self.exp_dir, "FindKey.t"))
            
                    
        return self.options_lst
    

  

