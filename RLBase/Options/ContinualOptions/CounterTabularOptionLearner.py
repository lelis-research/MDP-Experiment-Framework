from . import BaseContinualOptionLearner
from ..ManualSymbolicOptions import FindKeyOption, FindGoalOption, OpenDoorOption

import numpy as np
from minigrid.core.constants import COLOR_NAMES, IDX_TO_OBJECT, DIR_TO_VEC, OBJECT_TO_IDX
from gymnasium.spaces import Discrete

class CounterTabularContinualOptionLearner(BaseContinualOptionLearner):
    name = "CounterTabularContinualOptionLearner"
    
    def __init__(self):
        
        self.agent_id = OBJECT_TO_IDX["agent"]
        self.key_id = OBJECT_TO_IDX["key"]
        self.wall_id = OBJECT_TO_IDX["wall"]
        self.door_id = OBJECT_TO_IDX["door"]

        self.counter = 0
    
    def evaluate_option_trigger(self, last_observation, last_action, observation, reward, options_lst):
        self.counter += 1
        if self.counter >= 50000:
            self.counter = 0
            return True
        return False

        
    
    def extract_options(self, options_lst):
        if len(options_lst) == 0:
            option = FindKeyOption(option_len=20)
            options_lst.append(option)
        elif len(options_lst) == 1:
            option = OpenDoorOption(option_len=20)
            options_lst.append(option)
        self.num_options = len(options_lst)
       
    
    def init_options(self, policy):
        mode = "reset"
        policy.action_space = Discrete(policy.action_space.n + self.num_options)
        policy.action_dim = int(policy.action_space.n)
        
        for state in policy.q_table:
            if mode == "reset":
                policy.q_table = np.zeros(policy.action_dim)
            # old_q_values = policy.q_table[state]
            # num_new = len(self.new_options)
            # new_q_values = np.zeros(num_new, dtype=old_q_values.dtype)
            # policy.q_table[state] = np.concatenate([old_q_values, new_q_values])
        
    
    def reset(self):
        self.agent_id = OBJECT_TO_IDX["agent"]
        self.key_id = OBJECT_TO_IDX["key"]
        self.wall_id = OBJECT_TO_IDX["wall"]
        self.counter = 0