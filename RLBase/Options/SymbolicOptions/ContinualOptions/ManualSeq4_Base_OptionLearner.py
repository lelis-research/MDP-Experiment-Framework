
from gymnasium.spaces import Discrete
from ..ManualSymbolicOptions import FindGoalOption

class ManualSeq4_Base_OptionLearner():
    name="ManualSeq4_Base_OptionLearner"
    
    def __init__(self):
        self.counter = 0
        self.dsa = 0
        
    def evaluate_option_trigger(self, last_observation, last_action, observation, reward, options_lst):
        self.counter += 1
        if self.counter == 10000:
            self.counter = 0
            return True
        return False
    
    def extract_options(self, options_lst):
        self.new_options = []
        if self.dsa == 0:
            option = FindGoalOption(option_len=20, goal_color="red")
            options_lst.append(option)
            self.new_options.append(option)
            self.dsa += 1
        
        elif self.dsa == 1:
            option = FindGoalOption(option_len=20, goal_color="yellow")
            options_lst.append(option)
            self.new_options.append(option) 
            self.dsa += 1

        elif self.dsa == 2:
            option = FindGoalOption(option_len=20, goal_color="blue")
            options_lst.append(option)
            self.new_options.append(option)
            self.dsa += 1
        
        elif self.dsa == 3:
            option = FindGoalOption(option_len=20, goal_color="green")
            options_lst.append(option)
            self.new_options.append(option)
            self.dsa += 1

    def init_options(self, policy, init_epsilon):
        policy.action_space = Discrete(policy.action_space.n + len(self.new_options))

        policy.reset()

        policy.hp.epsilon = init_epsilon

    def reset(self):
        self.new_options = []

  

    
