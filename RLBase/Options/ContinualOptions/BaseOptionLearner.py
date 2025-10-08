
class BaseContinualOptionLearner():
    name="BaseContinualOptionLearner"
    
    def evaluate_option_trigger(self, last_observation, last_action, observation, reward, options_lst):
        pass
    
    def extract_options(self, options_lst):
        pass
    
    def init_options(self, policy):
        pass
    
    def reset(self):
        pass

  

    
