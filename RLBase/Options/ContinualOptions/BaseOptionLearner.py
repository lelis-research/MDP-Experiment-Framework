
class BaseContinualOptionLearner():
    name="BaseContinualOptionLearner"
    
    def evaluate_option_trigger(self, last_observation, last_action, observation, reward, options_lst):
        pass
    
    def extract_options(self, current_options):
        pass
    
    def init_options(self, policy):
        pass
    
    def reset(self):
        pass

  

    
