import torch

class BaseContinualOptionLearner():
    name = "BaseContinualOptionLearner"
    
    def __init__(self):
        pass
    
    def update(self):
        # get necessary data from the agent
        pass

        
    
    def learn(self):
        # return a new set of options
        pass
    
        
    
    def reset(self, seed):
       pass
   
    
    def save(self, file_path=None):
        checkpoint = {}
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_option_learner.t")
        
        return checkpoint

   
    def load_from_checkpoint(self, checkpoint):
       pass