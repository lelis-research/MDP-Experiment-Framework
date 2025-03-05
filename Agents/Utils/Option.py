from typing import Any
import pickle

class MaskedOption():
 
    def __init__(self, base_policy, mask):
        """
        Initialize the Option with a base policy.
        
        Args:
            base_policy: The underlying policy or agent to be used by this option.
        """
        self.policy = base_policy
        self.mask = mask
        self.reset()
    
    def reset(self):
        self.step_counter = 0

    def learn(self):
        """
        Update the option's policy based on experience.
        
        Subclasses should override this method to implement their specific learning logic.
        """
        pass

    def act(self, state):
        """
        Select an action based on the current state.
        
        Args:
            state: The current state of the environment.
            
        Returns:
            An action selected according to the option's policy.
        """
        action = self.policy.select_action_masked(state, self.mask)
        self.step_counter += 1
        return action
    
    def terminate(self, state):
        if self.step_counter > 5:
            self.step_counter = 0
            return True
        return False
    
    def save(self, file_path):
        file_path_policy = f"{file_path}_policy.t"
        file_path_mask = f"{file_path}_mask.t"
        self.policy.save(file_path_policy)
        with open(file_path_mask, "wb") as file:
            pickle.dump(self.mask, file)