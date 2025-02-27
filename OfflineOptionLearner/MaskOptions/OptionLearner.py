from .PolicyMaskers import POLICY_TO_MASKER
from OfflineOptionLearner.Utils import calculate_levin_loss
import random
import torch


class MaskedOptionLearner_v1:
    def __init__(self, base_agent, transitions):
        self.base_agent = base_agent
        self.base_policy = base_agent.policy
        self.base_policy.__class__ = POLICY_TO_MASKER[self.base_policy.__class__]
        self.feature_extractor = base_agent.feature_extractor
        self.num_actions = self.base_policy.action_dim

        self.transitions = transitions
        self.trajectories = self.get_trajectories(transitions)

    def get_trajectories(self, transitions):
        # get (state, action) trajectories from list of all (observation, action, reward, terminated, truncated) transitions
        trajectories = []
        for run in transitions:
            for episode in run:
                trajectory = []
                for transition in episode:
                    observation, action, reward, terminated, truncated = transition
                    state = self.feature_extractor(observation)
                    trajectory.append((state, action))
                trajectories.append(trajectory)
        return trajectories
    
    def random_search(self):
        for _ in range(10):
            loss = 0
            masks = self.generate_mask_dicts(mask_key="3", mask_length=128, num_masks=20)
            for trajectory in self.trajectories:
                loss += calculate_levin_loss(trajectory, self.base_policy, masks, self.num_actions)
            loss /= len(self.trajectories)
            print(loss)

    def generate_mask_dicts(self, mask_key, mask_length, num_masks):
        """
        Generates a list of dictionaries, each containing a random mask.
        
        Parameters:
            mask_key (str): The key to use for each mask dictionary.
            mask_length (int): The length of each random mask.
            num_masks (int): The number of mask dictionaries to generate.
            
        Returns:
            List[dict]: A list of dictionaries with the specified key and a random mask of the specified length.
        """
        def generate_random_mask(length):
            return [random.choice([-1, 0, 1]) for _ in range(length)]
        
        return [{mask_key: generate_random_mask(mask_length)} for _ in range(num_masks)]
