from .PolicyMaskers import POLICY_TO_MASKER
from OfflineOptionLearner.Utils import levin_loss_for_masks

import random
import torch
import numpy as np

class MaskedOptionLearner_v1:
    def __init__(self, base_agent, transitions):
        """
        Initialize the MaskedOptionLearner instance.

        This constructor accepts a pre-trained agent (base_agent) and a set of transitions,
        which are organized as a nested list with the following structure:
        - A list of runs,
        - Each run is a list of episodes,
        - Each episode is a list of transitions,
        - Each transition is a tuple: (observation, action, reward, terminated, truncated).

        The agent's policy is augmented by modifying its __class__ attribute to the corresponding
        masker class from POLICY_TO_MASKER. This "hack" injects the additional method select_action_masked
        into the existing policy instance without creating a new object.

        The feature_extractor is taken directly from the base agent, and the total number of actions
        is obtained from the policy's action_dim attribute.

        Lastly, the transitions are processed into trajectories, where each trajectory is a list of
        (state, action) pairs. The state is computed by applying the feature_extractor to the observation.

        Parameters:
            base_agent: A pre-trained agent instance containing attributes such as 'policy' and 'feature_extractor'.
            transitions (list): A nested list of transitions with the structure:
                                [run1, run2, ...], where each run is a list of episodes, and each episode
                                is a list of transitions formatted as (observation, action, reward, terminated, truncated).
        """
        self.base_agent = base_agent
        self.base_policy = base_agent.policy
        self.base_policy.__class__ = POLICY_TO_MASKER[self.base_policy.__class__]  # Injects select_action_masked into the policy.
        
        self.feature_extractor = base_agent.feature_extractor
        self.num_actions = self.base_policy.action_dim
        
        self.transitions = transitions
        self.trajectories = self._extract_trajectories(transitions)
    
    
    def _extract_trajectories(self, transitions):
        """
        Extract state-action trajectories from nested transition data.

        The input data is organized as a list of runs, where each run is a list of episodes,
        and each episode is a list of transitions. Each transition is expected to be a tuple 
        containing at least (observation, action, ...), where additional elements (like reward,
        terminated, truncated) are ignored in this process. The function applies the feature_extractor
        to each observation to produce a state and pairs it with the corresponding action.

        Parameters:
            transitions (list): A nested list structured as:
                - runs: list of runs,
                - each run: list of episodes,
                - each episode: list of transitions,
                - each transition: tuple (observation, action, reward, terminated, truncated, ...)

        Returns:
            List[List[Tuple[Any, Any]]]: A list of trajectories. Each trajectory is a list of (state, action) pairs.
        """
        trajectories = []
        # Iterate over each run in the transitions data.
        for run in transitions:
            # Each run can contain multiple episodes.
            for episode in run:
                # For each episode, create a trajectory by extracting the state and action.
                # The state is obtained by applying the feature_extractor to the observation.
                trajectory = [(self.feature_extractor(obs), action) for obs, action, *_ in episode]
                trajectories.append(trajectory)
        return trajectories
    
    def random_search(self, masked_layers, num_options, iteration=10):
        min_loss, best_masks = np.inf, []
        for _ in range(iteration):
            loss = 0
            random_masks = self.generate_random_maskings(layer_names=masked_layers, number_of_masks=num_options)
            for trajectory in self.trajectories:
                loss += levin_loss_for_masks(trajectory, self.base_policy, random_masks, self.num_actions)
            loss /= len(self.trajectories)
            if loss < min_loss:
                min_loss = loss
                best_masks = random_masks
        return best_masks

    def generate_random_maskings(self, layer_names, number_of_masks):
        """
        Generate a list of random mask dictionaries for the specified layers in the network.
        
        Each mask dictionary has keys corresponding to the layer names provided in `layer_names`.
        For each layer, the value is a list of randomly chosen integers (-1, 0, or 1). The length
        of this list is determined by the output size of the layer, which is obtained from 
        self.base_policy.network[layer_name]. For linear layers, this is `out_features`; for 
        convolutional layers, this is `out_channels`.

        Parameters:
            layer_names (List[str]): A list of layer names (keys in the network) for which to create masks.
            number_of_masks (int): The total number of mask dictionaries to generate.

        Returns:
            List[Dict[str, List[int]]]: A list of mask dictionaries. Each dictionary maps every layer name
                                        to a list of random mask values of the appropriate length.

        Raises:
            ValueError: If the output size for a layer cannot be determined.
        """
        masks = []
        # Retrieve the mapping of maskable layers from the base policy.
        maskable_layers = self.base_policy.maskable_layers
        
        for _ in range(number_of_masks):
            mask_dict = {}
            for layer_name in layer_names:
                if layer_name not in maskable_layers:
                    raise ValueError(f"Layer '{layer_name}' not found in base_policy maskable layers.")
                size = maskable_layers[layer_name]['size']
                # Generate a random mask: a list of length 'size' with values chosen from -1, 0, or 1.
                mask_dict[layer_name] = [random.choice([-1, 0, 1]) for _ in range(size)]
            masks.append(mask_dict)
        return masks
