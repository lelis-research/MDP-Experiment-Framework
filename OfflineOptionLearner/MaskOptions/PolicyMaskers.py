import torch
import random
from torch import nn

from Agents.DeepAgent.ValueBased import DQNPolicy
from .MaskedNetwork import MaskedNetwork

class DQNPolicyMasker(DQNPolicy):
    def select_action_masked(self, state, mask_dict):
        masked_network = MaskedNetwork(self.network, mask_dict)
        if random.random() < self.hp.epsilon:
            return self.action_space.sample()
        else:
            state_t = torch.FloatTensor(state)
            with torch.no_grad():
                q_values = masked_network(state_t)
            return int(torch.argmax(q_values, dim=1).item())
    @property
    def maskable_layers(self):
        """
        Returns a dictionary mapping incremental keys (as strings) to maskable activation layers 
        and their output sizes.

        Only activation layers (ReLU, LeakyReLU, Sigmoid, Tanh) are considered maskable.
        The output size is inferred from the most recent layer with a defined output size 
        (e.g. from a Linear or Conv2d layer via the attributes 'out_features' or 'out_channels').

        The returned dictionary maps keys (e.g. "0", "1", ...) to dictionaries with:
            - 'layer': the activation layer module,
            - 'size': the inferred output size for that activation.

        Returns:
            dict: A mapping from incremental string keys to dictionaries with keys 'layer' and 'size'.
        """
        maskable_layers = {}

        # Retrieve the underlying sequential module.
        seq_net = self.network.network
        
        last_output_size = None

        # Iterate through layers to detect activation functions.
        for idx, module in enumerate(seq_net):
            # Update the most recent output size if the module defines it.
            if hasattr(module, 'out_features'):
                last_output_size = module.out_features
            elif hasattr(module, 'out_channels'):
                last_output_size = module.out_channels

            # Only activation layers are maskable.
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)):
                if last_output_size is None:
                    # If we haven't encountered a preceding layer with a size, skip this activation.
                    continue
                maskable_layers[str(idx)] = {
                    'layer': module,
                    'size': last_output_size
                }

        return maskable_layers
        
POLICY_TO_MASKER = {
    DQNPolicy: DQNPolicyMasker
}