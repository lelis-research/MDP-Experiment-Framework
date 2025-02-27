import torch
import random

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
        
POLICY_TO_MASKER = {
    DQNPolicy: DQNPolicyMasker
}