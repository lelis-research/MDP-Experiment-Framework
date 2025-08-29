import torch
import random
from torch import nn
from torch.distributions import Categorical

from ...registry import register_policy
from ...Agents.DeepAgent.ValueBased import (
    DQNPolicy, 
    DoubleDQNPolicy,
    NStepDQNPolicy
)
from ...Agents.DeepAgent.PolicyGradient import (
    ReinforcePolicy,
    ReinforceWithBaselinePolicy,
    PPOPolicyDiscrete,
    A2CPolicyDiscrete,
)
from .NetworkMasker import NetworkMasker

    
@register_policy
class A2CPolicyDiscreteMasker(A2CPolicyDiscrete):
    def select_action_masked(self, state, mask_dict):
        masked_actor = NetworkMasker(self.actor)

        state_t = state.to(dtype=torch.float32, device=self.device) if torch.is_tensor(state) else torch.tensor(state, dtype=torch.float32, device=self.device)
        logits = masked_actor(state_t, mask_dict)
        action = logits.argmax().item()
    
        return action, logits
    
    @property
    def maskable_layers(self):
        return NetworkMasker.maskable_layers(self.actor) 
    

          
POLICY_TO_MASKER = {
    A2CPolicyDiscrete: A2CPolicyDiscreteMasker,
}