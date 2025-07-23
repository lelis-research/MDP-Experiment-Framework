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
    PPOPolicy,
    A2CPolicyDiscrete,
)
from .NetworkMasker import NetworkMasker

@register_policy
class DQNPolicyMasker(DQNPolicy):
    def select_action_masked(self, state, mask_dict):
        masked_network = NetworkMasker(self.network, mask_dict)
        if random.random() < self.hp.epsilon:
            return self.action_space.sample()
        else:
            state_t = state.to(dtype=torch.float32, device=self.device) if torch.is_tensor(state) else torch.tensor(state, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                q_values = masked_network(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    @property
    def maskable_layers(self):
        return NetworkMasker.maskable_layers(self.network)

@register_policy
class DoubleDQNPolicyMasker(DoubleDQNPolicy):
    def select_action_masked(self, state, mask_dict):
        masked_online_network = NetworkMasker(self.online_network, mask_dict)
        if random.random() < self.hp.epsilon:
            return self.action_space.sample()
        else:
            state_t = state.to(dtype=torch.float32, device=self.device) if torch.is_tensor(state) else torch.tensor(state, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                q_values = masked_online_network(state_t)
            return int(torch.argmax(q_values, dim=1).item())
    @property
    def maskable_layers(self):
        return NetworkMasker.maskable_layers(self.online_network)

@register_policy
class NStepDQNPolicyMasker(NStepDQNPolicy):
    def select_action_masked(self, state, mask_dict):
        masked_network = NetworkMasker(self.network, mask_dict)
        if random.random() < self.hp.epsilon:
            return self.action_space.sample()
        else:
            state_t = state.to(dtype=torch.float32, device=self.device) if torch.is_tensor(state) else torch.tensor(state, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                q_values = masked_network(state_t)
            return int(torch.argmax(q_values, dim=1).item())
    @property
    def maskable_layers(self):
        return NetworkMasker.maskable_layers(self.network)
    
@register_policy
class ReinforcePolicyMasker(ReinforcePolicy):
    def select_action_masked(self, state, mask_dict):
        masked_actor = NetworkMasker(self.actor, mask_dict)

        state_t = state.to(dtype=torch.float32, device=self.device) if torch.is_tensor(state) else torch.tensor(state, dtype=torch.float32, device=self.device)
        logits = masked_actor(state_t)
        dist = Categorical(logits=logits)
        action_t = dist.sample()
        log_prob_t = dist.log_prob(action_t)
        
        return action_t.item(), log_prob_t
    
    @property
    def maskable_layers(self):
        return NetworkMasker.maskable_layers(self.actor)

@register_policy
class ReinforceWithBaselinePolicyMasker(ReinforceWithBaselinePolicy):
    def select_action_masked(self, state, mask_dict):
        masked_actor = NetworkMasker(self.actor, mask_dict)

        state_t = state.to(dtype=torch.float32, device=self.device) if torch.is_tensor(state) else torch.tensor(state, dtype=torch.float32, device=self.device)
        logits = masked_actor(state_t)
        dist = Categorical(logits=logits)
        action_t = dist.sample()
        log_prob_t = dist.log_prob(action_t)
        
        return action_t.item(), log_prob_t
    
    @property
    def maskable_layers(self):
        return NetworkMasker.maskable_layers(self.actor)
    
@register_policy
class A2CPolicyDiscreteMasker(A2CPolicyDiscrete):
    def select_action_masked(self, state, mask_dict):
        masked_actor = NetworkMasker(self.actor, mask_dict)

        state_t = state.to(dtype=torch.float32, device=self.device) if torch.is_tensor(state) else torch.tensor(state, dtype=torch.float32, device=self.device)
        logits = masked_actor(state_t)
        dist = Categorical(logits=logits)
        action_t = dist.sample()
        log_prob_t = dist.log_prob(action_t)
        
        return action_t.item(), log_prob_t
    
    @property
    def maskable_layers(self):
        return NetworkMasker.maskable_layers(self.actor) 
    
@register_policy
class PPOPolicyMasker(PPOPolicy):
    def select_action_masked(self, state, mask_dict):
        masked_actor = NetworkMasker(self.actor, mask_dict)

        state_t = state.to(dtype=torch.float32, device=self.device) if torch.is_tensor(state) else torch.tensor(state, dtype=torch.float32, device=self.device)
        logits = masked_actor(state_t)
        dist = Categorical(logits=logits)
        action_t = dist.sample()
        log_prob_t = dist.log_prob(action_t)
        with torch.no_grad():
            value_t = self.critic(state_t)

        return action_t.item(), log_prob_t.detach(), value_t
    
    @property
    def maskable_layers(self):
        return NetworkMasker.maskable_layers(self.actor) 
          
POLICY_TO_MASKER = {
    DQNPolicy: DQNPolicyMasker,
    DoubleDQNPolicy: DoubleDQNPolicyMasker,
    NStepDQNPolicy: NStepDQNPolicyMasker,

    ReinforcePolicy: ReinforcePolicyMasker,
    ReinforceWithBaselinePolicy: ReinforceWithBaselinePolicyMasker,
    A2CPolicyDiscrete: A2CPolicyDiscreteMasker,
    PPOPolicy: PPOPolicyMasker,
}