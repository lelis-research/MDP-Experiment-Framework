import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from Agents.Utils.BaseAgent import BaseAgent, BasePolicy
from Agents.Utils.FeatureExtractor import FLattenFeature
from Agents.Utils.Buffer import BasicBuffer
from Agents.Utils.HelperFunction import *

class ActorNetwork(nn.Module):
    """
    The 'actor': outputs logits for a discrete action distribution.
    We'll still do a Categorical over these logits.
    """
    def __init__(self, input_dim, action_dim, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


class ValueNetwork(nn.Module):
    """
    The 'critic': outputs a scalar baseline V(s).
    """
    def __init__(self, input_dim, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # single value output
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value.squeeze(-1)  # shape [...], removing last dim if 1


class ReinforcePolicyWithBaseline(BasePolicy):
    """
    A Policy for REINFORCE with a learned baseline (value network).
    
    We'll store states, actions, log_probs, and rewards for each episode.
    Then, at episode end, we:
      1) compute the returns G_t,
      2) update ValueNetwork (the baseline),
      3) use advantage = (G_t - V(s_t)) in the policy gradient.
    """
    def __init__(self, action_space, features_dim, hyper_params):
        """
        Args:
            action_space: The environment's action space (assumed to be Discrete).
            features_dim: Integer showing the number of features (From here we start with fully connected)
            hyper-parameters:
                - gamma
                - actor_step_size  
                - critic_step_size
        """
        super().__init__(action_space, hyper_params)
        
        self.features_dim = features_dim
        self.action_dim = action_space.n

    def reset(self, seed):
        super().reset(seed)
        
        # Actor (policy) network & optimizer
        self.actor_network = ActorNetwork(self.features_dim, self.action_dim)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.hp.actor_step_size)

        # Critic (value) network & optimizer
        self.critic_network = ValueNetwork(self.features_dim)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.hp.critic_step_size)

    def select_action(self, state):
        """
        Sample an action from the policy distribution (Categorical).
        We'll return the action and store the log_prob for the update.
        """
        state_t = torch.FloatTensor(state)
        logits = self.actor_network(state_t) 
        
        dist = Categorical(logits=logits)
        action_t = dist.sample()
        
        # Store the log_prob
        log_prob_t = dist.log_prob(action_t)
        
        return action_t.item(), log_prob_t

    def update(self, states, log_probs, rewards):
        """
        End of episode => compute returns, train the baseline, do the policy gradient update with advantage.
        """
        states_t = torch.FloatTensor(np.array(states))
        log_probs_t = torch.stack(log_probs)
        
        # Compute discounted returns from the end of the episode to the beginning
        returns = calculate_n_step_returns(rewards, 0.0, self.hp.gamma)
        returns_t = torch.FloatTensor(returns)
        
        # (Optional) Normalize returns to help training stability
        # Made the performance much worse !
        # returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Critic update: we train the value network V(s) to approximate the returns
        predicted_values_t = self.critic_network(states_t)   
       
        # Backprop Critic
        critic_loss = F.mse_loss(predicted_values_t, returns_t)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute advantage: A_t = G_t - V(s_t)
        with torch.no_grad():
            predicted_values_t = self.critic_network(states_t)      
        advantages_t = returns_t - predicted_values_t                

        # Policy update:  - sum( log_prob_t * advantage_t )
        actor_loss = - (log_probs_t * advantages_t).sum()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


class ReinforceAgentWithBaseline(BaseAgent):
    """
    A REINFORCE-like agent that uses a learned baseline (value network)
    to reduce variance => 'Actor + Baseline'.
    """

    def __init__(self, action_space, observation_space, hyper_params, num_envs):
        BaseAgent.__init__(action_space, observation_space, hyper_params, num_envs)

        self.feature_extractor = FLattenFeature(observation_space)

        self.policy = ReinforcePolicyWithBaseline(
            action_space,
            self.feature_extractor.features_dim,
            hyper_params
        )
        self.rollout_buffer = BasicBuffer(np.inf)

    def act(self, observation):
        # Convert observation to features, select an action stochastically
        state = self.feature_extractor(observation)
        action, log_prob = self.policy.select_action(state)

        self.last_state = state
        self.last_log_prob = log_prob
        return action
    
    def update(self, observation, reward, terminated, truncated):
        """
        Called every step by the experiment loop:
        - If the episode ends, do a policy gradient update
        """
        transition = (self.last_state[0], self.last_log_prob, reward)
        self.rollout_buffer.add_single_item(transition)

        if terminated or truncated:
            rollout = self.rollout_buffer.get_all()
            states, log_probs, rewards, = zip(*rollout)
            self.policy.update(states, log_probs, rewards)    
            self.rollout_buffer.reset()

    def reset(self, seed):
        """
        Reset the agent at the start of a new run (multiple seeds).
        """
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.rollout_buffer.reset()
