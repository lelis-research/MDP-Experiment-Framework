import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from Agents.Utils.BaseAgent import BaseAgent, BasePolicy
from Agents.Utils.FeatureExtractor import FLattenFeature

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
        
    def forward(self, x):
        # x shape: [batch_size, input_dim]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)  # We'll pass them to Categorical(logits=...)
        return logits

###############################################################################
# The Policy class: Stores episode log_probs, rewards, does a Monte Carlo update.
###############################################################################
class ReinforcePolicy(BasePolicy):
    """
    A pure REINFORCE policy (no baseline). We store all log_probs and rewards
    for each episode, then do a single update at episode end.
    """
    def __init__(self, action_space, features_dim, hyper_params):
        """
        Args:
            action_space: The environment's action space (assumed to be Discrete).
            features_dim: Integer showing the number of features (From here we start with fully connected)
            hyper-parameters:
                - gamma
                - step_size  
        """
        super().__init__(action_space, hyper_params)
        
        self.features_dim = features_dim
        self.action_dim = action_space.n

    def reset(self, seed):
        """
        Called when we reset the entire agent. We create the policy network
        and its optimizer, and clear any episode storage.
        """
        super().reset(seed)
        
        # Build the policy network
        self.network = PolicyNetwork(self.features_dim, self.action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.hp.step_size)

        # Per-episode storage
        self.episode_log_probs = []
        self.episode_rewards = []

    def select_action(self, observation_features):
        """
        Sample an action from the policy distribution (Categorical).
        Store the log_prob for the update.
        """
        state_t = torch.FloatTensor(observation_features).unsqueeze(0)
        logits = self.network(state_t)  # shape [1, action_dim]
        
        dist = Categorical(logits=logits)
        action = dist.sample()
        
        # Store the log_prob
        log_prob = dist.log_prob(action)
        self.episode_log_probs.append(log_prob)
        
        return action.item()

    def store_reward(self, reward):
        """
        Track the immediate reward for each time step.
        """
        self.episode_rewards.append(reward)

    def finish_episode(self):
        """
        Once an episode ends, compute discounted returns and do the policy update.
        """
        #Compute discounted returns from the end of the episode to the beginning
        returns = []
        G = 0.0
        for r in reversed(self.episode_rewards):
            G = r + self.hp.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        # (Optional) Normalize returns to help training stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute the policy loss:  -(sum of log_probs * returns)
        log_probs_t = torch.stack(self.episode_log_probs)  # shape [T]
        policy_loss = - (log_probs_t * returns).sum()

        # Backprop
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Clear episode storage
        self.episode_log_probs = []
        self.episode_rewards = []

###############################################################################
# The Agent class: integrates the policy with the environment loop.
###############################################################################
class ReinforceAgent(BaseAgent):
    """
    Minimal REINFORCE agent for discrete actions, no baseline.
    """
    def __init__(self, action_space, observation_space, hyper_params):
        super().__init__(action_space, hyper_params)

        # Flatten the observation
        self.feature_extractor = FLattenFeature(observation_space)

        # Create the policy
        self.policy = ReinforcePolicy(
            action_space,
            self.feature_extractor.features_dim,
            hyper_params
        )
        
    def act(self, observation):
        # Convert observation to features, select an action stochastically
        features = self.feature_extractor(observation)
        action = self.policy.select_action(features)
        return action

    def update(self, observation, reward, terminated, truncated):
        """
        Called each time-step by the experiment loop. We store the reward,
        and if the episode ends, we do a policy update.
        """
        self.policy.store_reward(reward)

        if terminated or truncated:
            self.policy.finish_episode()

    def reset(self, seed):
        """
        Reset the agent at the start of a new run (multiple seeds).
        """
        super().reset(seed)
        self.feature_extractor.reset(seed)