import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from Agents.Utils.BaseAgent import BaseAgent, BasePolicy
from Agents.Utils.FeatureExtractor import FLattenFeature
from Agents.Utils.ReplayBuffer import BasicReplayBuffer


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
        self.value_network = ValueNetwork(self.features_dim)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.hp.critic_step_size)


        # We'll store the entire episode trajectory
        self.episode_states = []
        self.episode_log_probs = []
        self.episode_rewards = []

    def select_action(self, observation_features):
        """
        Sample an action from the policy distribution (Categorical).
        We'll return the action and store the log_prob for the update.
        """
        state_t = torch.FloatTensor(observation_features).unsqueeze(0)  # shape [1, features_dim]
        logits = self.actor_network(state_t)  # shape [1, action_dim]
        
        # Convert logits -> categorical distribution
        dist = Categorical(logits=logits)  # applies softmax internally
        action = dist.sample()  # shape [1]
        
        # Store the log_prob for this action
        log_prob = dist.log_prob(action)

        self.episode_states.append(observation_features)
        self.episode_log_probs.append(log_prob)
        
        return action.item()

    def store_reward(self, reward):
        """
        We'll call this at each step to track rewards.
        """
        self.episode_rewards.append(reward)

    def finish_episode(self):
        """
        End of episode => compute returns, train the baseline, do the policy gradient update with advantage.
        """
        # Compute discounted returns from the end of the episode to the beginning
        returns = []
        G = 0.0
        for r in reversed(self.episode_rewards):
            G = r + self.hp.gamma * G
            returns.insert(0, G)
        returns_t = torch.FloatTensor(returns)  # shape [T]

        

        # Critic update: we train the value network V(s) to approximate the returns
        #    value_loss = MSE(returns_t, V(s))
        states_t = torch.FloatTensor(np.asarray(self.episode_states))      # shape [T, features_dim]
        predicted_values = self.value_network(states_t)         # shape [T]
        value_loss = F.mse_loss(predicted_values, returns_t)

        # Backprop Critic
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Compute advantage: A_t = G_t - V(s_t)
        with torch.no_grad():
            updated_values = self.value_network(states_t)       # shape [T]
        advantages = returns_t - updated_values                 # shape [T]

        # Policy update:  - sum( log_prob_t * advantage_t )
        log_probs_t = torch.stack(self.episode_log_probs)      # shape [T]
        policy_loss = - (log_probs_t * advantages).sum()

        # Backprop Actor
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Clear episode storage
        self.episode_states = []
        self.episode_log_probs = []
        self.episode_rewards = []


class ReinforceAgentWithBaseline(BaseAgent):
    """
    A REINFORCE-like agent that uses a learned baseline (value network)
    to reduce variance => 'Actor + Baseline'.
    """

    def __init__(self, action_space, observation_space, hyper_params):
        super().__init__(action_space, hyper_params)
        
        self.feature_extractor = FLattenFeature(observation_space)
        self.policy = ReinforcePolicyWithBaseline(
            action_space,
            self.feature_extractor.features_dim,
            hyper_params
        )

    def act(self, observation):
        """
        Convert observation to features, then let the policy pick an action stochastically.
        """
        features = self.feature_extractor(observation)
        action = self.policy.select_action(features)
        return action
    
    def update(self, observation, reward, terminated, truncated):
        """
        Called every step by the experiment loop:
        - We store the reward in the policy
        - If the episode ends, do a policy gradient update
        """
        self.policy.store_reward(reward)

        if terminated or truncated:
            self.policy.finish_episode()
    
    def reset(self, seed):
        """
        Reset for a new run or new experiment.
        """
        super().reset(seed)
        self.feature_extractor.reset(seed)