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
from Agents.Utils.HelperFunction import calculate_n_step_returns


class ActorNetwork(nn.Module):
    """
    Outputs logits for a discrete action distribution.
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


class CriticNetwork(nn.Module):
    """
    Outputs V(s), a scalar value.
    """
    def __init__(self, input_dim, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value.squeeze(-1)  # shape [batch]


class PPOPolicy(BasePolicy):
    """
    Proximal Policy Optimization policy for discrete actions.
    
    Assumes the hyper-parameters (accessible via self.hp) include:
        - gamma
        - actor_step_size
        - critic_step_size
        - clip_range
        - num_epochs
        - mini_batch_size
        - entropy_coef
        - rollout_steps
    """
    def __init__(self, action_space, features_dim, hyper_params):
        super().__init__(action_space, hyper_params)
        self.features_dim = features_dim
        self.action_dim = action_space.n

    def reset(self, seed):
        super().reset(seed)
        self.actor = ActorNetwork(self.features_dim, self.action_dim)
        self.critic = CriticNetwork(self.features_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.hp.actor_step_size)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.hp.critic_step_size)

    def select_action(self, state):
        """
        Sample an action using the actor network and return its log-probability.
        """
        state_t = torch.FloatTensor(state)
        logits = self.actor(state_t)
        dist = Categorical(logits=logits)
        action_t = dist.sample()
        log_prob_t = dist.log_prob(action_t)
        return action_t.item(), log_prob_t

    def update(self, states, actions, old_log_probs, rewards, next_states, dones):
        """
        Perform a PPO update using the collected rollout buffer.
        
        Args:
            states: list of states.
            actions: list of actions taken.
            old_log_probs: list of log probabilities (from the old policy) for the actions.
            rewards: list of rewards collected.
            next_states: list of next states.
            dones: list of done flags.
        """
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(np.array(actions))
        old_log_probs_t = torch.stack(old_log_probs)
        next_states_t = torch.FloatTensor(np.array(next_states))
        
        # Compute n-step returns (with bootstrapping from the critic)
        with torch.no_grad():
            bootstrap_value = self.critic(next_states_t)[-1] if not dones[-1] else 0.0
        returns = calculate_n_step_returns(rewards, bootstrap_value, self.hp.gamma)
        returns_t = torch.FloatTensor(returns)
        
        # Compute advantages (using a simple baseline subtraction)
        with torch.no_grad():
            predicted_values_t = self.critic(states_t)
        advantages_t = returns_t - predicted_values_t
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        dataset_size = states_t.shape[0]
        indices = np.arange(dataset_size)
        
        # Perform multiple PPO epochs over the rollout
        for epoch in range(self.hp.num_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.hp.mini_batch_size):
                batch_indices = indices[start:start + self.hp.mini_batch_size]
                batch_states = states_t[batch_indices]
                batch_actions = actions_t[batch_indices]
                batch_old_log_probs = old_log_probs_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_returns = returns_t[batch_indices]
                
                # Recompute action probabilities under current policy
                logits = self.actor(batch_states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                
                # Calculate the probability ratio (new / old)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.hp.clip_range, 1 + self.hp.clip_range) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.hp.entropy_coef * dist.entropy().mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Critic update: MSE loss between predicted state value and the n-step return
                values_pred = self.critic(batch_states)
                critic_loss = F.mse_loss(values_pred, batch_returns)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) agent using a single environment and on-policy rollouts.
    
    Rollouts are collected until either the specified rollout steps or an episode termination/truncation.
    The rollout buffer stores tuples of (state, action, log_prob, reward, next_state, done).
    """
    def __init__(self, action_space, observation_space, hyper_params, num_envs):
        super().__init__(action_space, observation_space, hyper_params, num_envs)
        self.feature_extractor = FLattenFeature(observation_space)
        
        self.policy = PPOPolicy(
            action_space, 
            self.feature_extractor.features_dim, 
            hyper_params
        )
        
        self.rollout_buffer = BasicBuffer(np.inf)

    def act(self, observation):
        """
        Sample an action from the current policy.
        """
        state = self.feature_extractor(observation)
        action, log_prob = self.policy.select_action(state)
        
        self.last_state = state
        self.last_action = action
        self.last_log_prob = log_prob
        return action

    def update(self, observation, reward, terminated, truncated):
        """
        Called every step. Stores the transition in the rollout buffer. When
        the buffer is full (or the episode ends), performs the PPO update.
        """
        state = self.feature_extractor(observation)
        # Store the transition tuple: (state, action, log_prob, reward, next_state, done)
        transition = (self.last_state[0], self.last_action, self.last_log_prob, 
                      reward, state[0], terminated)
        self.rollout_buffer.add_single_item(transition)
        
        # Update if rollout is complete or the episode has ended.
        if self.rollout_buffer.size >= self.hp.rollout_steps or terminated or truncated:
            print(self.rollout_buffer.size)
            exit(0)
            rollout = self.rollout_buffer.get_all()
            states, actions, log_probs, rewards, next_states, dones = zip(*rollout)
            self.policy.update(states, actions, log_probs, rewards, next_states, dones)
            self.rollout_buffer.reset()

    def reset(self, seed):
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.rollout_buffer.reset()