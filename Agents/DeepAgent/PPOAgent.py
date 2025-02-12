import numpy as np
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from Agents.Utils.BaseAgent import BaseAgent, BasePolicy
from Agents.Utils.FeatureExtractor import FLattenFeature
from Agents.Utils.Buffer import BasicBuffer

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
    Outputs a scalar value for state V(s).
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
    Proximal Policy Optimization (PPO) for discrete actions, single environment.
    We'll collect rollouts for 'rollout_steps', then do 'ppo_epochs' passes
    of minibatch updates with the clipped objective.
    """

    def __init__(self, action_space, features_dim, hyper_params):
        super().__init__(action_space, hyper_params)
        self.features_dim = features_dim
        self.action_dim = action_space.n

    def reset(self, seed):
        super().reset(seed)

        # Build actor & critic
        self.actor = ActorNetwork(self.features_dim, self.action_dim)
        self.critic = CriticNetwork(self.features_dim)

        # Separate optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.hp.actor_step_size)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.hp.critic_step_size)

    def select_action(self, observation_features):
        """
        Returns: action, log_prob, value
        We'll store them in the rollout buffer.
        """
        state_t = torch.FloatTensor(observation_features)
        # Actor forward
        logits = self.actor(state_t)
        dist = Categorical(logits=logits)
        action = dist.sample()

        # log_prob for the chosen action
        log_prob = dist.log_prob(action)

        # Critic forward
        value = self.critic(state_t)  # shape [1]

        return action.item(), log_prob.detach(), value.detach()
        

    def update(self, states, actions, log_probs, values, rewards, next_states, dones):
        # Compute the bootstrapped n-step returns
        if dones[-1].item():  # i.e. last step was terminal
            returns = self.calculate_returns(rewards, 0.0) #bootstrap with 0
        else:
            with torch.no_grad():
                next_values = self.critic(next_states)
            returns = self.calculate_returns(rewards, next_values[-1].item()) #bootstrap from last state
        returns_t = torch.FloatTensor(returns).unsqueeze(1)
        advantages = returns_t - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(states)
        # For multiple epochs, we shuffle indices and break into mini-batches
        indices = np.arange(n)
        for _ in range(self.hp.num_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.hp.batch_size):                
                idx = indices[start:start + self.hp.batch_size]

                b_states = states[idx]           # shape [batch, input_dim]
                b_actions = actions[idx]         # [batch]
                b_log_probs = log_probs[idx]  # [batch]
                b_returns = returns_t[idx]         # [batch]
                b_advantages = advantages[idx]   # [batch]

                # Forward pass actor
                logits = self.actor(b_states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                # Critic
                new_values = self.critic(b_states).unsqueeze(1)

                # Ratio
                ratio = torch.exp(new_log_probs - b_log_probs)

                # Clipped objective
                unclipped = ratio * b_advantages
                clipped = torch.clamp(ratio, 1.0 - self.hp.clip_range, 1.0 + self.hp.clip_range) * b_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_values, b_returns)

                # Combine losses
                loss = policy_loss + self.hp.value_loss_coef * value_loss - self.hp.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

    
    def calculate_returns(self, rollout_rewards, bootstrap_value):
        returns = []
        G = bootstrap_value
        for r in reversed(rollout_rewards):
            G = r + self.hp.gamma * G
            # returns.append(G)
            returns.insert(0, G)
        return returns

class PPOAgent(BaseAgent):
    """
    Single-environment PPO agent. We collect rollout_steps transitions or until
    the episode ends, then do a PPO update. We repeat for multiple episodes.
    """
    def __init__(self, action_space, observation_space, hyper_params):
        super().__init__(action_space, hyper_params)
        self.feature_extractor = FLattenFeature(observation_space)
        self.policy = PPOPolicy(
            action_space,
            self.feature_extractor.features_dim,
            hyper_params
        )

        self.rollout_buffer = BasicBuffer(np.inf)

    def act(self, observation):
        """
        Convert obs, sample action from actor, store (log_prob, value) for the rollout.
        """
        state = self.feature_extractor(observation)
        action, log_prob, value = self.policy.select_action(state)

        self.last_state = state
        self.last_action = action
        self.last_log_prob = log_prob
        self.last_value = value

        return action

    def update(self, observation, reward, terminated, truncated):
        """
        Called every step. Store the transition in the rollout buffer.
        If we reach rollout_steps or the episode ends, do a PPO update.
        """
        state = self.feature_extractor(observation)
        transition = (self.last_state[0], self.last_action, self.last_log_prob, 
                      self.last_value, reward, state[0], terminated)
        self.rollout_buffer.add_single_item(transition)

        # If we've reached rollout_steps or the episode ended, do an update
        if self.rollout_buffer.size >= self.hp.rollout_steps or terminated or truncated:
            rollout = self.rollout_buffer.get_all()
            states, actions, log_probs, values, rewards, next_states, dones = zip(*rollout)
            
            states_t = torch.FloatTensor(np.array(states))
            actions_t = torch.LongTensor(np.array(actions)).unsqueeze(1)
            log_probs_t = torch.stack(log_probs)
            values_t = torch.stack(values)
            rewards_t = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
            next_states_t = torch.FloatTensor(np.array(next_states))
            dones_t = torch.FloatTensor(np.array(dones)).unsqueeze(1)

            self.policy.update(states_t, actions_t, log_probs_t, 
                               values_t, rewards_t, next_states_t,
                               dones_t)
            self.rollout_buffer.reset()
            

    def reset(self, seed):
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.rollout_buffer.reset()
        