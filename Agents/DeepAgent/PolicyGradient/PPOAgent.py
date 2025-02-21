import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from Agents.Utils.BaseAgent import BaseAgent, BasePolicy
from Agents.Utils.Buffer import BasicBuffer
from Agents.Utils.HelperFunction import calculate_n_step_returns
from Agents.Utils.NetworkGenerator import NetworkGen, prepare_network_config

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

    def reset(self, seed):
        super().reset(seed)
        # Actor and Critic networks
        actor_description = prepare_network_config(self.hp.actor_network, 
                                                   input_dim= self.features_dim, 
                                                   output_dim=self.action_dim)
        critic_description = prepare_network_config(self.hp.critic_network,
                                                    input_dim=self.features_dim,
                                                    output_dim=1)

        self.actor = NetworkGen(layer_descriptions=actor_description)
        self.critic = NetworkGen(layer_descriptions=critic_description)
        
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
        with torch.no_grad():
            value_t = self.critic(state_t)

        return action_t.item(), log_prob_t.detach(), value_t

    def update(self, states, actions, old_log_probs, states_values, rewards, next_states, dones):
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
        log_probs_t = torch.stack(old_log_probs)
        states_values_t = torch.stack(states_values)
        next_states_t = torch.FloatTensor(np.array(next_states))
        
        # Compute n-step returns (with bootstrapping from the critic)
        with torch.no_grad():
            bootstrap_value = self.critic(next_states_t)[-1] if not dones[-1] else 0.0
        returns = calculate_n_step_returns(rewards, bootstrap_value, self.hp.gamma)
        returns_t = torch.FloatTensor(returns).unsqueeze(1) #Correct the dims
        
        advantages_t = (returns_t - states_values_t)
        # advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        datasize = len(states)
        indices = np.arange(datasize)
        # Perform multiple PPO epochs over the rollout
        for epoch in range(self.hp.num_epochs):
            np.random.shuffle(indices)
            for start in range(0, datasize, self.hp.mini_batch_size):
                batch_indices = indices[start:start + self.hp.mini_batch_size]

                batch_states = states_t[batch_indices]
                batch_actions = actions_t[batch_indices]                
                batch_advantages = advantages_t[batch_indices]
                batch_returns = returns_t[batch_indices]
                batch_log_prob = log_probs_t[batch_indices]

                # Recompute action probabilities under current policy
                logits = self.actor(batch_states)
                dist = Categorical(logits=logits)
                new_log_probs_t = dist.log_prob(batch_actions).unsqueeze(1) # Correct the dims
                entropy = dist.entropy().unsqueeze(1) # Correct the dims            
                
                # Calculate the probability ratio (new / old)
                ratios = torch.exp(new_log_probs_t - batch_log_prob)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.hp.clip_range, 1 + self.hp.clip_range) * batch_advantages
                actor_loss = -torch.min(surr1, surr2) - self.hp.entropy_coef * entropy
                
                self.actor_optimizer.zero_grad()
                actor_loss.mean().backward()
                self.actor_optimizer.step()
                
                # Critic update: MSE loss between predicted state value and the n-step return
                new_state_values_t = self.critic(batch_states)
                critic_loss = F.mse_loss(new_state_values_t, batch_returns)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
    
    def save(self, file_path):
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'hyper_params': self.hp,   # Ensure that self.hp is pickle-serializable
            'features_dim': self.features_dim,
            'action_dim': self.action_dim,
        }
        torch.save(checkpoint, file_path)


    def load(self, file_path):
        checkpoint = torch.load(file_path, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.hp = checkpoint.get('hyper_params', self.hp)
        self.features_dim = checkpoint.get('features_dim', self.features_dim)
        self.action_dim = checkpoint.get('action_dim', self.action_dim)

class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) agent using a single environment and on-policy rollouts.
    
    Rollouts are collected until either the specified rollout steps or an episode termination/truncation.
    The rollout buffer stores tuples of (state, action, log_prob, reward, next_state, done).
    """
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class):
        super().__init__(action_space, observation_space, hyper_params, num_envs)
        self.feature_extractor = feature_extractor_class(observation_space)
        
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
        action, log_prob, state_value = self.policy.select_action(state)
        
        self.last_state = state
        self.last_action = action
        self.last_log_prob = log_prob
        self.last_state_value = state_value
        return action

    def update(self, observation, reward, terminated, truncated):
        """
        Called every step. Stores the transition in the rollout buffer. When
        the buffer is full (or the episode ends), performs the PPO update.
        """
        state = self.feature_extractor(observation)
        # Store the transition tuple: (state, action, log_prob, reward, next_state, done)
        transition = (self.last_state[0], self.last_action, self.last_log_prob, 
                      self.last_state_value[0], reward, state[0], terminated)
        self.rollout_buffer.add_single_item(transition)
        
        # Update if rollout is complete or the episode has ended.
        if self.rollout_buffer.size >= self.hp.rollout_steps or terminated or truncated:
            rollout = self.rollout_buffer.get_all()
            states, actions, log_probs, states_values, rewards, next_states, dones = zip(*rollout)
            self.policy.update(states, actions, log_probs, states_values, rewards, next_states, dones)
            self.rollout_buffer.reset()

    def reset(self, seed):
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.rollout_buffer.reset()