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


    def select_action(self, observation_features):
        """
        Sample an action from the policy distribution (Categorical).
        We'll return the action and store the log_prob for the update.
        """
        state_t = torch.FloatTensor(observation_features)
        logits = self.actor_network(state_t) 
        
        dist = Categorical(logits=logits)
        action = dist.sample()
        
        # Store the log_prob
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob

    def update(self, states, log_probs, rewards):
        """
        End of episode => compute returns, train the baseline, do the policy gradient update with advantage.
        """
        # Compute discounted returns from the end of the episode to the beginning
        returns = self.calculate_returns(rewards, 0.0)
        returns = torch.FloatTensor(returns)
        
        # Critic update: we train the value network V(s) to approximate the returns
        #    value_loss = MSE(returns_t, V(s))
        predicted_values = self.value_network(states)   
       
        # Backprop Critic
        value_loss = F.mse_loss(predicted_values, returns)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Compute advantage: A_t = G_t - V(s_t)
        with torch.no_grad():
            updated_values = self.value_network(states)      
        advantages = returns - updated_values                

        # Policy update:  - sum( log_prob_t * advantage_t )
        # Backprop Actor
        policy_loss = - (log_probs * advantages).sum()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

    def calculate_returns(self, rollout_rewards, bootstrap_value):
        returns = []
        G = bootstrap_value
        for r in reversed(rollout_rewards):
            G = r + self.hp.gamma * G
            returns.append(G)
        return returns


class ReinforceAgentWithBaseline(BaseAgent):
    """
    A REINFORCE-like agent that uses a learned baseline (value network)
    to reduce variance => 'Actor + Baseline'.
    """

    def __init__(self, action_space, observation_space, hyper_params):
        BaseAgent.__init__(action_space, hyper_params)

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
        self.last_action = action
        self.last_log_prob = log_prob
        return action
    
    def update(self, observation, reward, terminated, truncated):
        """
        Called every step by the experiment loop:
        - If the episode ends, do a policy gradient update
        """
        state = self.feature_extractor(observation)
        transition = (self.last_state[0], self.last_action, self.last_log_prob, reward, terminated)
        self.rollout_buffer.add_single_item(transition)

        if terminated or truncated:
            rollout = self.rollout_buffer.get_all()
            states, actions, log_probs, rewards, dones = zip(*rollout)

            states_t = torch.FloatTensor(np.array(states))
            log_probs_t = torch.stack(log_probs)
            rewards_t = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
            states_t = torch.FloatTensor(np.array(states))

            self.policy.update(states_t, log_probs_t, rewards_t)    
            self.rollout_buffer.reset()

    def reset(self, seed):
        """
        Reset the agent at the start of a new run (multiple seeds).
        """
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.rollout_buffer.reset()
