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
        self.actor_network = ActorNetwork(self.features_dim, self.action_dim)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.hp.step_size)

    def select_action(self, observation_features):
        """
        Sample an action from the policy distribution (Categorical).
        Store the log_prob for the update.
        """
        state_t = torch.FloatTensor(observation_features)
        logits = self.actor_network(state_t) 
        
        dist = Categorical(logits=logits)
        action = dist.sample()
        
        # Store the log_prob
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob

    def update(self, log_probs, rewards):
        """
        Once an episode ends, compute discounted returns and do the policy update.
        """
        #Compute discounted returns from the end of the episode to the beginning
        returns = self.calculate_returns(rewards, 0.0)
        returns = torch.FloatTensor(returns)

        # (Optional) Normalize returns to help training stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute the policy loss:  -(sum of log_probs * returns)
        policy_loss = - (log_probs * returns).sum()

        # Backprop
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

    def calculate_returns(self, rollout_rewards, bootstrap_value):
        returns = []
        G = bootstrap_value
        for r in reversed(rollout_rewards):
            G = r + self.hp.gamma * G
            # returns.append(G)
            returns.insert(0, G)
        return returns



class ReinforceAgent(BaseAgent):
    """
    Minimal REINFORCE agent for discrete actions, no baseline.
    """
    def __init__(self, action_space, observation_space, hyper_params):
        super().__init__(action_space, hyper_params)

        self.feature_extractor = FLattenFeature(observation_space)
        self.policy = ReinforcePolicy(
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
        Called each time-step by the experiment loop. We store the reward,
        and if the episode ends, we do a policy update.
        """
        state = self.feature_extractor(observation)
        transition = (self.last_action, self.last_log_prob, reward, terminated)
        self.rollout_buffer.add_single_item(transition)

        if terminated or truncated:
            rollout = self.rollout_buffer.get_all()
            actions, log_probs, rewards, dones = zip(*rollout)
            
            log_probs_t = torch.stack(log_probs)
            rewards_t = torch.FloatTensor(np.array(rewards)).unsqueeze(1)

            self.policy.update(log_probs_t, rewards_t)    
            self.rollout_buffer.reset()

    def reset(self, seed):
        """
        Reset the agent at the start of a new run (multiple seeds).
        """
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.rollout_buffer.reset()