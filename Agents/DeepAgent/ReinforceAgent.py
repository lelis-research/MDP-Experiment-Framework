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

    def select_action(self, state):
        """
        Sample an action from the policy distribution (Categorical).
        Store the log_prob for the update.
        """
        state_t = torch.FloatTensor(state)
        logits = self.actor_network(state_t) 
        
        dist = Categorical(logits=logits)
        action_t = dist.sample()
        
        # Store the log_prob
        log_prob_t = dist.log_prob(action_t)
        
        return action_t.item(), log_prob_t

    def update(self, log_probs, rewards):
        """
        Once an episode ends, compute discounted returns and do the policy update.
        """
        log_probs_t = torch.stack(log_probs)

        #Compute discounted returns from the end of the episode to the beginning
        returns = calculate_n_step_returns(rewards, 0.0, self.hp.gamma)
        returns_t = torch.FloatTensor(returns)

        # (Optional) Normalize returns to help training stability
        # Made the performance much worse !
        # returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Compute the policy loss:  -(sum of log_probs * returns)
        policy_loss = - (log_probs_t * returns_t).sum()

        # Backprop
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

class ReinforceAgent(BaseAgent):
    """
    Minimal REINFORCE agent for discrete actions, no baseline.
    """
    def __init__(self, action_space, observation_space, hyper_params, num_envs):
        super().__init__(action_space, observation_space, hyper_params, num_envs)

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
        transition = (self.last_log_prob, reward)
        self.rollout_buffer.add_single_item(transition)

        if terminated or truncated:
            rollout = self.rollout_buffer.get_all()
            log_probs, rewards = zip(*rollout)
        
            self.policy.update(log_probs, rewards)    
            self.rollout_buffer.reset()

    def reset(self, seed):
        """
        Reset the agent at the start of a new run (multiple seeds).
        """
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.rollout_buffer.reset()