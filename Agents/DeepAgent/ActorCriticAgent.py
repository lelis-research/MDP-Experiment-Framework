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


class ActorCriticPolicy(BasePolicy):
    """
    Advantage Actor-Critic policy for discrete actions with single-env n-step rollout.
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
                - n_steps
        """
        super().__init__(action_space, hyper_params)
        self.features_dim = features_dim
        self.action_dim = action_space.n

    def reset(self, seed):
        super().reset(seed)

        # Actor and Critic networks
        self.actor = ActorNetwork(self.features_dim, self.action_dim)
        self.critic = CriticNetwork(self.features_dim)

        # Separate optimizers for actor and critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.hp.actor_step_size)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.hp.critic_step_size)


    def select_action(self, observation_features):
        """
        Sample an action from actor's logits, store log_prob for advantage update.
        """
        state_t = torch.FloatTensor(observation_features)
        logits = self.actor(state_t)
        dist = Categorical(logits=logits)
        action = dist.sample()

        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def update(self, states, actions, log_probs, rewards, next_states, dones):
        """
        All inputs are tensors
        Perform an n-step A2C update using the rollout buffer.
        1) We compute the n-step returns (or if the rollout ends, the full return).
        2) Critic update: fit V(s) to these returns (or bootstrapped from V(s_{t+n})).
        3) Actor update: advantage = returns - V(s). Maximize log_prob * advantage.
        """        
        # Compute the bootstrapped n-step returns
        if dones[-1]:  # i.e. last step was terminal
            returns = self.calculate_returns(rewards, 0.0) #bootstrap with 0
        else:
            with torch.no_grad():
                next_values = self.critic(next_states)
            returns = self.calculate_returns(rewards, next_values[-1].item()) #bootstrap from last state
        
        returns = torch.FloatTensor(returns) 
        values_t = self.critic(states)  

        # Critic Loss: MSE( V(s), returns )
        critic_loss = F.mse_loss(values_t, returns)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 3) Actor Loss: -log_probs_t * advantage
        #    advantage = returns_t - V(s)
        with torch.no_grad():
            updated_values_t = self.critic(states)
        advantages_t = returns - updated_values_t

        actor_loss = - (log_probs * advantages_t).sum()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def calculate_returns(self, rollout_rewards, bootstrap_value):
        returns = []
        G = bootstrap_value
        for r in reversed(rollout_rewards):
            G = r + self.hp.gamma * G
            returns.append(G)
        return returns
    
class ActorCriticAgent(BaseAgent):
    """
    An Advantage Actor-Critic agent using a single environment + n-step rollouts.
    """
    def __init__(self, action_space, observation_space, hyper_params):
        super().__init__(action_space, hyper_params)
        self.feature_extractor = FLattenFeature(observation_space)
        self.policy = ActorCriticPolicy(
            action_space, 
            self.feature_extractor.features_dim, 
            hyper_params
        )
        self.rollout_buffer = BasicBuffer(np.inf)

    def act(self, observation):
        """
        Sample an action from the actor, storing the log_prob.
        """
        state = self.feature_extractor(observation)
        action, log_prob = self.policy.select_action(state)

        self.last_state = state
        self.last_action = action
        self.last_log_prob = log_prob
        return action

    def update(self, observation, reward, terminated, truncated):
        """
        Called every step. We store the transition. If we hit n_steps or the episode ends,
        we do an update. 
        """

        state = self.feature_extractor(observation)
        transition = (self.last_state[0], self.last_action, self.last_log_prob, 
                      reward, state[0], terminated)
        self.rollout_buffer.add_single_item(transition)

        # If we've collected n_steps or the episode ended, do an A2C update
        if self.rollout_buffer.size >= self.hp.rollout_steps or terminated or truncated:
            rollout = self.rollout_buffer.get_all()
            states, actions, log_probs, rewards, next_states, dones = zip(*rollout)
            
            states_t = torch.FloatTensor(np.array(states))
            actions_t = torch.LongTensor(np.array(actions)).unsqueeze(1)
            log_probs_t = torch.stack(log_probs)
            rewards_t = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
            next_states_t = torch.FloatTensor(np.array(next_states))
            dones_t = torch.FloatTensor(np.array(dones)).unsqueeze(1)
            
            self.policy.update(states_t, actions_t, log_probs_t, rewards_t, next_states_t, dones_t)
            
            self.rollout_buffer.reset()


    def reset(self, seed):
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.rollout_buffer.reset()
