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
from Agents.Utils.NetworkGenerator import NetworkGen, prepare_network_config

class A2CPolicyV1(BasePolicy):
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
        actor_description = prepare_network_config(self.hp.actor_network, 
                                                   input_dim= self.features_dim, 
                                                   output_dim=self.action_dim)
        critic_description = prepare_network_config(self.hp.critic_network,
                                                    input_dim=self.features_dim,
                                                    output_dim=1)

        self.actor = NetworkGen(layer_descriptions=actor_description)
        self.critic = NetworkGen(layer_descriptions=critic_description)

        # Separate optimizers for actor and critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.hp.actor_step_size)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.hp.critic_step_size)

    def select_action(self, state):
        """
        Sample an action from actor's logits, store log_prob for advantage update.
        """
        state_t = torch.FloatTensor(state)
        logits = self.actor(state_t)
        dist = Categorical(logits=logits)
        action_t = dist.sample()
        log_prob_t = dist.log_prob(action_t)
        return action_t.item(), log_prob_t

    def update(self, states, log_probs, rewards, next_states, dones):
        """
        All inputs are tensors
        Perform an n-step A2C update using the rollout buffer.
        1) We compute the n-step returns (or if the rollout ends, the full return).
        2) Critic update: fit V(s) to these returns (or bootstrapped from V(s_{t+n})).
        3) Actor update: advantage = returns - V(s). Maximize log_prob * advantage.
        """     
        states_t = torch.FloatTensor(np.array(states))
        log_probs_t = torch.stack(log_probs)
        next_states_t = torch.FloatTensor(np.array(next_states))

        with torch.no_grad():
            #bootstrap from the last state if not terminated
            bootstrap_value = self.critic(next_states_t)[-1] if not dones[-1] else 0.0
        
        returns = calculate_n_step_returns(rewards, bootstrap_value, self.hp.gamma)
        returns_t = torch.FloatTensor(returns).unsqueeze(1) #Correct the dims
        predicted_values_t = self.critic(states_t)  
        
        # Critic Loss: MSE( V(s), returns )
        critic_loss = F.mse_loss(predicted_values_t, returns_t)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        #  advantage = returns_t - V(s)
        with torch.no_grad():
            predicted_values_t = self.critic(states_t)
        advantages_t = returns_t - predicted_values_t
        
        #  Actor Loss: -log_probs_t * advantage
        actor_loss = - (log_probs_t * advantages_t).sum()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    
class A2CAgentV1(BaseAgent):
    """
    An Advantage Actor-Critic agent using a single environment + n-step rollouts.
    """
    def __init__(self, action_space, observation_space, hyper_params, num_envs):
        super().__init__(action_space, observation_space, hyper_params, num_envs)
        self.feature_extractor = FLattenFeature(observation_space)
        
        self.policy = A2CPolicyV1(
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
        transition = (self.last_state[0], self.last_log_prob, 
                      reward, state[0], terminated)
        self.rollout_buffer.add_single_item(transition)

        # If we've collected n_steps or the episode ended, do an A2C update
        if self.rollout_buffer.size >= self.hp.rollout_steps or terminated or truncated:
            rollout = self.rollout_buffer.get_all()
            states, log_probs, rewards, next_states, dones = zip(*rollout)
            
            self.policy.update(states, log_probs, rewards, next_states, dones)
            self.rollout_buffer.reset()


    def reset(self, seed):
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.rollout_buffer.reset()
