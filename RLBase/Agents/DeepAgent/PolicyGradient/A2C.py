import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Independent, TransformedDistribution
from torch.distributions.transforms import TanhTransform, AffineTransform

import gymnasium
import math

from ...Utils import (
    BaseAgent,
    BasePolicy,
    BasicBuffer,
    calculate_gae,
    NetworkGen,
    prepare_network_config,
)
from ....registry import register_agent, register_policy

@register_policy
class A2CPolicyDiscrete(BasePolicy):
    """
    Advantage Actor-Critic policy for discrete actions using n-step rollouts.
    """
    def __init__(self, action_space, features_dim, hyper_params, device="cpu"):
        """
        Args:
            action_space (gym.spaces.Discrete): Environment's action space.
            features_dim (int): Dimension of the flattened features.
            hyper_params: Hyper-parameters container, must include:
                - gamma (float)
                - actor_step_size (float)
                - critic_step_size (float)
                - n_steps (int)
                - actor_network (list): Network configuration for actor.
                - critic_network (list): Network configuration for critic.
        """
        super().__init__(action_space, hyper_params, device=device)
        self.features_dim = features_dim

    def reset(self, seed):
        """
        Reset the policy and initialize actor & critic networks.
        
        Args:
            seed (int): Random seed.
        """
        super().reset(seed)
        
        # Build the actor network
        actor_description = prepare_network_config(
            self.hp.actor_network, 
            input_dim=self.features_dim, 
            output_dim=self.action_dim
        )
        # Build the critic network (outputs scalar value)
        critic_description = prepare_network_config(
            self.hp.critic_network,
            input_dim=self.features_dim,
            output_dim=1
        )
        self.actor = NetworkGen(layer_descriptions=actor_description).to(self.device)
        self.critic = NetworkGen(layer_descriptions=critic_description).to(self.device)

        # Create separate optimizers for actor and critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.hp.actor_step_size)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.hp.critic_step_size)
       
    def select_action(self, state, greedy=False):
        """
        Sample an action from the actor's policy.
        
        Args:
            state (np.array): Flat feature vector of shape [features_dim] or [1, features_dim].
            
        Returns:
            tuple: (action (int), log_prob (torch.Tensor))
        """
        state_t = state.to(dtype=torch.float32, device=self.device) if torch.is_tensor(state) \
                    else torch.tensor(state, dtype=torch.float32, device=self.device)
        logits = self.actor(state_t)
        dist = Categorical(logits=logits)
        if greedy:
            action_t = torch.argmax(logits, dim=-1)
        else:
            action_t = dist.sample()
        return action_t.item()
    
    def select_parallel_actions(self, states, greedy=False):
        state_t = states.to(dtype=torch.float32, device=self.device) if torch.is_tensor(states) \
                    else torch.tensor(states, dtype=torch.float32, device=self.device)
        logits = self.actor(state_t)
        dist = Categorical(logits=logits)
        if greedy:
            action_t = torch.argmax(logits, dim=-1)
        else:
            action_t = dist.sample()
        return action_t.cpu().numpy()

    def update(self, states, actions, rewards, next_states, dones, call_back=None):
        """
        Perform an n-step A2C update.
        
        Args:
            states (list or np.array): Batch of states; shape [n_steps, features_dim].
            action (list of torch.Tensor): actions taken; length n_steps.
            rewards (list of float): Rewards collected over the rollout; length n_steps.
            next_states (list or np.array): Batch of next states; shape [n_steps, features_dim].
            dones (list of bool): Done flags for each transition; length n_steps.
            call_back (function, optional): Function to report losses.
        """
        states_t = torch.cat(states).to(dtype=torch.float32, device=self.device) if torch.is_tensor(states[0]) else torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(np.array(actions), dtype=torch.int64, device=self.device)
        next_states_t = torch.cat(next_states).to(dtype=torch.float32, device=self.device) if torch.is_tensor(next_states[0]) else torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        
        values = self.critic(states_t).squeeze(-1)
        with torch.no_grad():
            next_values = self.critic(next_states_t).squeeze(-1)
        
        # Compute n-step returns
        returns, advantages = calculate_gae(
            rewards,
            values,
            next_values,
            dones,
            gamma=self.hp.gamma,
            lamda=self.hp.lamda,
        )
        
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t    = torch.tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(1) # shape [n_steps, 1]
        
        if self.hp.norm_adv_flag:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
            
        logits = self.actor(states_t)
        dist = Categorical(logits=logits)
        log_probs_t = dist.log_prob(actions_t)
        entropy = dist.entropy()
        
        # Critic prediction: value estimates V(s)
        critic_loss = F.mse_loss(values.unsqueeze(1), returns_t)
        
        # Actor loss: maximize expected return => minimize -log_prob * advantage
        actor_loss = - (log_probs_t * advantages_t).mean() # mean is more stable
        
        loss = actor_loss + critic_loss - self.hp.entropy_coef * entropy.mean()

        
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()        
        self.actor_optimizer.step()

        if call_back is not None:
            call_back({
                "critic_loss": critic_loss.item(),
                "actor_loss": actor_loss.item()
            })

    def save(self, file_path=None):
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),

            'action_space': self.action_space,
            'features_dim': self.features_dim,
            'hyper_params': self.hp,
            
            'policy_class': self.__class__.__name__,
        }
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_policy.t")
        return checkpoint

    @classmethod
    def load_from_file(cls, file_path, seed=0, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        instance = cls(checkpoint['action_space'], checkpoint['features_dim'], checkpoint['hyper_params'])

        instance.reset(seed)
        instance.actor.load_state_dict(checkpoint['actor_state_dict'])
        instance.critic.load_state_dict(checkpoint['critic_state_dict'])
        instance.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        instance.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        return instance

    def load_from_checkpoint(self, checkpoint):
        """
        Load network states, optimizer state, and hyper-parameters.
        
        Args:
            checkpoint (dictionary)
        """
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        self.action_space = checkpoint.get('action_space')
        self.features_dim = checkpoint.get('features_dim')
        self.hp = checkpoint.get('hyper_params')

@register_policy
class A2CPolicyContinuous(BasePolicy):
    """
    Advantage Actor-Critic policy for continuous actions using n-step rollouts.
    """
    def __init__(self, action_space, features_dim, hyper_params, device="cpu"):
        """
        Args:
            action_space (gym.spaces.Box): Environment's action space.
            features_dim (int): Dimension of the flattened features.
            hyper_params: must include
                - gamma (float)
                - lamda (float)
                - actor_step_size (float)
                - critic_step_size (float)
                - n_steps (int)
                - actor_network (list): network config (outputs 2*action_dim for mean and log_std)
                - critic_network (list)
              Optional:
                - entropy_coef (float, default 0.0)
        """
        assert hasattr(action_space, "shape") and len(action_space.shape) == 1, \
            "A2CPolicyContinuous supports 1D Box action spaces."
        super().__init__(action_space, hyper_params, device=device)
        self.features_dim = features_dim
            
    def reset(self, seed):
        super().reset(seed)
        # Actor outputs concatenated [mean, log_std] of size 2 * action_dim
        actor_description = prepare_network_config(
            self.hp.actor_network,
            input_dim=self.features_dim,
            output_dim=self.action_dim
        )
        critic_description = prepare_network_config(
            self.hp.critic_network,
            input_dim=self.features_dim,
            output_dim=1
        )
        self.actor = NetworkGen(layer_descriptions=actor_description).to(self.device)
        self.critic = NetworkGen(layer_descriptions=critic_description).to(self.device)
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(self.action_space.shape), device=self.device))
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.hp.critic_step_size)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()) + [self.actor_logstd],lr=self.hp.actor_step_size, eps=1e-5)
        
        self.update_counter = 0


    def _log_prob_and_entropy(self, state_t, action_t=None):
        action_mean_t = self.actor(state_t)
        action_logstd_t = self.actor_logstd.expand_as(action_mean_t)
        action_std_t = torch.exp(action_logstd_t)
        probs = Normal(action_mean_t, action_std_t)
        if action_t is None:
            action_t = probs.sample()
        return action_t, probs.log_prob(action_t).sum(1), probs.entropy().sum(1)
    
    def select_action(self, state, greedy=False):
        """
        Returns a numpy action of shape [action_dim].
        """
        state_t = state.to(dtype=torch.float32, device=self.device) if torch.is_tensor(state) \
                  else torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if greedy:
                action_t = self.actor(state_t)
            else:
                action_t, _, _ = self._log_prob_and_entropy(state_t)
        return action_t.squeeze(0).detach().cpu().numpy()

    def update(self, states, actions, rewards, next_states, dones, call_back=None):
        """
        n-step A2C update with GAE for continuous actions.
        Args are identical to the discrete version, except:
          - actions: list/array of shape [n_steps, action_dim]
        """
        
        self.update_counter += 1

        # Update the step size
        if self.hp.anneal_step_size_flag:
            frac = 1.0 - (self.update_counter - 1.0) / self.hp.total_updates
            self.critic_optimizer.param_groups[0]["lr"] = frac * self.hp.critic_step_size
            self.actor_optimizer.param_groups[0]["lr"] = frac * self.hp.actor_step_size
            
        states_t = torch.cat(states).to(dtype=torch.float32, device=self.device) if torch.is_tensor(states[0]) \
                   else torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        next_states_t = torch.cat(next_states).to(dtype=torch.float32, device=self.device) if torch.is_tensor(next_states[0]) \
                        else torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        actions_t = torch.cat(actions).to(dtype=torch.float32, device=self.device) if torch.is_tensor(actions[0]) \
                    else torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)  # [n_steps, action_dim]

        # Critic values
        values = self.critic(states_t).squeeze(-1)               # [n_steps]
        with torch.no_grad():
            next_values = self.critic(next_states_t).squeeze(-1) # [n_steps]
        # n-step returns + advantages
        returns, advantages = calculate_gae(
            rewards,
            values,
            next_values,
            dones,
            gamma=self.hp.gamma,
            lamda=self.hp.lamda,
        )
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)  # [n_steps]
        returns_t    = torch.tensor(returns,    dtype=torch.float32, device=self.device).unsqueeze(1)  # [n_steps,1]
        # normalize advantage
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # Actor log-probs
        _, log_probs_t, entropy_t = self._log_prob_and_entropy(states_t, actions_t)  # [n_steps], [n_steps]
        
        # Critic update
        critic_loss = F.mse_loss(values.unsqueeze(1), returns_t)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update (maximize advantage-weighted log-prob + entropy bonus)
        actor_loss = - (log_probs_t * advantages_t).mean() - self.hp.entropy_coef * entropy_t.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if call_back is not None:
            call_back({
                "critic_loss": float(critic_loss.item()),
                "actor_loss":  float(actor_loss.item())
            })
    
    def save(self, file_path=None):
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_logstd': self.actor_logstd.detach().cpu(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),

            'action_space': self.action_space,
            'features_dim': self.features_dim,
            'hyper_params': self.hp,

            'policy_class': self.__class__.__name__,
        }
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_policy.t")
        return checkpoint

    @classmethod
    def load_from_file(cls, file_path, seed=0, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        instance = cls(checkpoint['action_space'], checkpoint['features_dim'], checkpoint['hyper_params'])
        
        instance.reset(seed)
        instance.actor.load_state_dict(checkpoint['actor_state_dict'])
        instance.critic.load_state_dict(checkpoint['critic_state_dict'])
        instance.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        instance.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        instance.actor_logstd = nn.Parameter(checkpoint['actor_logstd'])

        return instance

    def load_from_checkpoint(self, checkpoint):
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.actor_logstd = nn.Parameter(checkpoint.get('actor_logstd'))

        self.action_space = checkpoint.get('action_space')
        self.features_dim = checkpoint.get('features_dim')
        self.hp = checkpoint.get('hyper_params')
        
@register_agent
class A2CAgent(BaseAgent):
    """
    Advantage Actor-Critic agent using n-step rollouts in a single environment.
    """
    name = "A2C"
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device="cpu"):
        """
        Args:
            action_space (gym.spaces.Discrete): Environment's action space.
            observation_space: Environment's observation space.
            hyper_params: Hyper-parameters container (see A2CPolicy for required keys).
            num_envs (int): Number of parallel environments.
            feature_extractor_class: Class to extract features from observations.
        """
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device=device)
        
        if isinstance(action_space, gymnasium.spaces.Discrete):
            if action_space.n < 2:
                raise ValueError("Discrete action space must have n >= 2.")
            self.policy = A2CPolicyDiscrete(
                action_space, 
                self.feature_extractor.features_dim, 
                hyper_params,
                device=device
            )
            
        elif isinstance(action_space, gymnasium.spaces.Box): 
            # Require 1D action vector
            if len(action_space.shape) != 1:
                raise ValueError(
                    f"Continuous A2C expects a 1D Box for actions; got shape {action_space.shape}."
                )
            if not np.issubdtype(action_space.dtype, np.floating):
                raise TypeError(
                    f"Box action dtype must be float; got {action_space.dtype}."
                )
            self.policy = A2CPolicyContinuous(
                action_space, 
                self.feature_extractor.features_dim, 
                hyper_params,
                device=device
            )
        else:
            raise NotImplementedError("Policy is not implemented")
        
        # Rollout buffer stores transitions: (state, log_prob, reward, next_state, done)
        self.rollout_buffer = BasicBuffer(np.inf)

    def act(self, observation, greedy=False):
        """
        Select an action and store log_prob for later update.
        
        Args:
            observation (np.array or similar): Raw observation.
            
        Returns:
            int: Chosen action.
        """
        state = self.feature_extractor(observation)
        action = self.policy.select_action(state, greedy=greedy)

        self.last_state = state
        self.last_action = action
        return action

    def parallel_act(self, observations, greedy=False):        
        states = self.feature_extractor(observations)
        actions = self.policy.select_parallel_actions(states, greedy=False)
        
        self.last_states = states
        self.last_actions = actions
        return actions

    def update(self, observation, reward, terminated, truncated, call_back=None):
        """
        Store the transition; if rollout is complete (n_steps reached or episode ended), perform update.
        
        Args:
            observation (np.array or similar): New observation after action.
            reward (float): Reward received.
            terminated (bool): True if episode terminated.
            truncated (bool): True if episode was truncated.
            call_back (function, optional): Callback to track losses.
        """
        state = self.feature_extractor(observation)
        transition = (self.last_state, self.last_action, reward, state, terminated)
        self.rollout_buffer.add_single_item(transition)

        # If rollout length reached or episode ended, perform update.
        if self.rollout_buffer.size >= self.hp.rollout_steps:
            rollout = self.rollout_buffer.get_all()
            states, actions, rewards, next_states, dones = zip(*rollout)
            
            self.policy.update(states, actions, rewards, next_states, dones, call_back=call_back)
            self.rollout_buffer.reset()
            
    
    def parallel_update(self, observations, rewards, terminateds, truncateds, call_back=None):
        """
        observations: next batched obs after step (num_envs, ...)
        rewards:     (num_envs,)
        terminateds: (num_envs,)
        truncateds:  (num_envs,)
        """
        states = self.feature_extractor(observations)
        
        for i in range(self.num_envs):
            #unsqueeze(0) to keep the batch dimension
            transition =  (self.last_states[i].unsqueeze(0), self.last_actions[i], rewards[i], states[i].unsqueeze(0), terminateds[i])
            self.rollout_buffer.add_single_item(transition)
            

        if self.rollout_buffer.size >= self.hp.rollout_steps:
            self.last_update_counter = 0
            
            rollout = self.rollout_buffer.get_all()
            states, actions, rewards, next_states, dones = zip(*rollout)
            
            self.policy.update(states, actions, rewards, next_states, dones, call_back=call_back)
            self.rollout_buffer.reset()
            

        
    
    def reset(self, seed):
        """
        Reset the agent's state, including feature extractor and rollout buffer.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.rollout_buffer.reset()
