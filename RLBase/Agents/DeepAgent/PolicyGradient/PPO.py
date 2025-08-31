import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical, Normal, Independent, TransformedDistribution
import math
import gymnasium

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
class PPOPolicyDiscrete(BasePolicy):
    """
    Proximal Policy Optimization (PPO) policy for discrete actions.
    
    Hyper-parameters (in self.hp) must include:
        - gamma (float)
        - actor_step_size (float)
        - critic_step_size (float)
        - clip_range (float)
        - num_epochs (int)
        - mini_batch_size (int)
        - entropy_coef (float)
        - rollout_steps (int)
        
    Actor and Critic networks are built from provided network configurations.
    """
    def __init__(self, action_space, features_dim, hyper_params, device="cpu"):
        """
        Args:
            action_space (gym.spaces.Discrete): Action space.
            features_dim (int): Dimension of the flattened feature vector.
            hyper_params: Hyper-parameters container.
        """
        super().__init__(action_space, hyper_params, device=device)
        self.features_dim = features_dim
        self.action_dim = int(action_space.n) #Only for discrete actions

    def reset(self, seed):
        """
        Initialize actor and critic networks and their optimizers.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        # Build network configurations
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

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.hp.actor_step_size)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.hp.critic_step_size)

    def select_action(self, state, greedy=False):
        """
        Sample an action from the actor network.
        
        Args:
            state (np.array): Flat feature vector; shape [features_dim] or [1, features_dim].
            
        Returns:
            tuple: (action (int), log_prob (torch.Tensor), state_value (torch.Tensor))
        """
        state_t = state.to(dtype=torch.float32, device=self.device) if torch.is_tensor(state) else torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.actor(state_t)
            dist = Categorical(logits=logits)
            if greedy:
                action_t = torch.argmax(logits, dim=-1)
            else:
                action_t = dist.sample()
            log_prob_t = dist.log_prob(action_t)
        return action_t.item(), log_prob_t.detach()

    def update(self, states, actions, old_log_probs, rewards, next_states, dones, call_back=None):
        """
        Perform PPO update over the collected rollout.
        
        Args:
            states (list or np.array): List of states; shape [rollout_steps, features_dim].
            actions (list): List of actions (int) taken.
            old_log_probs (list): List of log probabilities (torch.Tensor) from the old policy.
            states_values (list): List of state value estimates (torch.Tensor); shape [rollout_steps, 1].
            rewards (list): List of rewards (float) collected.
            next_states (list or np.array): List of next states; shape [rollout_steps, features_dim].
            dones (list): List of done flags (bool).
            call_back (function, optional): Callback to report loss metrics.
        """
        states_t = torch.cat(states).to(dtype=torch.float32, device=self.device) if torch.is_tensor(states[0]) \
                    else torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(np.array(actions), dtype=torch.int64, device=self.device)
        log_probs_old_t = torch.stack(old_log_probs).squeeze().to(dtype=torch.float32, device=self.device)
        next_states_t = torch.cat(next_states).to(dtype=torch.float32, device=self.device) if torch.is_tensor(next_states[0]) \
                        else torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            prev_state_values_t = self.critic(states_t) # shape (rollout_steps, 1)
            next_state_values_t = self.critic(next_states_t) # shape (rollout_steps, 1)
                    
        returns, advantages = calculate_gae(
            rewards,
            prev_state_values_t,
            next_state_values_t,
            dones,
            gamma=self.hp.gamma,
            lamda=self.hp.lamda,
        )
        
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(1)  # [rollout_steps,1]
        
        if self.hp.norm_adv_flag:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        datasize = len(states)
        indices = np.arange(datasize)
        for epoch in range(self.hp.num_epochs):
            np.random.shuffle(indices)
            for start in range(0, datasize, self.hp.mini_batch_size):
                batch_indices = indices[start:start + self.hp.mini_batch_size]
                
                batch_states_t     = states_t[batch_indices] # shape (B, obs_dim)
                batch_actions_t    = actions_t[batch_indices] # shape (B, action_dim)
                batch_log_probs_t  = log_probs_old_t[batch_indices] # shape (B, )
                batch_advantages_t = advantages_t[batch_indices]     # shape (B, )
                batch_returns_t    = returns_t[batch_indices]        # shape (B, )
                batch_values_t     = prev_state_values_t[batch_indices].squeeze() # shape (B, )

                logits = self.actor(batch_states_t)
                dist = Categorical(logits=logits)
                batch_new_log_probs_t = dist.log_prob(batch_actions_t)
                entropy = dist.entropy()
                
                # actor loss
                ratios = torch.exp(batch_new_log_probs_t - batch_log_probs_t)  # [B, ]
                surr1 = - batch_advantages_t * ratios
                surr2 = - batch_advantages_t * torch.clamp(ratios, 1 - self.hp.clip_range, 1 + self.hp.clip_range)
                actor_loss = torch.max(surr1, surr2).mean()
                
                # critic loss
                batch_new_values_t = self.critic(batch_states_t).squeeze()

                if self.hp.clip_critic_loss_flag:
                    critic_loss_unclipped = (batch_returns_t - batch_new_values_t).pow(2)
                    value_clipped = batch_values_t + torch.clamp(batch_new_values_t - batch_values_t, 
                                                               -self.hp.clip_range, self.hp.clip_range)
                    critic_loss_clipped = (batch_returns_t - value_clipped).pow(2)
                    critic_loss = 0.5 * torch.max(critic_loss_clipped, critic_loss_unclipped).mean()
                else:
                    critic_loss = 0.5 * (batch_returns_t - batch_new_values_t).pow(2).mean()
                
                entropy_bonus = entropy.mean()
                
                loss = actor_loss + self.hp.critic_coef * critic_loss - self.hp.entropy_coef * entropy_bonus
                
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.hp.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.hp.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                if call_back is not None:
                    call_back({"critic_loss": critic_loss.item(),
                               "actor_loss": actor_loss.item()})
    
    def save(self, file_path=None):
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),

            'action_space': self.action_space,
            'features_dim': self.features_dim,
            'hyper_params': self.hp,
            
            'action_dim': self.action_dim,            
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
class PPOPolicyContinuous(BasePolicy):
    def __init__(self, action_space, features_dim, hyper_params, device="cpu"):
        assert hasattr(action_space, "shape") and len(action_space.shape) == 1, \
            "PPOPolicyContinuous supports 1D Box action spaces."
        super().__init__(action_space, hyper_params, device=device)
        self.features_dim = features_dim
        self.action_dim   = int(action_space.shape[0])
    
    def reset(self, seed):
        """
        Builds actor/critic and their optimizers.
        Actor outputs 2*action_dim (means and log-stds).
        """
        super().reset(seed)

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
        self.actor  = NetworkGen(layer_descriptions=actor_description).to(self.device)
        self.critic = NetworkGen(layer_descriptions=critic_description).to(self.device)
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(self.action_space.shape), device=self.device))

        # self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=self.hp.actor_step_size)
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
        Sample an action and return (action_np, log_prob_t, value_t).
        """
        state_t = state.to(dtype=torch.float32, device=self.device) if torch.is_tensor(state) \
                  else torch.tensor(state, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if greedy:
                action_t = self.actor(state_t)
                action_t, log_prob_t, _ = self._log_prob_and_entropy(state_t, action_t)
            else:
                action_t, log_prob_t, _ = self._log_prob_and_entropy(state_t)
        return action_t.squeeze(0).detach().cpu().numpy(), log_prob_t

    def update(self, states, actions, old_log_probs, rewards, next_states, dones, call_back=None):
        """
        PPO update over one rollout.
        Mirrors PPOPolicyDiscrete.update but with continuous log-probs.

        Args layout matches your discrete PPO:
            states, next_states: list/np/tensors of shape [T, obs_dim]
            actions: list/np/tensors of shape [T, action_dim] (bounded in Box range)
            old_log_probs: list of torch.Tensors shape [] or [1]
            states_values: list of torch.Tensors shape [1] (critic values at states)
            rewards: list[float]
            dones: list[bool]
        """

        self.update_counter += 1

        # Update the step size
        if self.hp.anneal_step_size_flag:
            frac = 1.0 - (self.update_counter - 1.0) / self.hp.total_updates
            self.critic_optimizer.param_groups[0]["lr"] = frac * self.hp.critic_step_size
            self.actor_optimizer.param_groups[0]["lr"] = frac * self.hp.actor_step_size
        
        states_t = torch.cat(states).to(dtype=torch.float32, device=self.device) if torch.is_tensor(states[0]) \
                   else torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.cat(actions).to(dtype=torch.float32, device=self.device) if torch.is_tensor(actions[0]) \
                    else torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        log_probs_old_t   = torch.stack(old_log_probs).squeeze().to(dtype=torch.float32, device=self.device)
        next_states_t = torch.cat(next_states).to(dtype=torch.float32, device=self.device) if torch.is_tensor(next_states[0]) \
                        else torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)

        
        
        with torch.no_grad():
            prev_state_values_t = self.critic(states_t) # shape (rollout_steps, 1)
            next_state_values_t = self.critic(next_states_t) # shape (rollout_steps, 1)
        
                    
        returns, advantages = calculate_gae(
            rewards,
            prev_state_values_t,
            next_state_values_t,
            dones,
            gamma=self.hp.gamma,
            lamda=self.hp.lamda,
        )
        
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device) # (rollout_steps, )
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device) # (rollout_steps, )

        if self.hp.norm_adv_flag:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        datasize = len(states_t)
        indices = np.arange(datasize)

        for epoch in range(self.hp.num_epochs):
            np.random.shuffle(indices)
            for start in range(0, datasize, self.hp.mini_batch_size):
                batch_indices = indices[start:start + self.hp.mini_batch_size]

                batch_states_t     = states_t[batch_indices] # shape (B, obs_dim)
                batch_actions_t    = actions_t[batch_indices] # shape (B, action_dim)
                batch_log_probs_t  = log_probs_old_t[batch_indices] # shape (B, )
                batch_advantages_t = advantages_t[batch_indices]     # shape (B, )
                batch_returns_t    = returns_t[batch_indices]        # shape (B, )
                batch_values_t     = prev_state_values_t[batch_indices].squeeze() # shape (B, )


                # new log-probs under current policy
                _, batch_log_prob_new_t, entropy = self._log_prob_and_entropy(batch_states_t, batch_actions_t)
                ratios = torch.exp(batch_log_prob_new_t - batch_log_probs_t)  # [B, ]
                
                surr1 = - batch_advantages_t * ratios
                surr2 = - batch_advantages_t * torch.clamp(ratios, 1 - self.hp.clip_range, 1 + self.hp.clip_range)
                actor_loss = torch.max(surr1, surr2).mean()

                # critic loss 
                batch_new_values_t = self.critic(batch_states_t).squeeze()

                if self.hp.clip_critic_loss_flag:
                    critic_loss_unclipped = (batch_returns_t - batch_new_values_t).pow(2)
                    value_clipped = batch_values_t + torch.clamp(batch_new_values_t - batch_values_t, 
                                                               -self.hp.clip_range, self.hp.clip_range)
                    critic_loss_clipped = (batch_returns_t - value_clipped).pow(2)
                    critic_loss = 0.5 * torch.max(critic_loss_clipped, critic_loss_unclipped).mean()
                else:
                    critic_loss = 0.5 * (batch_returns_t - batch_new_values_t).pow(2).mean()
                    
                entropy_bonus = entropy.mean()    
                
                loss = actor_loss + self.hp.critic_coef * critic_loss - self.hp.entropy_coef * entropy_bonus

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.hp.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.hp.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                if call_back is not None:
                    call_back({
                        "critic_loss": critic_loss.item(),
                        "actor_loss": actor_loss.item()
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

            'action_dim': self.action_dim,
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
        instance.actor_logstd = checkpoint['actor_logstd']
        return instance

    def load_from_checkpoint(self, checkpoint):
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.actor_logstd = checkpoint.get('actor_logstd')
        
        self.action_space = checkpoint.get('action_space')
        self.features_dim = checkpoint.get('features_dim')
        self.hp = checkpoint.get('hyper_params')
        
@register_agent
class PPOAgent(BaseAgent):
    """
    PPO agent using on-policy rollouts with n-step trajectories.
    
    Rollout buffer stores tuples of:
        (state, action, log_prob, state_value, reward, next_state, done)
    """
    name = "PPO"
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device="cpu"):
        """
        Args:
            action_space (gym.spaces.Discrete): Action space.
            observation_space: Observation space.
            hyper_params: Hyper-parameters container.
            num_envs (int): Number of parallel environments.
            feature_extractor_class: Class to extract features from observations.
        """
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device=device)        
        if isinstance(action_space, gymnasium.spaces.Discrete):
            if action_space.n < 2:
                raise ValueError("Discrete action space must have n >= 2.")
            self.policy = PPOPolicyDiscrete(
                action_space, 
                self.feature_extractor.features_dim, 
                hyper_params,
                device=device
            )   
        elif isinstance(action_space, gymnasium.spaces.Box): 
            # Require 1D action vector
            if len(action_space.shape) != 1:
                raise ValueError(
                    f"Continuous PPO expects a 1D Box for actions; got shape {action_space.shape}."
                )
            if not np.issubdtype(action_space.dtype, np.floating):
                raise TypeError(
                    f"Box action dtype must be float; got {action_space.dtype}."
                )
            self.policy = PPOPolicyContinuous(
                action_space, 
                self.feature_extractor.features_dim, 
                hyper_params,
                device=device
            )
        else:
            raise NotImplementedError("Policy is not implemented")
        
        self.rollout_buffer = BasicBuffer(np.inf)

    def act(self, observation, greedy=False):
        """
        Sample an action from the policy.
        
        Args:
            observation (np.array or similar): Raw observation.
        
        Returns:
            int: Selected action.
        """
        state = self.feature_extractor(observation)
        action, log_prob = self.policy.select_action(state, greedy=greedy)
        
        self.last_state = state
        self.last_action = action
        self.last_log_prob = log_prob
        return action

    def update(self, observation, reward, terminated, truncated, call_back=None):
        """
        Store transition and update policy when rollout is complete.
        
        Args:
            observation (np.array or similar): New observation.
            reward (float): Reward received.
            terminated (bool): True if episode terminated.
            truncated (bool): True if episode truncated.
            call_back (function, optional): Callback to report loss metrics.
        """
        state = self.feature_extractor(observation)
        transition = (self.last_state, self.last_action, self.last_log_prob, reward, state, terminated)
        self.rollout_buffer.add_single_item(transition)
        
        if self.rollout_buffer.size >= self.hp.rollout_steps:
            rollout = self.rollout_buffer.get_all()
            states, actions, log_probs, rewards, next_states, dones = zip(*rollout)
            
            self.policy.update(states, actions, log_probs, rewards, next_states, dones, call_back=call_back)
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