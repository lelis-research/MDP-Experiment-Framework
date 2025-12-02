import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical, Normal, Independent, TransformedDistribution
import math
import gymnasium
from gymnasium.spaces import Discrete, Box


from ...Base import BaseAgent, BasePolicy
from ....Buffers import BaseBuffer
from ...Utils import calculate_gae, get_single_observation, stack_observations
from ....registry import register_agent, register_policy
from ....Networks.NetworkFactory import NetworkGen, prepare_network_config
from ....FeatureExtractors import get_batch_features

def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    1 - Var[y_true - y_pred] / Var[y_true]
    Returns 0 if Var[y_true] == 0.
    """
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    var_y = np.var(y_true_np)
    if var_y < 1e-10:
        return 0.0
    return float(1.0 - np.var(y_true_np - y_pred_np) / (var_y + 1e-10))

@register_policy
class PPOPolicy(BasePolicy):
    def __init__(self, action_space, features_dict, hyper_params, device="cpu"):
        super().__init__(action_space, hyper_params, device=device)
        self.features_dict = features_dict
        
        actor_description = prepare_network_config(
            self.hp.actor_network,
            input_dims=self.features_dict,
            output_dim=self.action_dim
        )
        critic_description = prepare_network_config(
            self.hp.critic_network,
            input_dims=self.features_dict,
            output_dim=1
        )
        
        self.actor = NetworkGen(layer_descriptions=actor_description).to(self.device)
        self.critic = NetworkGen(layer_descriptions=critic_description).to(self.device)
        
        if isinstance(self.action_space, Box):
            self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(self.action_space.shape), device=self.device))
            self.actor_optimizer = optim.Adam(list(self.actor.parameters()) + [self.actor_logstd],lr=self.hp.actor_step_size, eps=1e-5)
            
            self.action_low = torch.as_tensor(self.action_space.low, device=self.device, dtype=torch.float32)
            self.action_high = torch.as_tensor(self.action_space.high, device=self.device, dtype=torch.float32)
        elif isinstance(self.action_space, Discrete):
            self.actor_logstd = None
            self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=self.hp.actor_step_size, eps=1e-5)
        else:
            raise NotImplementedError(f"PPOPolicy does not support action space of type {type(self.action_space)}")
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=self.hp.critic_step_size, eps=1e-5)
        self.update_counter = 0
        self.hp.update(total_updates=self.hp.total_steps // self.hp.rollout_steps)
        
    
    def reset(self, seed):
        super().reset(seed)

        actor_description = prepare_network_config(
            self.hp.actor_network,
            input_dims=self.features_dict,
            output_dim=self.action_dim
        )
        critic_description = prepare_network_config(
            self.hp.critic_network,
            input_dims=self.features_dict,
            output_dim=1
        )
        
        self.actor = NetworkGen(layer_descriptions=actor_description).to(self.device)
        self.critic = NetworkGen(layer_descriptions=critic_description).to(self.device)
        
        if isinstance(self.action_space, Box):
            self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(self.action_space.shape), device=self.device))
            self.actor_optimizer = optim.Adam(list(self.actor.parameters()) + [self.actor_logstd],lr=self.hp.actor_step_size, eps=1e-5)
            
            self.action_low = torch.as_tensor(self.action_space.low, device=self.device, dtype=torch.float32)
            self.action_high = torch.as_tensor(self.action_space.high, device=self.device, dtype=torch.float32)
        elif isinstance(self.action_space, Discrete):
            self.actor_logstd = None
            self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=self.hp.actor_step_size, eps=1e-5)
        else:
            raise NotImplementedError(f"A2CPolicy does not support action space of type {type(self.action_space)}")
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=self.hp.critic_step_size, eps=1e-5)
        self.update_counter = 0   
        self.hp.update(total_updates=self.hp.total_steps // self.hp.rollout_steps)


    def _log_prob_and_entropy(self, state, action_t=None):
        """
        Returns:
            action_t: sampled or given action
              - Discrete: shape [B]  (dtype long)
              - Continuous: shape [B, action_dim]
            log_prob: shape [B]
            entropy: shape [B]
        """

        # Forward actor
        logits_or_mean = self.actor(**state)
        
        if isinstance(self.action_space, Discrete):
            # ----- DISCRETE: Categorical over actions -----
            # logits_or_mean: [B, action_dim]
            dist = Categorical(logits=logits_or_mean)
            
            # Greedy = argmax over logits
            greedy_action = torch.argmax(logits_or_mean, dim=-1)  # [B]
                
            if action_t is None:
                action_t = dist.sample()             # [B]
            # log_prob: [B]
            log_prob = dist.log_prob(action_t)
            # entropy: [B]
            entropy = dist.entropy()

        elif isinstance(self.action_space, Box):
            # ----- CONTINUOUS: Normal over Box actions -----
            # logits_or_mean: [B, action_dim] = mean
            mean = logits_or_mean
            logstd = self.actor_logstd.expand_as(mean)   # [1, A] -> [B, A]
            std = torch.exp(logstd)

            dist = Normal(mean, std)
                
            greedy_action = mean
            if action_t is None:
                action_t = dist.sample()                 # [B, A]

            # For multivariate Normal with factorized dims, sum over action dim
            log_prob = dist.log_prob(action_t).sum(dim=-1)   # [B, A] -> [B]
            entropy = dist.entropy().sum(dim=-1)             # [B, A] -> [B]
        else:
            raise NotImplementedError(
                f"PPOPolicy:_log_prob_and_entropy does not support action space of type {type(self.action_space)}"
            )

        return action_t, log_prob, entropy, greedy_action
    
    def select_action(self, state, greedy=False):
        """
        Returns a numpy action of shape [action_dim].
        """
        
        with torch.no_grad():
            action, log_prob_t, _, greedy_action = self._log_prob_and_entropy(state)
    
            if greedy:
                return greedy_action.cpu().numpy(), log_prob_t.cpu().numpy()
            else:
                return action.cpu().numpy(), log_prob_t.cpu().numpy()

    def update(self, states, actions, old_log_probs, next_states, rewards, dones, call_back=None):

        self.update_counter += 1
            
        # LR annealing (optional)
        if self.hp.anneal_step_size_flag:
            frac = 1.0 - (self.update_counter - 1.0) / float(self.hp.total_updates)  # linear from initial->0
            for param_groups in self.actor_optimizer.param_groups:
                param_groups["lr"] = frac * self.hp.actor_step_size
            for param_groups in self.critic_optimizer.param_groups:
                param_groups["lr"] = frac * self.hp.critic_step_size    
                
        if self.hp.anneal_clip_range_actor:
            frac = 1.0 - (self.update_counter - 1.0) / float(self.hp.total_updates) 
            self.hp.update(clip_range_actor = float(frac * self.hp.clip_range_actor_init))
        else:
            self.hp.update(clip_range_actor = self.hp.clip_range_actor_init)
        
        if self.hp.anneal_clip_range_critic:
            frac = 1.0 - (self.update_counter - 1.0) / float(self.hp.total_updates) 
            self.hp.update(clip_range_critic = float(frac * self.hp.clip_range_critic_init))
        else:
            self.hp.update(clip_range_critic = self.hp.clip_range_critic_init)
            
        if isinstance(self.action_space, Discrete):
            actions_t = torch.tensor(np.array(actions), dtype=torch.int64, device=self.device)  # (T,)
        elif isinstance(self.action_space, Box):
            actions_t = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)  # (T, A)
        
        log_probs_old_t = torch.tensor(np.array(old_log_probs), dtype=torch.float32, device=self.device)  # (T,)

        
        with torch.no_grad():
            values = self.critic(**states).squeeze(-1)               # (T, )
            next_values = self.critic(**next_states).squeeze(-1) # (T, )
                    
        returns, advantages = calculate_gae(
            rewards,
            values,
            next_values,
            dones,
            gamma=self.hp.gamma,
            lamda=self.hp.lamda,
        )
        
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device) # (T, )
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device) # (T, )

        if self.hp.norm_adv_flag and advantages_t.numel() > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)


        datasize = self.hp.rollout_steps
        indices = np.arange(datasize)
        continue_training = True
        
        # for logging
        entropy_losses, actor_losses, critic_losses, clip_fractions = [], [], [], []

        for epoch in range(self.hp.num_epochs):
            if not continue_training:
                break
            approx_kl_divs = []
            
            indices = self._rand_permutation(datasize)
            for start in range(0, datasize, self.hp.mini_batch_size):
                batch_indices = indices[start:start + self.hp.mini_batch_size]
                
                batch_states     = get_batch_features(states, batch_indices) # shape (B, obs_dim)
                batch_actions_t    = actions_t[batch_indices] # shape (B, action_dim)
                batch_log_probs_t  = log_probs_old_t[batch_indices] # shape (B, )
                batch_advantages_t = advantages_t[batch_indices]     # shape (B, )
                batch_returns_t    = returns_t[batch_indices]        # shape (B, )
                batch_values_t     = values[batch_indices].squeeze() # shape (B, )


                # new log-probs under current policy
                _, batch_log_probs_new_t, entropy, _ = self._log_prob_and_entropy(batch_states, batch_actions_t)
                
                log_ratio = batch_log_probs_new_t - batch_log_probs_t
                ratios = torch.exp(log_ratio)  # [B, ]
                surr1 = - batch_advantages_t * ratios
                surr2 = - batch_advantages_t * torch.clamp(ratios, 1 - self.hp.clip_range_actor, 1 + self.hp.clip_range_actor)
                actor_loss = torch.max(surr1, surr2).mean()
                actor_losses.append(actor_loss.item())
                
                #logging
                clip_fraction = torch.mean((torch.abs(ratios - 1) > self.hp.clip_range_actor).float()).item()
                clip_fractions.append(clip_fraction)
                
                # critic loss 
                batch_new_values_t = self.critic(**batch_states).squeeze()
                
                if self.hp.clip_range_critic is None:
                    values_pred = batch_new_values_t
                else:
                    values_pred = batch_values_t + torch.clamp(batch_new_values_t - batch_values_t,
                                                           -self.hp.clip_range_critic, self.hp.clip_range_critic)
                critic_loss = F.mse_loss(batch_returns_t, values_pred)
                critic_losses.append(critic_loss.item())
                
                entropy_bonus = entropy.mean()    
                entropy_losses.append(entropy_bonus.item())
                
                loss = actor_loss + self.hp.critic_coef * critic_loss - self.hp.entropy_coef * entropy_bonus

                # early stopping
                with torch.no_grad():
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item()
                    approx_kl_divs.append(approx_kl)
                
                if self.hp.target_kl is not None and approx_kl > 1.5 * self.hp.target_kl:
                    continue_training = False
                    break
                    
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.hp.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.hp.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                if call_back is not None:
                    call_back({"critic_loss": float(np.mean(critic_losses)),
                                "actor_loss": float(np.mean(actor_losses)),
                                "entropy_loss": float(np.mean(entropy_losses)),
                                "loss": float(np.mean(actor_losses)) + 
                                        self.hp.critic_coef * float(np.mean(critic_losses)) + 
                                        self.hp.entropy_coef * float(np.mean(entropy_losses)),
                                        
                                "clip_fraction": float(np.mean(clip_fractions)),
                                "approx_kl": float(np.mean(approx_kl_divs)),
                                "explained_variance": explained_variance(values, returns_t)                                
                                })

    def save(self, file_path=None):
        checkpoint = super().save(file_path=None)
        
        checkpoint['actor_state_dict'] = self.actor.state_dict()
        checkpoint['critic_state_dict'] = self.critic.state_dict()
        checkpoint['actor_logstd'] = self.actor_logstd.cpu() if self.actor_logstd is not None else None
        checkpoint['actor_optimizer_state_dict'] = self.actor_optimizer.state_dict()
        checkpoint['critic_optimizer_state_dict'] = self.critic_optimizer.state_dict()
        
        checkpoint['features_dict'] = self.features_dict
        
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_policy.t")
        return checkpoint

    @classmethod
    def load(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        instance = cls(checkpoint['action_space'], checkpoint['features_dict'], checkpoint['hyper_params'], checkpoint['device'])
        instance.set_rng_state(checkpoint['rng_state']) #copied from the BasePolicy.load
        
        instance.actor.load_state_dict(checkpoint['actor_state_dict'])
        instance.actor_logstd = checkpoint['actor_logstd'].to(instance.device) if checkpoint['actor_logstd'] is not None else None
        instance.critic.load_state_dict(checkpoint['critic_state_dict'])
        instance.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        instance.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        instance.features_dict = checkpoint['features_dict']

        return instance
        
@register_agent
class PPOAgent(BaseAgent):
    """
    PPO agent using on-policy rollouts.
    
    Rollout buffer stores tuples of:
        (state, action, log_prob, state_value, reward, next_state, done)
    """
    name = "PPO"
    SUPPORTED_ACTION_SPACES = (Discrete, Box)
    
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device="cpu"):
        
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device=device)        
        self.policy = PPOPolicy(
                action_space, 
                self.feature_extractor.features_dict, 
                hyper_params,
                device=device
            )
        
        self.rollout_buffer = [BaseBuffer(self.hp.rollout_steps) for _ in range(self.num_envs)]  # Buffer is used for n-step

    def act(self, observation, greedy=False):
        """
        Select an action based on the observation.
        observation is a batch
        action is a batch 
        """
        state = self.feature_extractor(observation)
        action, log_prob = self.policy.select_action(state, greedy=greedy)
        
        self.last_observation = observation
        self.last_action = action
        self.last_log_prob = log_prob
        return action

    def update(self, observation, reward, terminated, truncated, call_back=None):
        """
        all arguments are batches
        """
        for i in range(self.num_envs):
            transition =  (get_single_observation(self.last_observation, i), 
                           self.last_action[i], self.last_log_prob[i], reward[i], 
                           get_single_observation(observation, i), terminated[i])
            self.rollout_buffer[i].add(transition)
            
            if self.rollout_buffer[i].is_full():
                rollout = self.rollout_buffer[i].all() 
                observations, actions, log_probs, rewards, next_observations, dones = zip(*rollout)
                observations, next_observations = stack_observations(observations), stack_observations(next_observations)
                states, next_states = self.feature_extractor(observations), self.feature_extractor(next_observations)
                self.policy.update(states, actions, log_probs, next_states, rewards, dones, call_back=call_back)
                self.rollout_buffer[i].clear()

    def reset(self, seed):
        super().reset(seed)
        
        self.last_observation = None
        self.last_action = None
        
        self.rollout_buffer = [BaseBuffer(self.hp.rollout_steps) for _ in range(self.num_envs)]