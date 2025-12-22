import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from gymnasium.spaces import Discrete, Box

from ....Networks.NetworkFactory import NetworkGen, prepare_network_config
from .PPO import PPOAgent, PPOPolicy
from ....Buffers import BaseBuffer
from ...Utils import (
    calculate_gae_with_discounts, 
    get_single_observation, 
    stack_observations, 
    grad_norm,
    explained_variance,
    get_single_state,
    get_single_observation_nobatch,
)
from ....registry import register_agent, register_policy
from ....Options import load_options_list, save_options_list
from ....FeatureExtractors import get_batch_features

@register_policy
class OptionPPOPolicy(PPOPolicy):
    def update(self, states, actions, old_log_probs, next_states, rewards, discounts, terminated, truncated, call_back=None):

        self.update_counter += 1
            
        # LR annealing (optional)
        if self.hp.enable_stepsize_anneal:
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
                    
        returns, advantages = calculate_gae_with_discounts(
            rewards,
            values.detach(),
            next_values.detach(),
            terminated,
            truncated,
            discounts,
            lamda=self.hp.lamda,
        )
        
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device) # (T, )
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device) # (T, )

        if self.hp.enable_advantage_normalization and advantages_t.numel() > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
            
        assert advantages_t.shape[0] == returns_t.shape[0] == log_probs_old_t.shape[0] == actions_t.shape[0] == values.shape[0] == next_values.shape[0]
        
        datasize = advantages_t.shape[0] #self.hp.rollout_steps
        indices = np.arange(datasize)
        continue_training = True
        
        # for logging
        entropy_losses, actor_losses, critic_losses, clip_fractions, losses, approx_kl_divs = [], [], [], [], [], []
        actor_grad_norms, critic_grad_norms = [], []

        for epoch in range(self.hp.num_epochs):
            if not continue_training:
                break
            
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
                _, batch_log_probs_new_t, entropy, _ = self.get_logprob_entropy(batch_states, batch_actions_t)
            
                log_ratio = batch_log_probs_new_t - batch_log_probs_t
                ratios = torch.exp(log_ratio)  # [B, ]
                surr1 = - batch_advantages_t * ratios
                surr2 = - batch_advantages_t * torch.clamp(ratios, 1 - self.hp.clip_range_actor, 1 + self.hp.clip_range_actor)
                actor_loss = torch.max(surr1, surr2).mean()
                actor_losses.append(actor_loss.item())
                
                if call_back is not None:
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
                losses.append(loss.item())
                
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
                if call_back is not None:
                    actor_grad_norms.append(grad_norm(self.actor.parameters()))
                    critic_grad_norms.append(grad_norm(self.critic.parameters()))
        
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.hp.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.hp.max_grad_norm)
                if self.actor_logstd is not None:
                    nn.utils.clip_grad_norm_([self.actor_logstd], self.hp.max_grad_norm)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
        if call_back is not None:
            payload = {
                "critic_loss": float(np.mean(critic_losses)),
                "actor_loss": float(np.mean(actor_losses)),
                "entropy_loss": float(np.mean(entropy_losses)),
                "loss": float(np.mean(losses)),
                        
                "clip_fraction": float(np.mean(clip_fractions)),
                "approx_kl": float(np.mean(approx_kl_divs)),
                "explained_variance": explained_variance(values, returns_t),
                
                "advantage_mean": float(advantages_t.mean().item()),
                "advantage_std": float(advantages_t.std().item()),
                
                "values_mean": float(values.mean().item()),
                "values_std": float(values.std().item()),
                "returns_mean": float(returns_t.mean().item()),
                "returns_std": float(returns_t.std().item()),
                
                "lr_actor": float(self.actor_optimizer.param_groups[0]["lr"]),
                "lr_critic": float(self.critic_optimizer.param_groups[0]["lr"]), 
                
                "actor_grad_norms": float(np.mean(actor_grad_norms)),
                "critic_grad_norms": float(np.mean(critic_grad_norms)),
            }
            if isinstance(self.action_space, Box) and self.actor_logstd is not None:
                payload["actor_logstd_mean"] = float(self.actor_logstd.mean().item())
                payload["actor_logstd_std"] = float(self.actor_logstd.std().item())
            
            call_back(payload)


@register_agent
class OptionPPOAgent(PPOAgent):
    name = "OptionPPO"
    SUPPORTED_ACTION_SPACES = (Discrete, )
    
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, init_option_lst=None, device="cpu"):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device=device)

        self.atomic_action_space = action_space
        self.options_lst = [] if init_option_lst is None else init_option_lst

        # Augment action space with options
        action_option_space = Discrete(self.atomic_action_space.n + len(self.options_lst))

        # Policy over (actions + options)
        self.policy = OptionPPOPolicy(
            action_option_space, 
            self.feature_extractor.features_dict,
            hyper_params,
            device=device
        )

        # Option execution bookkeeping
        self.running_option_index = [None for _ in range(self.num_envs)]       # index into options_lst (or None)
        self.option_start_obs = [None for _ in range(self.num_envs)]         # encoded state where option began
        self.option_cumulative_reward = [0.0 for _ in range(self.num_envs)]    # discounted return accumulator R_{t:t+k}
        self.option_multiplier = [1.0 for _ in range(self.num_envs)]           # current gamma^t during option
        self.option_num_steps = [0 for _ in range(self.num_envs)]
        self.option_log_prob = [None for _ in range(self.num_envs)]
        
    def act(self, observation, greedy=False):
        """
        Select a primitive action. If currently executing an option, return the option's next action.
        Otherwise, sample from the policy over (actions + options). If an option is chosen, initialize
        bookkeeping and return the option's first primitive action.
        
        Args:
            observation (np.array or similar): Raw observation.
            
        Returns:
            int: Primitive action to execute in the environment.
        """
        state = self.feature_extractor(observation) 
        
        action = []
        log_prob = []
        for i in range(self.num_envs):
            # If an option is currently running, either continue it or end it here.
            st = get_single_state(state, i)
            obs = get_single_observation(observation, i)
            obs_option = get_single_observation_nobatch(observation, i)
            curr_option_idx = self.running_option_index[i]
            if curr_option_idx is not None:
                a = self.options_lst[curr_option_idx].select_action(obs_option)
                lp = None
            else:
                # Choose an extended action (might be a primitive or an option)
                a, lp = self.policy.select_action(st, greedy=greedy) 
                a, lp = a[0], lp[0] # because 'a', 'lp' would be a batch of size 1 (for over the envs)
                
                if a >= self.atomic_action_space.n:
                    # Start an option
                    curr_option_idx = a - self.atomic_action_space.n
                    self.option_start_obs[i] = obs
                    self.option_log_prob[i] = lp
                    self.option_cumulative_reward[i] = 0.0
                    self.option_multiplier[i] = 1.0
                    self.option_num_steps[i] = 0
                    a = self.options_lst[curr_option_idx].select_action(obs_option)
                else:
                    curr_option_idx = None
            
            self.running_option_index[i] = curr_option_idx
            action.append(a) 
            log_prob.append(lp)
            
        self.last_observation = observation
        self.last_action = action
        self.last_log_prob = log_prob
        return action
    
    def update(self, observation, reward, terminated, truncated, call_back=None):
        if self.hp.update_type == "sync":
            self.hp.update(total_updates=self.hp.total_steps // (self.hp.rollout_steps * self.num_envs))
            return self.update_sync(observation, reward, terminated, truncated, call_back)
        elif self.hp.update_type == "per_env":
            self.hp.update(total_updates=self.hp.total_steps // self.hp.rollout_steps)
            return self.update_per_env(observation, reward, terminated, truncated, call_back)
        else:
            raise NotImplementedError("update_type not defined")
    
    def update_sync(self, observation, reward, terminated, truncated, call_back=None):
        """
        all arguments are batches
        """
        # add to the rollouts
        for i in range(self.num_envs):
            obs = get_single_observation(observation, i)
            obs_option = get_single_observation_nobatch(observation, i)
            curr_option_idx = self.running_option_index[i]
            
            if curr_option_idx is not None:
                # if an option is running
                # Accumulate SMDP return while option runs 
                self.option_cumulative_reward[i] += self.option_multiplier[i] * float(reward[i])
                self.option_multiplier[i] *= self.hp.gamma
                self.option_num_steps[i] += 1
                if self.options_lst[curr_option_idx].is_terminated(obs_option) or terminated[i] or truncated[i]:
                    transition = (
                        self.option_start_obs[i], 
                        self.atomic_action_space.n + curr_option_idx, 
                        self.option_log_prob[i],
                        self.option_cumulative_reward[i], 
                        self.option_multiplier[i], 
                        obs, 
                        terminated[i], 
                        truncated[i]
                    )                  
                    self.rollout_buffer[i].add(transition)
                    self.options_lst[curr_option_idx].reset()
                    self.running_option_index[i] = None
            else:
                transition = (
                    get_single_observation(self.last_observation, i), 
                    self.last_action[i], 
                    self.last_log_prob[i],
                    reward[i], 
                    self.hp.gamma,
                    obs, 
                    terminated[i],
                    truncated[i]
                )
                    
                self.rollout_buffer[i].add(transition)
        
        # update with the rollouts
        if not all(buf.is_full() for buf in self.rollout_buffer):
            return

        all_observations        = []
        all_next_observations   = []
        all_actions             = []
        all_log_probs           = []
        all_rewards             = []
        all_terminated          = []
        all_truncated           = []
        all_discounts           = []
        
        for i in range(self.num_envs):
            rollout = self.rollout_buffer[i].all()
            (
                rollout_observations,
                rollout_actions,
                rollout_log_probs,
                rollout_rewards,
                rollout_discounts,
                rollout_next_observations,
                rollout_terminated,
                rollout_truncated,
            ) = zip(*rollout)
            
            all_observations.extend(rollout_observations)
            all_next_observations.extend(rollout_next_observations)
            all_actions.extend(rollout_actions)
            all_log_probs.extend(rollout_log_probs)
            all_rewards.extend(rollout_rewards)
            all_discounts.extend(rollout_discounts)
            all_terminated.extend(rollout_terminated)
            all_truncated.extend(rollout_truncated)
            
            # --- IMPORTANT: cut GAE at env boundary with a fake truncation ---
            # If the last step of this env's rollout is not truly terminated,
            # mark it as truncated so advantages don't leak into the next env's segment.
            last_idx = len(all_truncated) - 1
            if not all_terminated[last_idx]:
                all_truncated[last_idx] = True
                
            # clear this env's buffer for next rollout
            self.rollout_buffer[i].clear()

        # 4) Stack and featurize the big batch
        observations, next_observations = (
            stack_observations(all_observations),
            stack_observations(all_next_observations),
        )
        states      = self.feature_extractor(observations)
        next_states = self.feature_extractor(next_observations)

        # 5) Single PPO update using data from *all* envs combined
        self.policy.update(
            states,
            all_actions,
            all_log_probs,
            next_states,
            all_rewards,
            all_discounts,
            all_terminated,
            all_truncated,
            call_back=call_back,
        )
            
    def update_per_env(self, observation, reward, terminated, truncated, call_back=None):
        """
        all arguments are batches
        """
        for i in range(self.num_envs):
            
            # add to the rollouts
            obs = get_single_observation(observation, i)
            obs_option = get_single_observation_nobatch(observation, i)
            curr_option_idx = self.running_option_index[i]
            
            if curr_option_idx is not None:
                # if an option is running
                # Accumulate SMDP return while option runs 
                self.option_cumulative_reward[i] += self.option_multiplier[i] * float(reward[i])
                self.option_multiplier[i] *= self.hp.gamma
                self.option_num_steps[i] += 1
                if self.options_lst[curr_option_idx].is_terminated(obs_option) or terminated[i] or truncated[i]:
                    transition = (
                        self.option_start_obs[i], 
                        self.atomic_action_space.n + curr_option_idx, 
                        self.option_log_prob[i],
                        self.option_cumulative_reward[i], 
                        self.option_multiplier[i], 
                        obs, 
                        terminated[i], 
                        truncated[i]
                    )                  
                    self.rollout_buffer[i].add(transition)
                    self.options_lst[curr_option_idx].reset()
                    self.running_option_index[i] = None
            else:
                transition = (
                    get_single_observation(self.last_observation, i), 
                    self.last_action[i], 
                    self.last_log_prob[i],
                    reward[i], 
                    self.hp.gamma,
                    obs, 
                    terminated[i],
                    truncated[i]
                )
                    
                self.rollout_buffer[i].add(transition)
            
            # update with the rollouts
            if self.rollout_buffer[i].is_full():
                rollout = self.rollout_buffer[i].all() 
                (
                    rollout_observations,
                    rollout_actions,
                    rollout_log_probs,
                    rollout_rewards,
                    rollout_discounts,
                    rollout_next_observations,
                    rollout_terminated,
                    rollout_truncated,
                ) = zip(*rollout)
                
                rollout_observations, rollout_next_observations = (
                    stack_observations(rollout_observations), 
                    stack_observations(rollout_next_observations)
                )
                rollout_states, rollout_next_states = (
                    self.feature_extractor(rollout_observations), 
                    self.feature_extractor(rollout_next_observations)
                )
                
                self.policy.update(rollout_states, 
                                   rollout_actions, 
                                   rollout_log_probs, 
                                   rollout_next_states, 
                                   rollout_rewards,
                                   rollout_discounts, 
                                   rollout_terminated, 
                                   rollout_truncated, 
                                   call_back=call_back)
                
                self.rollout_buffer[i].clear()             
            
    def reset(self, seed):
        """
        Reset the agent's state, including feature extractor and rollout buffer.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        
        # Option execution bookkeeping
        self.running_option_index = [None for _ in range(self.num_envs)]       # index into options_lst (or None)
        self.option_start_obs = [None for _ in range(self.num_envs)]         # encoded state where option began
        self.option_cumulative_reward = [0.0 for _ in range(self.num_envs)]    # discounted return accumulator R_{t:t+k}
        self.option_multiplier = [1.0 for _ in range(self.num_envs)]           # current gamma^t during option
        self.option_num_steps = [0 for _ in range(self.num_envs)]
        self.option_log_prob = [None for _ in range(self.num_envs)]
    
    
    def log(self):
        logs = []
        for i in range(self.num_envs):
            curr_option_idx = self.running_option_index[i]
            if curr_option_idx is None:
                logs.append({
                    "OptionUsageLog": False,
                    "NumOptions": len(self.options_lst),
                    "OptionIndex": None,
                    "OptionClass": None,
                })
            else:
                logs.append({
                    "OptionUsageLog": True,
                    "NumOptions": len(self.options_lst),
                    "OptionIndex": curr_option_idx,
                    "OptionClass": self.options_lst[curr_option_idx].__class__,
                })
        return logs
    
    def save(self, file_path=None):
        checkpoint = super().save(file_path=None)  # parent saves feature_extractor, policy, hp, etc.
        
        # Save options list payload
        checkpoint['options_lst'] = save_options_list(self.options_lst, file_path=None)
        checkpoint['atomic_action_space'] = self.atomic_action_space
        
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_agent.t")
        return checkpoint

    @classmethod
    def load(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        
        instance = super().load(file_path, checkpoint)
        instance.options_lst = load_options_list(file_path=None, checkpoint=checkpoint['options_lst'])
        instance.atomic_action_space = checkpoint['atomic_action_space']
        
        expected = instance.atomic_action_space.n + len(instance.options_lst)
        assert instance.policy.action_dim == expected
                
        return instance
    
