import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Independent, TransformedDistribution
from gymnasium.spaces import Discrete, Box



from ....Networks.NetworkFactory import NetworkGen, prepare_network_config
from ...Base import BaseAgent, BasePolicy
from ....Buffers import BaseBuffer
from ...Utils import calculate_gae, get_single_observation, stack_observations, grad_norm, explained_variance
from ....registry import register_agent, register_policy



        
@register_policy
class A2CPolicy(BasePolicy):
    """
    Advantage Actor-Critic policy for continuous actions using n-step rollouts.
    """
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
            self.actor_optimizer = optim.Adam(list(self.actor.parameters()) + [self.actor_logstd],lr=self.hp.actor_step_size, eps=self.hp.actor_eps)
            
            # self.action_low = torch.as_tensor(self.action_space.low, device=self.device, dtype=torch.float32)
            # self.action_high = torch.as_tensor(self.action_space.high, device=self.device, dtype=torch.float32)
        elif isinstance(self.action_space, Discrete):
            self.actor_logstd = None
            self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=self.hp.actor_step_size, eps=self.hp.actor_eps)
        else:
            raise NotImplementedError(f"A2CPolicy does not support action space of type {type(self.action_space)}")
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=self.hp.critic_step_size, eps=self.hp.critic_eps)
        self.update_counter = 0
        
            
    def reset(self, seed):
        super().reset(seed)
        # Actor outputs concatenated [mean, log_std] of size 2 * action_dim
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
            self.actor_optimizer = optim.Adam(list(self.actor.parameters()) + [self.actor_logstd],lr=self.hp.actor_step_size, eps=self.hp.actor_eps)
            
            # self.action_low = torch.as_tensor(self.action_space.low, device=self.device, dtype=torch.float32)
            # self.action_high = torch.as_tensor(self.action_space.high, device=self.device, dtype=torch.float32)
        elif isinstance(self.action_space, Discrete):
            self.actor_logstd = None
            self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=self.hp.actor_step_size, eps=self.hp.actor_eps)
        else:
            raise NotImplementedError(f"A2CPolicy does not support action space of type {type(self.action_space)}")
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=self.hp.critic_step_size, eps=self.hp.critic_eps)
        self.update_counter = 0
    
    
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
                f"A2CPolicy:_log_prob_and_entropy does not support action space of type {type(self.action_space)}"
            )

        return action_t, log_prob, entropy, greedy_action
    
    def select_action(self, state, greedy=False):
        """
        Returns a numpy action of shape [action_dim].
        """
        
        with torch.no_grad():
            action, _, _, greedy_action = self._log_prob_and_entropy(state)
            if greedy:
                return greedy_action.cpu().numpy()
            else:
                return action.cpu().numpy()

    def update(self, states, actions, next_states, rewards, terminated, truncated, call_back=None):
        # T: rollout depth
        
        self.update_counter += 1

        # Update the step size
         # LR annealing (optional)
        if self.hp.anneal_step_size_flag:
            frac = 1.0 - (self.update_counter - 1.0) / float(self.hp.total_updates)  # linear from initial->0
            for param_groups in self.actor_optimizer.param_groups:
                param_groups["lr"] = frac * self.hp.actor_step_size
            for param_groups in self.critic_optimizer.param_groups:
                param_groups["lr"] = frac * self.hp.critic_step_size  

        if isinstance(self.action_space, Discrete):
            actions_t = torch.tensor(np.array(actions), dtype=torch.int64, device=self.device)  # (T,)
        elif isinstance(self.action_space, Box):
            actions_t = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)  # (T, A)
           
        # Critic values
        values = self.critic(**states).squeeze(-1)               # (T, )
        with torch.no_grad():
            next_values = self.critic(**next_states).squeeze(-1) # (T, )
            
        
        # returns + advantages
        returns, advantages = calculate_gae(
            rewards,
            values,
            next_values,
            terminated,
            truncated,
            gamma=self.hp.gamma,
            lamda=self.hp.lamda,
        )
        
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)  # (T, )
        returns_t    = torch.tensor(returns,    dtype=torch.float32, device=self.device) # (T, )
        
        if self.hp.norm_adv_flag:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # Actor log-probs
        _, log_probs_t, entropy_t, _ = self._log_prob_and_entropy(states, actions_t)  # (T, ), (T, )
        

        critic_loss = F.mse_loss(values, returns_t)
        actor_loss = - (log_probs_t * advantages_t).mean() 
        loss = actor_loss + self.hp.critic_coef * critic_loss - self.hp.entropy_coef * entropy_t.mean()        
        
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        
        if call_back is not None:
            actor_grad_norm = grad_norm(self.actor.parameters())
            critic_grad_norm = grad_norm(self.critic.parameters())

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        if call_back is not None:
            payload = {
                "critic_loss": float(critic_loss.item()),
                "actor_loss": float(actor_loss.item()),
                "entropy_loss": float(entropy_t.mean().item()),
                "loss": float(loss.item()),
                        
                "explained_variance": explained_variance(values, returns_t),
                
                "advantage_mean": float(advantages_t.mean().item()),
                "advantage_std": float(advantages_t.std().item()),
                
                "values_mean": float(values.mean().item()),
                "values_std": float(values.std().item()),
                "returns_mean": float(returns_t.mean().item()),
                "returns_std": float(returns_t.std().item()),
                
                "lr_actor": float(self.actor_optimizer.param_groups[0]["lr"]),
                "lr_critic": float(self.critic_optimizer.param_groups[0]["lr"]), 
                
                "actor_grad_norms": float(actor_grad_norm),
                "critic_grad_norms": float(critic_grad_norm),
            }
            if isinstance(self.action_space, Box) and self.actor_logstd is not None:
                payload["actor_logstd_mean"] = float(self.actor_logstd.mean().item())
                payload["actor_logstd_std"] = float(self.actor_logstd.std().item())
            call_back(payload)
    
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
        instance.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        # Restore actor_logstd correctly (only for continuous / Box)
        if checkpoint['actor_logstd'] is not None:
            # instance.actor_logstd is already an nn.Parameter from __init__
            actor_logstd_loaded = checkpoint['actor_logstd'].to(instance.device)
            with torch.no_grad():
                instance.actor_logstd.data.copy_(actor_logstd_loaded)
        else:
            instance.actor_logstd = None
            
        instance.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        instance.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        instance.features_dict = checkpoint['features_dict']
        return instance

        
@register_agent
class A2CAgent(BaseAgent):
    """
    Advantage Actor-Critic agent using n-step rollouts in a single environment.
    """
    name = "A2C"
    SUPPORTED_ACTION_SPACES = (Discrete, Box)
    
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device="cpu"):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device=device)
        self.policy = A2CPolicy(
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
        action = self.policy.select_action(state, greedy=greedy)
        self.last_observation = observation
        self.last_action = action
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
        for i in range(self.num_envs):
            transition =  (
                get_single_observation(self.last_observation, i), 
                self.last_action[i], 
                reward[i], 
                get_single_observation(observation, i), 
                terminated[i], 
                truncated[i],
            )
            self.rollout_buffer[i].add(transition)
        
        
        if not all(buf.is_full() for buf in self.rollout_buffer):
            return

        all_observations        = []
        all_next_observations   = []
        all_actions             = []
        all_rewards             = []
        all_terminated          = []
        all_truncated           = []
        
        for i in range(self.num_envs):
            rollout = self.rollout_buffer[i].all()
            (
                rollout_observations,
                rollout_actions,
                rollout_rewards,
                rollout_next_observations,
                rollout_terminated,
                rollout_truncated,
            ) = zip(*rollout)
            
            all_observations.extend(rollout_observations)
            all_next_observations.extend(rollout_next_observations)
            all_actions.extend(rollout_actions)
            all_rewards.extend(rollout_rewards)
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

        # 5) Single A2C update using data from *all* envs combined
        self.policy.update(
            states,
            all_actions,
            next_states,
            all_rewards,
            all_terminated,
            all_truncated,
            call_back=call_back,
        )
    
    def update_per_env(self, observation, reward, terminated, truncated, call_back=None):
        """
        all arguments are batches
        """
        for i in range(self.num_envs): 
            transition =  (
                get_single_observation(self.last_observation, i), 
                self.last_action[i], 
                reward[i], 
                get_single_observation(observation, i), 
                terminated[i], 
                truncated[i]
            )
            self.rollout_buffer[i].add(transition)
            
            if self.rollout_buffer[i].is_full():
                rollout = self.rollout_buffer[i].all() 
                (
                    rollout_observations,
                    rollout_actions,
                    rollout_rewards,
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
                                   rollout_next_states, 
                                   rollout_rewards, 
                                   rollout_terminated, 
                                   rollout_truncated, 
                                   call_back=call_back)
                
                self.rollout_buffer[i].clear()  
                
            
            
   
    def reset(self, seed):
        """
        Reset the agent's state, including rollout buffer.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        
        self.last_observation = None
        self.last_action = None
        
        self.rollout_buffer = [BaseBuffer(self.hp.rollout_steps) for _ in range(self.num_envs)]
