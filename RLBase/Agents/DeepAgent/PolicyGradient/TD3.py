import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from gymnasium.spaces import Box

from ...Base import BaseAgent, BasePolicy
from ....Buffers import ReplayBuffer
from ...Utils import get_single_observation, stack_observations, grad_norm
from ....registry import register_agent, register_policy
from ....Networks.NetworkFactory import NetworkGen, prepare_network_config
from ....FeatureExtractors import get_batch_features


@register_policy
class TD3Policy(BasePolicy):
    """
    TD3:
      - Actor: a = pi(s)
      - Critics: Q1(s,a), Q2(s,a)
      - Target nets: actor_targ, critic1_targ, critic2_targ
      - Tricks:
          * clipped double Q
          * target policy smoothing
          * delayed policy updates
          * Polyak (EMA) target updates
    """
    def __init__(self, action_space, features_dict, hyper_params, device="cpu"):
        super().__init__(action_space, hyper_params, device=device)
        self.features_dict = features_dict

        # Action bounds
        self.action_low = torch.as_tensor(self.action_space.low, device=self.device, dtype=torch.float32)
        self.action_high = torch.as_tensor(self.action_space.high, device=self.device, dtype=torch.float32)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        # Actor
        actor_description = prepare_network_config(
            self.hp.actor_network,
            input_dims=self.features_dict,
            output_dim=self.action_dim,
        )
        self.actor = NetworkGen(layer_descriptions=actor_description).to(self.device)

        # Critic inputs = state features + action
        critic_input_dims = self.features_dict
        critic_input_dims["a"] = self.action_dim

        critic_description = prepare_network_config(
            self.hp.critic_network,
            input_dims=critic_input_dims,
            output_dim=1,
        )
        self.critic1 = NetworkGen(layer_descriptions=critic_description).to(self.device)
        self.critic2 = NetworkGen(layer_descriptions=critic_description).to(self.device)

        # Targets
        self.actor_target = NetworkGen(layer_descriptions=actor_description).to(self.device)
        self.critic1_target = NetworkGen(layer_descriptions=critic_description).to(self.device)
        self.critic2_target = NetworkGen(layer_descriptions=critic_description).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Opt
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.hp.actor_step_size, eps=self.hp.actor_eps)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.hp.critic_step_size,
            eps=self.hp.critic_eps,
        )

        self.update_counter = 0

    def reset(self, seed):
        super().reset(seed)

        actor_description = prepare_network_config(
            self.hp.actor_network,
            input_dims=self.features_dict,
            output_dim=self.action_dim,
        )
        critic_input_dims = dict(self.features_dict)
        critic_input_dims["a"] = self.action_dim
        critic_description = prepare_network_config(
            self.hp.critic_network,
            input_dims=critic_input_dims,
            output_dim=1,
        )

        self.actor = NetworkGen(layer_descriptions=actor_description).to(self.device)
        self.critic1 = NetworkGen(layer_descriptions=critic_description).to(self.device)
        self.critic2 = NetworkGen(layer_descriptions=critic_description).to(self.device)

        self.actor_target = NetworkGen(layer_descriptions=actor_description).to(self.device)
        self.critic1_target = NetworkGen(layer_descriptions=critic_description).to(self.device)
        self.critic2_target = NetworkGen(layer_descriptions=critic_description).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.hp.actor_step_size, eps=self.hp.actor_eps)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.hp.critic_step_size,
            eps=self.hp.critic_eps,
        )
        self.update_counter = 0

    def _add_exploration_noise(self, actions: torch.Tensor) -> torch.Tensor:
        if self.hp.exploration_noise <= 0.0:
            return actions

        noise = torch.randn_like(actions) * self.hp.exploration_noise
        a = actions + noise
        return torch.clamp(a, self.action_low, self.action_high)
    
    def _squash_and_scale(self, raw_action: torch.Tensor) -> torch.Tensor:
        # raw_action -> tanh -> scale to env bounds
        if self.hp.need_squash:
            a = torch.tanh(raw_action)
        else:
            a = raw_action
        return a * self.action_scale + self.action_bias

    def _critic_forward(self, critic_net, state_dict, action_t):
        inp = dict(state_dict)
        inp["a"] = action_t
        return critic_net(**inp).squeeze(-1)  # [B]

    def _polyak_update(self, network, target_network):
        with torch.no_grad():
            for p, p_targ in zip(network.parameters(), target_network.parameters()):
                p_targ.data.mul_(1.0 - self.hp.target_network_update_tau).add_(self.hp.target_network_update_tau * p.data)

    def select_action(self, state, greedy=False):
        with torch.no_grad():
            raw_action = self.actor(**state)
            greedy_action = self._squash_and_scale(raw_action)
            noisy_action = self._add_exploration_noise(greedy_action)
            if greedy:
                return greedy_action.cpu().numpy()
            return noisy_action.cpu().numpy()

    def update(self, states, actions, rewards, next_states, dones, call_back=None):
        """
        actions_t: torch [B, A]
        rewards_t: torch [B]
        dones_t: torch [B] (0/1 float)
        """
        self.update_counter += 1

        actions_t = torch.tensor(np.array(actions), device=self.device, dtype=torch.float32)      # [B, A]
        rewards_t = torch.tensor(np.array(rewards), device=self.device, dtype=torch.float32)     # [B]
        dones_t = torch.tensor(np.array(dones), device=self.device, dtype=torch.float32)      # [B]
        # ---------------------
        # Critic update
        # ---------------------
        with torch.no_grad():
            raw_next_action = self.actor_target(**next_states)
            next_action = self._squash_and_scale(raw_next_action)

            # Target policy smoothing
            if self.hp.target_policy_noise > 0.0:
                eps = torch.randn_like(next_action) * self.hp.target_policy_noise
                eps = torch.clamp(eps, -self.hp.target_policy_noise_clip, self.hp.target_policy_noise_clip)
                next_action = next_action + eps
                next_action = torch.clamp(next_action, self.action_low, self.action_high)

            q1_t = self._critic_forward(self.critic1_target, next_states, next_action)
            q2_t = self._critic_forward(self.critic2_target, next_states, next_action)
            q_t = torch.min(q1_t, q2_t)

            y = rewards_t + self.hp.gamma * (1.0 - dones_t) * q_t  # [B]

        q1 = self._critic_forward(self.critic1, states, actions_t)
        q2 = self._critic_forward(self.critic2, states, actions_t)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            self.hp.max_grad_norm
        )
        self.critic_optimizer.step()

        # ---------------------
        # Delayed actor + targets
        # ---------------------
        actor_loss = None
        if (self.update_counter % self.hp.policy_delay) == 0:
            raw_action = self.actor(**states)
            pi_action = self._squash_and_scale(raw_action)
            q_pi = self._critic_forward(self.critic1, states, pi_action)
            actor_loss = (-q_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()        
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.hp.max_grad_norm)
            self.actor_optimizer.step()

            # Polyak targets
            self._polyak_update(self.actor, self.actor_target)
            self._polyak_update(self.critic1, self.critic1_target)
            self._polyak_update(self.critic2, self.critic2_target)

        # ---------------------
        # Logging (optional)
        # ---------------------
        if call_back is not None:
            payload = {
                "critic_loss": float(critic_loss.item()),
                "q1_mean": float(q1.mean().item()),
                "q2_mean": float(q2.mean().item()),
                "target_q_mean": float(y.mean().item()),
                "lr_actor": float(self.actor_optimizer.param_groups[0]["lr"]),
                "lr_critic": float(self.critic_optimizer.param_groups[0]["lr"]),
                "critic_grad_norm": float(grad_norm(list(self.critic1.parameters()) + list(self.critic2.parameters()))),
            }
            if actor_loss is not None:
                payload["actor_loss"] = float(actor_loss.item())
                payload["actor_grad_norm"] = float(grad_norm(self.actor.parameters()))
            call_back(payload)

    def save(self, file_path=None):
        checkpoint = super().save(file_path=None)
        # checkpoint["actor_state_dict"] = self.actor.state_dict()
        # checkpoint["critic1_state_dict"] = self.critic1.state_dict()
        # checkpoint["critic2_state_dict"] = self.critic2.state_dict()
        # checkpoint["actor_targ_state_dict"] = self.actor_targ.state_dict()
        # checkpoint["critic1_targ_state_dict"] = self.critic1_targ.state_dict()
        # checkpoint["critic2_targ_state_dict"] = self.critic2_targ.state_dict()
        # checkpoint["actor_optimizer_state_dict"] = self.actor_optimizer.state_dict()
        # checkpoint["critic_optimizer_state_dict"] = self.critic_optimizer.state_dict()
        # checkpoint["features_dict"] = self.features_dict
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_policy.t")
        return checkpoint

    @classmethod
    def load(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)

        instance = cls(
            checkpoint["action_space"],
            checkpoint["features_dict"],
            checkpoint["hyper_params"],
            device=checkpoint["device"],
        )
        instance.set_rng_state(checkpoint["rng_state"])

        instance.actor.load_state_dict(checkpoint["actor_state_dict"])
        instance.critic1.load_state_dict(checkpoint["critic1_state_dict"])
        instance.critic2.load_state_dict(checkpoint["critic2_state_dict"])
        instance.actor_targ.load_state_dict(checkpoint["actor_targ_state_dict"])
        instance.critic1_targ.load_state_dict(checkpoint["critic1_targ_state_dict"])
        instance.critic2_targ.load_state_dict(checkpoint["critic2_targ_state_dict"])
        instance.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        instance.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        instance.features_dict = checkpoint["features_dict"]
        return instance


@register_agent
class TD3Agent(BaseAgent):
    """
    Off-policy TD3 agent using ReplayBuffer (your implementation).
    """
    name = "TD3"
    SUPPORTED_ACTION_SPACES = (Box,)

    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device="cpu"):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device=device)

        self.policy = TD3Policy(
            action_space,
            self.feature_extractor.features_dict,
            hyper_params,
            device=device,
        )

        self.replay_buffer = ReplayBuffer(capacity=self.hp.replay_buffer_size)
        # (optional) seed buffer rng if you want deterministic sampling
        # self.replay.set_seed(self.hp.seed)

        self.last_observation = None
        self.last_action = None



    def act(self, observation, greedy=False):
        """
        observation: batch
        returns: batch actions
        """
        # random actions (classic TD3 trick)
        if (not greedy) and (len(self.replay_buffer) <= int(self.hp.initial_random_steps)):
            action = self.np_random.uniform(self.policy.action_low, self.policy.action_high, size=(self.num_envs, self.policy.action_dim)).astype(np.float32)
        else:
            state = self.feature_extractor(observation)
            action = self.policy.select_action(state, greedy=greedy)  

        self.last_observation = observation
        self.last_action = action
        return action

    def update(self, observation, reward, terminated, truncated, call_back=None):
        """
        Stores (s, a, r, s', done) into replay and does gradient updates.
        """

        # 1) Store transitions in replay (per env)
        for i in range(self.num_envs):
            transition = (
                get_single_observation(self.last_observation, i),
                self.last_action[i],
                float(reward[i]),
                get_single_observation(observation, i),
                bool(terminated[i]),
            )
            self.replay_buffer.add(transition)

        # 2) Start learning after enough experience
        if len(self.replay_buffer) >= self.hp.warmup_buffer_size: 
            # 3) Potentially do multiple updates per env step
            for _ in range(self.hp.num_updates):
                batch = self.replay_buffer.sample(int(self.hp.batch_size))
                (batch_observations, batch_actions, batch_rewards, batch_next_observations, batch_dones) = zip(*batch)

                batch_observations = stack_observations(batch_observations)
                batch_next_observations = stack_observations(batch_next_observations)

                batch_states = self.feature_extractor(batch_observations)
                batch_next_states = self.feature_extractor(batch_next_observations)

                self.policy.update(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, call_back=call_back)

    def reset(self, seed):
        super().reset(seed)
        self.replay_buffer.clear()  
        self.replay_buffer.set_seed(seed) 
        
        self.last_observation = None
        self.last_action = None
        
    