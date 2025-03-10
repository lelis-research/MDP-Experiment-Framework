import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from ...Utils import (
    BaseAgent,
    BasePolicy,
    BasicBuffer,
    calculate_n_step_returns,
    NetworkGen,
    prepare_network_config,
)
from ....registry import register_agent, register_policy


@register_policy
class PPOPolicy(BasePolicy):
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

    def select_action(self, state):
        """
        Sample an action from the actor network.
        
        Args:
            state (np.array): Flat feature vector; shape [features_dim] or [1, features_dim].
            
        Returns:
            tuple: (action (int), log_prob (torch.Tensor), state_value (torch.Tensor))
        """
        state_t = state.to(dtype=torch.float32, device=self.device) if torch.is_tensor(state) else torch.tensor(state, dtype=torch.float32, device=self.device)

        logits = self.actor(state_t)
        dist = Categorical(logits=logits)
        action_t = dist.sample()
        log_prob_t = dist.log_prob(action_t)
        with torch.no_grad():
            value_t = self.critic(state_t)
        return action_t.item(), log_prob_t.detach(), value_t

    def update(self, states, actions, old_log_probs, states_values, rewards, next_states, dones, call_back=None):
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
        states_t = torch.cat(states).to(dtype=torch.float32, device=self.device) if torch.is_tensor(states[0]) else torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(np.array(actions), dtype=torch.int64, device=self.device)
        log_probs_t = torch.stack(old_log_probs).to(dtype=torch.float32, device=self.device)
        states_values_t = torch.stack(states_values).to(dtype=torch.float32, device=self.device)
        next_states_t = torch.cat(next_states).to(dtype=torch.float32, device=self.device) if torch.is_tensor(next_states[0]) else torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
    
        with torch.no_grad():
            bootstrap_value = self.critic(next_states_t)[-1] if not dones[-1] else 0.0
        returns = calculate_n_step_returns(rewards, bootstrap_value, self.hp.gamma)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(1)  # shape [rollout_steps, 1]
        
        advantages_t = returns_t - states_values_t
        
        datasize = len(states)
        indices = np.arange(datasize)
        for epoch in range(self.hp.num_epochs):
            np.random.shuffle(indices)
            for start in range(0, datasize, self.hp.mini_batch_size):
                batch_indices = indices[start:start + self.hp.mini_batch_size]
                batch_states = states_t[batch_indices]
                batch_actions = actions_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_returns = returns_t[batch_indices]
                batch_log_prob = log_probs_t[batch_indices]

                logits = self.actor(batch_states)
                dist = Categorical(logits=logits)
                new_log_probs_t = dist.log_prob(batch_actions).unsqueeze(1)
                entropy = dist.entropy().unsqueeze(1)
                
                ratios = torch.exp(new_log_probs_t - batch_log_prob)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.hp.clip_range, 1 + self.hp.clip_range) * batch_advantages
                actor_loss = torch.mean(-torch.min(surr1, surr2) - self.hp.entropy_coef * entropy)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                new_state_values_t = self.critic(batch_states)
                critic_loss = F.mse_loss(new_state_values_t, batch_returns)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
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

        self.action_dim = checkpoint.get('action_dim')

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

        self.policy = PPOPolicy(action_space, self.feature_extractor.features_dim, hyper_params, device=device)
        self.rollout_buffer = BasicBuffer(np.inf)

    def act(self, observation):
        """
        Sample an action from the policy.
        
        Args:
            observation (np.array or similar): Raw observation.
        
        Returns:
            int: Selected action.
        """
        state = self.feature_extractor(observation)
        action, log_prob, state_value = self.policy.select_action(state)
        self.last_state = state
        self.last_action = action
        self.last_log_prob = log_prob
        self.last_state_value = state_value
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
        transition = (self.last_state, self.last_action, self.last_log_prob, 
                      self.last_state_value[0], reward, state, terminated)
        self.rollout_buffer.add_single_item(transition)
        
        if self.rollout_buffer.size >= self.hp.rollout_steps or terminated or truncated:
            rollout = self.rollout_buffer.get_all()
            states, actions, log_probs, states_values, rewards, next_states, dones = zip(*rollout)
            self.policy.update(states, actions, log_probs, states_values, rewards, next_states, dones, call_back=call_back)
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