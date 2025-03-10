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
class A2CPolicyV2(BasePolicy):
    """
    Advantage Actor-Critic (A2C) policy using n-step rollouts.
    """
    def __init__(self, action_space, features_dim, hyper_params, device="cpu"):
        """
        Args:
            action_space (gym.spaces.Discrete): Action space.
            features_dim (int): Dimensionality of the extracted features.
            hyper_params: Contains gamma, actor_step_size, critic_step_size, n_steps, actor_network, critic_network.
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

        # Build actor and critic network configurations.
        actor_description = prepare_network_config(self.hp.actor_network, 
                                                   input_dim=self.features_dim, 
                                                   output_dim=self.action_dim)
        critic_description = prepare_network_config(self.hp.critic_network,
                                                    input_dim=self.features_dim,
                                                    output_dim=1)

        self.actor = NetworkGen(layer_descriptions=actor_description).to(self.device)
        self.critic = NetworkGen(layer_descriptions=critic_description).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.hp.actor_step_size)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.hp.critic_step_size)

    def select_action(self, state):
        """
        Sample an action from the actor network.
        
        Args:
            state (np.array): Input state (flat feature vector).
        
        Returns:
            int: Selected action.
        """
        state_t = state.to(dtype=torch.float32, device=self.device) if torch.is_tensor(state) else torch.tensor(state, dtype=torch.float32, device=self.device)
        logits = self.actor(state_t)
        dist = Categorical(logits=logits)
        action_t = dist.sample()
        return action_t.item()

    def update(self, states, actions, rewards, next_states, dones, call_back=None):
        """
        Perform n-step A2C update.
        
        Args:
            states (list or np.array): States from rollout; shape [n_steps, features_dim].
            actions (list): Actions taken (not used for update here).
            rewards (list of float): Collected rewards.
            next_states (list or np.array): Next states; shape [n_steps, features_dim].
            dones (list of bool): Done flags for each step.
            call_back (function, optional): To report loss values.
        """
        # Convert states and next_states to tensors.
        states_t = torch.cat(states).to(dtype=torch.float32, device=self.device) if torch.is_tensor(states[0]) else torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(np.array(actions), dtype=torch.int64, device=self.device)
        next_states_t = torch.cat(next_states).to(dtype=torch.float32, device=self.device) if torch.is_tensor(next_states[0]) else torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)

        # Compute log-probabilities for actions using actor.
        logits = self.actor(states_t)
        dist = Categorical(logits=logits)
        log_probs_t = dist.log_prob(actions_t) # shape [n_steps, 1]

        with torch.no_grad():
            # Bootstrap from the last state if not terminal.
            bootstrap_value = self.critic(next_states_t)[-1] if not dones[-1] else 0.0
        
        # Calculate n-step returns.
        returns = calculate_n_step_returns(rewards, bootstrap_value, self.hp.gamma)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        # Critic update: minimize MSE between V(s) and returns.
        predicted_values_t = self.critic(states_t)
        critic_loss = F.mse_loss(predicted_values_t, returns_t)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute advantage for actor update.
        with torch.no_grad():
            predicted_values_t = self.critic(states_t)
        advantages_t = returns_t - predicted_values_t

        # Actor update: maximize log_prob * advantage.
        actor_loss = - (log_probs_t * advantages_t).sum()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

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
class A2CAgentV2(BaseAgent):
    """
    Advantage Actor-Critic agent using n-step rollouts.
    """
    name = "A2C_v2"
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device="cpu"):
        """
        Args:
            action_space (gym.spaces.Discrete): Environment's action space.
            observation_space: Environment's observation space.
            hyper_params: Contains gamma, actor_step_size, critic_step_size, n_steps, rollout_steps, etc.
            num_envs (int): Number of parallel environments.
            feature_extractor_class: Class to extract features from observations.
        """
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device=device)

        self.policy = A2CPolicyV2(action_space, self.feature_extractor.features_dim, hyper_params, device=device)
        # Rollout buffer stores transitions as (state, action, reward, next_state, done)
        self.rollout_buffer = BasicBuffer(np.inf)

    def act(self, observation):
        """
        Select an action from the actor network.
        
        Args:
            observation (np.array or similar): Raw observation.
        
        Returns:
            int: Selected action.
        """
        state = self.feature_extractor(observation)
        action = self.policy.select_action(state)
        self.last_state = state
        self.last_action = action
        return action

    def update(self, observation, reward, terminated, truncated, call_back=None):
        """
        Store transition and perform update if rollout is complete.
        
        Args:
            observation (np.array or similar): New observation.
            reward (float): Reward received.
            terminated (bool): True if episode terminated.
            truncated (bool): True if episode truncated.
            call_back (function, optional): Callback to report losses.
        """
        state = self.feature_extractor(observation)
        transition = (self.last_state, self.last_action, reward, state, terminated)
        self.rollout_buffer.add_single_item(transition)

        if self.rollout_buffer.size >= self.hp.rollout_steps or terminated or truncated:
            rollout = self.rollout_buffer.get_all()
            states, actions, rewards, next_states, dones = zip(*rollout)
            self.policy.update(states, actions, rewards, next_states, dones, call_back=call_back)
            self.rollout_buffer.reset()

    def reset(self, seed):
        """
        Reset the agent's state.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.rollout_buffer.reset()