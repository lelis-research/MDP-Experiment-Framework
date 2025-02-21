import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from Agents.Utils import (
    BaseAgent,
    BasePolicy,
    BasicBuffer,
    calculate_n_step_returns,
    NetworkGen,
    prepare_network_config,
)
from Agents.Utils.NetworkGenerator import NetworkGen, prepare_network_config

class A2CPolicyV2(BasePolicy):
    """
    Advantage Actor-Critic (A2C) policy using n-step rollouts.
    """
    def __init__(self, action_space, features_dim, hyper_params):
        """
        Args:
            action_space (gym.spaces.Discrete): Action space.
            features_dim (int): Dimensionality of the extracted features.
            hyper_params: Contains gamma, actor_step_size, critic_step_size, n_steps, actor_network, critic_network.
        """
        super().__init__(action_space, hyper_params)
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

        self.actor = NetworkGen(layer_descriptions=actor_description)
        self.critic = NetworkGen(layer_descriptions=critic_description)

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
        state_t = torch.FloatTensor(state)
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
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.FloatTensor(np.array(actions))
        next_states_t = torch.FloatTensor(np.array(next_states))
        
        # Compute log-probabilities for actions using actor.
        logits = self.actor(states_t)
        dist = Categorical(logits=logits)
        log_probs_t = dist.log_prob(actions_t).unsqueeze(1)  # shape [n_steps, 1]
        
        with torch.no_grad():
            # Bootstrap from the last state if not terminal.
            bootstrap_value = self.critic(next_states_t)[-1] if not dones[-1] else 0.0
        
        # Calculate n-step returns.
        returns = calculate_n_step_returns(rewards, bootstrap_value, self.hp.gamma)
        returns_t = torch.FloatTensor(returns).unsqueeze(1)
        
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

    def save(self, file_path):
        """
        Save the actor and critic networks and optimizers.
        
        Args:
            file_path (str): File path to save the checkpoint.
        """
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'hyper_params': self.hp,
            'features_dim': self.features_dim,
            'action_dim': self.action_dim,
        }
        torch.save(checkpoint, file_path)

    def load(self, file_path):
        """
        Load checkpoint for actor and critic networks and optimizers.
        
        Args:
            file_path (str): File path of the checkpoint.
        """
        checkpoint = torch.load(file_path, map_location='cpu')
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.hp = checkpoint.get('hyper_params', self.hp)
        self.features_dim = checkpoint.get('features_dim', self.features_dim)
        self.action_dim = checkpoint.get('action_dim', self.action_dim)

class A2CAgentV2(BaseAgent):
    """
    Advantage Actor-Critic agent using n-step rollouts.
    """
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class):
        """
        Args:
            action_space (gym.spaces.Discrete): Environment's action space.
            observation_space: Environment's observation space.
            hyper_params: Contains gamma, actor_step_size, critic_step_size, n_steps, rollout_steps, etc.
            num_envs (int): Number of parallel environments.
            feature_extractor_class: Class to extract features from observations.
        """
        super().__init__(action_space, observation_space, hyper_params, num_envs)
        self.feature_extractor = feature_extractor_class(observation_space)
        self.policy = A2CPolicyV2(action_space, self.feature_extractor.features_dim, hyper_params)
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
        transition = (self.last_state[0], self.last_action, reward, state[0], terminated)
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