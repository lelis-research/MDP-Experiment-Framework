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

class ReinforceWithBaselinePolicy(BasePolicy):
    """
    REINFORCE with a learned baseline (value network).
    Stores states, log probabilities, and rewards for an episode, then updates
    both the actor (policy) and critic (baseline) at episode end.
    """
    def __init__(self, action_space, features_dim, hyper_params):
        """
        Args:
            action_space (gym.spaces.Discrete): Discrete action space.
            features_dim (int): Number of features from the feature extractor.
            hyper_params: Hyper-parameters; must include gamma, actor_step_size, and critic_step_size.
        """
        super().__init__(action_space, hyper_params)
        self.features_dim = features_dim

    def reset(self, seed):
        """
        Initialize the actor and critic networks and their optimizers.
        
        Args:
            seed (int): Random seed.
        """
        super().reset(seed)
        # Build actor network configuration
        actor_description = prepare_network_config(
            self.hp.actor_network, 
            input_dim=self.features_dim, 
            output_dim=self.action_dim
        )
        # Build critic network configuration (outputs scalar value)
        critic_description = prepare_network_config(
            self.hp.critic_network,
            input_dim=self.features_dim,
            output_dim=1
        )
        self.actor = NetworkGen(layer_descriptions=actor_description)
        self.critic = NetworkGen(layer_descriptions=critic_description)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.hp.actor_step_size)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.hp.critic_step_size)

    def select_action(self, state):
        """
        Sample an action from the policy distribution.
        
        Args:
            state (np.array): Flat feature vector.
        
        Returns:
            tuple: (action (int), log_prob (torch.Tensor))
        """
        state_t = torch.FloatTensor(state)
        logits = self.actor(state_t)
        dist = Categorical(logits=logits)
        action_t = dist.sample()
        log_prob_t = dist.log_prob(action_t)
        return action_t.item(), log_prob_t

    def update(self, states, log_probs, rewards, call_back=None):
        """
        At episode end, compute returns, update the critic (baseline) with MSE loss,
        and update the actor using the advantage (return - baseline).
        
        Args:
            states (list or np.array): List of states from the episode.
            log_probs (list): List of log probabilities for the actions taken.
            rewards (list): List of rewards received.
            call_back (function, optional): Callback to report losses.
        """
        states_t = torch.FloatTensor(np.array(states))
        log_probs_t = torch.stack(log_probs)
        
        # Compute discounted returns (G_t) from the episode
        returns = calculate_n_step_returns(rewards, 0.0, self.hp.gamma)
        returns_t = torch.FloatTensor(returns).unsqueeze(1)
        
        # Update the critic (value network) with MSE loss
        predicted_values_t = self.critic(states_t)
        critic_loss = F.mse_loss(predicted_values_t, returns_t)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Compute advantage: A_t = G_t - V(s_t)
        with torch.no_grad():
            predicted_values_t = self.critic(states_t)
        advantages_t = returns_t - predicted_values_t
                
        # Actor loss: negative log-likelihood weighted by advantage
        actor_loss = - (log_probs_t * advantages_t).sum()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if call_back is not None:
            call_back({
                "critic_loss": critic_loss.item(),
                "actor_loss": actor_loss.item()
            })
    
    def save(self, file_path):
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
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.hp = checkpoint.get('hyper_params', self.hp)
        self.features_dim = checkpoint.get('features_dim', self.features_dim)
        self.action_dim = checkpoint.get('action_dim', self.action_dim)


class ReinforceWithBaselineAgent(BaseAgent):
    """
    REINFORCE agent with a learned baseline (value network) to reduce variance.
    """
    name = "ReinforceWithBaseline"
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class):
        """
        Args:
            action_space (gym.spaces.Discrete): Discrete action space.
            observation_space: Environment's observation space.
            hyper_params: Must include gamma, actor_step_size, and critic_step_size.
            num_envs (int): Number of parallel environments.
            feature_extractor_class: Class to extract features from observations.
        """
        super().__init__(action_space, observation_space, hyper_params, num_envs)
        self.feature_extractor = feature_extractor_class(observation_space)
        self.policy = ReinforceWithBaselinePolicy(
            action_space,
            self.feature_extractor.features_dim,
            hyper_params
        )
        self.rollout_buffer = BasicBuffer(np.inf)

    def act(self, observation):
        """
        Convert the observation to features and select an action stochastically.
        
        Args:
            observation: Raw observation.
        
        Returns:
            int: Selected action.
        """
        state = self.feature_extractor(observation)
        action, log_prob = self.policy.select_action(state)
        self.last_state = state
        self.last_log_prob = log_prob
        return action

    def update(self, observation, reward, terminated, truncated, call_back=None):
        """
        Store the reward and, if the episode is finished, update the policy.
        
        Args:
            observation: New observation.
            reward (float): Reward received.
            terminated (bool): True if episode terminated.
            truncated (bool): True if episode was truncated.
            call_back (function, optional): Callback to report losses.
        """
        # Store log_prob and reward for the last time step.
        transition = (self.last_state[0], self.last_log_prob, reward)
        self.rollout_buffer.add_single_item(transition)

        # If episode ends, perform policy update.
        if terminated or truncated:
            rollout = self.rollout_buffer.get_all()
            states, log_probs, rewards = zip(*rollout)
            self.policy.update(states, log_probs, rewards, call_back=call_back)    
            self.rollout_buffer.reset()

    def reset(self, seed):
        """
        Reset the agent's state for a new run.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.rollout_buffer.reset()