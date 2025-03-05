import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from Agents.Utils import (
    BaseAgent,
    BasePolicy,
    BasicBuffer,
    calculate_n_step_returns,
    NetworkGen,
    prepare_network_config,
    register_agent,
    register_policy,
)

@register_policy
class A2CPolicyV1(BasePolicy):
    """
    Advantage Actor-Critic policy for discrete actions using n-step rollouts.
    """
    def __init__(self, action_space, features_dim, hyper_params):
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
        super().__init__(action_space, hyper_params)
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
        self.actor = NetworkGen(layer_descriptions=actor_description)
        self.critic = NetworkGen(layer_descriptions=critic_description)

        # Create separate optimizers for actor and critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.hp.actor_step_size)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.hp.critic_step_size)

    def select_action(self, state):
        """
        Sample an action from the actor's policy.
        
        Args:
            state (np.array): Flat feature vector of shape [features_dim] or [1, features_dim].
            
        Returns:
            tuple: (action (int), log_prob (torch.Tensor))
        """
        state_t = torch.FloatTensor(state)
        logits = self.actor(state_t)
        dist = Categorical(logits=logits)
        action_t = dist.sample()
        log_prob_t = dist.log_prob(action_t)
        return action_t.item(), log_prob_t

    def update(self, states, log_probs, rewards, next_states, dones, call_back=None):
        """
        Perform an n-step A2C update.
        
        Args:
            states (list or np.array): Batch of states; shape [n_steps, features_dim].
            log_probs (list of torch.Tensor): Log-probabilities for actions taken; length n_steps.
            rewards (list of float): Rewards collected over the rollout; length n_steps.
            next_states (list or np.array): Batch of next states; shape [n_steps, features_dim].
            dones (list of bool): Done flags for each transition; length n_steps.
            call_back (function, optional): Function to report losses.
        """
        states_t = torch.FloatTensor(np.array(states))
        log_probs_t = torch.stack(log_probs)  # shape [n_steps]
        next_states_t = torch.FloatTensor(np.array(next_states))

        # Bootstrap from the last state if not terminated.
        with torch.no_grad():
            bootstrap_value = self.critic(next_states_t)[-1] if not dones[-1] else 0.0
        
        # Compute n-step returns
        returns = calculate_n_step_returns(rewards, bootstrap_value, self.hp.gamma)
        returns_t = torch.FloatTensor(returns).unsqueeze(1)  # shape [n_steps, 1]
        
        # Critic prediction: value estimates V(s)
        predicted_values_t = self.critic(states_t)  
        critic_loss = F.mse_loss(predicted_values_t, returns_t)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Compute advantage: A = return - V(s)
        with torch.no_grad():
            predicted_values_t = self.critic(states_t)
        advantages_t = returns_t - predicted_values_t
        
        # Actor loss: maximize expected return => minimize -log_prob * advantage
        actor_loss = - (log_probs_t * advantages_t).sum()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
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
            
            'action_dim': self.action_dim,            
            'policy_class': self.__class__.__name__,
        }
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_policy.t")
        return checkpoint

    @classmethod
    def load_from_file(cls, file_path, seed=0):
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
class A2CAgentV1(BaseAgent):
    """
    Advantage Actor-Critic agent using n-step rollouts in a single environment.
    """
    name = "A2C_v1"
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class):
        """
        Args:
            action_space (gym.spaces.Discrete): Environment's action space.
            observation_space: Environment's observation space.
            hyper_params: Hyper-parameters container (see A2CPolicyV1 for required keys).
            num_envs (int): Number of parallel environments.
            feature_extractor_class: Class to extract features from observations.
        """
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class)
        
        self.policy = A2CPolicyV1(
            action_space, 
            self.feature_extractor.features_dim, 
            hyper_params
        )
        
        # Rollout buffer stores transitions: (state, log_prob, reward, next_state, done)
        self.rollout_buffer = BasicBuffer(np.inf)

    def act(self, observation):
        """
        Select an action and store log_prob for later update.
        
        Args:
            observation (np.array or similar): Raw observation.
            
        Returns:
            int: Chosen action.
        """
        state = self.feature_extractor(observation)
        action, log_prob = self.policy.select_action(state)

        self.last_state = state
        self.last_action = action
        self.last_log_prob = log_prob
        return action

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
        # Store transition; using [0] assumes feature_extractor returns batch-like output.
        transition = (self.last_state[0], self.last_log_prob, reward, state[0], terminated)
        self.rollout_buffer.add_single_item(transition)

        # If rollout length reached or episode ended, perform update.
        if self.rollout_buffer.size >= self.hp.rollout_steps or terminated or truncated:
            rollout = self.rollout_buffer.get_all()
            states, log_probs, rewards, next_states, dones = zip(*rollout)
            
            self.policy.update(states, log_probs, rewards, next_states, dones, call_back=call_back)
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