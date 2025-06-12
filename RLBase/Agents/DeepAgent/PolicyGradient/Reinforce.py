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
class ReinforcePolicy(BasePolicy):
    """
    A pure REINFORCE policy (no baseline). 
    Stores log probabilities and rewards over an episode and performs a single update at the end.
    """
    def __init__(self, action_space, features_dim, hyper_params, device="cpu"):
        """
        Args:
            action_space (gym.spaces.Discrete): Discrete action space.
            features_dim (int): Number of features in the flattened observation.
            hyper_params: Hyper-parameters; must include gamma and step_size.
        """
        super().__init__(action_space, hyper_params, device=device)
        self.features_dim = features_dim

    def reset(self, seed):
        """
        Initialize the policy network and optimizer.
        
        Args:
            seed (int): Random seed.
        """
        super().reset(seed)
        # Build the policy network based on the provided configuration.
        actor_description = prepare_network_config(
            self.hp.actor_network, 
            input_dim=self.features_dim, 
            output_dim=self.action_dim
        )
        self.actor = NetworkGen(layer_descriptions=actor_description).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.hp.step_size)

    def select_action(self, state):
        """
        Sample an action from the policy distribution.
        
        Args:
            state (np.array): Flat feature vector; shape [features_dim] or [1, features_dim].
        
        Returns:
            tuple: (action (int), log_prob (torch.Tensor))
        """
        state_t = state.to(dtype=torch.float32, device=self.device) if torch.is_tensor(state) else torch.tensor(state, dtype=torch.float32, device=self.device)
        logits = self.actor(state_t)
        dist = Categorical(logits=logits)
        action_t = dist.sample()
        log_prob_t = dist.log_prob(action_t)
        return action_t.item(), log_prob_t

    def update(self, log_probs, rewards, call_back=None):
        """
        Compute discounted returns and update the policy at episode end.
        
        Args:
            log_probs (list): List of log probabilities (torch.Tensor) per time step.
            rewards (list): List of rewards (float) per time step.
            call_back (function, optional): Callback to report loss metrics.
        """
        log_probs_t = torch.stack(log_probs).to(dtype=torch.float32, device=self.device)
        # Compute discounted returns from the episode.
        returns = calculate_n_step_returns(rewards, 0.0, self.hp.gamma)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(1)
        # Policy loss: negative sum of (log_prob * return)
        policy_loss = - (log_probs_t * returns_t).sum()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        if call_back is not None:
            call_back({"actor_loss": policy_loss.item()})

    def save(self, file_path=None):
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),

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
        instance.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        
        return instance

    def load_from_checkpoint(self, checkpoint):
        """
        Load network states, optimizer state, and hyper-parameters.
        
        Args:
            checkpoint (dictionary)
        """
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

        self.action_space = checkpoint.get('action_space')
        self.features_dim = checkpoint.get('features_dim')
        self.hp = checkpoint.get('hyper_params')

@register_agent
class ReinforceAgent(BaseAgent):
    """
    Minimal REINFORCE agent for discrete actions without a baseline.
    """
    name = "Reinforce"
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device="cpu"):
        """
        Args:
            action_space (gym.spaces.Discrete): Discrete action space.
            observation_space: Environment's observation space.
            hyper_params: Hyper-parameters; must include gamma and step_size.
            num_envs (int): Number of parallel environments.
            feature_extractor_class: Class for feature extraction.
        """
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device=device)

        self.policy = ReinforcePolicy(
            action_space,
            self.feature_extractor.features_dim,
            hyper_params,
            device=device
        )
        self.rollout_buffer = BasicBuffer(np.inf)
        
    def act(self, observation):
        """
        Select an action based on the current observation.
        
        Args:
            observation: Raw observation.
        
        Returns:
            int: Selected action.
        """
        state = self.feature_extractor(observation)
        action, log_prob = self.policy.select_action(state)
        self.last_state = state
        self.last_action = action
        self.last_log_prob = log_prob
        return action

    def update(self, observation, reward, terminated, truncated, call_back=None):
        """
        Store reward and, if the episode is finished, update the policy.
        
        Args:
            observation: New observation.
            reward (float): Reward received.
            terminated (bool): True if episode terminated.
            truncated (bool): True if episode truncated.
            call_back (function, optional): Callback to report loss.
        """
        transition = (self.last_log_prob, reward)
        self.rollout_buffer.add_single_item(transition)

        if terminated or truncated:
            rollout = self.rollout_buffer.get_all()
            log_probs, rewards = zip(*rollout)
            self.policy.update(log_probs, rewards, call_back=call_back)    
            self.rollout_buffer.reset()

    def reset(self, seed):
        """
        Reset the agent for a new run.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.rollout_buffer.reset()