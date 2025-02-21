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
    A Policy for REINFORCE with a learned baseline (value network).
    
    We'll store states, actions, log_probs, and rewards for each episode.
    Then, at episode end, we:
      1) compute the returns G_t,
      2) update ValueNetwork (the baseline),
      3) use advantage = (G_t - V(s_t)) in the policy gradient.
    """
    def __init__(self, action_space, features_dim, hyper_params):
        """
        Args:
            action_space: The environment's action space (assumed to be Discrete).
            features_dim: Integer showing the number of features (From here we start with fully connected)
            hyper-parameters:
                - gamma
                - actor_step_size  
                - critic_step_size
        """
        super().__init__(action_space, hyper_params)
        
        self.features_dim = features_dim

    def reset(self, seed):
        super().reset(seed)
        actor_description = prepare_network_config(self.hp.actor_network, 
                                                   input_dim= self.features_dim, 
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
        Sample an action from the policy distribution (Categorical).
        We'll return the action and store the log_prob for the update.
        """
        state_t = torch.FloatTensor(state)
        logits = self.actor(state_t) 
        
        dist = Categorical(logits=logits)
        action_t = dist.sample()
        
        # Store the log_prob
        log_prob_t = dist.log_prob(action_t)
        
        return action_t.item(), log_prob_t

    def update(self, states, log_probs, rewards, call_back=None):
        """
        End of episode => compute returns, train the baseline, do the policy gradient update with advantage.
        """
        states_t = torch.FloatTensor(np.array(states))
        log_probs_t = torch.stack(log_probs)
        
        # Compute discounted returns from the end of the episode to the beginning
        returns = calculate_n_step_returns(rewards, 0.0, self.hp.gamma)
        returns_t = torch.FloatTensor(returns).unsqueeze(1) #Correct the dims
        
        # (Optional) Normalize returns to help training stability
        # Made the performance much worse !
        # returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Critic update: we train the value network V(s) to approximate the returns
        predicted_values_t = self.critic(states_t)   

        # Backprop Critic
        critic_loss = F.mse_loss(predicted_values_t, returns_t)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute advantage: A_t = G_t - V(s_t)
        with torch.no_grad():
            predicted_values_t = self.critic(states_t)      
        advantages_t = returns_t - predicted_values_t                
        
        # Policy update:  - sum( log_prob_t * advantage_t )
        actor_loss = - (log_probs_t * advantages_t).sum()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if call_back is not None:
            call_back({"critic_loss":critic_loss.item(),
                       "actor_loss":actor_loss.item()})
    
    def save(self, file_path):
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'hyper_params': self.hp,  # Ensure that self.hp is serializable
            'features_dim': self.features_dim,
            'action_dim': self.action_dim,
        }
        torch.save(checkpoint, file_path)


    def load(self, file_path):
        checkpoint = torch.load(file_path, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.hp = checkpoint.get('hyper_params', self.hp)
        self.features_dim = checkpoint.get('features_dim', self.features_dim)
        self.action_dim = checkpoint.get('action_dim', self.action_dim)


class ReinforceWithBaselineAgent(BaseAgent):
    """
    A REINFORCE-like agent that uses a learned baseline (value network)
    to reduce variance => 'Actor + Baseline'.
    """

    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class):
        super().__init__(action_space, observation_space, hyper_params, num_envs)

        self.feature_extractor = feature_extractor_class(observation_space)
        self.policy = ReinforceWithBaselinePolicy(
            action_space,
            self.feature_extractor.features_dim,
            hyper_params
        )
        self.rollout_buffer = BasicBuffer(np.inf)

    def act(self, observation):
        # Convert observation to features, select an action stochastically
        state = self.feature_extractor(observation)
        action, log_prob = self.policy.select_action(state)

        self.last_state = state
        self.last_log_prob = log_prob
        return action
    
    def update(self, observation, reward, terminated, truncated, call_back=None):
        """
        Called every step by the experiment loop:
        - If the episode ends, do a policy gradient update
        """
        transition = (self.last_state[0], self.last_log_prob, reward)
        self.rollout_buffer.add_single_item(transition)

        if terminated or truncated:
            rollout = self.rollout_buffer.get_all()
            states, log_probs, rewards, = zip(*rollout)
            self.policy.update(states, log_probs, rewards, call_back=call_back)    
            self.rollout_buffer.reset()

    def reset(self, seed):
        """
        Reset the agent at the start of a new run (multiple seeds).
        """
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.rollout_buffer.reset()
