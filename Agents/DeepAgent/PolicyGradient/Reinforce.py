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

class ReinforcePolicy(BasePolicy):
    """
    A pure REINFORCE policy (no baseline). We store all log_probs and rewards
    for each episode, then do a single update at episode end.
    """
    def __init__(self, action_space, features_dim, hyper_params):
        """
        Args:
            action_space: The environment's action space (assumed to be Discrete).
            features_dim: Integer showing the number of features (From here we start with fully connected)
            hyper-parameters:
                - gamma
                - step_size  
        """
        super().__init__(action_space, hyper_params)
        
        self.features_dim = features_dim

    def reset(self, seed):
        """
        Called when we reset the entire agent. We create the policy network
        and its optimizer, and clear any episode storage.
        """
        super().reset(seed)
        # Build the policy network
        actor_description = prepare_network_config(self.hp.actor_network, 
                                                   input_dim= self.features_dim, 
                                                   output_dim=self.action_dim)
        self.actor = NetworkGen(layer_descriptions=actor_description)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.hp.step_size)

    def select_action(self, state):
        """
        Sample an action from the policy distribution (Categorical).
        Store the log_prob for the update.
        """
        state_t = torch.FloatTensor(state)
        logits = self.actor(state_t) 
        
        dist = Categorical(logits=logits)
        action_t = dist.sample()
        
        # Store the log_prob
        log_prob_t = dist.log_prob(action_t)
        
        return action_t.item(), log_prob_t

    def update(self, log_probs, rewards, call_back=None):
        """
        Once an episode ends, compute discounted returns and do the policy update.
        """
        log_probs_t = torch.stack(log_probs)

        #Compute discounted returns from the end of the episode to the beginning
        returns = calculate_n_step_returns(rewards, 0.0, self.hp.gamma)
        returns_t = torch.FloatTensor(returns).unsqueeze(1) #Correct the dims

        # (Optional) Normalize returns to help training stability
        # Made the performance much worse !
        # returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Compute the policy loss:  -(sum of log_probs * returns)
        policy_loss = - (log_probs_t * returns_t).sum()

        # Backprop
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        if call_back is not None:
            call_back({"actor_loss":policy_loss.item()})
    
    def save(self, file_path):
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'hyper_params': self.hp,  # Ensure that self.hp is pickle-serializable
            'features_dim': self.features_dim,
            'action_dim': self.action_dim,
        }
        torch.save(checkpoint, file_path)


    def load(self, file_path):
        checkpoint = torch.load(file_path, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.hp = checkpoint.get('hyper_params', self.hp)
        self.features_dim = checkpoint.get('features_dim', self.features_dim)
        self.action_dim = checkpoint.get('action_dim', self.action_dim)

class ReinforceAgent(BaseAgent):
    """
    Minimal REINFORCE agent for discrete actions, no baseline.
    """
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class):
        super().__init__(action_space, observation_space, hyper_params, num_envs)

        self.feature_extractor = feature_extractor_class(observation_space)

        self.policy = ReinforcePolicy(
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
        self.last_action = action
        self.last_log_prob = log_prob
        return action

    def update(self, observation, reward, terminated, truncated, call_back=None):
        """
        Called each time-step by the experiment loop. We store the reward,
        and if the episode ends, we do a policy update.
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
        Reset the agent at the start of a new run (multiple seeds).
        """
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.rollout_buffer.reset()