import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from Agents.Utils import (
    BaseAgent,
    BasePolicy,
    BasicBuffer,
    NetworkGen,
    prepare_network_config,
)

class DQNPolicy(BasePolicy):
    """
    Epsilon-greedy DQN policy using a Q-network to estimate action values.
    
    Hyper-parameters (in hp) must include:
        - epsilon (float)
        - gamma (float)
        - step_size (float)
        - replay_buffer_cap (int)
        - batch_size (int)
        - target_update_freq (int)
        - value_network (list): Network layer configuration.
    
    Args:
        action_space (gym.spaces.Discrete): The environment's action space.
        features_dim (int): Dimension of the flattened observation.
        hyper_params: Container for hyper-parameters.
    """
    def __init__(self, action_space, features_dim, hyper_params):
        super().__init__(action_space, hyper_params)
        self.features_dim = features_dim
         
    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state (np.array): Flat feature vector.
        
        Returns:
            int: Selected action.
        """
        if random.random() < self.hp.epsilon:
            return self.action_space.sample()
        else:
            state_t = torch.FloatTensor(state)
            with torch.no_grad():
                q_values = self.network(state_t)
            return int(torch.argmax(q_values, dim=1).item())
    
    def reset(self, seed):
        """
        Reset the policy: initialize networks, optimizer, and counter.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        self.update_counter = 0
        # Prepare network configuration using the value network description.
        network_description = prepare_network_config(
            self.hp.value_network,
            input_dim=self.features_dim,
            output_dim=self.action_dim
        )
        self.network = NetworkGen(layer_descriptions=network_description)
        self.target_network = NetworkGen(layer_descriptions=network_description)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.hp.step_size)
        self.loss_fn = nn.MSELoss()
        
    def update(self, states, actions, rewards, next_states, dones, call_back=None):
        """
        Update the Q-network using a batch of transitions.
        
        Args:
            states (list/np.array): Batch of current states; shape [batch, features_dim].
            actions (list/np.array): Batch of actions; shape [batch].
            rewards (list/np.array): Batch of rewards; shape [batch].
            next_states (list/np.array): Batch of next states; shape [batch, features_dim].
            dones (list/np.array): Batch of done flags; shape [batch].
            call_back (function, optional): Callback to track loss.
        """
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(np.array(actions)).unsqueeze(1)
        rewards_t = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(np.array(dones)).unsqueeze(1)
        
        # Compute Q-values for actions taken.
        qvalues_t = self.network(states_t).gather(1, actions_t)

        with torch.no_grad():
            # Use target network for bootstrap.
            bootstrap_value_t = self.target_network(next_states_t).max(1)[0].unsqueeze(1)
            target_t = rewards_t + self.hp.gamma * (1 - dones_t) * bootstrap_value_t
        
        loss = self.loss_fn(qvalues_t, target_t)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.hp.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        if call_back is not None:
            call_back({"value_loss": loss.item()})
            
    def save(self, file_path):
        """
        Save the network states, optimizer state, and hyper-parameters.
        
        Args:
            file_path (str): File path for saving.
        """
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyper_params': self.hp,
            'features_dim': self.features_dim,
            'action_dim': self.action_dim,
        }
        torch.save(checkpoint, file_path)

    def load(self, file_path):
        """
        Load network states, optimizer state, and hyper-parameters.
        
        Args:
            file_path (str): File path to load the checkpoint from.
        """
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.hp = checkpoint.get('hyper_params', self.hp)
        self.features_dim = checkpoint.get('features_dim', self.features_dim)
        self.action_dim = checkpoint.get('action_dim', self.action_dim)


class DQNAgent(BaseAgent):
    """
    Deep Q-Network (DQN) agent that uses experience replay and target networks.
    
    Args:
        action_space (gym.spaces.Discrete): The environment's action space.
        observation_space: The environment's observation space.
        hyper_params: Hyper-parameters container (see DQNPolicy).
        num_envs (int): Number of parallel environments.
        feature_extractor_class: Class to extract features from observations.
    """
    name = "DQN"
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class):
        super().__init__(action_space, observation_space, hyper_params, num_envs)
        self.feature_extractor = feature_extractor_class(observation_space)
        
        # Experience Replay Buffer
        self.replay_buffer = BasicBuffer(hyper_params.replay_buffer_cap)  

        # Create DQNPolicy using the feature extractor's feature dimension.
        self.policy = DQNPolicy(
            action_space, 
            self.feature_extractor.features_dim, 
            hyper_params
        )
        
    def act(self, observation):
        """
        Select an action based on the current observation.
        
        Args:
            observation (np.array or similar): Raw observation from the environment.
        
        Returns:
            int: Selected action.
        """
        state = self.feature_extractor(observation)
        action = self.policy.select_action(state)
        self.last_state = state
        self.last_action = action
        return action
    
    def update(self, observation, reward, terminated, truncated, call_back):
        """
        Store the transition and, if enough samples are available, perform a learning step.
        
        Args:
            observation (np.array or similar): New observation after action.
            reward (float): Reward received.
            terminated (bool): True if the episode has terminated.
            truncated (bool): True if the episode was truncated.
            call_back (function): Callback function to track training progress.
        """
        state = self.feature_extractor(observation)
        transition = (self.last_state[0], self.last_action, reward, state[0], terminated)
        self.replay_buffer.add_single_item(transition)
        
        if self.replay_buffer.size >= self.hp.batch_size:
            batch = self.replay_buffer.get_random_batch(self.hp.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            self.policy.update(states, actions, rewards, next_states, dones, call_back=call_back)
          
    def reset(self, seed):
        """
        Reset the agent's learning state, including feature extractor and replay buffer.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.replay_buffer.reset()