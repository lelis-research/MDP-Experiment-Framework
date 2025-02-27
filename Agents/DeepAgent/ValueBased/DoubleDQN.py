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

class DoubleDQNPolicy(BasePolicy):
    """
    Epsilon-greedy policy for Double DQN with online and target networks.
    
    Hyper-parameters in hp should include:
        - epsilon (float)
        - gamma (float)
        - step_size (float, learning rate)
        - replay_buffer_cap (int)
        - batch_size (int)
        - target_update_freq (int)
        - value_network (list): Layer configuration for the network.
    """
    def __init__(self, action_space, features_dim, hyper_params):
        """
        Args:
            action_space (gym.spaces.Discrete): Environment's action space.
            features_dim (int): Flattened observation dimension.
            hyper_params: Hyper-parameters container.
        """
        super().__init__(action_space, hyper_params)
        self.features_dim = features_dim

    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        
        Args:
            state (np.array, shape=[...]): Input state (already processed by feature extractor).
        
        Returns:
            int: Selected action.
        """
        if random.random() < self.hp.epsilon:
            return self.action_space.sample()
        else:
            # Convert state to tensor; shape expected: [batch, features_dim]
            state_t = torch.FloatTensor(state)
            with torch.no_grad():
                q_values = self.online_network(state_t)
            # Returns action from the first (and only) batch element.
            return int(torch.argmax(q_values, dim=1).item())
        
    def reset(self, seed):
        """
        Reset the policy: initialize networks, optimizer, and counter.
        
        Args:
            seed (int): Random seed.
        """
        super().reset(seed)
        self.update_counter = 0

        # Prepare network configuration and initialize online & target networks.
        network_description = prepare_network_config(
            self.hp.value_network,
            input_dim=self.features_dim,
            output_dim=self.action_dim
        )
        self.online_network = NetworkGen(layer_descriptions=network_description)
        self.target_network = NetworkGen(layer_descriptions=network_description)
        self.target_network.load_state_dict(self.online_network.state_dict())

        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.hp.step_size)
        self.loss_fn = nn.MSELoss()
        
    def update(self, states, actions, rewards, next_states, dones, call_back=None):
        """
        Perform a Double DQN update.
        
        Args:
            states (list or np.array): Batch of states, shape [batch, features_dim].
            actions (list or np.array): Batch of actions, shape [batch].
            rewards (list or np.array): Batch of rewards, shape [batch].
            next_states (list or np.array): Batch of next states, shape [batch, features_dim].
            dones (list or np.array): Batch of done flags (bool or 0/1), shape [batch].
            call_back (function, optional): Callback to track training progress.
        """
        # Convert inputs to tensors.
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(np.array(actions)).unsqueeze(1)
        rewards_t = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(np.array(dones)).unsqueeze(1)
        
        # Get current Q-values for the taken actions.
        qvalues_t = self.online_network(states_t).gather(1, actions_t)

        with torch.no_grad():
            # 1) Select best actions from online network for next_states.
            next_qvalues_t = self.online_network(next_states_t)
            best_actions = torch.argmax(next_qvalues_t, dim=1, keepdim=True)
            # 2) Evaluate these actions using the target network.
            bootstrap_value_t = self.target_network(next_states_t).gather(1, best_actions)
            # Compute target using Bellman equation.
            target_t = rewards_t + self.hp.gamma * (1 - dones_t) * bootstrap_value_t
        
        loss = self.loss_fn(qvalues_t, target_t)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        # Update target network periodically.
        if self.update_counter % self.hp.target_update_freq == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
        
        if call_back is not None:
            call_back({"value_loss": loss.item()})
    
    def save(self, file_path):
        """
        Save network states, optimizer state, and hyper-parameters.
        
        Args:
            file_path (str): Path to save the checkpoint.
        """
        checkpoint = {
            'online_network_state_dict': self.online_network.state_dict(),
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
            file_path (str): Path to the checkpoint file.
        """
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        self.online_network.load_state_dict(checkpoint['online_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.hp = checkpoint.get('hyper_params', self.hp)
        self.features_dim = checkpoint.get('features_dim', self.features_dim)
        self.action_dim = checkpoint.get('action_dim', self.action_dim)


class DoubleDQNAgent(BaseAgent):
    """
    Double DQN agent with experience replay and target network.
    
    Args:
        action_space (gym.spaces.Discrete): Environment's action space.
        observation_space: Environment's observation space.
        hyper_params: Hyper-parameters container (see DoubleDQNPolicy).
        num_envs (int): Number of parallel environments.
        feature_extractor_class: Class to extract features from observations.
    """
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class):
        super().__init__(action_space, observation_space, hyper_params, num_envs)
        self.feature_extractor = feature_extractor_class(observation_space)
        
        # Initialize replay buffer with capacity hyper_params.replay_buffer_cap.
        self.replay_buffer = BasicBuffer(hyper_params.replay_buffer_cap)
        
        # Initialize the Double DQN policy.
        self.policy = DoubleDQNPolicy(
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
    
    def update(self, observation, reward, terminated, truncated, call_back=None):
        """
        Store transition in replay buffer and update policy if batch is ready.
        
        Args:
            observation (np.array or similar): New observation after action.
            reward (float): Reward received.
            terminated (bool): True if episode terminated.
            truncated (bool): True if episode truncated.
            call_back (function, optional): Callback to track training progress.
        """
        state = self.feature_extractor(observation)
        # Store transition as (state, action, reward, next_state, done)
        self.replay_buffer.add_single_item(
            (self.last_state[0], self.last_action, reward, state[0], terminated)
        )
        
        # If enough transitions, sample a random batch and update.
        if self.replay_buffer.size >= self.hp.batch_size:
            batch = self.replay_buffer.get_random_batch(self.hp.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            self.policy.update(states, actions, rewards, next_states, dones, call_back=call_back)

    def reset(self, seed):
        """
        Reset the agent's state, feature extractor, and replay buffer.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.replay_buffer.reset()