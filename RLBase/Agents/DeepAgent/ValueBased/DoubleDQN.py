import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from ...Utils import (
    BaseAgent,
    BasePolicy,
    BasicBuffer,
    NetworkGen,
    prepare_network_config,
)
from ....registry import register_agent, register_policy


@register_policy
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
    def __init__(self, action_space, features_dim, hyper_params, device="cpu"):
        """
        Args:
            action_space (gym.spaces.Discrete): Environment's action space.
            features_dim (int): Flattened observation dimension.
            hyper_params: Hyper-parameters container.
        """
        super().__init__(action_space, hyper_params, device=device)
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
            state_t = state.to(dtype=torch.float32, device=self.device) if torch.is_tensor(state) else torch.tensor(state, dtype=torch.float32, device=self.device)
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
        self.online_network = NetworkGen(layer_descriptions=network_description).to(self.device)
        self.target_network = NetworkGen(layer_descriptions=network_description).to(self.device)
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
        states_t = torch.cat(states).to(dtype=torch.float32, device=self.device) if torch.is_tensor(states[0]) else torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        next_states_t = torch.cat(next_states).to(dtype=torch.float32, device=self.device) if torch.is_tensor(next_states[0]) else torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(np.array(actions), dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device).unsqueeze(1)
        
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
    
    def save(self, file_path=None):
        """
        Save network states, optimizer state, and hyper-parameters.
        
        Args:
            file_path (str): Path to save the checkpoint.
        """
        
        checkpoint = {
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),

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
        instance.online_network.load_state_dict(checkpoint['online_network_state_dict'])
        instance.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return instance

    def load_from_checkpoint(self, checkpoint):
        """
        Load network states, optimizer state, and hyper-parameters.
        
        Args:
            checkpoint (dictionary)
        """
        self.online_network.load_state_dict(checkpoint['online_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.action_space = checkpoint.get('action_space')
        self.features_dim = checkpoint.get('features_dim')
        self.hp = checkpoint.get('hyper_params')

@register_agent
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
    name = "DoubleDQN"
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device="cpu"):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device=device)
        
        # Initialize replay buffer with capacity hyper_params.replay_buffer_cap.
        self.replay_buffer = BasicBuffer(hyper_params.replay_buffer_cap)
        
        # Initialize the Double DQN policy.
        self.policy = DoubleDQNPolicy(
            action_space, 
            self.feature_extractor.features_dim, 
            hyper_params,
            device=device
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
            (self.last_state, self.last_action, reward, state, terminated)
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