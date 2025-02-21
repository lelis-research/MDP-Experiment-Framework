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
    Epsilon-greedy policy for DQN.
    
    Uses a Q-network to estimate action values.
    """
    def __init__(self, action_space, features_dim, hyper_params):
        """
        Args:
            action_space: The environment's action space (assumed to be Discrete).
            features_dim: Integer showing the number of features (From here we start with fully connected)
            hyper-parameters:
                - epsilon
                - gamma
                - step_size  
                - replay_buffer_cap
                - batch_size
                - target_update_freq
        """
        super().__init__(action_space, hyper_params)
        
        self.features_dim = features_dim
         
    def select_action(self, state):
        """
        Select an action based on epsilon-greedy policy.
        """
        # With probability epsilon choose a random action.
        if random.random()  < self.hp.epsilon:
            return self.action_space.sample()
        else:
            # Convert features to a PyTorch tensor.
            # Assume features are a flat NumPy array.
            state_t = torch.FloatTensor(state)
            with torch.no_grad():
                q_values = self.network(state_t)
            return int(torch.argmax(q_values, dim=1).item())
    
    def reset(self, seed):
        super().reset(seed)

        self.update_counter = 0
        network_description = prepare_network_config(self.hp.value_network,
                                                    input_dim=self.features_dim,
                                                    output_dim=self.action_dim)
        
        self.network = NetworkGen(layer_descriptions=network_description)
        self.target_network = NetworkGen(layer_descriptions=network_description)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.hp.step_size)
        self.loss_fn = nn.MSELoss()
        
    def update(self, states, actions, rewards, next_states, dones, call_back=None):
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(np.array(actions)).unsqueeze(1)
        rewards_t = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(np.array(dones)).unsqueeze(1)
        
        # Compute Q-values for current states using the online network.
        qvalues_t = self.network(states_t).gather(1, actions_t)

        # Compute target Q-values using the target network.
        with torch.no_grad():
            bootstrap_value_t = self.target_network(next_states_t).max(1)[0].unsqueeze(1)
            target_t = rewards_t + self.hp.gamma * (1 - dones_t) * bootstrap_value_t
        
        loss = self.loss_fn(qvalues_t, target_t)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        # Periodically update the target network.
        if self.update_counter % self.hp.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        if call_back is not None:
            call_back({"value_loss": loss.item()})
            
    def save(self, file_path):
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyper_params': self.hp,  # Ensure that self.hp is pickle-serializable
            'features_dim': self.features_dim,
            'action_dim': self.action_dim,
        }
        torch.save(checkpoint, file_path)


    def load(self, file_path):
        checkpoint = torch.load(file_path, weights_only=False)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.hp = checkpoint.get('hyper_params', self.hp)
        self.features_dim = checkpoint.get('features_dim', self.features_dim)
        self.action_dim = checkpoint.get('action_dim', self.action_dim)


class DQNAgent(BaseAgent):
    """
    Deep Q-Network (DQN) agent.
    """
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class):
        """
        Args:
            action_space: The environment's action space (assumed to be Discrete).
            observation_space: The environment's action space
            hyper-parameters:
                - epsilon
                - gamma
                - step_size  
                - replay_buffer_cap
                - batch_size
                - target_update_freq
        """
        super().__init__(action_space, observation_space, hyper_params, num_envs)
        
        self.feature_extractor = feature_extractor_class(observation_space)
        
        # Experience Replay Buffer
        self.replay_buffer = BasicBuffer(hyper_params.replay_buffer_cap)  

        # Create DQNPolicy using the online network.
        self.policy = DQNPolicy(action_space, 
                                self.feature_extractor.features_dim, 
                                hyper_params)
        
    def act(self, observation):
        """
        Choose an action based on the current observation.
        """
        # Convert observation to a flat vector.
        state = self.feature_extractor(observation)
        action = self.policy.select_action(state)
        
        # Store current state and action (for use in update).
        self.last_state = state
        self.last_action = action
        return action
    
    def update(self, observation, reward, terminated, truncated, call_back):
        """
        Called at every time step.
        
        Stores the transition and, if enough samples are available,
        performs a learning step by sampling a minibatch from the replay buffer.
        """
        
        # Process the new observation.
        state = self.feature_extractor(observation)

        # Store transition in replay buffer.
        transition = (self.last_state[0], self.last_action, reward, state[0], terminated)
        self.replay_buffer.add_single_item(transition)
        
        # Perform a learning step if there are enough samples.
        if self.replay_buffer.size >= self.hp.batch_size:
            batch = self.replay_buffer.get_random_batch(self.hp.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            self.policy.update(states, actions, rewards, next_states, dones, call_back=call_back)
          
    def reset(self, seed):        
        """
        Reset the agent's learning state.
        """
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.replay_buffer.reset()

        