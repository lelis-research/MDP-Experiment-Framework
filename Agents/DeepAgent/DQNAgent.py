import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from Agents.Utils.BaseAgent import BaseAgent, BasePolicy
from Agents.Utils.FeatureExtractor import FLattenFeature
from Agents.Utils.ReplayBuffer import BasicReplayBuffer
#########################################
# Define the Q-Network used by DQNAgent #
#########################################
class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        A simple fully-connected network.
        """
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

###############################################
# Define the DQNPolicy using an epsilon greedy #
###############################################
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
        self.action_dim = action_space.n
         
    def select_action(self, observation_features):
        """
        Select an action based on epsilon-greedy policy.
        """
        # With probability epsilon choose a random action.
        if random.random()  < self.hp.epsilon:
            return self.action_space.sample()
        else:
            # Convert features to a PyTorch tensor.
            # Assume features are a flat NumPy array.
            state = torch.FloatTensor(observation_features).unsqueeze(0)  # add batch dim
            with torch.no_grad():
                q_values = self.network(state)
            return int(torch.argmax(q_values, dim=1).item())
    
    def reset(self, seed):
        super().reset(seed)

        self.update_counter = 0

        self.network = DQNNetwork(self.features_dim, self.action_dim)
        self.target_network = DQNNetwork(self.features_dim, self.action_dim)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.hp.step_size)
        self.loss_fn = nn.MSELoss()
        
    def update(self, states, actions, rewards, next_states, dones):
         # Compute Q-values for current states using the online network.
        q_values = self.network(states).gather(1, actions)

        # Compute target Q-values using the target network.
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target = rewards + self.hp.gamma * (1 - dones) * next_q_values
        
        loss = self.loss_fn(q_values, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1

        # Periodically update the target network.
        if self.update_counter % self.hp.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

#####################################
# Define the DQNAgent (Deep RL Agent) #
#####################################
class DQNAgent(BaseAgent):
    """
    Deep Q-Network (DQN) agent.
    """
    def __init__(self, action_space, observation_space, hyper_params):
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
        super().__init__(action_space, hyper_params)
        
        self.feature_extractor = FLattenFeature(observation_space)
        self.action_dim = action_space.n
        
        # Experience Replay Buffer
        self.replay_buffer = BasicReplayBuffer(hyper_params.replay_buffer_cap)  
        
        # Create DQNPolicy using the online network.
        self.policy = DQNPolicy(action_space, 
                                self.feature_extractor.features_dim, 
                                hyper_params)
        
        # For storing the last state and action.
        self.last_state = None
        self.last_action = None
        
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
    
    def update(self, observation, reward, terminated, truncated):
        """
        Called at every time step.
        
        Stores the transition and, if enough samples are available,
        performs a learning step by sampling a minibatch from the replay buffer.
        """
        
        # Process the new observation.
        state = self.feature_extractor(observation)
        
        # Store transition in replay buffer.
        transition = (self.last_state, self.last_action, reward, state, terminated)
        self.replay_buffer.add_single_item(transition)
        
        # Perform a learning step if there are enough samples.
        if self.replay_buffer.size >= self.hp.batch_size:
            batch = self.replay_buffer.get_random_batch(self.hp.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(np.array(actions)).unsqueeze(1)
            rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)
            
            self.policy.update(states, actions, reward, next_states, dones)

        
        # If the episode ended, clear the last state and action.
        if terminated or truncated:
            self.last_state = None
            self.last_action = None
    
  
    def reset(self, seed):
        """
        Reset the agent's learning state.
        """
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.replay_buffer.reset()

        self.last_state = None
        self.last_action = None

        