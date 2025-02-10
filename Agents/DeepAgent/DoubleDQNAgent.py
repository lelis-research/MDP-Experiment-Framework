import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from Agents.Utils.BaseAgent import BaseAgent, BasePolicy
from Agents.Utils.FeatureExtractor import FLattenFeature
from Agents.Utils.ReplayBuffer import BasicReplayBuffer


#########################################
# Define the Q-Network used by DoubleDQN #
#########################################
class DoubleDQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DoubleDQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
##################################################
# Define the DoubleDQNPolicy using epsilon-greedy #
##################################################
class DoubleDQNPolicy(BasePolicy):
    """
    Epsilon-greedy policy for Double DQN.
    
    - We maintain an online network and a target network.
    - The difference from standard DQN is in how the target is computed:
        y = r + gamma * Q_target(s', argmax_a Q_online(s', a))
    """
    def __init__(self, action_space, features_dim, hyper_params):
        """
        Args:
            action_space: The environment's action space (Discrete).
            features_dim: Flattened observation dimension.
            hyper_params expected:
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
        Epsilon-greedy action selection.
        """
        if random.random() < self.hp.epsilon:
            return self.action_space.sample()
        else:
            state = torch.FloatTensor(observation_features).unsqueeze(0)  # add batch dim
            with torch.no_grad():
                q_values = self.online_network(state)
            return int(torch.argmax(q_values, dim=1).item())

    def reset(self, seed):
        super().reset(seed)

        self.update_counter = 0

        # Create two networks: online and target
        self.online_network = DoubleDQNNetwork(self.features_dim, self.action_dim)
        self.target_network = DoubleDQNNetwork(self.features_dim, self.action_dim)
        self.target_network.load_state_dict(self.online_network.state_dict())

        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.hp.step_size)
        self.loss_fn = nn.MSELoss()
        
    def update(self, states, actions, rewards, next_states, dones):
        """
        Double DQN update:
          1) a_star = argmax_a Q_online(next_states, a)
          2) target = r + gamma * Q_target(next_states, a_star)
        """
        # Q(s, a) with the online network
        q_values = self.online_network(states).gather(1, actions)

        with torch.no_grad():
            # 1) pick best actions wrt the *online* network
            online_next_q = self.online_network(next_states)
            best_actions = torch.argmax(online_next_q, dim=1, keepdim=True)  # shape [batch, 1]

            # 2) evaluate these best actions on the *target* network
            target_q = self.target_network(next_states).gather(1, best_actions)

            # compute target
            target = rewards + self.hp.gamma * (1 - dones) * target_q
        
        loss = self.loss_fn(q_values, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.hp.target_update_freq == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

######################################
# The DoubleDQNAgent
######################################
class DoubleDQNAgent(BaseAgent):
    """
    Double DQN agent with experience replay & target network.
    """
    def __init__(self, action_space, observation_space, hyper_params):
        super().__init__(action_space, hyper_params)
        self.feature_extractor = FLattenFeature(observation_space)
        self.action_dim = action_space.n
        
        # Replay buffer
        self.replay_buffer = BasicReplayBuffer(hyper_params.replay_buffer_cap)
        
        # The policy
        self.policy = DoubleDQNPolicy(
            action_space, 
            self.feature_extractor.features_dim, 
            hyper_params
        )

        self.last_state = None
        self.last_action = None
    
    def act(self, observation):
        state = self.feature_extractor(observation)
        action = self.policy.select_action(state)
        self.last_state = state
        self.last_action = action
        return action
    
    def update(self, observation, reward, terminated, truncated):
        new_state = self.feature_extractor(observation)
        
        # Add to replay buffer
        self.replay_buffer.add_single_item(
            (self.last_state, self.last_action, reward, new_state, terminated)
        )
        
        # Sample random batch if enough data
        if self.replay_buffer.size >= self.hp.batch_size:
            batch = self.replay_buffer.get_random_batch(self.hp.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(np.array(actions)).unsqueeze(1)
            rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)
            
            self.policy.update(states, actions, rewards, next_states, dones)

        if terminated or truncated:
            self.last_state = None
            self.last_action = None

    def reset(self, seed):
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.replay_buffer.reset()
        self.last_state = None
        self.last_action = None