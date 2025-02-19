import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from Agents.Utils.BaseAgent import BaseAgent, BasePolicy
from Agents.Utils.FeatureExtractor import FLattenFeature
from Agents.Utils.Buffer import BasicBuffer    
from Agents.Utils.NetworkGenerator import NetworkGen, prepare_network_config

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

    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        if random.random() < self.hp.epsilon:
            return self.action_space.sample()
        else:
            state_t = torch.FloatTensor(state)
            with torch.no_grad():
                q_values = self.online_network(state_t)
            return int(torch.argmax(q_values, dim=1).item())
        
    def reset(self, seed):
        super().reset(seed)

        self.update_counter = 0

        # Create two networks: online and target
        network_description = prepare_network_config(self.hp.value_network,
                                                    input_dim=self.features_dim,
                                                    output_dim=self.action_dim)
        self.online_network = NetworkGen(layer_descriptions=network_description)
        self.target_network = NetworkGen(layer_descriptions=network_description)

        self.target_network.load_state_dict(self.online_network.state_dict())

        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.hp.step_size)
        self.loss_fn = nn.MSELoss()
        
    def update(self, states, actions, rewards, next_states, dones):
        """
        Double DQN update:
          1) a_star = argmax_a Q_online(next_states, a)
          2) target = r + gamma * Q_target(next_states, a_star)
        """
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(np.array(actions)).unsqueeze(1)
        rewards_t = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(np.array(dones)).unsqueeze(1)
        
        # Q(s, a) with the online network
        qvalues_t = self.online_network(states_t).gather(1, actions_t)

        with torch.no_grad():
            # 1) pick best actions wrt the *online* network
            next_qvalues_t = self.online_network(next_states_t)
            best_actions = torch.argmax(next_qvalues_t, dim=1, keepdim=True)  # shape [batch, 1]

            # 2) evaluate these best actions on the *target* network
            bootstrap_value_t = self.target_network(next_states_t).gather(1, best_actions)

            # compute target
            target_t = rewards_t + self.hp.gamma * (1 - dones_t) * bootstrap_value_t
        
        loss = self.loss_fn(qvalues_t, target_t)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.hp.target_update_freq == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())


class DoubleDQNAgent(BaseAgent):
    """
    Double DQN agent with experience replay & target network.
    """
    def __init__(self, action_space, observation_space, hyper_params, num_envs):
        super().__init__(action_space, observation_space, hyper_params, num_envs)
        self.feature_extractor = FLattenFeature(observation_space)
        self.action_dim = action_space.n
        
        # Replay buffer
        self.replay_buffer = BasicBuffer(hyper_params.replay_buffer_cap)
        
        # The policy
        self.policy = DoubleDQNPolicy(
            action_space, 
            self.feature_extractor.features_dim, 
            hyper_params
        )

    
    def act(self, observation):
        state = self.feature_extractor(observation)
        action = self.policy.select_action(state)
        self.last_state = state
        self.last_action = action
        return action
    
    def update(self, observation, reward, terminated, truncated):
        state = self.feature_extractor(observation)
        
        # Add to replay buffer
        self.replay_buffer.add_single_item(
            (self.last_state[0], self.last_action, reward, state[0], terminated)
        )
        
        # Sample random batch if enough data
        if self.replay_buffer.size >= self.hp.batch_size:
            batch = self.replay_buffer.get_random_batch(self.hp.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            self.policy.update(states, actions, rewards, next_states, dones)

    def reset(self, seed):
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.replay_buffer.reset()