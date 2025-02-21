import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from Agents.Utils.BaseAgent import BaseAgent, BasePolicy
from Agents.Utils.Buffer import BasicBuffer
from Agents.Utils.HelperFunction import *
from Agents.Utils.NetworkGenerator import NetworkGen, prepare_network_config

class NStepDQNPolicy(BasePolicy):
    """
    Epsilon-greedy policy for n-step DQN.
    """
    def __init__(self, action_space, features_dim, hyper_params):
        """
        hyper_params must include:
            - epsilon
            - gamma
            - step_size  
            - target_update_freq
        """
        super().__init__(action_space, hyper_params)
        self.features_dim = features_dim
        
    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        if random.random() < self.hp.epsilon:
            return self.action_space.sample()
        else:
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
        
    def update(self, states, actions, n_step_returns, next_states, dones, n_steps):
        """
        Update Q-network using a batch of transitions.
        Each transition is a tuple:
          (state, action, cumulative_reward, next_state, done, n_step)
        """
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(np.array(actions)).unsqueeze(1)
        n_step_returns_t = torch.FloatTensor(np.array(n_step_returns)).unsqueeze(1)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(np.array(dones)).unsqueeze(1)
        n_steps_t = torch.FloatTensor(np.array(n_steps)).unsqueeze(1)
        
        # Online network Q-value for the taken action.
        qvalues_t = self.network(states_t).gather(1, actions_t)
        
        with torch.no_grad():
            bootstrap_values = self.target_network(next_states_t).max(1)[0].unsqueeze(1)
            discount = self.hp.gamma ** n_steps_t
            targets = n_step_returns_t + discount * (1 - dones_t) * bootstrap_values

        loss = self.loss_fn(qvalues_t, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_counter += 1
        if self.update_counter % self.hp.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def save(self, file_path):
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyper_params': self.hp,  # Ensure self.hp is pickle-serializable
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
class NStepDQNAgent(BaseAgent):
    """
    n-step Deep Q-Network (DQN) agent that uses a single replay buffer for all transitions.
    """
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class):
        """
        hyper_params must include:
            - epsilon
            - gamma
            - step_size  
            - replay_buffer_cap
            - batch_size
            - target_update_freq
            - n_steps   (number of steps for multi-step return)
        """
        super().__init__(action_space, observation_space, hyper_params, num_envs)
        self.feature_extractor = feature_extractor_class(observation_space)
        
        # Unified Replay Buffer: will store transitions as:
        # (state, action, cumulative_reward, next_state, done, n_steps)
        self.replay_buffer = BasicBuffer(hyper_params.replay_buffer_cap)
        
        self.policy = NStepDQNPolicy(action_space, self.feature_extractor.features_dim, hyper_params)
        
        # n-step buffer for accumulating transitions before computing the n-step return.
        # Each element: (state, action, reward, next_state, done)
        self.n_step_buffer = BasicBuffer(hyper_params.n_steps)  
    
    def act(self, observation):
        """
        Select an action based on the current observation.
        """
        state = self.feature_extractor(observation)
        action = self.policy.select_action(state)

        self.last_state = state  # Store current state
        self.last_action = action
        return action
    
    def update(self, observation, reward, terminated, truncated):
        """
        Called at every time step.
        Accumulate transitions into the n-step buffer, compute n-step returns,
        and add the resulting transitions to the replay buffer. Then sample a batch
        and update the network.
        """
        state = self.feature_extractor(observation)
        
        # Append the latest transition.
        transition = (self.last_state[0], self.last_action, reward, state[0], terminated)
        self.n_step_buffer.add_single_item(transition)

        # If we have at least n_step transitions, compute the n-step transition.
        if self.n_step_buffer.size >= self.hp.n_steps or terminated or truncated:
            rollout = self.n_step_buffer.get_all()
            states, actions, rewards, next_states, dones = zip(*rollout)
            if terminated or truncated:
                returns = calculate_n_step_returns(rewards, 0.0, self.hp.gamma)
                for i in range(self.n_step_buffer.size):
                    transition = states[i], actions[i], returns[i], next_states[-1], dones[-1], self.n_step_buffer.size - i
                    self.replay_buffer.add_single_item(transition)
                self.n_step_buffer.reset()
            else:
                returns = calculate_n_step_returns(rewards, 0.0, self.hp.gamma)
                transition = states[0], actions[0], returns[0], next_states[-1], dones[-1], self.n_step_buffer.size
                self.replay_buffer.add_single_item(transition)
                self.n_step_buffer.remove_oldest()
        

        # Update network if the replay buffer has enough samples.
        if self.replay_buffer.size >= self.hp.batch_size:
            batch = self.replay_buffer.get_random_batch(self.hp.batch_size)
            states, actions, n_step_returns, next_states, dones, n_steps = zip(*batch)
            self.policy.update(states, actions, n_step_returns, next_states, dones, n_steps)
    
    
    def reset(self, seed):
        """
        Reset the agent's learning state.
        """
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.replay_buffer.reset()
        self.n_step_buffer.reset()