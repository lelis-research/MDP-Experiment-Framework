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
    calculate_n_step_returns,
    register_agent,
    register_policy,
)

@register_policy
class NStepDQNPolicy(BasePolicy):
    """
    Epsilon-greedy policy for n-step DQN.
    
    Hyper-parameters (in hp) must include:
        - epsilon (float)
        - gamma (float)
        - step_size (float)
        - target_update_freq (int)
        - value_network (list): configuration for network layers.
    """
    def __init__(self, action_space, features_dim, hyper_params):
        """
        Args:
            action_space (gym.spaces.Discrete): Action space.
            features_dim (int): Dimension of the flattened features.
            hyper_params: Container with hyper-parameters.
        """
        super().__init__(action_space, hyper_params)
        self.features_dim = features_dim
        
    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state (np.array): Flat feature vector.
            
        Returns:
            int: Chosen action.
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
            seed (int): Random seed.
        """
        super().reset(seed)
        self.update_counter = 0
        
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
        
    def update(self, states, actions, n_step_returns, next_states, dones, n_steps, call_back=None):
        """
        Update Q-network using a batch of n-step transitions.
        
        Args:
            states (list/np.array): Batch of states; shape [batch, features_dim].
            actions (list/np.array): Batch of actions; shape [batch].
            n_step_returns (list/np.array): Batch of computed n-step returns; shape [batch].
            next_states (list/np.array): Batch of next states; shape [batch, features_dim].
            dones (list/np.array): Batch of done flags; shape [batch].
            n_steps (list/np.array): Number of steps for each transition; shape [batch].
            call_back (function, optional): Callback to track training progress.
        """
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(np.array(actions)).unsqueeze(1)
        n_step_returns_t = torch.FloatTensor(np.array(n_step_returns)).unsqueeze(1)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(np.array(dones)).unsqueeze(1)
        n_steps_t = torch.FloatTensor(np.array(n_steps)).unsqueeze(1)
        
        # Q-values for current states (for chosen actions)
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

        if call_back is not None:
            call_back({"value_loss": loss.item()})

    def save(self, file_path=None):
        """
        Save network states, optimizer state, and hyper-parameters.
        
        Args:
            file_path (str): File path for saving.
        """
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
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
    def load_from_file(cls, file_path, seed=0):
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        instance = cls(checkpoint['action_space'], checkpoint['features_dim'], checkpoint['hyper_params'])

        instance.reset(seed)
        instance.network.load_state_dict(checkpoint['network_state_dict'])
        instance.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return instance

    def load_from_checkpoint(self, checkpoint):
        """
        Load network states, optimizer state, and hyper-parameters.
        
        Args:
            checkpoint (dictionary)
        """
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.action_space = checkpoint.get('action_space')
        self.features_dim = checkpoint.get('features_dim')
        self.hp = checkpoint.get('hyper_params')

        self.action_dim = checkpoint.get('action_dim')

@register_agent
class NStepDQNAgent(BaseAgent):
    """
    n-step DQN agent that uses a unified replay buffer and an n-step buffer.
    
    Transitions are stored as:
        (state, action, cumulative_reward, next_state, done, n_step)
    
    Args:
        action_space (gym.spaces.Discrete): Environment's action space.
        observation_space: Environment's observation space.
        hyper_params: Hyper-parameters container (must include n_steps, epsilon, gamma, step_size, etc.).
        num_envs (int): Number of parallel environments.
        feature_extractor_class: Class for extracting features from observations.
    """
    name = "NStepDQN"
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class)
        
        # Replay buffer to store n-step transitions.
        self.replay_buffer = BasicBuffer(hyper_params.replay_buffer_cap)
        
        # The DQN policy.
        self.policy = NStepDQNPolicy(action_space, 
                                     self.feature_extractor.features_dim, 
                                     hyper_params)
        
        # Buffer to accumulate n-step transitions.
        self.n_step_buffer = BasicBuffer(hyper_params.n_steps)
    
    def act(self, observation):
        """
        Select an action based on the current observation.
        
        Args:
            observation (np.array or similar): Raw observation.
            
        Returns:
            int: Selected action.
        """
        state = self.feature_extractor(observation)
        action = self.policy.select_action(state)
        self.last_state = state  # Store for later update
        self.last_action = action
        return action
    
    def update(self, observation, reward, terminated, truncated, call_back=None):
        """
        At each time step, accumulate transitions, compute n-step returns, and update the network.
        
        Args:
            observation (np.array or similar): New observation after action.
            reward (float): Reward received.
            terminated (bool): True if the episode terminated.
            truncated (bool): True if the episode was truncated.
            call_back (function, optional): Callback to track training progress.
        """
        state = self.feature_extractor(observation)
        # Append current transition to n-step buffer:
        # (state, action, reward, next_state, done)
        transition = (self.last_state[0], self.last_action, reward, state[0], terminated)
        self.n_step_buffer.add_single_item(transition)

        # If enough transitions are accumulated or if episode ends:
        if self.n_step_buffer.size >= self.hp.n_steps or terminated or truncated:
            rollout = self.n_step_buffer.get_all()
            states, actions, rewards, next_states, dones = zip(*rollout)
            # Compute n-step returns using the accumulated rewards.
            returns = calculate_n_step_returns(rewards, 0.0, self.hp.gamma)
            if terminated or truncated:
                # For episode end, flush all transitions.
                for i in range(self.n_step_buffer.size):
                    trans = (states[i], actions[i], returns[i], next_states[-1], dones[-1], self.n_step_buffer.size - i)
                    self.replay_buffer.add_single_item(trans)
                self.n_step_buffer.reset()
            else:
                # Otherwise, add only the oldest transition.
                trans = (states[0], actions[0], returns[0], next_states[-1], dones[-1], self.n_step_buffer.size)
                self.replay_buffer.add_single_item(trans)
                self.n_step_buffer.remove_oldest()
        
        # Sample a minibatch and update the network if enough samples are available.
        if self.replay_buffer.size >= self.hp.batch_size:
            batch = self.replay_buffer.get_random_batch(self.hp.batch_size)
            states_b, actions_b, n_step_returns_b, next_states_b, dones_b, n_steps_b = zip(*batch)
            self.policy.update(states_b, actions_b, n_step_returns_b, next_states_b, dones_b, n_steps_b, call_back=call_back)
    
    def reset(self, seed):
        """
        Reset the agent: clear buffers and reset the feature extractor.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.replay_buffer.reset()
        self.n_step_buffer.reset()