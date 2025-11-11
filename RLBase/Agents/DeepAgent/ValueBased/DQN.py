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
    calculate_n_step_returns,
)
from ....registry import register_agent, register_policy

@register_policy
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
    def __init__(self, action_space, features_dim, num_features, hyper_params, device="cpu"):
        super().__init__(action_space, hyper_params, device=device)
        self.features_dim = features_dim
        self.num_features = num_features
        self.epsilon_step_counter = 0
        self.epsilon = self.hp.epsilon_start
         
    def select_action(self, state, greedy=False):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state (np.array): Flat feature vector.
        
        Returns:
            int: Selected action.
        """         
        self.act_counter += 1
        
        self.epsilon_step_counter += 1
              
        if random.random() < self.epsilon and not greedy:
            return self.action_space.sample()
        else:
            if self.num_features == 1:
                with torch.no_grad():
                    q_values = self.network(state)
            else:
                with torch.no_grad():
                    q_values = self.network(*state)
            
            # for ties
            max_actions = (q_values == torch.max(q_values)).nonzero()[:, 1] #the indexing gets rid of the batch dim
            action = int(np.random.choice(max_actions.cpu()))
            
            # has no ties
            # action = torch.argmax(q_values).item()
            return action
        
    def select_parallel_actions(self, states_vec, greedy=False, num_envs=None):
        self.act_counter += num_envs
        
        self.epsilon_step_counter += num_envs

        if random.random() < self.epsilon and not greedy:
            actions_vec = np.stack([self.action_space.sample() for _ in range(num_envs)])
            return actions_vec
        else:
            if self.num_features == 1:
                with torch.no_grad():
                    q_values = self.network(states_vec)
            else:
                with torch.no_grad():
                    q_values = self.network(*states_vec)
            
            # for ties
            max_actions_vec = (q_values == q_values.max(dim=1, keepdim=True).values).nonzero()
            actions_vec = [int(np.random.choice(max_actions_vec[max_actions_vec[:,0]==i,1].cpu())) for i in range(q_values.size(0))]
            
            # has no ties
            # actions_vec = torch.argmax(q_values, dim=1).cpu().numpy()
            return actions_vec

    
    def reset(self, seed):
        """
        Reset the policy: initialize networks, optimizer, and counter.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        self.target_update_counter = 0
        
        # Prepare network configuration using the value network description.
        network_description = prepare_network_config(
            self.hp.value_network,
            input_dims={"img": self.features_dim[0], "dir_carry": self.features_dim[1]},
            output_dim=self.action_dim
        )
        self.network = NetworkGen(layer_descriptions=network_description).to(self.device)
        self.target_network = NetworkGen(layer_descriptions=network_description).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.hp.step_size)
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.SmoothL1Loss()
        
        self.epsilon = self.hp.epsilon_start
        self.epsilon_step_counter = 0
        
        self.act_counter = 0 #for logging
        
        
    def update(self, states, actions, n_step_returns, next_states, dones, n_steps, call_back=None):
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
        actions_t = torch.tensor(np.array(actions), dtype=torch.int64, device=self.device).unsqueeze(1) # if n is 1 then it is rewards
        n_step_returns_t = torch.tensor(np.array(n_step_returns), dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device).unsqueeze(1)
        n_steps_t = torch.tensor(np.array(n_steps), dtype=torch.float32, device=self.device).unsqueeze(1)
        if self.num_features == 1:
            #feature extractor is just one feature
            states_t = torch.cat(states).to(dtype=torch.float32, device=self.device) if torch.is_tensor(states[0]) else torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
            next_states_t = torch.cat(next_states).to(dtype=torch.float32, device=self.device) if torch.is_tensor(next_states[0]) else torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
            qvalues_t = self.network(states_t).gather(1, actions_t)
            with torch.no_grad():
                if self.hp.flag_double_dqn_target:
                    q_next  = self.network(next_states_t)              # [B, A]
                    a_next  = q_next.argmax(dim=1, keepdim=True)  # [B,1]
                    bootstrap_value_t  = self.target_network(next_states_t).gather(1, a_next)
                else:
                    bootstrap_value_t = self.target_network(next_states_t).max(1)[0].unsqueeze(1)
        else:
            states_t = [torch.cat(x, dim=0).to(dtype=torch.float32, device=self.device) for x in zip(*states)]
            next_states_t = [torch.cat(x, dim=0).to(dtype=torch.float32, device=self.device) for x in zip(*next_states)]
            qvalues_t = self.network(*states_t).gather(1, actions_t)
            with torch.no_grad():
                if self.hp.flag_double_dqn_target:
                    q_next  = self.network(*next_states_t)             # [B, A]
                    a_next  = q_next.argmax(dim=1, keepdim=True)
                    bootstrap_value_t  = self.target_network(*next_states_t).gather(1, a_next)
                else:
                    bootstrap_value_t = self.target_network(*next_states_t).max(1)[0].unsqueeze(1)
        
 
        # calculate loss       
        discount = self.hp.gamma ** n_steps_t
        target_t = n_step_returns_t + discount * (1 - dones_t) * bootstrap_value_t
        
        loss = self.loss_fn(qvalues_t, target_t)
        
        # *** begin logging ******
        with torch.no_grad():
            td = (qvalues_t - target_t)                  # (B,1)
            td_abs = td.abs()
            q_mean = qvalues_t.mean().item()
            q_max  = qvalues_t.max().item()
            td_mean = td.mean().item()
            td_abs_mean = td_abs.mean().item()
            target_mean = target_t.mean().item()
        # *** end logging ******
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # *** begin logging ******
        total_grad_norm = 0.0
        for p in self.network.parameters():
            if p.grad is not None:
                total_grad_norm += (p.grad.detach().data.norm(2)**2).item()
        total_grad_norm = total_grad_norm ** 0.5
        # *** end logging ******

        self.optimizer.step()

        self.target_update_counter += 1
        if self.target_update_counter >= self.hp.target_update_freq:
            self.target_network.load_state_dict(self.network.state_dict())
            self.target_update_counter = 0
            
        #Update Epsilon
        frac = 1.0 - (self.epsilon_step_counter / self.hp.epsilon_decay_steps)
        frac = max(0.0, frac)
        self.epsilon = max(self.hp.epsilon_end, self.hp.epsilon_end + (self.hp.epsilon_start - self.hp.epsilon_end) * frac)



        if call_back is not None:            
            call_back({
                "train/loss": loss.item(),
                "train/epsilon": self.epsilon,
                "train/q_mean": q_mean,
                "train/q_max": q_max,
                "train/target_mean": target_mean,
                "train/td_mean": td_mean,
                "train/td_abs_mean": td_abs_mean,
                "train/grad_norm": total_grad_norm,
            }, counter=self.act_counter)
            
    def save(self, file_path=None):
        """
        Save the network states, optimizer state, and hyper-parameters.
        
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
            
            'policy_class': self.__class__.__name__,
            
            'epsilon_step_counter': self.epsilon_step_counter,
            'epsilon': self.epsilon,

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
        instance.load_from_checkpoint(checkpoint)
        
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
        
        self.epsilon = checkpoint.get('epsilon')
        self.epsilon_step_counter=checkpoint.get('epsilon_step_counter')

@register_agent
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
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device="cpu"):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device=device)
        # Experience Replay Buffer
        self.replay_buffer = BasicBuffer(hyper_params.replay_buffer_cap)
        
        # Create DQNPolicy using the feature extractor's feature dimension.
        self.policy = DQNPolicy(
            action_space, 
            self.feature_extractor.features_dim,
            self.feature_extractor.num_features, 
            hyper_params,
            device=device
        )
        
        # Buffer to accumulate n-step transitions.
        self.n_step_buffer = [BasicBuffer(hyper_params.n_steps)] * num_envs if num_envs > 1 else BasicBuffer(hyper_params.n_steps)
        
    def act(self, observation, greedy=False):
        """
        Select an action based on the current observation.
        
        Args:
            observation (np.array or similar): Raw observation from the environment.
        
        Returns:
            int: Selected action.
        """
        state = self.feature_extractor(observation)
        action = self.policy.select_action(state, greedy=greedy)
        self.last_observation = observation
        self.last_action = action
        return action
        
    def update(self, observation, reward, terminated, truncated, call_back=None):
        """
        Store the transition and, if enough samples are available, perform a learning step.
        
        Args:
            observation (np.array or similar): New observation after action.
            reward (float): Reward received.
            terminated (bool): True if the episode has terminated.
            truncated (bool): True if the episode was truncated.
            call_back (function): Callback function to track training progress.
        """
        reward = np.clip(reward, -1, 1) # for stability
        transition = (self.last_observation, self.last_action, reward, observation, terminated)
        self.n_step_buffer.add_single_item(transition)
        
        # If enough transitions are accumulated or if episode ends:
        if self.n_step_buffer.size >= self.hp.n_steps or terminated or truncated:
            rollout = self.n_step_buffer.get_all()
            observations, actions, rewards, next_observations, dones = zip(*rollout)
            # Compute n-step returns using the accumulated rewards.
            returns = calculate_n_step_returns(rewards, 0.0, self.hp.gamma)
            if terminated or truncated:
                # For episode end, flush all transitions.
                for i in range(self.n_step_buffer.size):
                    trans = (observations[i], actions[i], returns[i], next_observations[-1], dones[-1], self.n_step_buffer.size - i)
                    self.replay_buffer.add_single_item(trans)
                self.n_step_buffer.reset()
            else:
                # Otherwise, add only the oldest transition.
                trans = (observations[0], actions[0], returns[0], next_observations[-1], dones[-1], self.n_step_buffer.size)
                self.replay_buffer.add_single_item(trans)
                self.n_step_buffer.remove_oldest()
        
        
        if self.replay_buffer.size >= self.hp.warmup_buffer_size:
            batch = self.replay_buffer.get_random_batch(self.hp.batch_size)
            observations, actions, rewards, next_observations, dones, n_steps = zip(*batch)
            states, next_states = [self.feature_extractor(obs) for obs in observations], [self.feature_extractor(obs) for obs in next_observations]
            self.policy.update(states, actions, rewards, next_states, dones, n_steps, call_back=call_back)
            
        if call_back is not None:            
            call_back({
                "train/buffer_size": self.replay_buffer.size,
            }, counter=self.policy.act_counter)
        

                

    def reset(self, seed):
        """
        Reset the agent's learning state, including feature extractor and replay buffer.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.replay_buffer.reset()
        
        if isinstance(self.n_step_buffer, list):
            for buffer in self.n_step_buffer: buffer.reset()
        else:
            self.n_step_buffer.reset()
        
        self.last_observation = None
        self.last_action = None
        
