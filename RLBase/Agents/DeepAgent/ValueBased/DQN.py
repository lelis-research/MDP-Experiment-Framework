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
    get_single_observation
)
from ....registry import register_agent, register_policy

@register_policy
class DQNPolicy(BasePolicy):

    def __init__(self, action_space, features_dict, hyper_params, device="cpu"):
        super().__init__(action_space, hyper_params, device=device)
        
        self.features_dict = features_dict
        
        network_description = prepare_network_config(
            self.hp.value_network,
            input_dims=self.features_dict,
            output_dim=self.action_dim
        )
        self.network = NetworkGen(layer_descriptions=network_description).to(self.device)
        self.target_network = NetworkGen(layer_descriptions=network_description).to(self.device)
   
        
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.hp.step_size)
        self.target_update_counter = 0
        
        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.SmoothL1Loss() #more stable loss
        
        self.epsilon = self.hp.epsilon_start
        self.epsilon_step_counter = 0
        self.log_counter = 0
        
    
    def get_values(self, state, use_target=False):
        # each state element should have the batch dimension
        keys = list(self.features_dict.keys())
        kwargs = {keys[0]: state} if len(keys) == 1 else {k: v for k, v in zip(keys, state)}
        q_values = self.network(**kwargs) if not use_target else self.target_network(**kwargs)
        return q_values
        
    def select_action(self, state, greedy=False):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state (np.array): Flat feature vector.
        
        Returns:
            int: Selected action.
        """         
        action = []
        
        with torch.no_grad():
            q_values = self.get_values(state).cpu().numpy()
        
        
        state_dim = q_values.shape[0]

        for i in range(state_dim):
            self.epsilon_step_counter += 1
            self.log_counter += 1

            if (self._rand_float(low=0, high=1) < self.epsilon) and not greedy:
                # Random action
                action.append(int(self._rand_int(0, self.action_dim)))
            else:
                # Greedy action with tie-breaking, consistent with QLearning
                row = q_values[i]  # shape [A]]
                max_actions = np.flatnonzero(row == np.max(row))
                action.append(int(self._rand_elem(max_actions)))

        return action
        
   
    def reset(self, seed):
        """
        Reset the policy: initialize networks, optimizer, and counter.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
            
        network_description = prepare_network_config(
            self.hp.value_network,
            input_dims=self.features_dict,
            output_dim=self.action_dim
        )
        self.network = NetworkGen(layer_descriptions=network_description).to(self.device)
        self.target_network = NetworkGen(layer_descriptions=network_description).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.hp.step_size)
        self.target_update_counter = 0
        
        self.epsilon = self.hp.epsilon_start
        self.epsilon_step_counter = 0
        
        self.log_counter = 0 #for logging
        
        
    def update(self, last_state, last_action, state, target_reward, terminated, truncated, effective_discount, call_back=None):
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
        actions_t = torch.tensor(np.array(last_action), dtype=torch.int64, device=self.device).unsqueeze(1) # if n is 1 then it is rewards
        target_reward_t = torch.tensor(np.array(target_reward), dtype=torch.float32, device=self.device).unsqueeze(1)
        terminated_t = torch.tensor(np.array(terminated), dtype=torch.float32, device=self.device).unsqueeze(1)
        effective_discount_t = torch.tensor(np.array(effective_discount), dtype=torch.float32, device=self.device).unsqueeze(1)
        
        
        last_state_t = torch.as_tensor(np.array(last_state), dtype=torch.float32, device=self.device)
        state_t = torch.as_tensor(np.array(state), dtype=torch.float32, device=self.device)
   
        # ---- Q(s_t, a_t) from online network ----
        # last_state has batch dimension already, so we can call network directly
        qvalues_all = self.get_values(last_state_t)          # [B, A]
        qvalues_t = qvalues_all.gather(1, actions_t)       # [B, 1]
        
        # ---- Bootstrap value from next state ----
        with torch.no_grad():
            if self.hp.flag_double_dqn_target:
                # Double DQN: argmax from online net, value from target net
                q_next_online = self.get_values(state_t)          # [B, A]
                a_next = q_next_online.argmax(dim=1, keepdim=True)  # [B, 1]
                q_next_target = self.get_values(state_t, use_target=True)       # [B, A]
                bootstrap_value = q_next_target.gather(1, a_next)   # [B, 1]
            else:
                # Standard DQN: max over target net's Q
                q_next_target = self.get_values(state_t, use_target=True)       # [B, A]
                bootstrap_value = q_next_target.max(1, keepdim=True)[0]  # [B, 1]
                
            # Do not bootstrap on terminal transitions
            bootstrap_value = (1.0 - terminated_t) * bootstrap_value  # [B, 1]
        
        
        # ---- TD target ----
        target_t = target_reward_t + effective_discount_t * bootstrap_value           # [B, 1]
        
        # ---- Loss ----
        loss = self.loss_fn(qvalues_t, target_t)
        
        
        # ******** Logging (without grad) ********
        with torch.no_grad():
            td = qvalues_t - target_t                 # (B,1)
            td_abs = td.abs()
            q_mean = qvalues_t.mean().item()
            q_max = qvalues_t.max().item()
            td_mean = td.mean().item()
            td_abs_mean = td_abs.mean().item()
            target_mean = target_t.mean().item()
        # ***************************************
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient norm logging
        total_grad_norm = 0.0
        for p in self.network.parameters():
            if p.grad is not None:
                total_grad_norm += (p.grad.detach().data.norm(2) ** 2).item()
        total_grad_norm = total_grad_norm ** 0.5

        self.optimizer.step()

        # ---- Target network update ----
        self.target_update_counter += 1
        if self.target_update_counter >= self.hp.target_update_freq:
            self.target_network.load_state_dict(self.network.state_dict())
            self.target_update_counter = 0

            
        #Update Epsilon
        frac = 1.0 - (self.epsilon_step_counter / self.hp.epsilon_decay_steps)
        frac = max(0.0, frac)
        self.epsilon = max(self.hp.epsilon_end, self.hp.epsilon_end + (self.hp.epsilon_start - self.hp.epsilon_end) * frac)

        if call_back is not None:
            call_back(
                {
                    "train/loss": loss.item(),
                    "train/epsilon": self.epsilon,
                    "train/q_mean": q_mean,
                    "train/q_max": q_max,
                    "train/target_mean": target_mean,
                    "train/td_mean": td_mean,
                    "train/td_abs_mean": td_abs_mean,
                    "train/grad_norm": total_grad_norm,
                },
                counter=self.log_counter,
            )
            
    def save(self, file_path=None):
        checkpoint = super().save(file_path=None)
        
        checkpoint['network_state_dict'] = self.network.state_dict()
        checkpoint['target_network_state_dict'] = self.target_network.state_dict()
        checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        checkpoint['features_dict'] = self.features_dict
        checkpoint['epsilon_step_counter'] = self.epsilon_step_counter
        checkpoint['epsilon'] = self.epsilon
        
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_policy.t")
        return checkpoint

    @classmethod
    def load(cls, file_path, checkpoint=None):
        instance = super().load(file_path, checkpoint)
        
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        
        instance.network.load_state_dict(checkpoint['network_state_dict'])
        instance.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        instance.features_dict = checkpoint['features_dict']
        
        instance.epsilon = checkpoint.get('epsilon')
        instance.epsilon_step_counter = checkpoint.get('epsilon_step_counter')
        
        return instance

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
            self.feature_extractor.features_dict, 
            hyper_params,
            device=device
        )
        
        # Buffer to accumulate n-step transitions.
        self.rollout_buffer = [BasicBuffer(self.hp.n_steps) for _ in range(self.num_envs)]  # Buffer is used for n-step
        
    def act(self, observation, greedy=False):
        """
        Select an action based on the observation.
        observation is a batch
        action is a batch 
        """
        state = self.feature_extractor(observation) # tuple (B, Features)
        action = self.policy.select_action(state, greedy=greedy) # (B, )
        # self.last_observation = observation
        self.last_state = state
        self.last_action = action
        return action
        
    def update(self, observation, reward, terminated, truncated, call_back=None):
        """
        all arguments are batches
        """
        # reward = np.clip(reward, -1, 1) # for stability
        state = self.feature_extractor(observation)
        for i in range(len(reward)):
            transition = self.last_state[i], self.last_action[i], state[i], reward[i], terminated[i], truncated[i], self.hp.gamma
            self.replay_buffer.add_single_item(transition)
            # # transition = get_single_observation(self.last_observation, i), self.last_action[i], reward[i]
            # transition = self.last_state[i], self.last_action[i], reward[i]
            # self.rollout_buffer[i].add_single_item(transition)
            # print (self.rollout_buffer[i].size)
            
            # if terminated[i] or truncated[i]:
            #     rollout = self.rollout_buffer[i].get_all()
            #     # rollout_observations, rollout_actions, rollout_rewards = zip(*rollout)
            #     rollout_states, rollout_actions, rollout_rewards = zip(*rollout)
                
            #     n_step_return = calculate_n_step_returns(rollout_rewards, 0, self.hp.gamma)
            #     for j in range(self.rollout_buffer[i].size):                    
            #         # trans = (rollout_observations[j], rollout_actions[j], get_single_observation(observation, i), \
            #         #         n_step_return[j], terminated[i], \
            #         #         truncated[i], self.hp.gamma**(self.rollout_buffer[i].size-j))
                    
            #         trans = (rollout_states[j], rollout_actions[j], state[i], \
            #                 n_step_return[j], terminated[i], \
            #                 truncated[i], self.hp.gamma**(self.rollout_buffer[i].size-j))
                    
            #         self.replay_buffer.add_single_item(trans)
            #     self.rollout_buffer[i].reset() 
            
            # elif self.rollout_buffer[i].size >= self.hp.n_steps:
            #     rollout = self.rollout_buffer[i].get_all() 
            #     # rollout_observations, rollout_actions, rollout_rewards = zip(*rollout)
            #     rollout_states, rollout_actions, rollout_rewards = zip(*rollout)
            #     n_step_return = calculate_n_step_returns(rollout_rewards, 0, self.hp.gamma)
            #     # trans = (rollout_observations[0], rollout_actions[0], get_single_observation(observation, i), \
            #     #          n_step_return[0], terminated[i], \
            #     #          truncated[i], self.hp.gamma**self.hp.n_steps)
            #     trans = (rollout_states[0], rollout_actions[0], state[i], \
            #              n_step_return[0], terminated[i], \
            #              truncated[i], self.hp.gamma**self.hp.n_steps)
            #     self.replay_buffer.add_single_item(trans)
                
            
        
        if self.replay_buffer.size >= self.hp.warmup_buffer_size:
            batch = self._rand_subset(self.replay_buffer.get_all(), self.hp.batch_size)

            # observations, actions, next_observations, n_step_return, terminated, truncated, effective_discount = zip(*batch)
            states, actions, next_states, n_step_return, terminated, truncated, effective_discount = zip(*batch)
            
            # states, next_states = [self.feature_extractor(obs) for obs in observations], [self.feature_extractor(obs) for obs in next_observations]
            self.policy.update(states, actions, next_states, n_step_return , terminated, truncated, effective_discount, call_back=call_back)
        
        if call_back is not None:            
            call_back({
                "train/buffer_size": self.replay_buffer.size,
            }, counter=self.policy.log_counter)
        

                

    def reset(self, seed):
        """
        Reset the agent's learning state, including feature extractor and replay buffer.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        
        self.last_observation = None
        self.last_action = None

        self.replay_buffer.reset()        
        self.rollout_buffer = [BasicBuffer(self.hp.n_steps) for _ in range(self.num_envs)]
        
        
