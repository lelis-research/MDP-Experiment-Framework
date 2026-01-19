import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import Discrete
import time

from ...Base import BaseAgent, BasePolicy
from ....Buffers import BaseBuffer, ReplayBuffer
from ....Networks.NetworkFactory import NetworkGen, prepare_network_config
from ...Utils import calculate_n_step_returns, get_single_observation, stack_observations
from ....registry import register_agent, register_policy

@register_policy
class DQNPolicy(BasePolicy):

    def __init__(self, action_space, features_dict, hyper_params, device="cpu"):
        super().__init__(action_space, hyper_params, device=device)
        
        self.features_dict = features_dict
        if self.hp.enable_dueling_networks:
            # Dueling: network returns dict {"V": [B,1], "A": [B,A]}
            # Your preset must include linear nodes with ids "V" and "A"
            network_description = prepare_network_config(
                self.hp.value_network,
                input_dims=self.features_dict,
                output_dims={"V": 1, "A": self.action_dim},
            )
            self.online_network = NetworkGen(layer_descriptions=network_description, output_ids=["V", "A"]).to(self.device)
            self.target_network = NetworkGen(layer_descriptions=network_description, output_ids=["V", "A"]).to(self.device)
            
        else:
            network_description = prepare_network_config(
                self.hp.value_network,
                input_dims=self.features_dict,
                output_dim=self.action_dim
            )
            self.online_network = NetworkGen(layer_descriptions=network_description).to(self.device)
            self.target_network = NetworkGen(layer_descriptions=network_description).to(self.device)
   
        
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.hp.step_size)
        self.target_update_counter = 0
        
        if self.hp.enable_huber_loss:
            self.loss_fn = nn.SmoothL1Loss() #more stable loss
        else:
            self.loss_fn = nn.MSELoss()
    
        
        self.epsilon = self.hp.epsilon_start if not self.hp.enable_noisy_nets else 0.0
        self.epsilon_step_counter = 0
        

    def get_values(self, state, use_target=False):
        # each state element should have the batch dimension
        net = self.target_network if use_target else self.online_network

        out = net(**state)

        if not self.hp.enable_dueling_networks:
            # out is [B, A]
            return out

        # out is dict: {"V": [B,1], "A": [B,A]}
        V = out["V"]
        A = out["A"]
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q
        
    
    def select_action(self, state, greedy=False):
        if self.hp.enable_noisy_nets:
            # NoisyNet exploration: resample noise each action selection
            self.online_network.reset_noise()

            with torch.no_grad():
                q = self.get_values(state)          # [B, A] torch
                a = torch.argmax(q, dim=1)          # [B]
            return a.tolist()

        # ---------- epsilon-greedy (your current logic) ----------
        action = []
        with torch.no_grad():
            q_values = self.get_values(state).cpu().numpy()

        for i in range(q_values.shape[0]):
            self.epsilon_step_counter += 1
            if (not greedy) and (self._rand_float(0, 1) < self.epsilon):
                action.append(int(self._rand_int(0, self.action_dim)))
            else:
                row = q_values[i]
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
            
        if self.hp.enable_dueling_networks:
            # Dueling: network returns dict {"V": [B,1], "A": [B,A]}
            # Your preset must include linear nodes with ids "V" and "A"
            network_description = prepare_network_config(
                self.hp.value_network,
                input_dims=self.features_dict,
                output_dims={"V": 1, "A": self.action_dim},
            )
            self.online_network = NetworkGen(layer_descriptions=network_description, output_ids=["V", "A"]).to(self.device)
            self.target_network = NetworkGen(layer_descriptions=network_description, output_ids=["V", "A"]).to(self.device)
        else:
            network_description = prepare_network_config(
                self.hp.value_network,
                input_dims=self.features_dict,
                output_dim=self.action_dim
            )
            self.online_network = NetworkGen(layer_descriptions=network_description).to(self.device)
            self.target_network = NetworkGen(layer_descriptions=network_description).to(self.device)
        
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.hp.step_size)
        self.target_update_counter = 0
        
        self.epsilon = self.hp.epsilon_start if not self.hp.enable_noisy_nets else 0.0
        self.epsilon_step_counter = 0
        
        
        
    def update(self, states, actions, next_states, target_reward, terminated, truncated, effective_discount, call_back=None):
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
        target_reward_t = torch.tensor(np.array(target_reward), dtype=torch.float32, device=self.device).unsqueeze(1)
        terminated_t = torch.tensor(np.array(terminated), dtype=torch.float32, device=self.device).unsqueeze(1)
        effective_discount_t = torch.tensor(np.array(effective_discount), dtype=torch.float32, device=self.device).unsqueeze(1)
        
        if self.hp.enable_noisy_nets:
            self.online_network.reset_noise()
            self.target_network.reset_noise()
   
        # ---- Q(s_t, a_t) from online network ----
        # states has batch dimension already, so we can call network directly
        qvalues_all = self.get_values(states)          # [B, A]
        qvalues_t = qvalues_all.gather(1, actions_t)       # [B, 1]
       
        # ---- Bootstrap value from next state ----
        with torch.no_grad():
            if self.hp.enable_double_dqn_target:
                # Double DQN: argmax from online net, value from target net
                q_next_online = self.get_values(next_states)          # [B, A]
                a_next = q_next_online.argmax(dim=1, keepdim=True)  # [B, 1]
                q_next_target = self.get_values(next_states, use_target=True)       # [B, A]
                bootstrap_value = q_next_target.gather(1, a_next)   # [B, 1]
            else:
                # Standard DQN: max over target net's Q
                q_next_target = self.get_values(next_states, use_target=True)       # [B, A]
                bootstrap_value = q_next_target.max(1, keepdim=True)[0]  # [B, 1]
                
            # Do not bootstrap on terminal transitions
            bootstrap_value = (1.0 - terminated_t) * bootstrap_value  # [B, 1]
        
        
        # ---- TD target ----
        target_t = target_reward_t + effective_discount_t * bootstrap_value           # [B, 1]

        # ---- Loss ----
        loss = self.loss_fn(qvalues_t, target_t)
        
        
        # ******** Logging (without grad) ********
        if call_back is not None:
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
        if self.hp.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), self.hp.max_grad_norm)
        
        # Gradient norm logging
        if call_back is not None:
            total_grad_norm = 0.0
            for p in self.online_network.parameters():
                if p.grad is not None:
                    total_grad_norm += (p.grad.detach().data.norm(2) ** 2).item()
            total_grad_norm = total_grad_norm ** 0.5

        self.optimizer.step()
        

        # ---- Target network update ----
        self.target_update_counter += 1
        if self.target_update_counter >= self.hp.target_update_freq:
            self.target_network.load_state_dict(self.online_network.state_dict())
            self.target_update_counter = 0

        
        if not self.hp.enable_noisy_nets:
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
            )
            
    def save(self, file_path=None):
        checkpoint = super().save(file_path=None)
        
        checkpoint['online_network_state_dict'] = self.online_network.state_dict()
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
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        # can't call parent class bc the __init__ args are different
        instance = cls(checkpoint['action_space'], checkpoint['features_dict'], checkpoint['hyper_params'], checkpoint['device'])
        instance.set_rng_state(checkpoint['rng_state']) #copied from the BasePolicy.load
        
        instance.online_network.load_state_dict(checkpoint['online_network_state_dict'])
        instance.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        instance.features_dict = checkpoint['features_dict']
        
        instance.epsilon = checkpoint.get('epsilon')
        instance.epsilon_step_counter = checkpoint.get('epsilon_step_counter')
        
        return instance

@register_agent
class DQNAgent(BaseAgent):
    """
    Hyper-params:
        "gamma": 0.99,
        "n_steps": 3,

        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 15_000,
        "enable_noisy_nets": True,

        "replay_buffer_size": 200_000,
        "warmup_buffer_size": 500,

        "batch_size": 64,
        "update_freq": 4,
        "target_update_freq": 20,

        "value_network": "MiniGrid/DQN/mlp_noisy",
        "step_size": 1e-3,
        "enable_double_dqn_target": True,
        "enable_dueling_networks": False,
        "enable_huber_loss": True,
        "max_grad_norm": None,
    """
    name = "DQN"
    SUPPORTED_ACTION_SPACES = (Discrete, )
    
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device="cpu"):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device=device)
        # Experience Replay Buffer
        self.replay_buffer = ReplayBuffer(hyper_params.replay_buffer_size)
        
        # Create DQNPolicy using the feature extractor's feature dimension.
        self.policy = DQNPolicy(
            action_space, 
            self.feature_extractor.features_dict, 
            hyper_params,
            device=device
        )
        
        # Buffer to accumulate n-step transitions.
        self.rollout_buffer = [BaseBuffer(self.hp.n_steps) for _ in range(self.num_envs)]  # Buffer is used for n-step
        self.update_call_counter = 0
        
    def act(self, observation, greedy=False):
        """
        Select an action based on the observation.
        observation is a batch
        action is a batch 
        """
        state = self.feature_extractor(observation) 
        action = self.policy.select_action(state, greedy=greedy)
        
        self.last_observation = observation
        self.last_action = action
        return action
        
    def update(self, observation, reward, terminated, truncated, call_back=None):
        """
        all arguments are batches
        """
        # reward = np.clip(reward, -1, 1) # for stability
        for i in range(self.num_envs):
            transition = get_single_observation(self.last_observation, i), self.last_action[i], reward[i]
            self.rollout_buffer[i].add(transition)
            
            if terminated[i] or truncated[i]:
                rollout = self.rollout_buffer[i].all()
                rollout_observations, rollout_actions, rollout_rewards = zip(*rollout)     
                n_step_return = calculate_n_step_returns(rollout_rewards, 0, self.hp.gamma)
                for j in range(len(self.rollout_buffer[i])):                    
                    trans = (rollout_observations[j], rollout_actions[j], get_single_observation(observation, i), \
                            n_step_return[j], terminated[i], \
                            truncated[i], self.hp.gamma**(len(self.rollout_buffer[i])-j))          
                    
                    self.replay_buffer.add(trans)
                self.rollout_buffer[i].clear() 
            
            elif self.rollout_buffer[i].is_full():
                rollout = self.rollout_buffer[i].all() 
                rollout_observations, rollout_actions, rollout_rewards = zip(*rollout)
                n_step_return = calculate_n_step_returns(rollout_rewards, 0, self.hp.gamma)
                trans = (rollout_observations[0], rollout_actions[0], get_single_observation(observation, i), \
                         n_step_return[0], terminated[i], \
                         truncated[i], self.hp.gamma**self.hp.n_steps)
                
                self.replay_buffer.add(trans)
     
        self.update_call_counter += 1
        if len(self.replay_buffer) >= self.hp.warmup_buffer_size and self.update_call_counter >= self.hp.update_freq:
            self.update_call_counter = 0
            batch = self.replay_buffer.sample(self.hp.batch_size)

            observations, actions, next_observations, n_step_return, terminated, truncated, effective_discount = zip(*batch)
            observations, next_observations = stack_observations(observations), stack_observations(next_observations)
            states, next_states = self.feature_extractor(observations), self.feature_extractor(next_observations)
            
            self.policy.update(states, actions, next_states, n_step_return, terminated, truncated, effective_discount, call_back=call_back)
        
        if call_back is not None:            
            call_back({
                "train/buffer_size": len(self.replay_buffer),
                })
        

    def reset(self, seed):
        """
        Reset the agent's learning state, including replay buffer.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        
        self.last_observation = None
        self.last_action = None

        self.replay_buffer.clear()
        self.replay_buffer.set_seed(seed)        
        self.rollout_buffer = [BaseBuffer(self.hp.n_steps) for _ in range(self.num_envs)]
        self.update_call_counter = 0
        
        
