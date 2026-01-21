import numpy as np
import torch
from gymnasium.spaces import Discrete

from ..Base import BaseAgent, BasePolicy
from ...Buffers import BaseBuffer, ReplayBuffer
from ..Utils import calculate_n_step_returns, get_single_observation, stack_observations
from ...registry import register_agent, register_policy


@register_policy
class QLearningPolicy(BasePolicy): 
    def __init__(self, action_space, hyper_params=None, device='cpu'):
        super().__init__(action_space, hyper_params, device)
        
        self.q_table = {}
        self.epsilon = self.hp.epsilon_start
        self.epsilon_step_counter = 0
        
        
    def select_action(self, state, greedy=False):
        self.epsilon_step_counter += 1 
            
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_dim)
        
        if self._rand_float(low=0, high=1) < self.epsilon and not greedy:
            action = self._rand_int(0, self.action_dim)
        else:
            q_values = self.q_table[state]
            max_actions = np.flatnonzero(q_values == np.max(q_values))
            action = int(self._rand_elem(max_actions))
        return action

    
    def update(self, last_state, last_action, state, target_reward, terminated, truncated, effective_discount, call_back=None):
        """
        Update the Q-table using the Q-Learning update rule.
        
        Args:
            last_state (hashable): Previous state (encoded) where the last action was taken.
            last_action (int): Action taken in the previous state.
            state (hashable): Current state (encoded) after action.
            reward (float): Reward received.
            terminated (bool): True if the episode has terminated.
            truncated (bool): True if the episode was truncated.
            call_back (function, optional): Callback for tracking training progress.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_dim)
        if last_state not in self.q_table:
            self.q_table[last_state] = np.zeros(self.action_dim)

    
        bootstrap_value = np.max(self.q_table[state]) if not terminated else 0.0
        td_target = float(target_reward) + float(effective_discount) * bootstrap_value
        td_error = td_target - self.q_table[last_state][last_action]
        self.q_table[last_state][last_action] += self.hp.step_size * td_error
            
        #Update Epsilon
        frac = 1.0 - (self.epsilon_step_counter / self.hp.epsilon_decay_steps)
        frac = max(0.0, frac)
        self.epsilon = max(self.hp.epsilon_end, self.hp.epsilon_end + (self.hp.epsilon_start - self.hp.epsilon_end) * frac)
        
        
        if call_back is not None:         
            call_back({
                "train/td_error": td_error,
                "train/epsilon": self.epsilon,
                "train/q_mean": np.mean(self.q_table[last_state]),
                "train/q_max": np.max(self.q_table[last_state]),
                "train/td_target": td_target,
            })
        
    def reset(self, seed):
        """
        Reset the Q-table and seed random generators.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        self.q_table = {}
        self.epsilon = self.hp.epsilon_start
        self.epsilon_step_counter = 0

    def save(self, file_path=None):
        """
        Save the Q-table and hyper-parameters.
        
        Args:
            file_path (str): File path for saving.
        """
        
        checkpoint = super().save(file_path=None)
        checkpoint['q_table'] = self.q_table
        checkpoint['epsilon'] = self.epsilon
        checkpoint['epsilon_step_counter'] = self.epsilon_step_counter

        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_policy.t")
        return checkpoint

    @classmethod
    def load(cls, file_path, checkpoint=None):
        """
        Load the Q-table and hyper-parameters.
        
        Args:
            file_path (str): File path from which to load the checkpoint.
        """
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
            
        instance = super().load(file_path, checkpoint)        
        instance.q_table = checkpoint.get('q_table')
        instance.epsilon = checkpoint.get('epsilon')
        instance.epsilon_step_counter = checkpoint.get('epsilon_step_counter')
        
        return instance
    

@register_agent        
class QLearningAgent(BaseAgent):
    """
    Hyper-params:
        "step_size": 0.1,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 40000,
        "n_steps": 3,

        "replay_buffer_size": 256,
        "batch_size": 1,
        "warmup_buffer_size": 128,
    """
    name = "QLearning"
    SUPPORTED_ACTION_SPACES = (Discrete, )
    
    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class):
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class)
        self.policy = QLearningPolicy(action_space, hyper_params)
        self.rollout_buffer = [BaseBuffer(capacity=self.hp.n_steps) for _ in range(self.num_envs)]  # Buffer is used for n-step
        
        if self.hp.replay_buffer_size is not None:
            self.replay_buffer = ReplayBuffer(self.hp.replay_buffer_size)
        else:
            self.replay_buffer = None
        
    def act(self, observation):
        """
        Select an action based on the observation.
        observation is a batch
        action is a batch 
        """
        state = self.feature_extractor(observation) 
        action = []
        for i in range(self.num_envs):
            st = state[i]
            action.append(self.policy.select_action(st, greedy=not self.training))
        
        self.last_action = action
        self.last_state = state
        return action
                
    def update(self, observation, reward, terminated, truncated, call_back=None):
        if self.training:
            state = self.feature_extractor(observation)
            for i in range(self.num_envs):
                st = state[i]
                transition = self.last_state[i], self.last_action[i], reward[i]
                self.rollout_buffer[i].add(transition)
                
                if terminated[i] or truncated[i]:
                    rollout = self.rollout_buffer[i].all()
                    rollout_states, rollout_actions, rollout_rewards = zip(*rollout)
                    n_step_return = calculate_n_step_returns(rollout_rewards, 0, self.hp.gamma)
                    for j in range(len(self.rollout_buffer[i])):
                        self.policy.update(
                            rollout_states[j], 
                            rollout_actions[j], 
                            st, 
                            n_step_return[j], 
                            terminated[i], 
                            truncated[i], 
                            self.hp.gamma**(len(self.rollout_buffer[i])-j), 
                            call_back=call_back
                        )
                        if self.replay_buffer is not None:
                            trans = (
                                rollout_states[j], 
                                rollout_actions[j], 
                                st, 
                                n_step_return[j], 
                                terminated[i], 
                                truncated[i], 
                                self.hp.gamma**(len(self.rollout_buffer[i])-j)
                            )
                            self.replay_buffer.add(trans)
                    self.rollout_buffer[i].clear() 
                
                elif self.rollout_buffer[i].is_full():
                    rollout = self.rollout_buffer[i].all() 
                    rollout_states, rollout_actions, rollout_rewards = zip(*rollout)
                    n_step_return = calculate_n_step_returns(rollout_rewards, 0, self.hp.gamma)
                    self.policy.update(
                        rollout_states[0], 
                        rollout_actions[0], 
                        st, 
                        n_step_return[0],
                        terminated[i], 
                        truncated[i], 
                        self.hp.gamma**self.hp.n_steps, 
                        call_back=call_back
                    )
                    if self.replay_buffer is not None:
                            trans = (
                                rollout_states[0], 
                                rollout_actions[0], 
                                st, 
                                n_step_return[0], 
                                terminated[i], 
                                truncated[i], 
                                self.hp.gamma**self.hp.n_steps
                            )
                            self.replay_buffer.add(trans)
                            
            
            if self.replay_buffer is not None and len(self.replay_buffer) >= self.hp.warmup_buffer_size:
                batch = self.replay_buffer.sample(self.hp.batch_size)

                states, actions, next_states, n_step_return, terminated, truncated, effective_discount = zip(*batch)  

                for i in range(len(batch)):       
                    self.policy.update(states[i], actions[i], next_states[i], 
                                    n_step_return[i], terminated[i], truncated[i], effective_discount[i], call_back=call_back)
        
            
             
                
    
    def reset(self, seed):
        super().reset(seed)

        self.last_state = None
        self.last_action = None
        self.rollout_buffer = [BaseBuffer(capacity=self.hp.n_steps) for _ in range(self.num_envs)] 
        if self.replay_buffer is not None:
            self.replay_buffer.clear()
            self.replay_buffer.set_seed(seed)
        
