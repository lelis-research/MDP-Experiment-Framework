import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from gymnasium.spaces import Discrete

from ...Utils import (
    BaseAgent,
    BasePolicy,
    BasicBuffer,
    calculate_gae,
    calculate_gae_with_discounts,
    NetworkGen,
    prepare_network_config,
)
from .A2C import A2CAgent, A2CPolicyDiscrete
from ....registry import register_agent, register_policy
from ....Options.Utils import load_options_list, save_options_list

@register_policy
class OptionA2CPolicyDiscrete(A2CPolicyDiscrete):
    def update(self, states, actions, rewards, next_states, dones, discounts=None, call_back=None):
        states_t = torch.cat(states).to(dtype=torch.float32, device=self.device) if torch.is_tensor(states[0]) else torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(np.array(actions), dtype=torch.int64, device=self.device)
        next_states_t = torch.cat(next_states).to(dtype=torch.float32, device=self.device) if torch.is_tensor(next_states[0]) else torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        
        values = self.critic(states_t).squeeze(-1)

        with torch.no_grad():
            next_values = self.critic(next_states_t).squeeze(-1)

       
        # numpy array to align with calculate_gae_with_discounts
        discounts_np = np.asarray(discounts, dtype=np.float32)

        # ---- GAE with per-transition discounts ----
        # returns, advantages = calculate_gae_with_discounts(
        #     rewards,
        #     values,
        #     next_values,
        #     dones,
        #     discounts_np,
        #     lamda=self.hp.lamda,
        # )
        
        returns, advantages = calculate_gae(
            rewards,
            values,
            next_values,
            dones,
            self.hp.gamma,
            lamda=self.hp.lamda,
        )

        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t    = torch.tensor(returns,    dtype=torch.float32, device=self.device).unsqueeze(1)

        if getattr(self.hp, "norm_adv_flag", False):
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        logits = self.actor(states_t)
        dist   = Categorical(logits=logits)
        log_probs_t   = dist.log_prob(actions_t)
        entropy = dist.entropy()

        critic_loss = F.mse_loss(values.unsqueeze(1), returns_t)
        actor_loss  = -(log_probs_t * advantages_t).mean()   # mean is more stable than sum across batch
        loss = actor_loss + critic_loss - self.hp.entropy_coef * entropy.mean()

        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        self.actor_optimizer.step()

        if call_back is not None:
            call_back({"critic_loss": float(critic_loss.item()),
                       "actor_loss": float(actor_loss.item())})

@register_agent
class OptionA2CAgent(BaseAgent):
    """
    Advantage Actor-Critic agent that can choose between primitive actions and options.
    Options execute low-level actions internally until termination; the agent treats each
    option execution as a single (macro) transition for learning.
    """
    name = "OptionA2C"

    def __init__(self, action_space, observation_space, hyper_params, num_envs, feature_extractor_class, options_lst, device="cpu"):
        """
        Args:
            action_space (gym.spaces.Discrete): The environment's primitive action space.
            observation_space: The environment's observation space.
            hyper_params: Hyper-parameters container (see OptionA2CPolicy).
            num_envs (int): Number of parallel environments.
            feature_extractor_class: Class to extract features from observations.
            options_lst (list): List of option objects, each implementing:
                                - select_action(observation)
                                - is_terminated(observation) -> bool
            device (str): Torch device.
        """
        super().__init__(action_space, observation_space, hyper_params, num_envs, feature_extractor_class, device=device)

        self.atomic_action_space = action_space
        self.options_lst = options_lst
        print(f"Number of options: {len(options_lst)}")

        # Augment action space with options
        action_option_space = Discrete(self.atomic_action_space.n + len(self.options_lst))

        # Policy over (actions + options)
        self.policy = OptionA2CPolicyDiscrete(
            action_option_space,
            self.feature_extractor.features_dim,
            hyper_params,
            device=device
        )

        # Rollout buffer stores transitions: (state, action_or_option_idx, reward, next_state, done)
        self.rollout_buffer = BasicBuffer(np.inf)

        # Option execution bookkeeping
        self.running_option_index = None            # index within options_lst
        self.option_start_state = None              # features at option initiation
        self.option_cumulative_reward = None        # discounted cumulative reward during option
        self.option_multiplier = None               # running gamma^k multiplier

    def act(self, observation, greedy=False):
        """
        Select a primitive action. If currently executing an option, return the option's next action.
        Otherwise, sample from the policy over (actions + options). If an option is chosen, initialize
        bookkeeping and return the option's first primitive action.
        
        Args:
            observation (np.array or similar): Raw observation.
            
        Returns:
            int: Primitive action to execute in the environment.
        """
        state = self.feature_extractor(observation)

        # If an option is currently running, check termination and continue if not done.
        if self.running_option_index is not None:
            if self.options_lst[self.running_option_index].is_terminated(observation):
                # Terminated at the start of this step; clear and fall through to pick again.
                self.running_option_index = None
            else:
                action = self.options_lst[self.running_option_index].select_action(observation)
        
        if self.running_option_index is None:
            # No running option: select (action or option) from policy
            action = self.policy.select_action(state, greedy=greedy)
            if action >= self.atomic_action_space.n:
                # Option selected
                self.running_option_index = action - self.atomic_action_space.n
                self.option_start_state = state
                self.option_cumulative_reward = 0.0
                self.option_multiplier = 1.0
                # Get the option's first primitive action
                action = self.options_lst[self.running_option_index].select_action(observation)
        
                
        self.last_state = state
        self.last_action = action

        return action

    def update(self, observation, reward, terminated, truncated, call_back=None):
        """
        Collect transitions. Primitive steps are stored directly. If executing an option,
        accumulate discounted rewards until the option terminates (or episode ends), then
        store a single macro transition for the whole option execution.
        
        When the number of stored transitions reaches `hp.rollout_steps`, perform an A2C update.
        
        Args:
            observation (np.array or similar): New observation after the executed primitive action.
            reward (float): Reward received.
            terminated (bool): True if episode terminated.
            truncated (bool): True if episode was truncated.
            call_back (function, optional): Callback to track losses.
        """
        state = self.feature_extractor(observation)

        if self.running_option_index is not None:
            # Accumulate discounted reward while the option is executing
            self.option_cumulative_reward += self.option_multiplier * reward
            self.option_multiplier *= self.hp.gamma
            
            # Check option termination (or env termination/truncation)
            if terminated or truncated or self.options_lst[self.running_option_index].is_terminated(observation):
                # Build a single macro transition for the option execution
                effective_discount = self.option_multiplier
                transition = (
                    self.option_start_state,
                    self.running_option_index + self.atomic_action_space.n,
                    self.option_cumulative_reward,
                    state,
                    terminated,
                    effective_discount,
                )
                self.rollout_buffer.add_single_item(transition)

                # Clear option bookkeeping
                self.running_option_index = None
                self.option_start_state = None
                self.option_cumulative_reward = None
                self.option_multiplier = None
        else:
            # Primitive action: store single-step transition
            transition = (self.last_state, self.last_action, reward, state, terminated, self.hp.gamma)
            self.rollout_buffer.add_single_item(transition)

        # If rollout length reached, perform update
        if self.rollout_buffer.size >= self.hp.rollout_steps:
            rollout = self.rollout_buffer.get_all()
            states, actions, rewards, next_states, dones, discounts = zip(*rollout)

            self.policy.update(states, actions, rewards, next_states, dones, discounts, call_back=call_back)
            self.rollout_buffer.reset()

    def reset(self, seed):
        """
        Reset the agent's state, including feature extractor and rollout buffer.
        
        Args:
            seed (int): Seed for reproducibility.
        """
        super().reset(seed)
        self.feature_extractor.reset(seed)
        self.rollout_buffer.reset()
        # Clear any running option state
        self.running_option_index = None
        self.option_start_state = None
        self.option_cumulative_reward = None
        self.option_multiplier = None

    def save(self, file_path=None):
        """
        Save agent checkpoint, including options and atomic action space,
        alongside the base components saved by BaseAgent.
        """
        checkpoint = super().save(file_path=None)
        options_checkpoint = save_options_list(self.options_lst, file_path=None)

        checkpoint['options_lst'] = options_checkpoint
        checkpoint['atomic_action_space'] = self.atomic_action_space

        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_agent.t")
        return checkpoint

    @classmethod
    def load_from_file(cls, file_path, seed=0, checkpoint=None):
        """
        Restore an OptionA2CAgent from file. Expects the BaseAgent checkpoint format
        plus 'options_lst' and 'atomic_action_space'.
        """
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        options_lst = load_options_list(file_path=None, checkpoint=checkpoint['options_lst'])

        instance = cls(
            checkpoint['atomic_action_space'],
            checkpoint['observation_space'],
            checkpoint['hyper_params'],
            checkpoint['num_envs'],
            checkpoint['feature_extractor_class'],
            options_lst
        )
        instance.reset(seed)

        instance.feature_extractor.load_from_checkpoint(checkpoint['feature_extractor'])
        instance.policy.load_from_checkpoint(checkpoint['policy'])

        return instance