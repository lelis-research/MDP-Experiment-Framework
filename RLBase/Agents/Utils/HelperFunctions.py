import numpy as np
import torch
from typing import Any, Dict, List, Union

ObsType = Union[Dict[str, Any], np.ndarray, torch.Tensor]

def get_single_observation(observation: ObsType, index: int) -> ObsType:
    """
    Extract a single observation from a batched observation.
    - If observation is a dict: return a dict with the same keys.
    - Slices arrays/tensors to keep batch dimension = 1.
    - For non-array values (e.g., int, None), returns them unchanged.
    
    NOTE: Assumes observation has a batch dimension as the first dimension and it will keep the batch dimension in the output.
    """

    def slice_one(x):
        """Slice batch dimension and keep it."""
        if isinstance(x, (np.ndarray, torch.Tensor)):
            # Add back batch dim = 1
            return x[index:index+1]
        else:
            raise NotImplementedError(
            f"Dict element type {type(x)} is not supported."
        )

    # ---- Handle dict observation ----
    if isinstance(observation, dict):
        return {k: slice_one(v) for k, v in observation.items()}

    # ---- Handle array / tensor observation ----
    if isinstance(observation, (np.ndarray, torch.Tensor)):
        return observation[index:index+1]

    # ---- Fallback ----
    raise NotImplementedError(
        f"Vectorized observation type {type(observation)} is not supported."
    )
        
def stack_observations(observations: List[ObsType]) -> ObsType:
    """
    Stack a list of observations into a batched observation.

    Supports:
    - dict[str, np.ndarray / torch.Tensor]
    - np.ndarray
    - torch.Tensor

    Assumes all observations in the list have the same structure and shapes.
    
    NOTE: This function assumes that the input observations are already batched (i.e. batch size of 1 or more)
    """

    if len(observations) == 0:
        raise ValueError("stack_observations got an empty list.")

    first = observations[0]

    # ---- Dict observation: stack per key ----
    if isinstance(first, dict):
        stacked: Dict[str, Any] = {}
        for k in first.keys():
            vals = [obs[k] for obs in observations]

            # All vals must be same type
            v0 = vals[0]
            if isinstance(v0, torch.Tensor):
                stacked[k] = torch.cat(vals, dim=0)
            elif isinstance(v0, np.ndarray):
                stacked[k] = np.concatenate(vals, axis=0)
            else:
                raise NotImplementedError(
                    f"Dict value type {type(v0)} for key '{k}' is not supported."
                )
        return stacked

    # ---- Tensor observation ----
    if isinstance(first, torch.Tensor):
        return torch.cat(observations, dim=0)

    # ---- Numpy observation ----
    if isinstance(first, np.ndarray):
        return np.concatenate(observations, axis=0)

    # ---- Fallback ----
    raise NotImplementedError(
        f"Observation type {type(first)} is not supported in stack_observations."
    )
    
    

def calculate_n_step_returns(rollout_rewards, bootstrap_value, gamma):
    # Initialize the return with the bootstrap value.
    returns = []
    G = bootstrap_value
    # Compute discounted returns backwards.
    for r in reversed(rollout_rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

def calculate_n_step_returns_with_discounts(rollout_rewards, bootstrap_value, discounts):
    """
    Compute discounted n-step returns given per-step or per-option discounts.

    Args:
        rollout_rewards (list[float]): Rewards for each step/option.
        bootstrap_value (float): Bootstrap value for the final state.
        discounts (list[float]): Discount factors for each step/option,
                                 e.g. [γ_1, γ_2, ..., γ_T]

    Returns:
        list[float]: Discounted returns aligned with rollout_rewards.
    """
    assert len(rollout_rewards) == len(discounts), \
        "rollout_rewards and discounts must have the same length."

    returns = []
    G = bootstrap_value
    # Walk backward through rewards/discounts
    for r, d in zip(reversed(rollout_rewards), reversed(discounts)):
        G = r + d * G
        returns.insert(0, G)
    return returns



def calculate_gae(
    rollout_rewards,         # list of length T
    values,          # tensor [T] of V(s_t)
    next_values,     # tensor [T] of V(s_{t+1})
    dones,           # list of bools length T
    gamma, 
    lamda,      # γ and λ
):
    T = len(rollout_rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns    = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        delta = rollout_rewards[t] + gamma * next_values[t].item() * mask - values[t].item()
        gae   = delta + gamma * lamda * mask * gae
        advantages[t] = gae
        returns[t] = gae + values[t].item()
    
    return returns, advantages

def calculate_gae_with_discounts(
    rollout_rewards,     # list/np.array length T
    values,              # torch tensor [T]
    next_values,         # torch tensor [T]
    dones,               # list/np.array[bool], True = true termination (no bootstrap)
    discounts,           # list/np.array length T, e.g., γ for primitive, γ**τ for options
    lamda,               # λ
):
    T = len(rollout_rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns    = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])  # 0 if true terminal, 1 otherwise
        d    = float(discounts[t])     # per-transition discount
        delta = rollout_rewards[t] + d * next_values[t].item() * mask - values[t].item()
        gae   = delta + d * lamda * mask * gae
        advantages[t] = gae
        returns[t]    = gae + values[t].item()
    return returns, advantages
