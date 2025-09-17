import numpy as np
import torch

def calculate_n_step_returns(rollout_rewards, bootstrap_value, gamma):
    # Initialize the return with the bootstrap value.
    returns = []
    G = bootstrap_value
    # Compute discounted returns backwards.
    for r in reversed(rollout_rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

def calculate_gae(
    rollout_rewards,         # list of length T
    values,          # tensor [T] of V(s_t)
    next_values,     # tensor [T] of V(s_{t+1})
    dones,           # list of bools length T
    gamma, lamda,      # γ and λ
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


def calculate_gae_torch(rewards, values, next_values, dones, gamma, lam):
    """
    rewards: list[float] or 1D tensor [T]
    values, next_values: 1D tensors [T] (critic outputs)
    dones: list[bool] of length T
    Returns (returns, advantages) as 1D tensors on values.device with no grad.
    """
    if not torch.is_tensor(values):
        values = torch.tensor(values, dtype=torch.float32)
    if not torch.is_tensor(next_values):
        next_values = torch.tensor(next_values, dtype=torch.float32)

    device = values.device
    T = len(rewards)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    dones_t   = torch.as_tensor(dones,   dtype=torch.float32, device=device)  # 1.0 if done else 0.0
    mask_t    = 1.0 - dones_t

    advantages = torch.zeros(T, dtype=torch.float32, device=device)
    gae = torch.zeros((), dtype=torch.float32, device=device)

    with torch.no_grad():
        for t in reversed(range(T)):
            delta = rewards_t[t] + gamma * next_values[t] * mask_t[t] - values[t]
            gae   = delta + gamma * lam * mask_t[t] * gae
            advantages[t] = gae
        returns = advantages + values

    return returns, advantages

def keep_batch(x, i):
    '''
    get index i element of input x but keep the batch dimension
    '''
    if isinstance(x, dict):
        return {k: keep_batch(v, i) for k, v in x.items()}
    if isinstance(x, (np.ndarray, torch.Tensor)):
        return x[i:i+1]  # preserves batch dim
    return x  # scalars or other types, just return as is