import numpy as np

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