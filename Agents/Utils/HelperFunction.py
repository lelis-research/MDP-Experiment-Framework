def calculate_n_step_returns(rollout_rewards, bootstrap_value, gamma):
    returns = []
    G = bootstrap_value
    for r in reversed(rollout_rewards):
        G = r + gamma * G
        # returns.append(G)
        returns.insert(0, G)
    return returns