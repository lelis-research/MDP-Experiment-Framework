import torch

from . import POLICY_REGISTRY, AGENT_REGISTRY

def calculate_n_step_returns(rollout_rewards, bootstrap_value, gamma):
    # Initialize the return with the bootstrap value.
    returns = []
    G = bootstrap_value
    # Compute discounted returns backwards.
    for r in reversed(rollout_rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

def load_policy(file_path):
    checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
    policy_class_name = checkpoint.get("policy_class")
    if policy_class_name is None:
        raise ValueError("Checkpoint does not contain policy class information.")
    policy_cls = POLICY_REGISTRY.get(policy_class_name)
    if policy_cls is None:
        raise ValueError(f"Unknown policy class: {policy_class_name}")
    return policy_cls.load_from_file(file_path)


def load_agent(file_path):
    checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
    agent_class_name = checkpoint.get("agent_class")
    if agent_class_name is None:
        raise ValueError("Checkpoint does not contain agent class information.")
    agent_cls = AGENT_REGISTRY.get(agent_class_name)
    if agent_cls is None:
        raise ValueError(f"Unknown agent class: {agent_class_name}")
    return agent_cls.load_from_file(file_path)