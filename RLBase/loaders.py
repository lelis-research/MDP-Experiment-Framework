import torch
from .registry import OPTION_REGISTRY, POLICY_REGISTRY, AGENT_REGISTRY, FEATURE_EXTRACTOR_REGISTRY

def load_option(file_path, checkpoint=None):
    if checkpoint is None:
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
    option_class_name = checkpoint.get("option_class")
    if option_class_name is None:
        raise ValueError("Checkpoint does not contain option class information.")
    option_cls = OPTION_REGISTRY.get(option_class_name)
    if option_cls is None:
        raise ValueError(f"Unknown policy class: {option_class_name}")
    return option_cls.load_from_file(file_path, checkpoint=checkpoint)


def load_policy(file_path, checkpoint=None):
    if checkpoint is None:
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
    policy_class_name = checkpoint.get("policy_class")
    if policy_class_name is None:
        raise ValueError("Checkpoint does not contain policy class information.")
    policy_cls = POLICY_REGISTRY.get(policy_class_name)
    if policy_cls is None:
        raise ValueError(f"Unknown policy class: {policy_class_name}")
    return policy_cls.load_from_file(file_path, checkpoint=checkpoint)


def load_agent(file_path, checkpoint=None):
    if checkpoint is None:
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
    agent_class_name = checkpoint.get("agent_class")
    if agent_class_name is None:
        raise ValueError("Checkpoint does not contain agent class information.")
    agent_cls = AGENT_REGISTRY.get(agent_class_name)
    if agent_cls is None:
        raise ValueError(f"Unknown agent class: {agent_class_name}")
    return agent_cls.load_from_file(file_path, checkpoint=checkpoint)

def load_feature_extractor(file_path, checkpoint=None):
    if checkpoint is None:
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
    feature_extractor_class_name = checkpoint.get("feature_extractor_class")
    if feature_extractor_class_name is None:
        raise ValueError("Checkpoint does not contain feature extractor class information.")
    feature_extractor_cls = FEATURE_EXTRACTOR_REGISTRY.get(feature_extractor_class_name)
    if feature_extractor_cls is None:
        raise ValueError(f"Unknown feature extractor class: {feature_extractor_class_name}")
    return feature_extractor_cls.load_from_file(file_path, checkpoint=checkpoint)