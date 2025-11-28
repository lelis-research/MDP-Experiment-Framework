POLICY_REGISTRY = {}
AGENT_REGISTRY = {}
OPTION_REGISTRY = {}
FEATURE_EXTRACTOR_REGISTRY = {}

def register_policy(cls):
    POLICY_REGISTRY[cls.__name__] = cls
    return cls

def register_agent(cls):
    AGENT_REGISTRY[cls.__name__] = cls
    return cls

def register_option(cls):
    OPTION_REGISTRY[cls.__name__] = cls
    return cls

def register_feature_extractor(cls):
    FEATURE_EXTRACTOR_REGISTRY[cls.__name__] = cls
    return cls