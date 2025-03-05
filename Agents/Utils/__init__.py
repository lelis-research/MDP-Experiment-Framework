POLICY_REGISTRY = {}
AGENT_REGISTRY = {}

def register_policy(cls):
    POLICY_REGISTRY[cls.__name__] = cls
    return cls

def register_agent(cls):
    AGENT_REGISTRY[cls.__name__] = cls
    return cls

from .BaseAgent import BaseAgent, BasePolicy
from .Buffer import *
from .FeatureExtractor import *
from .HelperFunctions import *
from .HyperParams import HyperParameters
from .NetworkGenerator import NetworkGen, prepare_network_config
from .Option import MaskedOption