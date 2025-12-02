"""
Collection of common layer initializations with guidance on when to use each.
"""

import math
import torch
import torch.nn as nn


def init_kaiming_normal(layer: nn.Module, nonlinearity: str = "relu", mode: str = "fan_out", bias_const: float = 0.0):
    """He/Kaiming normal: good default for ReLU/LeakyReLU conv/linear stacks."""
    nn.init.kaiming_normal_(layer.weight, mode=mode, nonlinearity=nonlinearity)
    if getattr(layer, "bias", None) is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


def init_kaiming_uniform(layer: nn.Module, nonlinearity: str = "relu", mode: str = "fan_in", bias_const: float = 0.0):
    """He/Kaiming uniform: similar to normal variant; works well with ReLU-family activations."""
    nn.init.kaiming_uniform_(layer.weight, mode=mode, nonlinearity=nonlinearity)
    if getattr(layer, "bias", None) is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


def init_xavier_uniform(layer: nn.Module, gain: float = 1.0, bias_const: float = 0.0):
    """Xavier/Glorot uniform: balanced fan-in/out; solid for tanh/sigmoid or linear heads."""
    nn.init.xavier_uniform_(layer.weight, gain=gain)
    if getattr(layer, "bias", None) is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


def init_xavier_normal(layer: nn.Module, gain: float = 1.0, bias_const: float = 0.0):
    """Xavier/Glorot normal: same use-cases as uniform; different variance scaling."""
    nn.init.xavier_normal_(layer.weight, gain=gain)
    if getattr(layer, "bias", None) is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


def init_lecun_normal(layer: nn.Module, bias_const: float = 0.0):
    """LeCun normal: pairs nicely with SELU/ELU activations (fan-in scaling)."""
    nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="linear")
    if getattr(layer, "bias", None) is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


def init_orthogonal(layer: nn.Module, gain: float = math.sqrt(2.0), bias_const: float = 0.0):
    """Orthogonal: preserves norms; helpful for RNNs or deep linear/conv stacks."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    if getattr(layer, "bias", None) is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


def init_dirac(layer: nn.Module, bias_const: float = 0.0):
    """Dirac: start convs as identity (useful for residual branches)."""
    nn.init.dirac_(layer.weight)
    if getattr(layer, "bias", None) is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


def init_trunc_normal(layer: nn.Module, std: float = 0.02, bias_const: float = 0.0):
    """Truncated normal: common for transformer embeddings/heads to keep logits small."""
    nn.init.trunc_normal_(layer.weight, std=std)
    if getattr(layer, "bias", None) is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


# Registry to map string keys to init functions.
INIT_REGISTRY = {
    "orthogonal": init_orthogonal,
    "kaiming_normal": init_kaiming_normal,
    "kaiming_uniform": init_kaiming_uniform,
    "xavier_uniform": init_xavier_uniform,
    "xavier_normal": init_xavier_normal,
    "lecun_normal": init_lecun_normal,
    "dirac": init_dirac,
    "trunc_normal": init_trunc_normal,
}

def apply_init(
    layer: nn.Module,
    name: str | None = None,
    bias_const: float = 0.0,
    std: float | None = None,
    gain: float | None = None,
    nonlinearity: str = "relu",
    mode: str = "fan_out",
):
    """
    Dispatch to a known initializer by name.

    Args:
        layer: module to initialize
        name: one of INIT_REGISTRY keys; defaults to DEFAULT_INIT
        bias_const: bias fill value
        std/gain/nonlinearity/mode: optional params forwarded to the chosen initializer
    """
    name = name.lower()
    if name not in INIT_REGISTRY:
        raise KeyError(f"Unknown init '{name}'. Available: {list(INIT_REGISTRY)}")

    if name == "orthogonal":
        return init_orthogonal(layer, gain=gain, bias_const=bias_const)
    if name == "kaiming_normal":
        return init_kaiming_normal(layer, nonlinearity=nonlinearity, mode=mode, bias_const=bias_const)
    if name == "kaiming_uniform":
        return init_kaiming_uniform(layer, nonlinearity=nonlinearity, mode=mode, bias_const=bias_const)
    if name == "xavier_uniform":
        return init_xavier_uniform(layer, gain=gain if gain is not None else 1.0, bias_const=bias_const)
    if name == "xavier_normal":
        return init_xavier_normal(layer, gain=gain if gain is not None else 1.0, bias_const=bias_const)
    if name == "lecun_normal":
        return init_lecun_normal(layer, bias_const=bias_const)
    if name == "dirac":
        return init_dirac(layer, bias_const=bias_const)
    if name == "trunc_normal":
        return init_trunc_normal(layer, std=std if std is not None else 0.02, bias_const=bias_const)

    # Fallback (should not hit thanks to earlier guard)
    return INIT_REGISTRY[name](layer)
