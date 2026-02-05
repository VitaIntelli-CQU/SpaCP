from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["init_weights", "buildNetwork", "get_activation"]

def init_weights(m: nn.Module) -> None:
    """Initialize linear layers with Xavier init."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def buildNetwork(layers, network="decoder", activation="relu", dropout=0., dtype=torch.float32, norm="batchnorm"):
    net = []
    if network == "encoder" and dropout > 0:
        net.append(nn.Dropout(p=dropout))
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if norm == "batchnorm":
            net.append(nn.BatchNorm1d(layers[i]))
        elif norm == "layernorm":
            net.append(nn.LayerNorm(layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        elif activation=="elu":
            net.append(nn.ELU())
        if dropout > 0:
            net.append(nn.Dropout(p=dropout))
    return nn.Sequential(*net)

def get_activation(activation="gelu"):
    return {
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "relu": nn.ReLU,
    }[activation]
# TimeEmbedder
