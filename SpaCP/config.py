from __future__ import annotations

import torch

__all__ = ["Config2"]

class Config2:
    def __init__(
        self,
        *,
        n_neighbors=20,        # Neighbor count
        d_model=960,           # Feature dimension
        d_edge_model=512,      # Edge feature dimension
        n_genes=768,           # Gene count (matches N in 2D input)
        n_heads=8,             # Attention heads
        act="swiglu",          # Activation
        attn_dropout=0.1,      # Attention dropout
        dropout=0.1,           # Projection dropout
        n_layers=3,            # Transformer layers
        gene_exp_non_negative=True,  # Enforce non-negative gene expression
        d_attr=3,
        device=None,
    ):
        self.n_neighbors = n_neighbors
        self.d_model = d_model
        self.d_edge_model = d_edge_model
        self.n_genes = n_genes
        self.n_heads = n_heads
        self.act = act
        self.attn_dropout = attn_dropout
        self.dropout = dropout
        self.n_layers = n_layers
        self.gene_exp_non_negative = gene_exp_non_negative
        self.d_attr = d_attr
        self.device = device or torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
