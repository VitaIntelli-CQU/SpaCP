from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from scvi.nn import FCLayers

from utils import buildNetwork, init_weights

__all__ = [
    "DenseEncoder",
    "LordEncoder",
    "RegularizedEmbedding",
    "TimestepEmbedder",
]

class DenseEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation="relu", dropout=0, dtype=torch.float32, norm="batchnorm"):
        super(DenseEncoder, self).__init__()
        self.layers = buildNetwork([input_dim]+hidden_dims, network="decoder", activation=activation, dropout=dropout, dtype=dtype, norm=norm)
        self.enc_mu = nn.Linear(hidden_dims[-1], output_dim)
        self.enc_var = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        h = self.layers(x)
        mu = self.enc_mu(h)
        var = torch.exp(self.enc_var(h).clamp(-15, 15))
        return mu, var

class RegularizedEmbedding(nn.Module):
    """Embedding layer with optional Gaussian noise for regularization."""

    def __init__(self, n_input: int, n_output: int, sigma: float = 0.0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=n_input, embedding_dim=n_output)
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.embedding(x)
        if self.training and self.sigma > 0.0:
            noise = torch.zeros_like(out).normal_(mean=0, std=self.sigma)
            out = out + noise
        return out

class LordEncoder(nn.Module):
    """
    LORD encoder for attribute-aware latent representations.

    Parameters
    ----------
    embedding_dim : List[int]
        Dimensions for each covariate embedding (order matches attributes).
    num_genes : int
        Number of genes in the dataset.
    attributes : List[str]
        Names of covariates (e.g., ["tissue", "perturbation"]).
    attributes_type : List[str]
        Types of covariates ("categorical" or "ordinal").
    labels : np.ndarray
        Matrix of covariate labels, shape (n_cells, n_covariates).
    device : str
        Device to place the model on.
    noise : float
        Standard deviation of Gaussian noise for basal embedding.
    append_layer_width : Optional[int]
        Override number of genes if using appended layers.
    """

    def __init__(
        self,
        embedding_dim: List[int],
        num_genes: int,
        attributes: List[str],
        attributes_type: List[str],
        labels: np.ndarray,
        device: str = "cuda:0",
        append_layer_width: Optional[int] = None,
        multi_task: bool = False,
        noise: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_genes = append_layer_width or num_genes
        self.device = device
        self.multi_task = multi_task
        self.noise = noise

        self.labels = labels
        self.num_covariates = labels.shape[1]
        logging.info("Number of covariates: %d", self.num_covariates)
        assert self.num_covariates < 10, "Too many covariates"

        # Track unique values per covariate
        self.unique_labels = [np.unique(labels[:, i]) for i in range(self.num_covariates)]
        self.attributes = attributes
        self.attributes_type = attributes_type
        self.attribute_type_map = {a: t for a, t in zip(attributes, attributes_type)}
        self.embedding_dim_map = {a: d for a, d in zip(attributes, embedding_dim)}

        # Build encoders for each covariate
        self.s_encoder = nn.ModuleDict()
        for i, attr in enumerate(attributes):
            if attributes_type[i] == "categorical":
                self.s_encoder[attr] = nn.Embedding(len(self.unique_labels[i]), embedding_dim[i])
            elif attributes_type[i] == "ordinal":
                self.s_encoder[attr] = FCLayers(
                    n_in=1,
                    n_out=embedding_dim[i],
                    n_layers=3,
                    n_hidden=128,
                    dropout_rate=0.3,
                    bias=False,
                    use_activation=True,
                )

        # Basal embedding per sample index
        self.z_encoder = RegularizedEmbedding(
            n_input=len(labels[:, 0]),
            n_output=embedding_dim[-1],
            sigma=self.noise,
        )

        # Init
        self.apply(init_weights)
        self.to(self.device)
        self.history = {"epoch": [], "stats_epoch": []}

    # -----------------------------------------------------------------
    # Forward utilities
    # -----------------------------------------------------------------
    def predict(
        self,
        sample_indices: torch.Tensor,
        batch_size: int,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute attribute-aware latent representations.

        Returns dict with:
            total_latent : concatenated basal + attribute embeddings
            basal_latent : basal embedding only
            {attr}: individual attribute embeddings
        """
        # Convert labels into dict by covariate
        label_dict = {a: labels[:, i] for i, a in enumerate(self.attributes)}

        # Basal embedding
        z = self.z_encoder(sample_indices)
        if torch.isnan(z).any():
            logging.warning("NaN detected in basal embedding")

        # Attribute-specific embeddings
        s: Dict[str, torch.Tensor] = {}
        for attr, encoder in self.s_encoder.items():
            if self.attribute_type_map[attr] == "ordinal":
                input_ = label_dict[attr].unsqueeze(1).float()
                s_val = encoder(input_)
            else:
                s_val = encoder(label_dict[attr])
            s_val = s_val.view(batch_size, self.embedding_dim_map[attr])
            s[attr] = s_val
            if torch.isnan(s_val).any():
                logging.warning("NaN detected in %s embedding", attr)

        # Concatenate all latents
        latent_vecs = [z] + [s[attr] for attr in self.attributes]
        latent = torch.cat(latent_vecs, dim=-1)

        return {
            "total_latent": latent,
            "basal_latent": z,
            **s,
        }

    def get_latent(self, att: torch.Tensor, type_: str, batch_size: int) -> torch.Tensor:
        """Get latent representation for a specific attribute."""
        if type_ not in self.s_encoder:
            raise ValueError(f"Unknown attribute type {type_}")
        out = self.s_encoder[type_](att)
        #repeat out for batch_size times
        out = out.repeat(batch_size, 1)
        return out

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    time emb to frequency_embedding_size dim, then to hidden_size
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[..., None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
# Config2
