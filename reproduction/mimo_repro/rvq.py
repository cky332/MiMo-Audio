"""Residual vector quantization with EMA codebook updates.

Structurally equivalent to the official src/mimo_audio_tokenizer/quantization.py
(which is the EnCodec/SoundStream recipe): per-layer Euclidean codebooks with
exponential-moving-average updates, straight-through estimator and a
commitment loss.  Kept minimal: no k-means init and no distributed hooks;
dead-code expiry is optional.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class EMACodebook(nn.Module):
    def __init__(self, dim: int, codebook_size: int, decay: float = 0.99, epsilon: float = 1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.decay = decay
        self.epsilon = epsilon
        embed = torch.empty(codebook_size, dim)
        nn.init.kaiming_uniform_(embed)
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("cluster_size", torch.zeros(codebook_size))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, dim] -> indices [N]
        dist = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ self.embed.t()
            + self.embed.pow(2).sum(1)[None, :]
        )
        return dist.argmin(dim=-1)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.embed)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        indices = self.encode(x)
        quantized = self.decode(indices)
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(indices, self.codebook_size).type_as(x)
                self.cluster_size.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
                embed_sum = onehot.t() @ x
                self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                smoothed = (
                    (self.cluster_size + self.epsilon)
                    / (n + self.codebook_size * self.epsilon)
                    * n
                )
                self.embed.copy_(self.embed_avg / smoothed.unsqueeze(1))
        return quantized, indices


class ResidualVQ(nn.Module):
    """RVQ over a list of codebook sizes (paper: 20 layers, 2x1024 + 18x128)."""

    def __init__(self, dim: int, codebook_sizes, decay: float = 0.99):
        super().__init__()
        self.layers = nn.ModuleList(
            EMACodebook(dim, size, decay=decay) for size in codebook_sizes
        )

    def forward(self, x: torch.Tensor, n_q: int | None = None):
        """Returns (quantized, indices [n_q, N], commit_loss)."""
        n_q = n_q or len(self.layers)
        residual = x
        quantized_out = torch.zeros_like(x)
        all_indices = []
        commit = x.new_zeros(())
        for layer in self.layers[:n_q]:
            quantized, indices = layer(residual)
            commit = commit + F.mse_loss(quantized.detach(), residual)
            if self.training:
                quantized = residual + (quantized - residual).detach()
            residual = residual - quantized.detach() if not self.training else residual - quantized
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
        return quantized_out, torch.stack(all_indices), commit / n_q

    @torch.no_grad()
    def encode(self, x: torch.Tensor, n_q: int | None = None) -> torch.Tensor:
        n_q = n_q or len(self.layers)
        residual = x
        all_indices = []
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            residual = residual - layer.decode(indices)
            all_indices.append(indices)
        return torch.stack(all_indices)  # [n_q, N]

    @torch.no_grad()
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        # indices: [n_q, N] — decoding from the first n_q codebooks only is the
        # protocol the paper evaluates with (first 8 of 20).
        out = None
        for i, layer_indices in enumerate(indices):
            q = self.layers[i].decode(layer_indices)
            out = q if out is None else out + q
        return out
