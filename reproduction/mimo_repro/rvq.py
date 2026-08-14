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


def _kmeans(samples: torch.Tensor, k: int, iters: int = 10) -> torch.Tensor:
    means = samples[torch.randperm(samples.shape[0])[:k]] if samples.shape[0] >= k \
        else samples[torch.randint(0, samples.shape[0], (k,))]
    for _ in range(iters):
        dist = (samples.pow(2).sum(1, keepdim=True) - 2 * samples @ means.t()
                + means.pow(2).sum(1)[None, :])
        buckets = dist.argmin(dim=-1)
        bins = torch.bincount(buckets, minlength=k).clamp_min(1)
        new = torch.zeros_like(means).index_add_(0, buckets, samples) / bins[:, None]
        means = torch.where((torch.bincount(buckets, minlength=k) == 0)[:, None], means, new)
    return means


class EMACodebook(nn.Module):
    """EMA codebook.  `kmeans_init` and `threshold_ema_dead_code` mirror the
    stabilizers in the official quantization.py (kmeans_init=True,
    threshold_ema_dead_code=10 in the released config); without them the EMA
    codebook collapses on real speech -- see experiments/exp_track3_ood.py."""

    def __init__(self, dim: int, codebook_size: int, decay: float = 0.99,
                 epsilon: float = 1e-5, kmeans_init: bool = False,
                 threshold_ema_dead_code: int = 0):
        super().__init__()
        self.codebook_size = codebook_size
        self.decay = decay
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code
        embed = torch.zeros(codebook_size, dim) if kmeans_init \
            else nn.init.kaiming_uniform_(torch.empty(codebook_size, dim))
        self.register_buffer("inited", torch.tensor(not kmeans_init))
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
        if self.training and not self.inited:
            with torch.no_grad():
                init = _kmeans(x.detach(), self.codebook_size)
                self.embed.copy_(init)
                self.embed_avg.copy_(init)
                self.cluster_size.fill_(1.0)
                self.inited.fill_(True)
        indices = self.encode(x)
        quantized = self.decode(indices)
        if self.training:
            with torch.no_grad():
                # dead-code expiry BEFORE the EMA update, as in the official code
                if self.threshold_ema_dead_code > 0:
                    dead = self.cluster_size < self.threshold_ema_dead_code
                    if dead.any():
                        replace = x[torch.randint(0, x.shape[0], (int(dead.sum()),))]
                        self.embed[dead] = replace
                        self.embed_avg[dead] = replace
                        self.cluster_size[dead] = self.threshold_ema_dead_code
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

    def __init__(self, dim: int, codebook_sizes, decay: float = 0.99,
                 kmeans_init: bool = False, threshold_ema_dead_code: int = 0):
        super().__init__()
        self.layers = nn.ModuleList(
            EMACodebook(dim, size, decay=decay, kmeans_init=kmeans_init,
                        threshold_ema_dead_code=threshold_ema_dead_code)
            for size in codebook_sizes
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
