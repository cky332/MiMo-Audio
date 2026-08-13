"""Shared minimal Transformer used by the tokenizer components.

Matches the structure of the official `TransformerLayer`
(src/mimo_audio_tokenizer/modeling_audio_tokenizer.py): pre-LN blocks with
RoPE attention and a GELU MLP.  The official code hard-requires
flash-attn's `flash_attn_varlen_func`; here plain SDPA with an explicit
additive mask is used so everything runs on CPU, including the sliding
window attention the vocoder needs (window = (left, right) frames).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def rope_frequencies(dim: int, positions: torch.Tensor, theta: float) -> tuple:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, heads, T, head_dim]; cos/sin: [T, head_dim]
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


def window_mask(
    seq_len: int,
    window: tuple[int, int] | None,
    causal: bool,
    device: torch.device,
) -> torch.Tensor | None:
    """Additive attention mask for causal and/or sliding-window attention.

    `window` = (left, right): position i may attend to j with
    i - left <= j <= i + right (flash-attn convention used by the official
    vocoder, window_size=[40, 10]).  Causal caps right at 0.
    """
    if window is None and not causal:
        return None
    idx = torch.arange(seq_len, device=device)
    rel = idx[None, :] - idx[:, None]  # rel[i, j] = j - i
    allowed = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    if causal:
        allowed &= rel <= 0
    if window is not None:
        left, right = window
        if left >= 0:
            allowed &= rel >= -left
        if right >= 0 and not causal:
            allowed &= rel <= right
    mask = torch.zeros(seq_len, seq_len, device=device)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        # same projection bias layout as the official code (k has no bias)
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor, rope: tuple, mask: torch.Tensor | None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        cos, sin = rope
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(out)


class TransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_dim: int):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.mlp_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, dim)

    def forward(self, x: torch.Tensor, rope: tuple, mask: torch.Tensor | None):
        x = x + self.attn(self.attn_norm(x), rope, mask)
        x = x + self.fc2(F.gelu(self.fc1(self.mlp_norm(x))))
        return x


class TransformerStack(nn.Module):
    """A stack of layers with optional causal / sliding-window attention and an
    optional skip connection from an early layer to the final output (the
    paper's "add the layer-3 hidden states to the final-layer output")."""

    def __init__(
        self,
        num_layers: int,
        dim: int,
        heads: int,
        ffn_dim: int,
        rope_theta: float = 10000.0,
        causal: bool = False,
        window: tuple[int, int] | None = None,
        skip_layer_id: int | None = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerLayer(dim, heads, ffn_dim) for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(dim)
        self.head_dim = dim // heads
        self.rope_theta = rope_theta
        self.causal = causal
        self.window = window
        self.skip_layer_id = skip_layer_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        positions = torch.arange(T, device=x.device)
        rope = rope_frequencies(self.head_dim, positions, self.rope_theta)
        rope = (rope[0].to(x.dtype).to(x.device), rope[1].to(x.dtype).to(x.device))
        mask = window_mask(T, self.window, self.causal, x.device)
        skip = None
        for i, layer in enumerate(self.layers):
            x = layer(x, rope, mask)
            if self.skip_layer_id is not None and i == self.skip_layer_id - 1:
                skip = x
        if skip is not None:
            x = x + skip
        return self.final_norm(x)
