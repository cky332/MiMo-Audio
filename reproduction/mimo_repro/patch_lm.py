"""MiMo-Audio LM reproduction (paper section 2.2, Eq. 8-15).

Three components on top of an LLM backbone:

- PatchEncoder (2.2.1): per-codebook embedding tables summed per frame
  (Eq. 11), a small Transformer over the G=4 frames of each patch, then the
  frame outputs are concatenated and linearly projected to the LLM width.
- the LLM consumes interleaved text tokens and audio patches (Eq. 9-10); a
  position is speech iff its text token is the <|empty|> placeholder — text
  embeddings and (projected) patch embeddings are added after zeroing the
  inactive branch, exactly as the official `_prepare_input_embeds` does.
- PatchDecoder (2.2.3): a causal Transformer over G + max(delay) steps that
  generates the R' codebook streams with per-codebook delays (Eq. 14-15).

Faithful-to-code details that differ from the paper's formalization:
- the "empty token" is an EXTRA index appended to each codebook
  (1025/129-sized vocabularies), not id 0 as Eq. 15 states; it is also the
  embedding padding_idx and is banned during sampling.
- the text token of a speech position is replicated across the G frames of
  the flattened storage layout, and the text channel of a text position
  carries the audio empty ids.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import MiMoLMConfig
from .transformer import TransformerStack


@dataclass
class LMOutput:
    text_logits: torch.Tensor          # [B, T_patch, text_vocab]
    audio_logits: list | None          # per-channel [B, T_patch, ctx, vocab_r]
    loss: torch.Tensor | None = None
    text_loss: torch.Tensor | None = None
    audio_loss: torch.Tensor | None = None


class PatchEncoder(nn.Module):
    """Eq. 11 + section 2.2.1."""

    def __init__(self, cfg: MiMoLMConfig):
        super().__init__()
        self.cfg = cfg
        self.embeddings = nn.ModuleList(
            nn.Embedding(v, cfg.patch_dim, padding_idx=e)
            for v, e in zip(cfg.audio_vocab_sizes, cfg.audio_empty_ids)
        )
        self.transformer = TransformerStack(
            cfg.patch_encoder_layers,
            cfg.patch_dim,
            cfg.patch_heads,
            cfg.patch_ffn_dim,
            rope_theta=cfg.rope_theta,
            causal=not cfg.patch_encoder_bidirectional,
        )
        self.proj = nn.Linear(cfg.patch_dim * cfg.group_size, cfg.llm_dim, bias=False)

    def frame_embeddings(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """audio_tokens: [..., channels] -> summed embeddings [..., patch_dim]."""
        out = None
        for r, table in enumerate(self.embeddings):
            e = table(audio_tokens[..., r])
            out = e if out is None else out + e
        return out

    def forward(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """audio_tokens: [B, T_patch, G, channels] -> [B, T_patch, llm_dim]."""
        B, T, G, _ = audio_tokens.shape
        frames = self.frame_embeddings(audio_tokens)          # [B, T, G, patch_dim]
        frames = self.transformer(frames.reshape(B * T, G, -1)).reshape(B, T, G, -1)
        return self.proj(frames.reshape(B, T, G * frames.shape[-1]))


def build_delayed_patch(
    patch: torch.Tensor, delay_pattern, empty_ids
) -> torch.Tensor:
    """Eq. 14-15: [B, G, channels] -> delayed [B, G + max(D), channels] where
    channel r holds a[i - d_r, r] for i - d_r in [0, G) and empty otherwise."""
    B, G, C = patch.shape
    ctx = G + max(delay_pattern)
    out = patch.new_empty(B, ctx, C)
    for r in range(C):
        out[:, :, r] = empty_ids[r]
        d = delay_pattern[r]
        out[:, d : d + G, r] = patch[:, :, r]
    return out


def undelay_patch(delayed: torch.Tensor, delay_pattern, group_size: int) -> torch.Tensor:
    """Inverse of build_delayed_patch."""
    B, _, C = delayed.shape
    out = delayed.new_empty(B, group_size, C)
    for r in range(C):
        d = delay_pattern[r]
        out[:, :, r] = delayed[:, d : d + group_size, r]
    return out


class PatchDecoder(nn.Module):
    """Section 2.2.3: causal Transformer with R' embedding tables (shared
    with the patch encoder in the official model — here independent tables
    with identical shapes, plus a `tie_embeddings` hook) and R' output heads,
    operating over G + max(D) delayed steps seeded by the LLM hidden state."""

    def __init__(self, cfg: MiMoLMConfig):
        super().__init__()
        self.cfg = cfg
        self.seed_proj = nn.Linear(cfg.llm_dim, cfg.patch_dim, bias=False)
        self.embeddings = nn.ModuleList(
            nn.Embedding(v, cfg.patch_dim, padding_idx=e)
            for v, e in zip(cfg.audio_vocab_sizes, cfg.audio_empty_ids)
        )
        self.transformer = TransformerStack(
            cfg.patch_decoder_layers,
            cfg.patch_dim,
            cfg.patch_heads,
            cfg.patch_ffn_dim,
            rope_theta=cfg.rope_theta,
            causal=True,
        )
        self.heads = nn.ModuleList(
            nn.Linear(cfg.patch_dim, v, bias=False) for v in cfg.audio_vocab_sizes
        )

    def tie_embeddings(self, encoder: PatchEncoder) -> None:
        """The paper says the decoder 'employs the same R' embedding tables as
        the patch encoder'."""
        for mine, theirs in zip(self.embeddings, encoder.embeddings):
            mine.weight = theirs.weight

    def step_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """Delayed-frame tokens [B, channels] -> input embedding [B, patch_dim].
        Empty ids contribute zero (padding_idx)."""
        out = None
        for r, table in enumerate(self.embeddings):
            e = table(tokens[..., r])
            out = e if out is None else out + e
        return out

    def forward(self, seed: torch.Tensor, delayed_patch: torch.Tensor) -> list:
        """Teacher-forced pass for training.

        seed: [B, llm_dim] LLM hidden state; delayed_patch: [B, ctx, channels].
        Input sequence: [seed, emb(step_0), ..., emb(step_{ctx-2})];
        output step t predicts delayed step t for every active channel.
        Returns per-channel logits [B, ctx, vocab_r].
        """
        B, ctx, C = delayed_patch.shape
        seed = self.seed_proj(seed).unsqueeze(1)                   # [B, 1, D]
        step_embeds = self.step_embedding(delayed_patch[:, :-1])   # [B, ctx-1, D]
        h = self.transformer(torch.cat([seed, step_embeds], dim=1))
        return [head(h) for head in self.heads]                    # [B, ctx, vocab_r]

    @torch.no_grad()
    def generate(self, seed: torch.Tensor, sample_fn=None) -> torch.Tensor:
        """Autoregressive generation of one patch, mirroring the official
        `local_forward`: at step t, channel r is emitted iff
        d_r <= t < d_r + G; the empty token is banned from sampling.
        Returns [B, G, channels]."""
        cfg = self.cfg
        B = seed.shape[0]
        G, D = cfg.group_size, cfg.delay_pattern
        ctx = cfg.patch_decoder_context
        if sample_fn is None:
            sample_fn = lambda logits: logits.argmax(dim=-1)

        tokens = torch.full(
            (B, G, cfg.audio_channels), 0, dtype=torch.long, device=seed.device
        )
        inputs = self.seed_proj(seed).unsqueeze(1)  # growing [B, t+1, D]
        for t in range(ctx):
            h = self.transformer(inputs)[:, -1]     # no KV cache: O(ctx^2), fine at this scale
            next_embed = torch.zeros_like(inputs[:, 0])
            for r in range(cfg.audio_channels):
                if D[r] <= t < D[r] + G:
                    logits = self.heads[r](h)
                    logits[:, cfg.audio_empty_ids[r]] = float("-inf")  # ban empty
                    tok = sample_fn(logits)
                    tokens[:, t - D[r], r] = tok
                    next_embed = next_embed + self.embeddings[r](tok)
            inputs = torch.cat([inputs, next_embed.unsqueeze(1)], dim=1)
        return tokens


class MiMoAudioLMRepro(nn.Module):
    """Interleaved text/audio LM (Eq. 8-10) with training loss and generation.

    The LLM backbone here is a from-scratch TransformerStack standing in for
    MiMo-7B; the official model wraps a pretrained Qwen2-architecture model.
    """

    def __init__(self, cfg: MiMoLMConfig):
        super().__init__()
        self.cfg = cfg
        self.text_embed = nn.Embedding(cfg.text_vocab_size, cfg.llm_dim)
        self.llm = TransformerStack(
            cfg.llm_layers, cfg.llm_dim, cfg.llm_heads, cfg.llm_ffn_dim,
            rope_theta=cfg.rope_theta, causal=True,
        )
        self.lm_head = nn.Linear(cfg.llm_dim, cfg.text_vocab_size, bias=False)
        self.patch_encoder = PatchEncoder(cfg)
        self.patch_decoder = PatchDecoder(cfg)
        self.patch_decoder.tie_embeddings(self.patch_encoder)

    # -- input preparation (official `_prepare_input_embeds`) --------------
    def prepare_inputs(self, text_tokens: torch.Tensor, audio_tokens: torch.Tensor):
        """text_tokens: [B, T_patch] (empty_text_id on speech positions);
        audio_tokens: [B, T_patch, G, channels] (audio empty ids on text
        positions).  Returns LLM input embeddings [B, T_patch, llm_dim]."""
        is_speech = text_tokens == self.cfg.empty_text_id           # [B, T]
        text_embeds = self.text_embed(text_tokens)
        text_embeds = text_embeds * (~is_speech).unsqueeze(-1)      # zero on speech
        patch_embeds = self.patch_encoder(audio_tokens)
        patch_embeds = patch_embeds * is_speech.unsqueeze(-1)       # zero on text
        return text_embeds + patch_embeds, is_speech

    def forward(
        self,
        text_tokens: torch.Tensor,          # [B, T_patch]
        audio_tokens: torch.Tensor,         # [B, T_patch, G, channels]
        text_targets: torch.Tensor | None = None,    # [B, T_patch], -100 = ignore
        audio_targets: torch.Tensor | None = None,   # [B, T_patch, G, channels], -100 = ignore
    ) -> LMOutput:
        cfg = self.cfg
        embeds, _ = self.prepare_inputs(text_tokens, audio_tokens)
        hidden = self.llm(embeds)                                  # [B, T, llm_dim]
        text_logits = self.lm_head(hidden)

        if text_targets is None:
            return LMOutput(text_logits=text_logits, audio_logits=None)

        # ---- text loss (positions t predict t+1) --------------------------
        text_loss = F.cross_entropy(
            text_logits[:, :-1].transpose(1, 2),
            text_targets[:, 1:],
            ignore_index=-100,
        )

        # ---- audio loss via teacher-forced patch decoder ------------------
        # hidden state at position t seeds the decoder for patch t+1
        B, T, G, C = audio_tokens.shape
        seeds = hidden[:, :-1].reshape(-1, cfg.llm_dim)            # [(B*(T-1)), llm_dim]
        target_patches = audio_targets[:, 1:].reshape(-1, G, C)
        # only train the decoder on patches that are actually speech
        active = (target_patches != -100).all(dim=(1, 2))
        audio_loss = hidden.new_zeros(())
        if active.any():
            seeds = seeds[active]
            target_patches = target_patches[active]
            delayed = build_delayed_patch(
                target_patches, cfg.delay_pattern, cfg.audio_empty_ids
            )
            logits = self.patch_decoder(seeds, delayed)
            audio_loss_terms = []
            for r in range(C):
                d = cfg.delay_pattern[r]
                # channel r is supervised on delayed steps [d, d+G)
                lr = F.cross_entropy(
                    logits[r][:, d : d + G].transpose(1, 2),
                    delayed[:, d : d + G, r],
                )
                audio_loss_terms.append(cfg.audio_loss_weights[r] * lr)
            audio_loss = sum(audio_loss_terms)

        loss = cfg.text_loss_weight * text_loss + audio_loss
        return LMOutput(
            text_logits=text_logits,
            audio_logits=None,
            loss=loss,
            text_loss=text_loss,
            audio_loss=audio_loss,
        )

    # -- generation (official `slm_sample` semantics) -----------------------
    @torch.no_grad()
    def generate(
        self,
        text_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        max_new_patches: int = 16,
        text_sample_fn=None,
        audio_sample_fn=None,
        stop_ids: set | None = None,
    ):
        """Group-level loop: each step samples one text token; if it is the
        <|empty|> placeholder, the patch decoder generates G frames of audio,
        otherwise the audio channels are filled with empty ids.  (The official
        implementation is the same loop with KV caches and batch=1.)"""
        cfg = self.cfg
        if text_sample_fn is None:
            text_sample_fn = lambda logits: logits.argmax(dim=-1)
        stop_ids = stop_ids or set()

        text_out, audio_out = [], []
        for _ in range(max_new_patches):
            embeds, _ = self.prepare_inputs(text_tokens, audio_tokens)
            hidden = self.llm(embeds)[:, -1]                       # [B, llm_dim]
            next_text = text_sample_fn(self.lm_head(hidden))       # [B]

            if (next_text == cfg.empty_text_id).all():
                next_patch = self.patch_decoder.generate(hidden, audio_sample_fn)
            else:
                next_patch = torch.tensor(
                    cfg.audio_empty_ids, device=hidden.device
                ).expand(text_tokens.shape[0], cfg.group_size, -1).clone()

            text_out.append(next_text)
            audio_out.append(next_patch)
            text_tokens = torch.cat([text_tokens, next_text[:, None]], dim=1)
            audio_tokens = torch.cat([audio_tokens, next_patch[:, None]], dim=1)
            if next_text[0].item() in stop_ids:
                break
        return torch.stack(text_out, dim=1), torch.stack(audio_out, dim=1)
