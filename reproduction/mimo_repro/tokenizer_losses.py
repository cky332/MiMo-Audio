"""Tokenizer training objectives (paper section 2.1.2, Eq. 1-7).

None of this exists in the official repository — it ships inference only.
Reimplemented from the paper text:

Stage 1 (unified representation learning):
  L_stage1 = 10 * L_A2T + 1 * L_recon + 1 * L_commit          (Eq. 3)
  - L_A2T: next-token loss of a jointly-trained LLM that reads the quantized
    audio representation and predicts the transcription (Eq. 1).
  - L_recon: multi-scale mel L1 with scales e = {5, 6, 7}: 2^i mel bins,
    STFT window 15 * 2^(i-1), hop 15 * 2^(i-2) (Eq. 2).

Stage 2 (adversarial fine-tuning, encoder + RVQ frozen):
  discriminators: Multi-Period Discriminator (HiFi-GAN) + Multi-Scale STFT
  discriminator (EnCodec), hinge objective (Eq. 4-5), spectral norm on all
  discriminator layers, feature matching (Eq. 6),
  L_G = 1 * L_recon + 1 * L_adv + 2 * L_fm                    (Eq. 7)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

from .tokenizer import mel_filterbank


# --------------------------------------------------------------------------
# Stage 1
# --------------------------------------------------------------------------

def multi_scale_mel_loss(
    wav_hat: torch.Tensor, wav: torch.Tensor, sr: int, scales=(5, 6, 7)
) -> torch.Tensor:
    """Eq. 2: sum_i || S_i(X) - S_i(X_hat) ||_1 with 2^i mel bins,
    window 15 * 2^(i-1), hop 15 * 2^(i-2)."""
    loss = wav.new_zeros(())
    n = min(wav_hat.shape[-1], wav.shape[-1])
    wav_hat, wav = wav_hat[..., :n], wav[..., :n]
    for i in scales:
        n_mels = 2 ** i
        win = 15 * 2 ** (i - 1)
        hop = 15 * 2 ** (i - 2)
        n_fft = 2 ** (win - 1).bit_length()  # next pow2 >= win
        window = torch.hann_window(win, device=wav.device)
        fb = mel_filterbank(n_mels, n_fft, sr, wav.device)

        def mel(x):
            s = torch.stft(
                x, n_fft, hop_length=hop, win_length=win, window=window,
                center=True, return_complex=True,
            ).abs()
            return torch.log(torch.clamp(fb @ s, min=1e-5))

        loss = loss + F.l1_loss(mel(wav_hat), mel(wav))
    return loss


class A2THead(nn.Module):
    """Stand-in for the LLM jointly trained with the tokenizer (Eq. 1).

    The paper trains a full from-scratch LLM on top of the quantized audio
    representation Q~ to predict the transcription; the LLM is discarded at
    release (it is not in the official repo).  Here: a small causal
    transformer decoder that cross-reads the audio features by prefixing
    them to the text sequence.
    """

    def __init__(self, audio_dim: int, vocab_size: int, dim: int = 128, layers: int = 2, heads: int = 4):
        super().__init__()
        from .transformer import TransformerStack

        self.audio_proj = nn.Linear(audio_dim, dim)
        self.text_embed = nn.Embedding(vocab_size, dim)
        self.decoder = TransformerStack(layers, dim, heads, dim * 4, causal=True)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, quantized_audio: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """quantized_audio: [B, T_a, D_audio]; text: [B, N] -> scalar CE loss."""
        prefix = self.audio_proj(quantized_audio)
        text_in = self.text_embed(text[:, :-1])
        h = self.decoder(torch.cat([prefix, text_in], dim=1))
        logits = self.lm_head(h[:, prefix.shape[1] :])
        return F.cross_entropy(logits.transpose(1, 2), text[:, 1:])


def stage1_loss(
    wav_hat: torch.Tensor,
    wav: torch.Tensor,
    commit: torch.Tensor,
    a2t: torch.Tensor,
    sr: int,
    lambda_a2t: float = 10.0,
    lambda_recon: float = 1.0,
    lambda_commit: float = 1.0,
) -> torch.Tensor:
    """Eq. 3."""
    recon = multi_scale_mel_loss(wav_hat, wav, sr)
    return lambda_a2t * a2t + lambda_recon * recon + lambda_commit * commit


# --------------------------------------------------------------------------
# Stage 2: discriminators
# --------------------------------------------------------------------------

class PeriodDiscriminator(nn.Module):
    """One period branch of the HiFi-GAN MPD, spectral-normalized (the paper
    applies spectral normalization to all discriminator layers)."""

    def __init__(self, period: int, channels: int = 8):
        super().__init__()
        self.period = period
        c = channels
        self.convs = nn.ModuleList(
            [
                spectral_norm(nn.Conv2d(1, c, (5, 1), (3, 1), padding=(2, 0))),
                spectral_norm(nn.Conv2d(c, c * 2, (5, 1), (3, 1), padding=(2, 0))),
                spectral_norm(nn.Conv2d(c * 2, c * 4, (5, 1), (3, 1), padding=(2, 0))),
            ]
        )
        self.post = spectral_norm(nn.Conv2d(c * 4, 1, (3, 1), padding=(1, 0)))

    def forward(self, wav: torch.Tensor):
        # wav: [B, T] -> logits and intermediate features
        B, T = wav.shape
        if T % self.period:
            wav = F.pad(wav, (0, self.period - T % self.period), mode="reflect")
        x = wav.view(B, 1, -1, self.period)
        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        x = self.post(x)
        features.append(x)
        return x.flatten(1), features


class STFTDiscriminator(nn.Module):
    """One scale of the EnCodec-style multi-scale STFT discriminator."""

    def __init__(self, n_fft: int, channels: int = 8):
        super().__init__()
        self.n_fft = n_fft
        self.hop = n_fft // 4
        c = channels
        self.convs = nn.ModuleList(
            [
                spectral_norm(nn.Conv2d(2, c, 3, padding=1)),
                spectral_norm(nn.Conv2d(c, c * 2, 3, stride=(2, 1), padding=1)),
                spectral_norm(nn.Conv2d(c * 2, c * 4, 3, stride=(2, 1), padding=1)),
            ]
        )
        self.post = spectral_norm(nn.Conv2d(c * 4, 1, 3, padding=1))

    def forward(self, wav: torch.Tensor):
        window = torch.hann_window(self.n_fft, device=wav.device)
        spec = torch.stft(
            wav, self.n_fft, hop_length=self.hop, window=window,
            center=True, return_complex=True,
        )
        x = torch.stack([spec.real, spec.imag], dim=1)  # [B, 2, F, T]
        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        x = self.post(x)
        features.append(x)
        return x.flatten(1), features


class Discriminators(nn.Module):
    """MPD (periods as in HiFi-GAN) + MS-STFT, jointly D = {D_k}."""

    def __init__(self, periods=(2, 3, 5, 7, 11), n_ffts=(512, 1024, 2048), channels: int = 8):
        super().__init__()
        self.subs = nn.ModuleList(
            [PeriodDiscriminator(p, channels) for p in periods]
            + [STFTDiscriminator(n, channels) for n in n_ffts]
        )

    def forward(self, wav: torch.Tensor):
        return [sub(wav) for sub in self.subs]


def discriminator_loss(disc: Discriminators, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """Eq. 4: hinge loss, averaged over the K sub-discriminators."""
    real_outs = disc(real)
    fake_outs = disc(fake.detach())
    loss = real.new_zeros(())
    for (real_logits, _), (fake_logits, _) in zip(real_outs, fake_outs):
        loss = loss + F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()
    return loss / len(real_outs)


def generator_adversarial_losses(
    disc: Discriminators, real: torch.Tensor, fake: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eq. 5 (adversarial, 1/K-normalized) and Eq. 6 (feature matching)."""
    real_outs = disc(real)
    fake_outs = disc(fake)
    adv = real.new_zeros(())
    fm = real.new_zeros(())
    for (_, real_feats), (fake_logits, fake_feats) in zip(real_outs, fake_outs):
        adv = adv - fake_logits.mean()
        layer_fm = real.new_zeros(())
        for rf, ff in zip(real_feats, fake_feats):
            layer_fm = layer_fm + F.l1_loss(ff, rf.detach())
        fm = fm + layer_fm / len(real_feats)
    return adv / len(real_outs), fm / len(real_outs)


def stage2_generator_loss(
    disc: Discriminators,
    wav_hat: torch.Tensor,
    wav: torch.Tensor,
    sr: int,
    lambda_recon: float = 1.0,
    lambda_adv: float = 1.0,
    lambda_fm: float = 2.0,
) -> torch.Tensor:
    """Eq. 7."""
    n = min(wav_hat.shape[-1], wav.shape[-1])
    wav_hat, wav = wav_hat[..., :n], wav[..., :n]
    recon = multi_scale_mel_loss(wav_hat, wav, sr)
    adv, fm = generator_adversarial_losses(disc, wav, wav_hat)
    return lambda_recon * recon + lambda_adv * adv + lambda_fm * fm
