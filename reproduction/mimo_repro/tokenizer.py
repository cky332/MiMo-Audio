"""MiMo-Audio-Tokenizer reproduction (paper section 2.1).

Pipeline (matching the official modeling_audio_tokenizer.py):

  24 kHz wav --log-mel(100 Hz)--> encoder conv (stride 2, ->50 Hz)
    -> Transformer encoder (skip connection from layer 3 to output)
    -> avg-pool conv (stride 2, ->25 Hz) -> RVQ (20 codebooks)
    -> causal Transformer decoder with transposed convs back to 100 Hz mel
    -> Vocos-style Transformer vocoder (sliding-window attention)
    -> ISTFT head -> 24 kHz wav
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import TokenizerConfig
from .rvq import ResidualVQ
from .transformer import TransformerStack


def log_mel_spectrogram(wav: torch.Tensor, cfg: TokenizerConfig) -> torch.Tensor:
    """[B, samples] -> [B, T_mel, n_mels] at 100 Hz (same shape convention the
    official wav2mel uses: magnitude mel, log with 1e-7 clamp)."""
    window = torch.hann_window(cfg.win_length, device=wav.device)
    spec = torch.stft(
        wav,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        center=True,
        return_complex=True,
    ).abs()
    mel_fb = mel_filterbank(cfg.n_mels, cfg.n_fft, cfg.sampling_rate, wav.device)
    mel = mel_fb @ spec
    return torch.log(torch.clamp(mel, min=1e-7)).transpose(1, 2)


def mel_filterbank(n_mels: int, n_fft: int, sr: int, device) -> torch.Tensor:
    """Slaney-free triangular mel filterbank (HTK mel scale), pure torch."""
    def hz_to_mel(f):
        return 2595.0 * torch.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    fmax = sr / 2
    mel_points = torch.linspace(
        hz_to_mel(torch.tensor(0.0)), hz_to_mel(torch.tensor(float(fmax))), n_mels + 2
    )
    hz_points = mel_to_hz(mel_points)
    bins = torch.floor((n_fft + 1) * hz_points / sr).long().clamp(0, n_fft // 2)
    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center > left:
            fb[i, left:center] = (torch.arange(left, center) - left) / max(1, (center - left))
        if right > center:
            fb[i, center:right] = (right - torch.arange(center, right)) / max(1, (right - center))
    return fb.to(device)


class ISTFTHead(nn.Module):
    """Vocos ISTFT head: predict magnitude and phase, inverse STFT to wav."""

    def __init__(self, dim: int, n_fft: int, hop_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.out = nn.Linear(dim, n_fft + 2)
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, dim] -> wav [B, T * hop_length]
        x = self.out(x).transpose(1, 2)
        mag, phase = x.chunk(2, dim=1)
        mag = torch.exp(mag).clamp(max=1e2)
        spec = mag * (torch.cos(phase) + 1j * torch.sin(phase))
        wav = torch.istft(
            spec,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            length=x.shape[-1] * self.hop_length,
        )
        return wav


class TokenizerEncoder(nn.Module):
    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv1d(cfg.n_mels, cfg.d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            cfg.d_model, cfg.d_model, kernel_size=3, stride=cfg.conv_stride, padding=1
        )
        self.transformer = TransformerStack(
            cfg.encoder_layers,
            cfg.d_model,
            cfg.encoder_heads,
            cfg.encoder_ffn_dim,
            rope_theta=cfg.rope_theta,
            causal=False,
            skip_layer_id=cfg.encoder_skip_layer_id,
        )
        self.down = nn.Conv1d(
            cfg.d_model, cfg.d_model, cfg.avg_pooler, stride=cfg.avg_pooler, bias=False
        )
        self.down_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [B, T_mel, n_mels] -> features [B, T_mel // 4, d_model] (25 Hz)
        x = mel.transpose(1, 2)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))          # 100 Hz -> 50 Hz
        x = x.transpose(1, 2)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        if x.shape[-1] % self.cfg.avg_pooler:
            pad = self.cfg.avg_pooler - x.shape[-1] % self.cfg.avg_pooler
            x = F.pad(x, (0, pad))
        x = F.gelu(self.down(x))           # 50 Hz -> 25 Hz
        return self.down_norm(x.transpose(1, 2))


class TokenizerDecoder(nn.Module):
    """Causal mirror of the encoder: 25 Hz tokens -> 100 Hz coarse mel."""

    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        self.cfg = cfg
        self.up1 = nn.ConvTranspose1d(
            cfg.d_model, cfg.d_model, cfg.avg_pooler, stride=cfg.avg_pooler
        )
        self.transformer = TransformerStack(
            cfg.decoder_layers,
            cfg.d_model,
            cfg.decoder_heads,
            cfg.decoder_ffn_dim,
            rope_theta=cfg.rope_theta,
            causal=True,  # paper: "employs causal self-attention to support streaming"
        )
        self.up2 = nn.ConvTranspose1d(cfg.d_model, cfg.n_mels, 3, stride=cfg.conv_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T25, d_model] -> coarse mel [B, T100, n_mels]
        y = self.up1(x.transpose(1, 2)).transpose(1, 2)   # 25 Hz -> 50 Hz
        y = self.transformer(y)
        y = self.up2(y.transpose(1, 2))                   # 50 Hz -> 100 Hz
        # causal trim as in the official CausalConvTranspose1d
        trim = max(0, 3 - self.cfg.conv_stride)
        if trim:
            y = y[..., :-trim]
        return y.transpose(1, 2)


class TransformerVocoder(nn.Module):
    """Vocos-style vocoder with sliding-window attention + ISTFT head."""

    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        self.embed = nn.Linear(cfg.n_mels, cfg.vocoder_dim, bias=False)
        self.transformer = TransformerStack(
            cfg.vocoder_layers,
            cfg.vocoder_dim,
            cfg.vocoder_heads,
            cfg.vocoder_ffn_dim,
            rope_theta=cfg.rope_theta,
            causal=False,
            window=cfg.vocoder_window,
        )
        self.head = ISTFTHead(cfg.vocoder_dim, cfg.n_fft, cfg.hop_length)

    def forward(self, coarse_mel: torch.Tensor) -> torch.Tensor:
        x = self.embed(coarse_mel)
        x = self.transformer(x)
        return self.head(x)


class MiMoAudioTokenizerRepro(nn.Module):
    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = TokenizerEncoder(cfg)
        self.quantizer = ResidualVQ(
            cfg.d_model, cfg.codebook_sizes, decay=cfg.rvq_decay,
            kmeans_init=True, threshold_ema_dead_code=cfg.threshold_ema_dead_code,
        )
        self.decoder = TokenizerDecoder(cfg)
        self.vocoder = TransformerVocoder(cfg)

    # -- training path -----------------------------------------------------
    def forward(self, mel: torch.Tensor, n_q: int | None = None):
        """Returns (reconstructed wav, quantized features, codes, commit loss)."""
        features = self.encoder(mel)
        B, T, D = features.shape
        quantized, codes, commit = self.quantizer(features.reshape(-1, D), n_q=n_q)
        quantized = quantized.reshape(B, T, D)
        coarse_mel = self.decoder(quantized)
        wav = self.vocoder(coarse_mel)
        return wav, quantized, codes.reshape(-1, B, T), commit

    # -- inference path (mirrors the official encode/decode API) -----------
    @torch.no_grad()
    def encode(self, mel: torch.Tensor, n_q: int | None = None) -> torch.Tensor:
        features = self.encoder(mel)
        B, T, D = features.shape
        codes = self.quantizer.encode(features.reshape(-1, D), n_q=n_q)
        return codes.reshape(-1, B, T)  # [n_q, B, T25]

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        # codes: [n_q, B, T25] — decoding with only the first 8 codebooks is
        # exactly what the downstream LM produces.
        n_q, B, T = codes.shape
        features = self.quantizer.decode(codes.reshape(n_q, -1)).reshape(B, T, -1)
        coarse_mel = self.decoder(features)
        return self.vocoder(coarse_mel)
