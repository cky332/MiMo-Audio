"""Shared utilities for the MiMo-Audio stress-test experiments.

Real speech comes from the repository's own examples/ directory (11 ESD
emotional-speech clips at 16 kHz, 4 prompt-speech clips and one 56 s
spoken-dialogue turn at 24 kHz).  Everything is resampled to the tokenizer's
24 kHz rate.
"""

from __future__ import annotations

import glob
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mimo_repro.config import TokenizerConfig
from mimo_repro.tokenizer import MiMoAudioTokenizerRepro, log_mel_spectrogram

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

SR = 24000


def load_wav(path: str, target_sr: int = SR) -> torch.Tensor:
    import soundfile as sf
    import torchaudio.functional as AF

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T).mean(dim=0)
    if sr != target_sr:
        wav = AF.resample(wav, sr, target_sr)
    return wav


def real_speech_files() -> dict:
    esd = sorted(glob.glob(os.path.join(EXAMPLES_DIR, "ESD", "*.wav")))
    prompts = sorted(glob.glob(os.path.join(EXAMPLES_DIR, "prompt_speech_*.wav")))
    long_turn = os.path.join(EXAMPLES_DIR, "spoken_dialogue_assistant_turn_1.wav")
    return {"esd": esd, "prompts": prompts, "long": long_turn}


def speech_training_segments(seg_samples: int = 24000) -> torch.Tensor:
    """Cut all short clips (ESD + prompts) into fixed 1 s segments -> [N, seg]."""
    files = real_speech_files()
    segs = []
    for path in files["esd"] + files["prompts"]:
        wav = load_wav(path)
        wav = wav / (wav.abs().max() + 1e-8)
        for start in range(0, wav.shape[0] - seg_samples + 1, seg_samples):
            segs.append(wav[start : start + seg_samples])
    return torch.stack(segs)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def mel_l1(wav_a: torch.Tensor, wav_b: torch.Tensor, cfg: TokenizerConfig) -> float:
    """Log-mel L1 distance between two waveforms (single scale, n_mels of cfg)."""
    n = min(wav_a.shape[-1], wav_b.shape[-1])
    ma = log_mel_spectrogram(wav_a[..., :n].reshape(1, -1), cfg)
    mb = log_mel_spectrogram(wav_b[..., :n].reshape(1, -1), cfg)
    return (ma - mb).abs().mean().item()


def code_usage_entropy(codes: torch.Tensor, codebook_size: int) -> float:
    """Empirical entropy (bits) of code usage for one quantizer level."""
    counts = torch.bincount(codes.flatten(), minlength=codebook_size).float()
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * torch.log2(p)).sum())


def train_tiny_tokenizer(
    steps: int = 400,
    batch: int = 4,
    lr: float = 2e-3,
    seed: int = 0,
    lambda_a2t: float = 0.0,
    text_labels: torch.Tensor | None = None,
    log_every: int = 50,
    stabilizers: bool = True,
):
    """Train the tiny reproduction tokenizer on real speech segments.

    Returns (tokenizer, cfg, losses). With lambda_a2t > 0, a toy A2T head is
    trained jointly on the provided per-segment text label sequences.
    stabilizers=False removes the kmeans init + dead-code expiry that the
    official quantization.py uses (ablation for the collapse experiment)."""
    from mimo_repro.rvq import ResidualVQ
    from mimo_repro.tokenizer_losses import A2THead, multi_scale_mel_loss

    torch.manual_seed(seed)
    cfg = TokenizerConfig.tiny()
    tok = MiMoAudioTokenizerRepro(cfg)
    if not stabilizers:
        tok.quantizer = ResidualVQ(cfg.d_model, cfg.codebook_sizes, decay=cfg.rvq_decay,
                                   kmeans_init=False, threshold_ema_dead_code=0)
    segs = speech_training_segments()

    params = list(tok.parameters())
    a2t_head = None
    if lambda_a2t > 0:
        a2t_head = A2THead(audio_dim=cfg.d_model, vocab_size=int(text_labels.max()) + 1)
        params += list(a2t_head.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    losses, recon_losses = [], []
    for step in range(steps):
        idx = torch.randint(0, segs.shape[0], (batch,))
        wav = segs[idx]
        mel = log_mel_spectrogram(wav, cfg)
        wav_hat, quantized, _, commit = tok(mel)
        recon = multi_scale_mel_loss(wav_hat, wav, cfg.sampling_rate)
        loss = recon + commit
        if a2t_head is not None:
            loss = loss + lambda_a2t * a2t_head(quantized, text_labels[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        recon_losses.append(recon.item())
        if log_every and step % log_every == 0:
            print(f"  step {step:4d}  loss {loss.item():7.3f}  recon {recon.item():7.3f}")
    tok.eval()
    return tok, cfg, {"loss": losses, "recon": recon_losses}
