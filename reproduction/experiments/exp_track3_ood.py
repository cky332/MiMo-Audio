"""Track 3: out-of-distribution signal robustness of the tokenizer design.

The tiny tokenizer trained on the repo's real speech is probed with a suite
of signals that a deployed "general audio" model will inevitably meet.  Two
questions per signal class:

1. how much does reconstruction degrade relative to held-out speech?
2. does the RVQ codebook usage collapse (entropy drop), i.e. does the
   tokenizer map diverse OOD content onto a few codes -- the failure mode
   behind 'the model cannot even hear the difference'?

Held-out speech = segments of the 56 s dialogue recording (never used in
training, different speaker/content).
"""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import ARTIFACTS_DIR, code_usage_entropy, load_wav, mel_l1, real_speech_files  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mimo_repro.config import TokenizerConfig  # noqa: E402
from mimo_repro.tokenizer import MiMoAudioTokenizerRepro, log_mel_spectrogram  # noqa: E402

# ---------------------------------------------------------------------------
# E3.0: RVQ training fragility -- the paper never mentions the kmeans init and
# dead-code expiry that the official quantization.py relies on.  Ablate them.
# ---------------------------------------------------------------------------
if os.environ.get("SKIP_E30") != "1":
    from common import train_tiny_tokenizer

    print("=== E3.0 EMA-RVQ collapse without the official (unmentioned) stabilizers ===")
    for stab in (False, True):
        tok_s, cfg_s, hist = train_tiny_tokenizer(steps=150, log_every=0, stabilizers=stab)
        held_probe = load_wav(real_speech_files()["long"])[: 4 * 24000]
        codes = tok_s.encode(log_mel_spectrogram(held_probe[None, :], cfg_s))
        ents = [code_usage_entropy(codes[q, 0], cfg_s.codebook_sizes[q]) for q in range(4)]
        uniq = [int(torch.unique(codes[q, 0]).numel()) for q in range(4)]
        name = "official recipe (kmeans + expiry)" if stab else "plain EMA-VQ (paper text only)  "
        print(f"  {name}: recon {sum(hist['recon'][-20:]) / 20:6.3f}  "
              f"q0-3 entropy {['%.2f' % e for e in ents]} bit  unique {uniq}")
    print("  (entropy near 0 / 1 unique code = collapsed codebook)\n")

torch.manual_seed(0)
cfg = TokenizerConfig.tiny()
tok = MiMoAudioTokenizerRepro(cfg)
tok.load_state_dict(torch.load(os.path.join(ARTIFACTS_DIR, "tok_realspeech.pt"))["state_dict"])
tok.eval()

SR = 24000
DUR = 2 * SR                                            # 2 s probes
t = torch.arange(DUR) / SR

held = load_wav(real_speech_files()["long"])
held = held / held.abs().max()
held_speech = held[10 * SR : 10 * SR + DUR]             # unseen speaker segment


def tone(freq, amp=0.5):
    return amp * torch.sin(2 * torch.pi * freq * t)


def chirp():
    f0, f1 = 100, 8000
    phase = 2 * torch.pi * (f0 * t + (f1 - f0) * t ** 2 / (2 * 2.0))
    return 0.5 * torch.sin(phase)


def pink_noise():
    white = torch.randn(DUR)
    spec = torch.fft.rfft(white)
    freqs = torch.fft.rfftfreq(DUR, 1 / SR)
    spec = spec / (freqs + 1.0).sqrt()
    x = torch.fft.irfft(spec, n=DUR)
    return 0.5 * x / x.abs().max()


signals = {
    "speech (held-out)": held_speech,
    "speech -40 dB": held_speech * 0.01,
    "speech clipped 5x": (held_speech * 5).clamp(-1, 1),
    "speech + DC 0.5": held_speech * 0.5 + 0.5,
    "speech 16k-as-24k": held_speech[::1][: DUR * 2 // 3].repeat_interleave(1)[:DUR]
        if False else torch.nn.functional.interpolate(
            held_speech[None, None, :], scale_factor=1.5, mode="linear", align_corners=False
        )[0, 0, :DUR],
    "silence": torch.zeros(DUR),
    "white noise": 0.5 * torch.randn(DUR).clamp(-1, 1),
    "pink noise": pink_noise(),
    "pure tone 440 Hz": tone(440),
    "chirp 0.1-8 kHz": chirp(),
    "3-note chord": (tone(220, 0.25) + tone(277, 0.25) + tone(330, 0.25)),
}

print(f"{'signal':22s} {'recon mel-L1':>12s} {'q0 entropy':>11s} {'q0 top-code':>11s} {'uniq q0':>8s}")
q0_size = cfg.codebook_sizes[0]
max_bits = math.log2(q0_size)
results = {}
for name, sig in signals.items():
    sig = sig.float()
    mel = log_mel_spectrogram(sig[None, :], cfg)
    with torch.no_grad():
        codes = tok.encode(mel)
        recon = tok.decode(codes)[0]
    err = mel_l1(sig, recon, cfg)
    q0 = codes[0, 0]
    ent = code_usage_entropy(q0, q0_size)
    top = torch.bincount(q0, minlength=q0_size).max().item() / q0.numel()
    uniq = int(torch.unique(q0).numel())
    results[name] = (err, ent, top, uniq)
    print(f"{name:22s} {err:12.3f} {ent:8.2f} bit {top:10.1%} {uniq:8d}")

print(f"\n(q0 codebook size {q0_size} -> max entropy {max_bits:.1f} bit; "
      f"'top-code' = share of frames mapped to the single most frequent code)")

base = results["speech (held-out)"][0]
print("\nrelative degradation vs held-out speech:")
for name, (err, ent, top, uniq) in results.items():
    if name == "speech (held-out)":
        continue
    print(f"  {name:22s} recon {err / base:5.2f}x   entropy {ent:5.2f} bit   top-code {top:6.1%}")

print("\nTrack 3 done.")
