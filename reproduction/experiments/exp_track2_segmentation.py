"""Track 2: segmentation artifacts of the official inference pipeline.

The official code (src/mimo_audio/mimo_audio.py) processes long audio in
independent, non-overlapping chunks in TWO places:

- encoding: the mel is split into 6000-frame (60 s) segments and each is
  encoded separately (preprocess_input, L258-266);
- decoding: generated codes are decoded in 1500-token (60 s) segments and
  the waveforms are concatenated (forward, L1150-1155) — NOT the overlap-
  aware streaming_decode the tokenizer also ships.

Both are emulated here at 1/12 scale (5 s chunks) on the repo's real 56 s
spoken-dialogue recording, with the tiny tokenizer trained on real speech.
Also measured: the group-size padding distortion (L271-277) and the token
length formula consistency.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import ARTIFACTS_DIR, load_wav, mel_l1, real_speech_files  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mimo_repro.config import TokenizerConfig  # noqa: E402
from mimo_repro.tokenizer import MiMoAudioTokenizerRepro, log_mel_spectrogram  # noqa: E402

torch.manual_seed(0)
cfg = TokenizerConfig.tiny()
tok = MiMoAudioTokenizerRepro(cfg)
tok.load_state_dict(torch.load(os.path.join(ARTIFACTS_DIR, "tok_realspeech.pt"))["state_dict"])
tok.eval()

wav = load_wav(real_speech_files()["long"])
wav = wav / wav.abs().max()
print(f"long recording: {wav.shape[0] / 24000:.1f} s")

mel = log_mel_spectrogram(wav[None, :], cfg)          # [1, T_mel, n_mels]
T_MEL = mel.shape[1]
SEG_MEL = 500                                          # 5 s of mel (official: 6000 = 60 s)

# ---------------------------------------------------------------------------
print("\n=== E2.2 Independent-segment ENCODING vs whole-signal encoding ===")
codes_full = tok.encode(mel)                           # [n_q, 1, T25]

parts = []
for start in range(0, T_MEL, SEG_MEL):
    parts.append(tok.encode(mel[:, start : start + SEG_MEL]))
codes_seg = torch.cat(parts, dim=-1)

n = min(codes_full.shape[-1], codes_seg.shape[-1])
codes_full_c, codes_seg_c = codes_full[..., :n], codes_seg[..., :n]
print(f"token count: full={codes_full.shape[-1]} segmented={codes_seg.shape[-1]}"
      f" (drift = {codes_seg.shape[-1] - codes_full.shape[-1]} tokens from per-segment conv padding)")

seg_tokens = SEG_MEL // 4                              # 25 Hz tokens per segment
boundaries = list(range(seg_tokens, n, seg_tokens))
dist_to_boundary = torch.full((n,), 10_000)
for b in boundaries:
    idx = torch.arange(n)
    dist_to_boundary = torch.minimum(dist_to_boundary, (idx - b).abs())

with torch.no_grad():
    feat_full = tok.encoder(mel)[0]                    # [T25, d]
    feat_parts = [tok.encoder(mel[:, s : s + SEG_MEL])[0]
                  for s in range(0, T_MEL, SEG_MEL)]
feat_seg = torch.cat(feat_parts, dim=0)[:n]
feat_full = feat_full[:n]
feat_delta = (feat_full - feat_seg).norm(dim=-1) / feat_full.norm(dim=-1).clamp_min(1e-8)

mismatch = (codes_full_c[:, 0] != codes_seg_c[:, 0])   # [n_q, n]
print(f"{'bucket (dist to seam, tokens)':35s} {'mismatch q0':>11s} {'q0-3 mean':>10s} {'feat delta':>10s}")
for lo, hi, name in [(0, 2, "0-1 (<=80 ms)"), (2, 5, "2-4"), (5, 13, "5-12"),
                     (13, 40, "13-39"), (40, 10_000, ">=40 (interior)")]:
    m = (dist_to_boundary >= lo) & (dist_to_boundary < hi)
    if m.sum() == 0:
        continue
    q0 = mismatch[0][m].float().mean().item()
    qa = mismatch[:4][:, m].float().mean().item()
    fd = feat_delta[m].mean().item()
    print(f"  {name:33s} {q0:10.1%} {qa:9.1%} {fd:10.4f}   (n={int(m.sum())})")
print("  (feat delta = relative L2 change of the pre-RVQ encoder feature; with the paper's"
      "\n   bidirectional 32-layer encoder the whole segment is one attention context, so any"
      "\n   nonzero delta is content the two encodings genuinely disagree on)")

# ---------------------------------------------------------------------------
print("\n=== E2.3 Independent-segment DECODING seams (official forward(), no overlap) ===")
SEG_TOK = 125                                          # 5 s of tokens (official: 1500 = 60 s)
with torch.no_grad():
    wav_full = tok.decode(codes_full)[0]
    chunks = [tok.decode(codes_full[..., s : s + SEG_TOK])[0]
              for s in range(0, codes_full.shape[-1], SEG_TOK)]
wav_seg = torch.cat(chunks, dim=-1)
n = min(wav_full.shape[-1], wav_seg.shape[-1])

print(f"overall mel-L1 (full-decode vs segment-decode): {mel_l1(wav_full[:n], wav_seg[:n], cfg):.4f}")

samples_per_tok = 960
seam_samples = [s * samples_per_tok for s in range(SEG_TOK, codes_full.shape[-1], SEG_TOK)]
win = 2400                                             # 100 ms analysis window
seam_err, interior_err = [], []
for s in range(0, n - win, win):
    err = (wav_full[s : s + win] - wav_seg[s : s + win]).pow(2).mean().item()
    if any(abs(s + win // 2 - b) < win for b in seam_samples):
        seam_err.append(err)
    else:
        interior_err.append(err)
seam_mse = sum(seam_err) / len(seam_err)
interior_mse = sum(interior_err) / len(interior_err)
print(f"waveform MSE between the two decodes, 100 ms windows:")
print(f"  at seams:    {seam_mse:.5f}   (n={len(seam_err)})")
print(f"  interior:    {interior_mse:.5f}   (n={len(interior_err)})")
print(f"  seam/interior ratio: {seam_mse / max(interior_mse, 1e-12):.1f}x")

jumps_seg = [abs(wav_seg[b].item() - wav_seg[b - 1].item()) for b in seam_samples if b < n]
typical = (wav_full[1:n] - wav_full[: n - 1]).abs().median().item()
print(f"first-difference at seam junctions (segment decode): "
      f"mean {sum(jumps_seg)/len(jumps_seg):.4f} vs whole-signal median step {typical:.4f}")

# ---------------------------------------------------------------------------
print("\n=== E2.4 group_size padding distortion on real clips (preprocess_input L271-277) ===")
files = real_speech_files()
rows = []
for path in files["esd"] + files["prompts"]:
    w = load_wav(path)
    m = log_mel_spectrogram(w[None, :], cfg)
    codes = tok.encode(m)
    n_tok = codes.shape[-1]
    pad = (4 - n_tok % 4) % 4
    dur_in = w.shape[0] / 24000
    dur_out = (n_tok + pad) * samples_per_tok / 24000
    rows.append((os.path.basename(path), dur_in, n_tok, pad, (dur_out - dur_in) * 1000))
pads = [r[3] for r in rows]
deltas = [r[4] for r in rows]
print(f"{len(rows)} real clips: pad frames distribution "
      f"{{0:{pads.count(0)}, 1:{pads.count(1)}, 2:{pads.count(2)}, 3:{pads.count(3)}}}")
print(f"duration change after tokenize(+pad)->decode: min {min(deltas):+.0f} ms, "
      f"max {max(deltas):+.0f} ms, mean {sum(deltas)/len(deltas):+.0f} ms")
print("  (padding repeats the LAST frame 1-3 times: up to 120 ms of frozen spectrum appended,")
print("   and every clip's length is quantized to the 160 ms patch grid)")

# ---------------------------------------------------------------------------
print("\n=== E2.5 Official token-length formula vs actual conv arithmetic ===")
import torch.nn as nn

conv1 = nn.Conv1d(80, 8, kernel_size=3, padding=1)
conv2 = nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1)


def official_formula(mel_len, kernel_size=3, stride=2):
    tgt = mel_len + 3 - kernel_size
    return (tgt + 2 - kernel_size) // stride + 1

bad = []
for L in range(1, 200):
    with torch.no_grad():
        real = conv2(conv1(torch.zeros(1, 80, L))).shape[-1]
    if real != official_formula(L):
        bad.append((L, real, official_formula(L)))
print(f"mel lengths 1..199: {len(bad)} mismatches between get_output_length and the real convs"
      + (f" -> {bad[:5]}" if bad else " (formula is consistent)"))

print("\nTrack 2 done.")
