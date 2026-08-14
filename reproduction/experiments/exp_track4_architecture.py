"""Track 4: controlled-training probes of the MiMo-Audio architecture.

Each probe constructs synthetic data with a KNOWN dependency structure and a
known entropy floor, trains the (tiny) reproduction end to end, and measures
where the architecture loses information:

E4.1  patch-boundary bottleneck   -- frame 0 of every patch is conditioned
      only through the LLM hidden state (one 1024-d vector at paper scale),
      frames 1..G-1 also see their patch-local past directly.  Per-frame NLL
      against the analytic floor quantifies the seam cost.
E4.2  delay-pattern ablation      -- same-frame cross-codebook dependencies
      are only modelable WITH the delay; measure per-channel NLL for
      delay (0,1,2,3) vs (0,0,0,0).
E4.3  loss-weight competition     -- paper weights (text 100, audio 12-8-6-4)
      vs uniform weights on identical interleaved data: what does the 100:1
      allocation buy and cost at fixed capacity?
E4.4  A2T vs reconstruction       -- tokenizer trained with lambda_A2T in
      {0, 10} on real speech: does the semantic objective tax reconstruction
      (the 'semantic-acoustic conflict' the paper claims scale resolves)?
E4.5  patch-encoder causality     -- the paper says bidirectional, the code
      defaults to causal: does the difference matter for modeling quality?
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mimo_repro.config import MiMoLMConfig  # noqa: E402
from mimo_repro.patch_lm import MiMoAudioLMRepro, build_delayed_patch  # noqa: E402

G, C, K = 4, 4, 16          # group size, channels, effective token alphabet
T = 8                        # patches per sequence
BATCH = 8


def lm_config(delay=(0, 1, 2, 3), bidir=True) -> MiMoLMConfig:
    return MiMoLMConfig(
        text_vocab_size=64,
        audio_vocab_sizes=(K + 1,) * C,
        audio_channels=C,
        group_size=G,
        delay_pattern=delay,
        llm_layers=2, llm_dim=64, llm_ffn_dim=128, llm_heads=4,
        patch_encoder_layers=2, patch_dim=32, patch_ffn_dim=64, patch_heads=4,
        patch_encoder_bidirectional=bidir,
        patch_decoder_layers=2,
        audio_loss_weights=(12.0, 8.0, 6.0, 4.0),
    )


def random_walk_batch(cfg, batch=BATCH, t=T):
    """All-speech batch; every channel an independent +-1/0 random walk mod K."""
    frames = t * G
    steps = torch.randint(-1, 2, (batch, frames, C))
    steps[:, 0] = torch.randint(0, K, (batch, C))
    audio = steps.cumsum(dim=1) % K
    audio = audio.reshape(batch, t, G, C)
    text = torch.full((batch, t), cfg.empty_text_id)
    return text, audio


def crosscode_batch(cfg, batch=BATCH, t=T):
    """Channel 0 iid uniform; channel r = (c0 + r) mod K, same frame."""
    frames = t * G
    c0 = torch.randint(0, K, (batch, frames, 1))
    audio = torch.cat([(c0 + r) % K for r in range(C)], dim=-1)
    audio = audio.reshape(batch, t, G, C)
    text = torch.full((batch, t), cfg.empty_text_id)
    return text, audio


def train(model, cfg, data_fn, steps=600, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        text, audio = data_fn(cfg)
        out = model(text, audio, text.clone(), audio.clone())
        opt.zero_grad()
        out.loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def per_frame_channel_nll(model, cfg, data_fn, n_batches=40):
    """Teacher-forced audio NLL broken down by frame-in-patch and channel."""
    nll = torch.zeros(G, C)
    count = 0
    for _ in range(n_batches):
        text, audio = data_fn(cfg)
        embeds, _ = model.prepare_inputs(text, audio)
        hidden = model.llm(embeds)
        seeds = hidden[:, :-1].reshape(-1, cfg.llm_dim)
        target = audio[:, 1:].reshape(-1, G, C)
        delayed = build_delayed_patch(target, cfg.delay_pattern, cfg.audio_empty_ids)
        logits = model.patch_decoder(seeds, delayed)
        for r in range(C):
            d = cfg.delay_pattern[r]
            lg = logits[r][:, d : d + G]                     # [N, G, V]
            ce = F.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), target[..., r].reshape(-1),
                reduction="none",
            ).reshape(-1, G)
            nll[:, r] += ce.mean(dim=0)
        count += 1
    return nll / count


def section(t):
    print(f"\n=== {t} ===")


# ---------------------------------------------------------------------------
section("E4.1 Patch-boundary bottleneck (per-frame NLL, random-walk data)")
cfg = lm_config()
model = train(MiMoAudioLMRepro(cfg), cfg, random_walk_batch)
nll = per_frame_channel_nll(model, cfg, random_walk_batch)
floor = math.log(3)
per_frame = nll.mean(dim=1)
print(f"analytic floor per frame (random +-1/0 walk): ln 3 = {floor:.3f} nats")
for f in range(G):
    marker = "  <- conditioned only through the LLM patch bottleneck" if f == 0 else ""
    print(f"  frame {f} in patch: NLL {per_frame[f]:.3f}  (excess {per_frame[f] - floor:+.3f}){marker}")
print(f"frame-0 excess over frames 1-3 mean: "
      f"{(per_frame[0] - per_frame[1:].mean()):+.3f} nats "
      f"({(per_frame[0] - per_frame[1:].mean()) / floor * 100:+.0f}% of the floor)")

# ---------------------------------------------------------------------------
section("E4.2 Delay-pattern ablation (same-frame cross-codebook dependency)")
for delay in [(0, 1, 2, 3), (0, 0, 0, 0)]:
    cfg_d = lm_config(delay=delay)
    m = train(MiMoAudioLMRepro(cfg_d), cfg_d, crosscode_batch, steps=500)
    nll = per_frame_channel_nll(m, cfg_d, crosscode_batch, n_batches=20).mean(dim=0)
    print(f"  delay {delay}: per-channel NLL {[f'{v:.3f}' for v in nll.tolist()]}"
          f"   (floors: ch0 = ln{K} = {math.log(K):.3f}, ch1-3 = 0.0 if conditioning possible)")

# ---------------------------------------------------------------------------
section("E4.3 Loss-weight competition (paper 100:12-8-6-4 vs uniform 1:1)")

def interleaved_batch(cfg, batch=BATCH, t=T):
    """First half text (sticky +1 Markov chain over 16 ids), second half speech."""
    text = torch.zeros(batch, t, dtype=torch.long)
    text[:, 0] = torch.randint(0, K, (batch,))
    for i in range(1, t // 2):
        stay = torch.rand(batch) < 0.8
        text[:, i] = torch.where(stay, (text[:, i - 1] + 1) % K,
                                 torch.randint(0, K, (batch,)))
    audio = torch.tensor(cfg.audio_empty_ids).expand(batch, t, G, C).clone()
    _, walk = random_walk_batch(cfg, batch, t - t // 2)
    audio[:, t // 2 :] = walk
    text[:, t // 2 :] = cfg.empty_text_id
    return text, audio


@torch.no_grad()
def unweighted_eval(model, cfg, n=40):
    tl, al = 0.0, 0.0
    for _ in range(n):
        text, audio = interleaved_batch(cfg)
        audio_targets = audio.clone()
        audio_targets[:, : T // 2] = -100
        out = model(text, audio, text.clone(), audio_targets)
        tl += out.text_loss.item()
        # normalize the weighted audio loss back to a plain mean CE
        al += out.audio_loss.item() / sum(cfg.audio_loss_weights)
    return tl / n, al / n

for name, weights in [("paper 100 / 12-8-6-4", (100.0, (12.0, 8.0, 6.0, 4.0))),
                      ("uniform 1 / 1-1-1-1", (1.0, (1.0, 1.0, 1.0, 1.0)))]:
    cfg_w = lm_config()
    cfg_w.text_loss_weight, cfg_w.audio_loss_weights = weights

    def data_fn(c):
        text, audio = interleaved_batch(c)
        return text, audio

    torch.manual_seed(0)
    m = MiMoAudioLMRepro(cfg_w)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(600):
        text, audio = interleaved_batch(cfg_w)
        audio_targets = audio.clone()
        audio_targets[:, : T // 2] = -100
        out = m(text, audio, text.clone(), audio_targets)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
    m.eval()
    tl, al = unweighted_eval(m, cfg_w)
    print(f"  {name:22s} -> text NLL {tl:.3f}, audio NLL (unweighted mean) {al:.3f}")

# ---------------------------------------------------------------------------
section("E4.4 A2T weight vs reconstruction (real speech, 250 steps each)")
from common import load_wav, real_speech_files  # noqa: E402
from mimo_repro.config import TokenizerConfig  # noqa: E402
from common import train_tiny_tokenizer  # noqa: E402

files = real_speech_files()
paths = files["esd"] + files["prompts"]
# per-segment class label = source file index, repeated as a toy transcription
segs_per_file = []
labels = []
for fi, path in enumerate(paths):
    wav = load_wav(path)
    wav = wav / (wav.abs().max() + 1e-8)
    n = wav.shape[0] // 24000
    labels += [fi] * n
label_seqs = torch.tensor(labels)[:, None].repeat(1, 6)

for lam in (0.0, 10.0):
    tok, cfg_t, hist = train_tiny_tokenizer(
        steps=250, log_every=0, lambda_a2t=lam,
        text_labels=label_seqs if lam > 0 else None,
    )
    recon = sum(hist["recon"][-20:]) / 20
    print(f"  lambda_A2T = {lam:4.1f}: final recon loss {recon:.3f}")

# ---------------------------------------------------------------------------
section("E4.5 Patch encoder: bidirectional (paper) vs causal (code default)")
for bidir in (True, False):
    cfg_b = lm_config(bidir=bidir)
    m = train(MiMoAudioLMRepro(cfg_b), cfg_b, random_walk_batch, steps=500, seed=1)
    nll = per_frame_channel_nll(m, cfg_b, random_walk_batch, n_batches=20)
    print(f"  bidirectional={str(bidir):5s}: audio NLL {nll.mean():.3f} "
          f"(floor {math.log(3):.3f})")

print("\nTrack 4 done.")
