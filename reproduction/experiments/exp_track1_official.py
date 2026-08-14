"""Track 1: behavioural tests of the OFFICIAL MiMo-Audio code on CPU.

The official LM stack (src/mimo_audio/modeling_mimo_audio.py,
process_speechdata.py) only needs torch + transformers, so unlike the
tokenizer (hard flash-attn dependency) it can be instantiated at tiny scale
on CPU and driven through its real generation loop.  Each experiment prints
an OBSERVED line; nothing here uses the reproduction code except as an
oracle for the delay schedule.
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

REPO = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, REPO)

from src.mimo_audio.modeling_mimo_audio import (  # noqa: E402
    MiMoAudioArguments,
    MiMoAudioConfig,
    MiMoAudioForCausalLM,
    MiMoSampler,
    MiMoStopper,
)
from src.mimo_audio.process_speechdata import InputSegment, StreamingInputSegment  # noqa: E402

torch.manual_seed(0)

G, C = 4, 4
VOCABS = [17, 19, 9, 11]           # distinct sizes so a recorder can identify channels
EMPTIES = [16, 18, 8, 10]
STEP = (C + 1) * G


def tiny_config() -> MiMoAudioConfig:
    return MiMoAudioConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        head_dim=8,
        speech_vocab_size="-".join(map(str, VOCABS)),
        speech_zeroemb_idx="-".join(map(str, EMPTIES)),
        delay_pattern="0-1-2-3",
        group_size=G,
        audio_channels=C,
        local_dim=16,
        local_layers=2,
        local_attn_heads=4,
        local_ffn_dim=32,
        input_local_layers=2,
    )


ARGS = MiMoAudioArguments(
    model_name_or_path="", sosp_idx=50, eosp_idx=51, sostm_idx=52,
    eostm_idx=53, eot_idx=54, empty_idx=55,
)


def build_prompt(n_patches: int, text_token: int = 7) -> torch.Tensor:
    """Flattened frame-major prompt [1, n_patches*STEP]: text positions with
    audio channels filled by empty ids (the layout slm_sample produces)."""
    frames = []
    for _ in range(n_patches * G):
        frames.append([text_token] + EMPTIES)
    return torch.tensor(frames, dtype=torch.long).reshape(1, -1)


def section(title):
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
section("E1.1 MiMoSampler.top_k at batch size > 1")
sampler = MiMoSampler(do_sample=False, top_k=5)
scores1 = torch.randn(1, 100)
ok1 = sampler.sample(scores1.clone())
print(f"OBSERVED B=1: works, token={ok1.item()}")
try:
    sampler.sample(torch.randn(2, 100))
    print("OBSERVED B=2: no crash (unexpected)")
except RuntimeError as e:
    print(f"OBSERVED B=2: RuntimeError -> {str(e)[:110]}")
# case where V == B by coincidence: silently WRONG instead of crashing.
# rows have different top-1 values, so the [B]-shaped threshold broadcasts
# across columns instead of rows and masks the true argmax of row 0
sampler2 = MiMoSampler(do_sample=False, top_k=1)
sq = torch.tensor([[0.0, 5.0], [9.0, 0.0]])  # B=2, V=2; correct top-1 sample = [1, 0]
try:
    out = sampler2.sample(sq.clone())
    print(f"OBSERVED B=2,V=2 (shape coincidence): returns {out.tolist()}, correct is [1, 0]"
          f" -> {'SILENTLY WRONG' if out.tolist() != [1, 0] else 'correct by luck'}")
except RuntimeError as e:
    print(f"OBSERVED B=2,V=2: RuntimeError -> {str(e)[:110]}")

# ---------------------------------------------------------------------------
section("E1.2 MiMoSampler.top_p vs reference nucleus filter (B=1)")
mismatch = 0
for trial in range(500):
    logits = torch.randn(1, 50) * 3
    p = 0.9
    got = MiMoSampler(top_p=p).process(logits.clone())
    kept_official = torch.isfinite(got[0])
    # reference: legacy HF algorithm (ascending sort, drop cumulative <= 1-p)
    sl, si = torch.sort(logits[0])
    cum = sl.softmax(-1).cumsum(-1)
    remove = cum <= (1 - p)
    kept_ref = torch.ones(50, dtype=torch.bool)
    kept_ref[si[remove]] = False
    mismatch += int(not torch.equal(kept_official, kept_ref))
print(f"OBSERVED: {mismatch}/500 mismatches vs reference nucleus masking")

# ---------------------------------------------------------------------------
section("E1.3 MiMoStopper: detection position, latency and batching")
stopper = MiMoStopper(group_size=G, audio_channels=C, stop_tokens=[53])
seq_stop_last = torch.cat([build_prompt(2), build_prompt(1, text_token=53)], dim=1)
seq_stop_mid = torch.cat([build_prompt(1), build_prompt(1, text_token=53), build_prompt(1)], dim=1)
print(f"OBSERVED stop token in final patch:      is_done={stopper(seq_stop_last, None)[0].item()}")
print(f"OBSERVED stop token one patch earlier:   is_done={stopper(seq_stop_mid, None)[0].item()}  (single-position check: an already-passed stop is never seen)")
stopper_min = MiMoStopper(group_size=G, audio_channels=C, stop_tokens=[53], min_length=10)
print(f"OBSERVED same stop but min_length=10:    is_done={stopper_min(seq_stop_last, None)[0].item()}")
batch = torch.cat([build_prompt(3), seq_stop_last], dim=0)  # stop only in row 1
res = stopper(batch, None)
print(f"OBSERVED B=2, stop only in row 1:        is_done={res.tolist()}  (row 0 decides for the whole batch)")

# ---------------------------------------------------------------------------
section("E1.4a Official model, default fp32 load -> hardcoded bf16 buffer")
model = MiMoAudioForCausalLM(tiny_config(), ARGS).eval()
prompt = build_prompt(3)
try:
    with torch.no_grad():
        model.generate(
            prompt,
            global_sampler=MiMoSampler(do_sample=False),
            local_sampler=MiMoSampler(do_sample=False),
            max_length=5,
        )
    # static analysis predicted a dtype crash here; empirically the fp32
    # embeddings are downcast into the hardcoded bf16 buffer (allowed: same
    # dtype category) and RMSNorm promotes activations back to fp32, so the
    # model runs with a silent precision squeeze instead of failing
    print("OBSERVED: fp32 generate RUNS -- the hardcoded bf16 buffer silently downcasts"
          " fp32 embeddings (precision loss), then RMSNorm promotes back to fp32.")
except RuntimeError as e:
    print(f"OBSERVED: RuntimeError -> {str(e)[:120]}")

# ---------------------------------------------------------------------------
section("E1.4b Same model cast to bf16 (the only supported dtype)")
model = model.to(torch.bfloat16)
with torch.no_grad():
    out = model.generate(
        prompt,
        global_sampler=MiMoSampler(do_sample=False),
        local_sampler=MiMoSampler(do_sample=False),
        max_length=6,
    )
new = (out.shape[1] - prompt.shape[1]) // STEP
print(f"OBSERVED: generation works, {new} new patches, output len {out.shape[1]}")

# ---------------------------------------------------------------------------
section("E1.4c generate() with default global_sampler=None")
try:
    with torch.no_grad():
        model.generate(prompt, local_sampler=MiMoSampler(do_sample=False), max_length=5)
    print("OBSERVED: ran (unexpected)")
except AttributeError as e:
    print(f"OBSERVED: AttributeError -> {str(e)[:100]}")

# ---------------------------------------------------------------------------
section("E1.4d generate() with batch size 2")
try:
    with torch.no_grad():
        out2 = model.generate(
            torch.cat([build_prompt(3, 7), build_prompt(3, 9)], dim=0),
            global_sampler=MiMoSampler(do_sample=False),
            local_sampler=MiMoSampler(do_sample=False),
            max_length=6,
        )
    t0 = out2[0].reshape(-1, C + 1)[:, 0][3 * G :: G]
    t1 = out2[1].reshape(-1, C + 1)[:, 0][3 * G :: G]
    print(f"OBSERVED: no crash; generated text rows row0={t0.tolist()} row1={t1.tolist()}"
          f" (row 1's text/speech branch is decided by row 0: `next_text_tokens[0]`)")
except Exception as e:
    print(f"OBSERVED: {type(e).__name__} -> {str(e)[:120]}")

# ---------------------------------------------------------------------------
section("E1.5 Official local_forward delay schedule vs Eq.15 staircase")

class RecorderSampler(MiMoSampler):
    def __init__(self):
        super().__init__(do_sample=False)
        self.calls = []

    def sample(self, scores, removed_tokens=None):
        self.calls.append(scores.shape[-1])
        return super().sample(scores, removed_tokens)

rec = RecorderSampler()
with torch.no_grad():
    tokens = model.local_forward(
        torch.randn(1, 1, 16, dtype=torch.bfloat16), torch.long, torch.device("cpu"), rec
    )
# reconstruct schedule: at step t, channels r with d_r <= t < d_r + G fire in ascending order
D = [0, 1, 2, 3]
expected = []
for t in range(G + max(D)):
    for r in range(C):
        if D[r] <= t < D[r] + G:
            expected.append(VOCABS[r])
print(f"OBSERVED call sequence (vocab sizes): {rec.calls}")
print(f"Eq.15 staircase prediction:           {expected}")
print(f"MATCH: {rec.calls == expected}; emitted patch shape {tuple(tokens.shape)}; "
      f"empty ids present: {any((tokens[0, :, r] == EMPTIES[r]).any().item() for r in range(C))}")

# ---------------------------------------------------------------------------
section("E1.6 Official patch encoder causality flag (paper says bidirectional)")
x = torch.randn(1, 1, G, 16, dtype=torch.bfloat16)
x2 = x.clone()
x2[0, 0, G - 1] += 5.0
with torch.no_grad():
    model.config.input_full_attention = None  # repository default
    d_default = (model.apply_input_local_transformer(x)[0, 0, 0]
                 - model.apply_input_local_transformer(x2)[0, 0, 0]).abs().max().item()
    model.config.input_full_attention = True  # what the paper describes
    d_full = (model.apply_input_local_transformer(x)[0, 0, 0]
              - model.apply_input_local_transformer(x2)[0, 0, 0]).abs().max().item()
model.config.input_full_attention = None
print(f"OBSERVED effect of perturbing frame 3 on frame 0's encoding:")
print(f"  default config (input_full_attention=None): delta={d_default:.6f}  -> CAUSAL")
print(f"  input_full_attention=True:                  delta={d_full:.6f}  -> bidirectional")

# ---------------------------------------------------------------------------
section("E1.7 InputSegment / StreamingInputSegment format behaviour")

class MockTokenizer:
    specials = {"<|sosp|>": 50, "<|eosp|>": 51, "<|sostm|>": 52, "<|eostm|>": 53, "<|eot|>": 54}

    def __call__(self, text, **kw):
        if text in self.specials:
            ids = [self.specials[text]]
        else:
            ids = [ord(c) % 40 for c in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def convert_tokens_to_ids(self, tok):
        return self.specials[tok]

mock = MockTokenizer()

seg = InputSegment(text="abc", speech_zeroemb_idx=EMPTIES, text_zeroemb_idx=55)
ids = seg.to_input_id(mock, G, C)
print(f"OBSERVED text segment 'abc' -> shape {tuple(ids.shape)}; text row: {ids[0].tolist()}")
print(f"  (each text token occupies {G} slots, filler is -100, not the empty id)")

audio = torch.arange(2 * G * C)  # 2 patches
seg = InputSegment(audio=audio, speech_zeroemb_idx=EMPTIES, text_zeroemb_idx=55)
ids = seg.to_input_id(mock, G, C)
print(f"OBSERVED audio segment (2 patches) -> shape {tuple(ids.shape)}; "
      f"text row: {ids[0].tolist()} (sosp/eosp patches added)")

try:
    InputSegment(audio=torch.arange(G * C + C), speech_zeroemb_idx=EMPTIES,
                 text_zeroemb_idx=55).to_input_id(mock, G, C)
    print("OBSERVED non-divisible audio: accepted (unexpected)")
except AssertionError as e:
    print(f"OBSERVED non-divisible audio: bare AssertionError -> '{str(e)[:80]}'")

stream = StreamingInputSegment(
    text="abcdefghijkl",                      # 12 tokens -> segments 5/5/2(+eot)
    audio=torch.arange(8 * G * C),            # 8 patches -> segments 5/3
    speech_zeroemb_idx=EMPTIES, text_zeroemb_idx=55,
    tokenizer=mock, group_size=G, audio_channels=C,
)
ids = stream.to_input_id(mock, G, C)
text_row = ids[0, ::G]
kinds = ["T" if t not in (55,) else "A" for t in text_row.tolist()]
print(f"OBSERVED streaming interleave, {ids.shape[1] // G} patch positions: {''.join(kinds)}")
print(f"  (T=text-ish token, A=audio position; pattern = sostm | 5T 5A 5T 3A | leftover text+eot | eostm)")

try:
    StreamingInputSegment(
        text="", audio=torch.arange(2 * G * C),
        speech_zeroemb_idx=EMPTIES, text_zeroemb_idx=55,
        tokenizer=mock, group_size=G, audio_channels=C,
    ).to_input_id(mock, G, C)
    print("OBSERVED empty text with audio: accepted")
except Exception as e:
    print(f"OBSERVED empty text with audio: {type(e).__name__} -> {str(e)[:90]}")

# ---------------------------------------------------------------------------
section("E1.8 Stop granularity: audio can only end on a 640 ms patch boundary")

class ScriptedSampler(MiMoSampler):
    """Emit <empty> (speech) for n patches, then the stop token."""

    def __init__(self, n_speech, empty_idx, stop_idx):
        super().__init__(do_sample=False)
        self.n = n_speech
        self.empty = empty_idx
        self.stop = stop_idx
        self.count = 0

    def sample(self, scores, removed_tokens=None):
        self.count += 1
        tok = self.empty if self.count <= self.n else self.stop
        return torch.full((scores.shape[0],), tok, dtype=torch.long)

scripted = ScriptedSampler(2, ARGS.empty_idx, 53)
stop = MiMoStopper(group_size=G, audio_channels=C, stop_tokens=[53])
with torch.no_grad():
    out = model.generate(
        prompt,
        global_sampler=scripted,
        local_sampler=MiMoSampler(do_sample=False),
        stopping_criteria=[stop],
        max_length=20,
    )
gen = out[0, prompt.shape[1]:].reshape(-1, C + 1)
text_channel = gen[::G, 0]
frames_of_audio = int((gen[:, 0] == ARGS.empty_idx).sum())
print(f"OBSERVED generated text channel per patch: {text_channel.tolist()}")
print(f"  2 speech patches (= {2*G} frames) then stop patch; the <|eostm|> itself consumes a full "
      f"patch of {G} frames ({G/25*1000:.0f} ms) filled with empty audio ids")
print(f"  audio frames emitted: {frames_of_audio}, minimum audio granularity = {G} frames = {G/25*1000:.0f} ms")

print("\nAll Track 1 experiments done.")
