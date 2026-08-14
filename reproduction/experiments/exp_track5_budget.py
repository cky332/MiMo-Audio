"""Track 5: context-budget and capacity arithmetic of the deployed system.

Pure arithmetic from constants that are hardcoded in the official code and
the paper -- no training.  Numbers the paper never states explicitly:

- one LLM position = 1 patch = G(4) frames @ 25 Hz = 160 ms of audio,
  OR exactly one text token (each text token also occupies a full
  group_size slot block, mimo_audio.py get_input_ids / InputSegment);
- generation cap: max_length = 8192 positions (mimo_audio.py L109);
- multiturn message_list mode replays and re-encodes ALL history each call.
"""

import math

POS_PER_SEC = 6.25       # patches per second of audio
MAX_POS = 8192

print("=== E5.1 What fits in the 8192-position context ===")
print(f"1 position = 160 ms audio or 1 text token; cap = {MAX_POS} positions")
print(f"pure audio ceiling: {MAX_POS / POS_PER_SEC:.0f} s = {MAX_POS / POS_PER_SEC / 60:.1f} min")

# spoken-dialogue round cost, from the official templates (get_spoken_dialogue_sft_prompt
# + StreamingInputSegment):  user turn = chat scaffold + audio; assistant turn =
# interleaved transcript (1 pos per text token) + audio + sostm/eot/eostm overhead.
def round_cost(user_sec, assistant_sec, transcript_tokens, scaffold_tokens=16):
    user = math.ceil(user_sec * POS_PER_SEC)
    assistant_audio = math.ceil(assistant_sec * POS_PER_SEC)
    return scaffold_tokens + user + assistant_audio + transcript_tokens + 3  # sostm/eot/eostm

for u, a, tr in [(5, 10, 35), (10, 20, 70), (30, 60, 210)]:
    cost = round_cost(u, a, tr)
    print(f"dialogue round ({u:2d}s user + {a:2d}s reply + {tr} transcript tokens): "
          f"{cost:4d} positions -> context full after {MAX_POS // cost} rounds "
          f"(~{MAX_POS // cost * (u + a) / 60:.0f} min of conversation)")

print("\ntext-vs-audio exchange rate: 1 s of speech costs 6.25 positions;")
print("a typical 15-token spoken sentence (~4 s) costs 25 audio positions vs 15 text positions")
print("history replay (message_list mode) re-tokenizes every prior audio turn on every call:")
for n in (5, 10, 20):
    print(f"  round {n:2d}: audio re-encoded so far ~{n * (n - 1) // 2}x one turn's cost (quadratic)")

print("\n=== E5.2 Patch-decoder conditioning capacity ===")
bits_per_frame = 2 * 10 + 6 * 7          # 8 codebooks: 2x1024 + 6x128
patch_bits = 4 * bits_per_frame
vec_bits = 1024 * 16                      # 1024-dim bf16 hidden state
print(f"audio content per patch: 4 frames x {bits_per_frame} bit = {patch_bits} bit")
print(f"conditioning channel: one 1024-dim bf16 vector = {vec_bits} raw bit "
      f"({vec_bits / patch_bits:.0f}x the content) -> the bottleneck is representational,"
      "\nnot information-theoretic (track 4 E4.1 measures the realized gap)")

print("\n=== E5.3 Decode chunking at deployment scale ===")
print("official forward() decodes generated audio in independent 1500-token (60 s) chunks")
print("with no overlap (mimo_audio.py L1150-1155) although streaming_decode with")
print("left/right overlap of 10 s / 1.6 s exists in the tokenizer -- every reply longer")
print("than 60 s contains hard decode seams (track 2 E2.3 quantifies them);")
print("encoding side likewise chunks input mel at 60 s (L258-266) with no overlap.")

print("\nTrack 5 done.")
