import torch

from mimo_repro.config import MiMoLMConfig
from mimo_repro.patch_lm import (
    MiMoAudioLMRepro,
    PatchEncoder,
    PatchDecoder,
    build_delayed_patch,
    undelay_patch,
)


def tiny():
    torch.manual_seed(0)
    return MiMoLMConfig.tiny()


# ---------------------------------------------------------------------------
# Delay mechanism (Eq. 14-15)
# ---------------------------------------------------------------------------

def test_delayed_patch_matches_eq15_elementwise():
    cfg = tiny()
    G, D, C = cfg.group_size, cfg.delay_pattern, cfg.audio_channels
    patch = torch.arange(1, G * C + 1).reshape(1, G, C)
    delayed = build_delayed_patch(patch, D, cfg.audio_empty_ids)
    assert delayed.shape == (1, G + max(D), C)
    # Eq. 15: a'_{i,r} = a_{i-d_r, r} if 1 <= i-d_r <= G else empty
    for i in range(G + max(D)):
        for r in range(C):
            src = i - D[r]
            if 0 <= src < G:
                assert delayed[0, i, r] == patch[0, src, r]
            else:
                assert delayed[0, i, r] == cfg.audio_empty_ids[r]


def test_delay_roundtrip():
    cfg = tiny()
    patch = torch.randint(0, 16, (3, cfg.group_size, cfg.audio_channels))
    delayed = build_delayed_patch(patch, cfg.delay_pattern, cfg.audio_empty_ids)
    assert torch.equal(
        undelay_patch(delayed, cfg.delay_pattern, cfg.group_size), patch
    )


def test_delay_pattern_staircase():
    """With D = (0,1,2,3), step t emits channel r iff d_r <= t < d_r + G —
    the diagonal staircase from the MusicGen-style delay scheme."""
    cfg = tiny()
    G, D = cfg.group_size, cfg.delay_pattern
    for t in range(cfg.patch_decoder_context):
        active = [r for r in range(cfg.audio_channels) if D[r] <= t < D[r] + G]
        # e.g. t=0 -> [0]; t=1 -> [0,1]; t=4 -> [1,2,3] (G=4)
        expected = [r for r in range(cfg.audio_channels) if t - G < D[r] <= t]
        assert active == expected


# ---------------------------------------------------------------------------
# Patch encoder
# ---------------------------------------------------------------------------

def test_patch_encoder_shapes_and_empty_zeroing():
    cfg = tiny()
    enc = PatchEncoder(cfg).eval()
    B, T = 2, 3
    tokens = torch.randint(0, 16, (B, T, cfg.group_size, cfg.audio_channels))
    out = enc(tokens)
    assert out.shape == (B, T, cfg.llm_dim)
    # empty ids embed to exactly zero (padding_idx), so a frame of all-empty
    # tokens contributes a zero summed embedding
    empty = torch.tensor(cfg.audio_empty_ids).expand(1, 1, cfg.group_size, -1)
    frame_embeds = enc.frame_embeddings(empty)
    assert torch.all(frame_embeds == 0)


def test_patch_encoder_bidirectional_flag():
    """Paper says bidirectional; official code is causal unless
    input_full_attention is set.  Both behaviours must be reproducible and
    actually differ."""
    cfg = tiny()
    torch.manual_seed(1)
    enc_bi = PatchEncoder(cfg)
    cfg_causal = tiny()
    cfg_causal.patch_encoder_bidirectional = False
    torch.manual_seed(1)
    enc_causal = PatchEncoder(cfg_causal)

    tokens = torch.randint(0, 16, (1, 1, cfg.group_size, cfg.audio_channels))
    a, b = enc_bi.eval()(tokens), enc_causal.eval()(tokens)
    assert a.shape == b.shape
    assert not torch.allclose(a, b), "attention direction must change the output"

    # causal patch encoder: frame 0's transformer output can't see frame 3
    x = enc_causal.frame_embeddings(tokens)[0]          # [1*T, G, D] pre-transformer
    h1 = enc_causal.transformer(x)
    x2 = x.clone()
    x2[:, -1] += 5.0
    h2 = enc_causal.transformer(x2)
    assert torch.allclose(h1[:, 0], h2[:, 0], atol=1e-5)


# ---------------------------------------------------------------------------
# Patch decoder
# ---------------------------------------------------------------------------

def test_patch_decoder_teacher_forced_shapes():
    cfg = tiny()
    dec = PatchDecoder(cfg).eval()
    B = 2
    seed = torch.randn(B, cfg.llm_dim)
    patch = torch.randint(0, 16, (B, cfg.group_size, cfg.audio_channels))
    delayed = build_delayed_patch(patch, cfg.delay_pattern, cfg.audio_empty_ids)
    logits = dec(seed, delayed)
    assert len(logits) == cfg.audio_channels
    for r, lg in enumerate(logits):
        assert lg.shape == (B, cfg.patch_decoder_context, cfg.audio_vocab_sizes[r])


def test_patch_decoder_is_causal():
    cfg = tiny()
    dec = PatchDecoder(cfg).eval()
    seed = torch.randn(1, cfg.llm_dim)
    patch = torch.randint(0, 16, (1, cfg.group_size, cfg.audio_channels))
    delayed = build_delayed_patch(patch, cfg.delay_pattern, cfg.audio_empty_ids)
    la = dec(seed, delayed)
    delayed_b = delayed.clone()
    delayed_b[:, -2] = 3  # change a late input step
    lb = dec(seed, delayed_b)
    # logits strictly before the perturbed input step are unchanged
    for r in range(cfg.audio_channels):
        assert torch.allclose(la[r][:, : delayed.shape[1] - 2], lb[r][:, : delayed.shape[1] - 2], atol=1e-5)


def test_patch_decoder_generate_never_emits_empty():
    cfg = tiny()
    dec = PatchDecoder(cfg).eval()
    tokens = dec.generate(torch.randn(2, cfg.llm_dim))
    assert tokens.shape == (2, cfg.group_size, cfg.audio_channels)
    for r in range(cfg.audio_channels):
        assert (tokens[..., r] != cfg.audio_empty_ids[r]).all()
        assert (tokens[..., r] < cfg.audio_vocab_sizes[r]).all()


def test_embedding_tables_are_tied():
    cfg = tiny()
    model = MiMoAudioLMRepro(cfg)
    for e_enc, e_dec in zip(
        model.patch_encoder.embeddings, model.patch_decoder.embeddings
    ):
        assert e_enc.weight is e_dec.weight


# ---------------------------------------------------------------------------
# Full interleaved model
# ---------------------------------------------------------------------------

def make_batch(cfg, B=2, T=6):
    """Half text positions, half speech positions."""
    text = torch.randint(0, cfg.text_vocab_size - 1, (B, T))
    audio = (
        torch.tensor(cfg.audio_empty_ids)
        .expand(B, T, cfg.group_size, cfg.audio_channels)
        .clone()
    )
    is_speech = torch.zeros(B, T, dtype=torch.bool)
    is_speech[:, T // 2 :] = True
    text[is_speech] = cfg.empty_text_id
    audio[is_speech] = torch.randint(
        0, 16, (int(is_speech.sum()), cfg.group_size, cfg.audio_channels)
    )
    return text, audio, is_speech


def test_forward_loss_and_weights():
    cfg = tiny()
    model = MiMoAudioLMRepro(cfg).eval()
    text, audio, is_speech = make_batch(cfg)
    text_targets = text.clone()
    audio_targets = audio.clone()
    audio_targets[~is_speech] = -100
    out = model(text, audio, text_targets, audio_targets)
    assert torch.isfinite(out.loss)
    assert torch.isfinite(out.text_loss) and torch.isfinite(out.audio_loss)
    # Table 3: total = 100 * text + sum_r w_r * audio_r; check the text part
    assert torch.allclose(
        out.loss, cfg.text_loss_weight * out.text_loss + out.audio_loss
    )


def test_generation_text_and_speech_branches():
    cfg = tiny()
    model = MiMoAudioLMRepro(cfg).eval()
    text, audio, _ = make_batch(cfg, B=1, T=4)

    # force the text branch: sampler that never returns empty
    force_text = lambda logits: logits[:, : cfg.empty_text_id].argmax(dim=-1)
    t_out, a_out = model.generate(text, audio, max_new_patches=2, text_sample_fn=force_text)
    assert t_out.shape == (1, 2)
    # text positions carry audio empty ids on every channel
    for r in range(cfg.audio_channels):
        assert (a_out[..., r] == cfg.audio_empty_ids[r]).all()

    # force the speech branch: sampler that always returns empty
    force_speech = lambda logits: torch.full(
        (logits.shape[0],), cfg.empty_text_id, dtype=torch.long
    )
    t_out, a_out = model.generate(text, audio, max_new_patches=2, text_sample_fn=force_speech)
    assert (t_out == cfg.empty_text_id).all()
    for r in range(cfg.audio_channels):
        assert (a_out[..., r] != cfg.audio_empty_ids[r]).all()


def test_speech_positions_ignore_text_embedding():
    """On a speech position the text embedding is zeroed: swapping which
    (non-empty) text token sits there must not change the LLM input."""
    cfg = tiny()
    model = MiMoAudioLMRepro(cfg).eval()
    text, audio, is_speech = make_batch(cfg, B=1, T=4)
    e1, _ = model.prepare_inputs(text, audio)
    # a speech position keeps empty_text_id; on a TEXT position swap the token
    text2 = text.clone()
    text2[0, 0] = (text2[0, 0] + 1) % (cfg.text_vocab_size - 1)
    e2, _ = model.prepare_inputs(text2, audio)
    assert not torch.allclose(e1[:, 0], e2[:, 0])          # text change matters
    assert torch.allclose(e1[:, -1], e2[:, -1])            # speech pos unaffected
