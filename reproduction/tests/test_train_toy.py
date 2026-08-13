"""End-to-end toy training: verify both stages of the reproduction optimize.

This is the strongest correctness signal available without the released
checkpoints (which need GPUs + flash-attn): if the wiring of encoder -> RVQ
-> decoder -> vocoder and patch encoder -> LLM -> delayed patch decoder is
consistent, a tiny model must be able to overfit a tiny batch.
"""

import torch

from mimo_repro.config import MiMoLMConfig, TokenizerConfig
from mimo_repro.patch_lm import MiMoAudioLMRepro
from mimo_repro.tokenizer import MiMoAudioTokenizerRepro, log_mel_spectrogram
from mimo_repro.tokenizer_losses import A2THead, stage1_loss


def test_tokenizer_stage1_overfits_a_batch():
    torch.manual_seed(0)
    cfg = TokenizerConfig.tiny()
    tok = MiMoAudioTokenizerRepro(cfg)
    a2t = A2THead(audio_dim=cfg.d_model, vocab_size=32)

    t = torch.arange(12000) / cfg.sampling_rate
    wav = torch.sin(2 * torch.pi * 220 * t)[None, :]
    mel = log_mel_spectrogram(wav, cfg)
    text = torch.randint(0, 32, (1, 8))

    opt = torch.optim.Adam(list(tok.parameters()) + list(a2t.parameters()), lr=1e-3)
    losses = []
    for _ in range(15):
        tok.train()
        wav_hat, quantized, _, commit = tok(mel)
        loss = stage1_loss(
            wav_hat, wav, commit, a2t(quantized, text), cfg.sampling_rate,
            lambda_a2t=cfg.lambda_a2t,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"stage-1 loss did not decrease: {losses}"


def test_lm_overfits_a_batch():
    torch.manual_seed(0)
    cfg = MiMoLMConfig.tiny()
    model = MiMoAudioLMRepro(cfg)

    B, T = 1, 6
    text = torch.randint(0, cfg.text_vocab_size - 1, (B, T))
    audio = (
        torch.tensor(cfg.audio_empty_ids)
        .expand(B, T, cfg.group_size, cfg.audio_channels)
        .clone()
    )
    text[:, T // 2 :] = cfg.empty_text_id
    audio[:, T // 2 :] = torch.randint(
        0, 16, (B, T - T // 2, cfg.group_size, cfg.audio_channels)
    )
    audio_targets = audio.clone()
    audio_targets[:, : T // 2] = -100

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(40):
        out = model(text, audio, text.clone(), audio_targets)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        losses.append(out.loss.item())
    assert losses[-1] < losses[0] * 0.5, f"LM loss did not drop: {losses[0]} -> {losses[-1]}"


def test_full_pipeline_tokens_roundtrip_through_lm():
    """wav -> tokenizer codes (first R' books) -> LM generate -> codes ->
    tokenizer decode -> wav.  Random weights: only shapes/ranges are checked."""
    torch.manual_seed(0)
    tcfg = TokenizerConfig.tiny()
    tok = MiMoAudioTokenizerRepro(tcfg).eval()
    lcfg = MiMoLMConfig.tiny()
    # tiny LM uses 4 channels; tokenizer has 4 quantizers -> aligned
    model = MiMoAudioLMRepro(lcfg).eval()

    wav = torch.randn(1, 24000)
    mel = log_mel_spectrogram(wav, tcfg)
    codes = tok.encode(mel)[: lcfg.audio_channels]        # [R', 1, T25]

    T25 = codes.shape[-1]
    T25 -= T25 % lcfg.group_size
    frames = codes[..., :T25].permute(1, 2, 0)            # [1, T25, R']
    patches = frames.reshape(1, -1, lcfg.group_size, lcfg.audio_channels)
    text = torch.full(patches.shape[:2], lcfg.empty_text_id)

    force_speech = lambda logits: torch.full(
        (logits.shape[0],), lcfg.empty_text_id, dtype=torch.long
    )
    _, new_patches = model.generate(
        text, patches, max_new_patches=2, text_sample_fn=force_speech
    )

    new_codes = new_patches.reshape(1, -1, lcfg.audio_channels).permute(2, 0, 1)
    wav_out = tok.decode(new_codes.long())
    # 2 patches * G=4 frames * ~960 samples/frame
    assert wav_out.shape[-1] == new_codes.shape[-1] * 960
