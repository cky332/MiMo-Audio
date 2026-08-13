"""Toy end-to-end demo of the reproduction on CPU.

Runs the full MiMo-Audio pipeline at tiny scale with random weights:

  1. stage-1 tokenizer training steps (A2T + multi-scale mel + commit, Eq. 3)
  2. stage-2 GAN steps with encoder/RVQ frozen (Eq. 4-7)
  3. LM training steps on an interleaved text/audio batch (Table 3 weights)
  4. generation: text branch -> speech branch -> waveform via the tokenizer

Usage: python demo_toy_pipeline.py
"""

import torch

from mimo_repro.config import MiMoLMConfig, TokenizerConfig
from mimo_repro.patch_lm import MiMoAudioLMRepro
from mimo_repro.tokenizer import MiMoAudioTokenizerRepro, log_mel_spectrogram
from mimo_repro.tokenizer_losses import (
    A2THead,
    Discriminators,
    discriminator_loss,
    stage1_loss,
    stage2_generator_loss,
)


def main() -> None:
    torch.manual_seed(0)
    tcfg = TokenizerConfig.tiny()
    lcfg = MiMoLMConfig.tiny()

    # ---- data: 0.5 s of a 220 Hz tone --------------------------------------
    t = torch.arange(12000) / tcfg.sampling_rate
    wav = torch.sin(2 * torch.pi * 220 * t)[None, :]
    mel = log_mel_spectrogram(wav, tcfg)
    text = torch.randint(0, 32, (1, 8))

    # ---- 1. tokenizer stage 1 ---------------------------------------------
    tok = MiMoAudioTokenizerRepro(tcfg)
    a2t = A2THead(audio_dim=tcfg.d_model, vocab_size=32)
    opt = torch.optim.Adam(list(tok.parameters()) + list(a2t.parameters()), lr=1e-3)
    for step in range(10):
        wav_hat, quantized, _, commit = tok(mel)
        loss = stage1_loss(wav_hat, wav, commit, a2t(quantized, text), tcfg.sampling_rate)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 3 == 0:
            print(f"[stage1] step {step:2d}  loss {loss.item():8.3f}")

    # ---- 2. tokenizer stage 2 (encoder + RVQ frozen) -----------------------
    for p in tok.encoder.parameters():
        p.requires_grad_(False)
    tok.quantizer.eval()
    disc = Discriminators(periods=(2, 3), n_ffts=(256,), channels=4)
    g_opt = torch.optim.Adam(
        list(tok.decoder.parameters()) + list(tok.vocoder.parameters()), lr=1e-4
    )
    d_opt = torch.optim.Adam(disc.parameters(), lr=1e-4)
    for step in range(4):
        wav_hat, _, _, _ = tok(mel)
        d_loss = discriminator_loss(disc, wav, wav_hat)
        d_opt.zero_grad(); d_loss.backward(); d_opt.step()
        wav_hat, _, _, _ = tok(mel)
        g_loss = stage2_generator_loss(disc, wav_hat, wav, tcfg.sampling_rate)
        g_opt.zero_grad(); g_loss.backward(); g_opt.step()
        print(f"[stage2] step {step:2d}  D {d_loss.item():6.3f}  G {g_loss.item():8.3f}")

    # ---- 3. LM training on interleaved text + audio ------------------------
    tok.eval()
    codes = tok.encode(mel)[: lcfg.audio_channels]           # [R', 1, T25]
    T25 = codes.shape[-1] - codes.shape[-1] % lcfg.group_size
    patches = codes[..., :T25].permute(1, 2, 0).reshape(
        1, -1, lcfg.group_size, lcfg.audio_channels
    )
    T = patches.shape[1] + 2
    text_tokens = torch.randint(0, lcfg.text_vocab_size - 1, (1, T))
    audio_tokens = (
        torch.tensor(lcfg.audio_empty_ids)
        .expand(1, T, lcfg.group_size, lcfg.audio_channels).clone()
    )
    text_tokens[:, 2:] = lcfg.empty_text_id                  # 2 text patches, rest speech
    audio_tokens[:, 2:] = patches
    audio_targets = audio_tokens.clone()
    audio_targets[:, :2] = -100

    lm = MiMoAudioLMRepro(lcfg)
    opt = torch.optim.Adam(lm.parameters(), lr=3e-3)
    for step in range(15):
        out = lm(text_tokens, audio_tokens, text_tokens.clone(), audio_targets)
        opt.zero_grad(); out.loss.backward(); opt.step()
        if step % 5 == 0:
            print(
                f"[lm]     step {step:2d}  loss {out.loss.item():9.3f}"
                f"  (text {out.text_loss.item():6.3f}, audio {out.audio_loss.item():7.3f})"
            )

    # ---- 4. generation ------------------------------------------------------
    lm.eval()
    force_speech = lambda logits: torch.full(
        (logits.shape[0],), lcfg.empty_text_id, dtype=torch.long
    )
    _, new_patches = lm.generate(
        text_tokens, audio_tokens, max_new_patches=3, text_sample_fn=force_speech
    )
    new_codes = new_patches.reshape(1, -1, lcfg.audio_channels).permute(2, 0, 1).long()
    wav_out = tok.decode(new_codes)
    print(
        f"[gen]    {new_patches.shape[1]} patches -> {new_codes.shape[-1]} frames"
        f" -> {wav_out.shape[-1]} samples ({wav_out.shape[-1] / tcfg.sampling_rate:.2f} s)"
    )


if __name__ == "__main__":
    main()
