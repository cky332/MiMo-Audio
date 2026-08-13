import torch

from mimo_repro.config import TokenizerConfig
from mimo_repro.tokenizer import MiMoAudioTokenizerRepro, log_mel_spectrogram
from mimo_repro.tokenizer_losses import (
    A2THead,
    Discriminators,
    discriminator_loss,
    generator_adversarial_losses,
    multi_scale_mel_loss,
    stage1_loss,
    stage2_generator_loss,
)


def test_multi_scale_mel_loss_zero_for_identity():
    torch.manual_seed(0)
    wav = torch.randn(1, 24000)
    assert multi_scale_mel_loss(wav, wav, 24000).item() == 0.0
    assert multi_scale_mel_loss(wav, torch.randn(1, 24000), 24000).item() > 0


def test_a2t_head_trains():
    torch.manual_seed(0)
    head = A2THead(audio_dim=16, vocab_size=32)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    audio = torch.randn(2, 10, 16)
    text = torch.randint(0, 32, (2, 12))
    first = None
    for _ in range(30):
        loss = head(audio, text)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first, "A2T loss should decrease when memorizing"


def test_hinge_gan_losses():
    torch.manual_seed(0)
    disc = Discriminators(periods=(2, 3), n_ffts=(256,), channels=4)
    real = torch.randn(1, 4096)
    fake = torch.randn(1, 4096)
    d = discriminator_loss(disc, real, fake)
    assert torch.isfinite(d) and d.item() > 0
    adv, fm = generator_adversarial_losses(disc, real, fake)
    assert torch.isfinite(adv) and torch.isfinite(fm) and fm.item() > 0


def test_discriminators_are_spectral_normalized():
    disc = Discriminators(periods=(2,), n_ffts=(256,), channels=4)
    names = [n for n, _ in disc.named_parameters()]
    assert any("weight_orig" in n for n in names), "spectral norm not applied"


def test_stage2_freezes_encoder_and_quantizer():
    """Paper: in stage 2 'all parameters involved in the audio tokenization
    process are frozen'.  Verify the intended optimization split leaves
    encoder+RVQ untouched."""
    torch.manual_seed(0)
    cfg = TokenizerConfig.tiny()
    tok = MiMoAudioTokenizerRepro(cfg)
    disc = Discriminators(periods=(2,), n_ffts=(256,), channels=4)

    for p in tok.encoder.parameters():
        p.requires_grad_(False)
    # RVQ is EMA-based (buffers); freezing = keeping it in eval mode
    tok.quantizer.eval()

    wav = torch.randn(1, 12000)
    mel = log_mel_spectrogram(wav, cfg)
    enc_before = [p.clone() for p in tok.encoder.parameters()]
    codebook_before = tok.quantizer.layers[0].embed.clone()

    gen_params = list(tok.decoder.parameters()) + list(tok.vocoder.parameters())
    opt = torch.optim.Adam(gen_params, lr=1e-3)
    wav_hat, _, _, _ = tok(mel)
    loss = stage2_generator_loss(disc, wav_hat, wav, cfg.sampling_rate)
    opt.zero_grad()
    loss.backward()
    opt.step()

    for before, p in zip(enc_before, tok.encoder.parameters()):
        assert torch.equal(before, p)
    assert torch.equal(codebook_before, tok.quantizer.layers[0].embed)


def test_stage1_loss_weighting():
    a2t = torch.tensor(2.0)
    commit = torch.tensor(0.5)
    wav = torch.randn(1, 24000)
    loss = stage1_loss(wav, wav, commit, a2t, 24000)
    # recon = 0 -> loss = 10*2.0 + 1*0 + 1*0.5
    assert torch.allclose(loss, torch.tensor(20.5))
