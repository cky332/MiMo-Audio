import torch

from mimo_repro.config import TokenizerConfig
from mimo_repro.tokenizer import MiMoAudioTokenizerRepro, log_mel_spectrogram


def make_tokenizer():
    torch.manual_seed(0)
    cfg = TokenizerConfig.tiny()
    return cfg, MiMoAudioTokenizerRepro(cfg).eval()


def test_mel_frontend_is_100hz():
    cfg = TokenizerConfig.tiny()
    wav = torch.randn(1, 24000)  # 1 s
    mel = log_mel_spectrogram(wav, cfg)
    assert mel.shape[-1] == cfg.n_mels
    assert abs(mel.shape[1] - 100) <= 1  # ~100 frames per second


def test_encode_downsamples_4x_to_25hz():
    cfg, tok = make_tokenizer()
    wav = torch.randn(1, 24000)
    mel = log_mel_spectrogram(wav, cfg)
    codes = tok.encode(mel)
    assert codes.shape[0] == cfg.num_quantizers
    t25 = codes.shape[-1]
    assert abs(t25 - 25) <= 1, f"expected ~25 Hz tokens for 1 s audio, got {t25}"


def test_decode_first_n_codebooks_only():
    cfg, tok = make_tokenizer()
    mel = log_mel_spectrogram(torch.randn(1, 12000), cfg)
    codes = tok.encode(mel)
    # decode using only the first 2 of 4 codebooks (the paper's LM uses 8/20)
    wav = tok.decode(codes[:2])
    assert wav.ndim == 2 and wav.shape[-1] > 0


def test_decoder_output_rate():
    cfg, tok = make_tokenizer()
    mel = log_mel_spectrogram(torch.randn(1, 24000), cfg)
    codes = tok.encode(mel)
    wav = tok.decode(codes)
    # 25 Hz tokens -> 24 kHz wav: ~960 samples per token
    ratio = wav.shape[-1] / codes.shape[-1]
    assert 900 < ratio < 1000, ratio


def test_tokenizer_decoder_is_causal():
    """Perturbing a later token must not change earlier decoder output.

    The paper motivates the causal decoder with streaming generation; the
    vocoder is NOT causal (sliding window with 10 frames of lookahead), so
    causality is asserted on the coarse-mel decoder, not the waveform."""
    cfg, tok = make_tokenizer()
    mel = log_mel_spectrogram(torch.randn(1, 48000), cfg)
    codes = tok.encode(mel)
    feats = tok.quantizer.decode(codes.reshape(codes.shape[0], -1))
    feats = feats.reshape(1, -1, cfg.d_model)

    out_a = tok.decoder(feats)
    feats_b = feats.clone()
    feats_b[:, -1] += 10.0  # perturb the last 25 Hz token
    out_b = tok.decoder(feats_b)

    # up1 upsamples by 2; the last token affects the final 2 pre-transformer
    # steps, so everything strictly before must be identical
    safe = out_a.shape[1] - 2 * cfg.conv_stride * cfg.avg_pooler
    assert torch.allclose(out_a[:, :safe], out_b[:, :safe], atol=1e-5)
    assert not torch.allclose(out_a[:, safe:], out_b[:, safe:], atol=1e-5)


def test_vocoder_sliding_window_locality():
    """With window (left=40, right=10), output frame i must not depend on
    inputs beyond i+right per layer; total lookahead = layers * right."""
    cfg, tok = make_tokenizer()
    T = 64
    mel = torch.randn(1, T, cfg.n_mels)
    out_a = tok.vocoder.transformer(tok.vocoder.embed(mel))
    mel_b = mel.clone()
    mel_b[:, -1] += 10.0
    out_b = tok.vocoder.transformer(tok.vocoder.embed(mel_b))
    max_reach = cfg.vocoder_layers * cfg.vocoder_window[1]
    safe = T - 1 - max_reach
    assert safe > 0, "test needs a longer sequence"
    assert torch.allclose(out_a[:, :safe], out_b[:, :safe], atol=1e-4)


def test_encoder_skip_connection_active():
    """The layer-3 (here layer-1 in tiny) skip connection must contribute."""
    cfg, tok = make_tokenizer()
    mel = log_mel_spectrogram(torch.randn(1, 12000), cfg)
    x = tok.encoder(mel)
    tok.encoder.transformer.skip_layer_id = None
    x_noskip = tok.encoder(mel)
    assert not torch.allclose(x, x_noskip, atol=1e-5)
