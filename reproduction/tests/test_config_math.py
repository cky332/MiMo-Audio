"""Numerical claims from the paper, checked against the config math."""

import math

from mimo_repro.config import MiMoLMConfig, TokenizerConfig


def test_frame_rate_pipeline():
    cfg = TokenizerConfig()
    # 24000 / 240 = 100 Hz mel; /2 conv; /2 pooler = 25 Hz
    assert cfg.sampling_rate / cfg.hop_length == 100
    assert cfg.frame_rate == 25


def test_bitrate_matches_table1():
    cfg = TokenizerConfig()
    # paper Table 1: 1.55 kbps for the 8 codebooks the LM uses
    # 25 Hz * (2*10 + 6*7) bits = 1550 bps
    assert abs(cfg.bitrate_bps(8) - 1550) < 1e-6


def test_200_tokens_per_second():
    cfg = TokenizerConfig()
    lm = MiMoLMConfig()
    # abstract: "200 tokens per second" = 25 Hz * 8 codebooks
    assert cfg.frame_rate * lm.audio_channels == 200


def test_patch_rate():
    lm = MiMoLMConfig()
    # patchification: 25 Hz / G=4 = 6.25 Hz at the LLM interface
    assert 25 / lm.group_size == 6.25


def test_patch_decoder_context_is_11():
    lm = MiMoLMConfig()
    # Table 2: patch decoder context length 11 = G + max(delay) = 4 + 7
    assert lm.patch_decoder_context == 11


def test_vocoder_receptive_field():
    cfg = TokenizerConfig()
    # paper: window [40, 10] over 16 layers at 100 Hz mel rate
    # -> [6.4 s, 1.6 s] receptive field
    left = cfg.vocoder_window[0] * cfg.vocoder_layers / 100
    right = cfg.vocoder_window[1] * cfg.vocoder_layers / 100
    assert left == 6.4 and right == 1.6


def test_codebook_layout():
    cfg = TokenizerConfig()
    assert len(cfg.codebook_sizes) == 20
    assert cfg.codebook_sizes[:2] == (1024, 1024)
    assert set(cfg.codebook_sizes[2:]) == {128}


def test_empty_ids_follow_official_code():
    lm = MiMoLMConfig()
    # official config: speech_vocab_size 1025-1025-129-..., zeroemb 1024-1024-128-...
    assert lm.audio_vocab_sizes[:2] == (1025, 1025)
    assert lm.audio_empty_ids[:2] == (1024, 1024)
    assert lm.audio_empty_ids[2] == 128
