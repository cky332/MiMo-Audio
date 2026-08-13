"""Configurations for the reproduction.

Default values follow the paper (section 2.1.1, section 2.2, Table 2).
`tiny()` constructors return CPU-friendly versions with the same structure
for tests and toy training runs.

Where the paper and the official code disagree, the field is documented with
both values; we follow the official code because it is what actually ran.
"""

from dataclasses import dataclass, field


@dataclass
class TokenizerConfig:
    # --- audio frontend (paper 2.1.1: 24 kHz input, 100 Hz mel frame rate) ---
    sampling_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 240          # 24000 / 240 = 100 Hz mel frames
    win_length: int = 1024
    n_mels: int = 80               # official config default; paper does not state it

    # --- encoder (paper: 32 layers, dim 1280, FFN 5120, 20 heads, RoPE+GELU) ---
    encoder_layers: int = 32
    d_model: int = 1280
    encoder_ffn_dim: int = 5120
    encoder_heads: int = 20
    # paper: "add the layer-3 hidden states to the final-layer output"
    # official code: `encoder_skip_layer_id` saved after layer index (id-1)
    encoder_skip_layer_id: int = 3
    # paper: "bracketed by 2x downsampling layers at the input and output";
    # official code: conv stride 2 before the transformer + avg_pooler conv
    # (stride 2) after it -> 100 Hz / 2 / 2 = 25 Hz
    conv_stride: int = 2
    avg_pooler: int = 2

    # --- RVQ (paper: 20 layers, first two codebooks 1024, the rest 128) ---
    num_quantizers: int = 20
    codebook_sizes: tuple = (1024, 1024) + (128,) * 18
    rvq_decay: float = 0.99
    threshold_ema_dead_code: int = 10   # official config default

    # --- decoder (paper: mirror of encoder but causal) ---
    decoder_layers: int = 32
    decoder_ffn_dim: int = 5120
    decoder_heads: int = 20

    # --- vocoder (paper: Vocos-style Transformer, 16 layers, 16 heads,
    #     dim 256, FFN 1024, sliding window [40, 10]) ---
    vocoder_layers: int = 16
    vocoder_dim: int = 256
    vocoder_ffn_dim: int = 1024
    vocoder_heads: int = 16
    vocoder_window: tuple = (40, 10)   # (left, right) in 100 Hz frames

    rope_theta: float = 10000.0

    # --- losses (paper Eq. 3 and Eq. 7) ---
    lambda_a2t: float = 10.0
    lambda_recon: float = 1.0
    lambda_commit: float = 1.0
    lambda_adv: float = 1.0
    lambda_fm: float = 2.0
    recon_scales: tuple = (5, 6, 7)    # mel bins 2^i, win 15*2^(i-1), hop 15*2^(i-2)

    @property
    def frame_rate(self) -> float:
        # 100 Hz mel -> /conv_stride -> /avg_pooler
        return self.sampling_rate / self.hop_length / self.conv_stride / self.avg_pooler

    def bitrate_bps(self, n_codebooks: int | None = None) -> float:
        """Effective bitrate of the token stream (paper Table 1: 1.55 kbps @ 8)."""
        import math

        sizes = self.codebook_sizes[: n_codebooks or self.num_quantizers]
        return self.frame_rate * sum(math.log2(s) for s in sizes)

    @classmethod
    def tiny(cls) -> "TokenizerConfig":
        return cls(
            n_mels=32,
            encoder_layers=2,
            d_model=64,
            encoder_ffn_dim=128,
            encoder_heads=4,
            encoder_skip_layer_id=1,
            num_quantizers=4,
            codebook_sizes=(64, 64, 16, 16),
            threshold_ema_dead_code=0,
            decoder_layers=2,
            decoder_ffn_dim=128,
            decoder_heads=4,
            vocoder_layers=2,
            vocoder_dim=32,
            vocoder_ffn_dim=64,
            vocoder_heads=4,
        )


@dataclass
class MiMoLMConfig:
    # --- interleaved sequence space (paper Table 2) ---
    text_vocab_size: int = 151680
    audio_channels: int = 8                      # R' = 8 RVQ layers for the LM
    # paper: vocab 1024-1024-128-... and "0 denotes an empty token" (Eq. 15).
    # official code: one EXTRA index per codebook is appended and used as the
    # empty token (speech_vocab_size 1025-1025-129-..., empty id 1024/128).
    # We follow the code: vocab_size = codebook_size + 1, empty = last index.
    audio_vocab_sizes: tuple = (1025, 1025, 129, 129, 129, 129, 129, 129)
    group_size: int = 4                          # G: 25 Hz -> 6.25 Hz patches
    delay_pattern: tuple = (0, 1, 2, 3, 4, 5, 6, 7)   # Table 3

    # --- LLM backbone (paper Table 2: 36 layers, dim 4096, FFN 11008, 32 heads) ---
    llm_layers: int = 36
    llm_dim: int = 4096
    llm_ffn_dim: int = 11008
    llm_heads: int = 32

    # --- patch encoder (paper Table 2: 6 layers, dim 1024, FFN 4096;
    #     Table 2 says 16 heads but the paper text (2.2.1) says 64 and the
    #     official code shares `local_attn_heads`=64 between encoder/decoder;
    #     we follow the code) ---
    patch_encoder_layers: int = 6
    patch_dim: int = 1024
    patch_ffn_dim: int = 4096
    patch_heads: int = 64
    # paper 2.2.1: "the encoder employs bidirectional self-attention".
    # official code: causal unless config.input_full_attention is truthy
    # (modeling_mimo_audio.py line 319).  We default to bidirectional per the
    # paper and expose the flag to reproduce the code's behaviour.
    patch_encoder_bidirectional: bool = True

    # --- patch decoder (paper Table 2: 16 layers, dim 1024, FFN 4096, 64 heads,
    #     context 11 = G + max delay) ---
    patch_decoder_layers: int = 16

    rope_theta: float = 10000.0

    # --- training loss weights (paper 3.2.2 / Table 3) ---
    text_loss_weight: float = 100.0
    audio_loss_weights: tuple = (12.0, 8.0, 6.0, 4.0, 2.0, 2.0, 1.0, 1.0)

    # special text tokens (appended after the text vocab, mirroring the
    # official repo's <|sosp|>/<|eosp|>/<|empty|>/<|sostm|>/<|eostm|>/<|eot|>)
    @property
    def empty_text_id(self) -> int:
        return self.text_vocab_size - 1

    @property
    def audio_empty_ids(self) -> tuple:
        return tuple(v - 1 for v in self.audio_vocab_sizes)

    @property
    def patch_decoder_context(self) -> int:
        return self.group_size + max(self.delay_pattern)

    @classmethod
    def tiny(cls) -> "MiMoLMConfig":
        return cls(
            text_vocab_size=256,
            # tokenizer tiny codebooks are (64, 64, 16, 16); LM vocab adds the
            # empty token: codebook_size + 1, same as 1025/129 vs 1024/128
            audio_vocab_sizes=(65, 65, 17, 17),
            audio_channels=4,
            group_size=4,
            delay_pattern=(0, 1, 2, 3),
            llm_layers=2,
            llm_dim=64,
            llm_ffn_dim=128,
            llm_heads=4,
            patch_encoder_layers=2,
            patch_dim=32,
            patch_ffn_dim=64,
            patch_heads=4,
            patch_decoder_layers=2,
            audio_loss_weights=(12.0, 8.0, 6.0, 4.0),
        )
