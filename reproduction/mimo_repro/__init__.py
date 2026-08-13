"""Minimal from-scratch reproduction of the MiMo-Audio architecture.

Implements the components described in the technical report
"MiMo-Audio: Audio Language Models are Few-Shot Learners" (arXiv:2512.23808):

- MiMo-Audio-Tokenizer: mel frontend, Transformer encoder with 2x downsampling
  at input and output, RVQ discretization, causal Transformer decoder and a
  Vocos-style Transformer vocoder with an ISTFT head (paper section 2.1),
  together with the stage-1 losses (A2T + multi-scale mel reconstruction +
  commitment, Eq. 1-3) and stage-2 GAN losses (MPD + MS-STFT discriminators,
  hinge objective, feature matching, Eq. 4-7).
- MiMo-Audio LM: patch encoder (Eq. 11), LLM backbone over interleaved
  text/audio-patch sequences (Eq. 8-10) and the patch decoder with the
  per-codebook delay mechanism (Eq. 12-15), plus the training loss weights
  from Table 3 and the group-wise autoregressive generation loop.

Everything runs on CPU at small scale; paper-scale hyperparameters are kept
as the documented defaults in `config.py`.
"""

from .config import TokenizerConfig, MiMoLMConfig
from .tokenizer import MiMoAudioTokenizerRepro
from .patch_lm import MiMoAudioLMRepro

__all__ = [
    "TokenizerConfig",
    "MiMoLMConfig",
    "MiMoAudioTokenizerRepro",
    "MiMoAudioLMRepro",
]
