"""
p4mbfa: the multilingual phoneme model behind the `std` phoneme set.

Inference-only port of `bfa_models/p4mbfa/`. p3cupe's trunk with three heads on it:

    ph_head     one LANGUAGE GROUP's phoneme tokens (208 for 'latin', 48 for 'cjk', ...)
    phg_head    15 phoneme groups, the same in every language
    tone_head   22 tone values, the same in every language

    model.py      waveform window -> the three per-frame posteriors, 5 ms frames, CNN only
    extractor.py  checkpoint loading, and the check that its token list is still the one
                  standard_g2p produces

The windowed forward pass and the segmental Viterbi are shared with p3cupe
(`bournemouth_aligner/segmental/`): the decoder that built p4mbfa's training targets is that
one, with silence at index 1 instead of 0.
"""

from .extractor import MbfaExtractor
from .model import MbfaCNN

__all__ = ["MbfaCNN", "MbfaExtractor"]
