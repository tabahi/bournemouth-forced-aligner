"""
What p3cupe and p4mbfa have in common: everything except the heads.

    windowing.py  slicing a clip into 120 ms windows and stitching the frames back
    extractor.py  checkpoint loading and the windowed forward pass over a whole clip
    aligner.py    the segmental Viterbi - the SAME decoder that built the training targets

The two families share a trunk, a frame grid and a decoder. They differ in what their heads
emit, which is why each has its own `model.py` and a small `extractor.py` and nothing else.
"""

from .aligner import align_batch, confidence_dip, refine_onsets, viterbi_align
from .extractor import BaseExtractor, checkpoint_family, load_checkpoint

__all__ = [
    "BaseExtractor",
    "load_checkpoint",
    "checkpoint_family",
    "align_batch",
    "viterbi_align",
    "refine_onsets",
    "confidence_dip",
]
