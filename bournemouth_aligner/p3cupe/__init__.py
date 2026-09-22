"""
p3cupe: the phoneme model and the forced aligner it was trained against.

Inference-only port of `bournemouth_aligner/model3p/` (the training package, whose README.md
and memory.md hold the design record). Three pieces:

    model.py      waveform window -> per-frame phoneme posteriors, 5 ms frames, CNN only
    extractor.py  checkpoint loading, and the windowed forward pass over a whole clip
    aligner.py    the segmental Viterbi - the SAME decoder that built the training targets

The model's output IS the boundary detector: each phone's posterior is trained to be a hat
over its own segment, so two phones cross at their shared boundary and `refine_onsets` reads
the crossing off by interpolation. There is no blank class and no onset snapping.
"""

from .aligner import align_batch, confidence_dip, refine_onsets, viterbi_align
from .extractor import CupeExtractor
from .model import CupeCNN

__all__ = [
    "CupeCNN",
    "CupeExtractor",
    "align_batch",
    "viterbi_align",
    "refine_onsets",
    "confidence_dip",
]
