"""
p3cupe: the ph66 phoneme model.

Inference-only port of `bournemouth_aligner/model3p/` (the training package, whose README.md
and memory.md hold the design record). Two pieces of its own:

    model.py      waveform window -> per-frame phoneme posteriors, 5 ms frames, CNN only
    extractor.py  checkpoint loading

The windowed forward pass over a whole clip and the segmental Viterbi - the SAME decoder that
built the training targets - are shared with p4mbfa and live in `bournemouth_aligner/segmental/`.
They are re-exported here because this is where they used to be.

The model's output IS the boundary detector: each phone's posterior is trained to be a hat
over its own segment, so two phones cross at their shared boundary and `refine_onsets` reads
the crossing off by interpolation. There is no blank class and no onset snapping.
"""

from ..segmental.aligner import align_batch, confidence_dip, refine_onsets, viterbi_align
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
