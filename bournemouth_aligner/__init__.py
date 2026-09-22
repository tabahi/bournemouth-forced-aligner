"""
Bournemouth Forced Aligner - phoneme-level timestamps from audio, using the p3cupe model.

URL: https://github.com/tabahi/bournemouth-forced-aligner
"""

__version__ = "2.0.0"

from .core import PhonemeTimestampAligner
from .ipamappers.ph66_phonemeizer import Phonemizer as Phonemizer_ph66

__all__ = [
    "PhonemeTimestampAligner",
    "Phonemizer_ph66",
    "__version__"
]
