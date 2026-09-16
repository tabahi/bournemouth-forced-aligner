"""
Bournemouth Forced Aligner - Phoneme-level timestamp extraction from audio using CUPE models.

URL: https://github.com/tabahi/bournemouth-forced-aligner
"""

__version__ = "1.2.0"

from .core import PhonemeTimestampAligner

__all__ = [
    "PhonemeTimestampAligner",
    "__version__"
]