"""
Bournemouth Forced Aligner - Phoneme-level timestamp extraction from audio using CUPE models.

URL: https://github.com/tabahi/bournemouth-forced-aligner
"""

__version__ = "0.1.0"

from .core import PhonemeTimestampAligner, process_single_clip, process_transcription

__all__ = [
    "PhonemeTimestampAligner",
    "process_single_clip", 
    "process_transcription",
    "__version__"
]