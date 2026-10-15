"""
Bournemouth Forced Aligner - phoneme-level timestamps from audio and text.

Two phoneme sets: "std" (standard_g2p phonemes, p4mbfa models, the default) and "ph66"
(espeak-ng + ph66, p3cupe models). See `core.py`.

URL: https://github.com/tabahi/bournemouth-forced-aligner
"""

__version__ = "2.0.0"

from .core import BFA

__all__ = [
    "BFA",
    "Phonemizer_ph66",
    "__version__"
]


def __getattr__(name):
    # Imported on first use, not at package import: the ph66 phonemizer needs the `phonemizer`
    # package and espeak-ng, which the default phoneme set does not.
    if name == "Phonemizer_ph66":
        from .ipamappers.ph66_phonemeizer import Phonemizer
        return Phonemizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
