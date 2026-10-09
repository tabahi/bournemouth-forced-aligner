"""
Phonemizers: text -> the id sequence a model head was trained on.

    std   standard_g2p (ByT5 G2P, 100 languages) -> one token list per language group
    ph66  espeak-ng -> 66 phonemes shared by every language

`make_phonemizer` is the one entry point; each backend is imported only when asked for, so the
std set does not need espeak-ng and ph66 does not need transformers.
"""

from .base import Inventory

__all__ = ["Inventory", "make_phonemizer"]


def make_phonemizer(phoneme_set, language, **std_options):
    """
    Build the phonemizer for a phoneme set.

    Args:
        phoneme_set: 'std' or 'ph66'.
        language: canonical BCP 47 code for std, espeak-ng code for ph66.
        **std_options: `device`, `model_dir`, `cache_size`, `punctuation_sil`, `verbose` -
            see `std.StdPhonemizer`. Ignored by ph66.
    """
    if phoneme_set == "ph66":
        from .ph66 import make_ph66_phonemizer
        return make_ph66_phonemizer(language)
    if phoneme_set == "std":
        from .std import StdPhonemizer
        return StdPhonemizer(language, **std_options)
    raise ValueError(f"unknown phoneme_set {phoneme_set!r}")
