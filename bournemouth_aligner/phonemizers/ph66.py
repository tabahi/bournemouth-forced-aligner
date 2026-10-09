"""
The ph66 phoneme set: espeak-ng IPA reduced to 66 phonemes and 16 groups.

The phonemizer itself is `ipamappers/ph66_phonemeizer.py`, unchanged. This module only builds
it and describes its label space, and is where the optional dependency is reported: ph66 needs
the `phonemizer` package and a system espeak-ng, which the default (std) phoneme set does not.
"""

from .base import Inventory


def ph66_inventory(phonemizer):
    """The ph66 label space, read off a `ph66_phonemeizer.Phonemizer`."""
    label_to_id = {label: idx for idx, label in phonemizer.index_to_plabel.items()}
    group_label_to_id = {label: idx for idx, label in phonemizer.index_to_glabel.items()}
    # 'noise' is the last id in both spaces. p3cupe has no blank class and never makes it a
    # target, so it exists only as the padding filler for ragged batches and as the slot the
    # classifier keeps for downstream tooling that indexes by it.
    return Inventory(
        name="ph66", phoneme_set="ph66",
        id_to_label=dict(phonemizer.index_to_plabel),
        group_id_to_label=dict(phonemizer.index_to_glabel),
        phoneme_id_to_group_id=dict(phonemizer.phoneme_id_to_group_id),
        sil_id=label_to_id['SIL'], sil_group=group_label_to_id['SIL'],
        pad_id=label_to_id['noise'], pad_group=group_label_to_id['noise'],
        phonemes_key=phonemizer.phonemes_key, groups_key=phonemizer.phoneme_groups_key)


def make_ph66_phonemizer(language):
    """A ph66 phonemizer for an espeak-ng language code, with `.inventory` attached."""
    try:
        from ..ipamappers.ph66_phonemeizer import Phonemizer
    except ImportError as e:
        raise ImportError(
            "phoneme_set='ph66' needs the `phonemizer` package and espeak-ng:\n"
            "    pip install 'bournemouth-forced-aligner[ph66]'\n"
            "    apt-get install espeak-ng        (or the equivalent for your system)\n"
            f"({e})") from e
    phonemizer = Phonemizer(language=language, remove_noise_phonemes=True)
    phonemizer.inventory = ph66_inventory(phonemizer)
    return phonemizer
