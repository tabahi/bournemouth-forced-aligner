"""
What `core.py` needs to know about a phoneme set, and nothing else.

A phoneme set is two things: a phonemizer (text -> id sequence) and the label space those ids
live in. `core.py` reads the label space from an `Inventory` and treats the phonemizer as
anything with `phonemize_sentence(text)` and `set_backend(language)`.

Every phonemizer returns the same dict from `phonemize_sentence`:

    inventory.phonemes_key   id per target phoneme, in the model's own label space
    inventory.groups_key     phoneme-group id per target phoneme
    "eipa"                   IPA string per target phoneme, as the G2P wrote it
    "mipa"                   the inventory label each one was mapped to
    "words", "word_num"      the words, and the word index of each target phoneme
    "text"                   the input

The std set adds `tone`, `stress` and `length` (one per target phoneme), which ph66 has no
notion of.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Inventory:
    """
    One model head's label space.

    Attributes:
        name: 'ph66', or 'std/<lang_group>' - what output files are stamped with.
        phoneme_set: 'ph66' or 'std'.
        lang_group: the language group the labels belong to; None for ph66, which has one
            label space for every language.
        id_to_label / group_id_to_label: id -> printable label.
        phoneme_id_to_group_id: many-to-one relabelling of phonemes into groups.
        sil_id / sil_group: silence. The aligner pads an optional one onto both ends of every
            sequence and may skip any it finds, so it must be the id the model was trained on.
        pad_id / pad_group: filler for ragged batches. Never a target: ph66 uses its unused
            'noise' slot, std uses '<blank>' (std's 'noise' IS a real target).
        n_classes / n_groups: logits the model's two heads must emit for this inventory.
        phonemes_key / groups_key: where the id sequences sit in a phonemized segment.
        fingerprint: for std, the gold inventory the token lists were built from.
    """
    name: str
    phoneme_set: str
    id_to_label: Dict[int, str]
    group_id_to_label: Dict[int, str]
    phoneme_id_to_group_id: Dict[int, int]
    sil_id: int
    sil_group: int
    pad_id: int
    pad_group: int
    phonemes_key: str
    groups_key: str
    lang_group: Optional[str] = None
    fingerprint: Optional[str] = None
    label_to_id: Dict[str, int] = field(default_factory=dict)
    group_label_to_id: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.label_to_id:
            self.label_to_id = {label: idx for idx, label in self.id_to_label.items()}
        if not self.group_label_to_id:
            self.group_label_to_id = {label: idx for idx, label
                                      in self.group_id_to_label.items()}

    @property
    def n_classes(self):
        return len(self.id_to_label)

    @property
    def n_groups(self):
        return len(self.group_id_to_label)

    @property
    def tokens(self):
        """Labels in id order."""
        return [self.id_to_label[i] for i in range(self.n_classes)]
