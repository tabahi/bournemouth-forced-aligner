"""
standard_g2p - VENDORED from https://github.com/tabahi/CharsiuG2P (MIT, see LICENSE).

Do not edit anything in this directory by hand: `tools/sync_standard_g2p.py` is its only
writer, and VENDORED.md records the commit it was taken from. The modules are verbatim copies;
this file is the one BFA-owned piece, and all it does is tell the Mandarin segmenter where the
dictionary keys shipped with this package live (upstream resolves them from its repository
root, which does not exist in an installed package).
"""

import os as _os

from . import word_segmentation as _word_segmentation

_word_segmentation.DICTS_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "dicts")
