"""
The std phoneme set: standard_g2p's labels, the ones the p4mbfa models were trained on.

Text goes through three stages, all of them in the vendored `standard_g2p` package:

    text -> words -> IPA (ByT5 G2P)       split into phonemes + tone / stress / length layers
         -> gold_ph                       one closed inventory of 270 tokens, all languages
         -> local_ph                      re-indexed into the language's LANGUAGE GROUP list

`local_ph` is what a p4mbfa `ph_head` emits, so it is the sequence the aligner decodes. A
phoneme can back off to two tokens (ʈ͡ʂ -> ʈ ʂ), so every per-phoneme layer is expanded to line
up with it, one entry per token.

## What is different from ph66

  - No silence at punctuation. ph66 writes a SIL for every comma and full stop; the p4mbfa
    models were trained on sequences with none inside a segment, so by default none is
    written here either. `punctuation_sil=True` writes them (the aligner may skip any SIL at
    no cost, so this only offers the path a place to put a pause). An explicit `<SIL>` in the
    text is always honoured.
  - No text normalisation. The G2P reads one word at a time and was trained on dictionary
    entries: digits, abbreviations and symbols are not expanded the way espeak-ng expands
    them. Spell them out in the transcript.
  - The G2P model is ~1.2 GB and is loaded on the first call that needs it, not at
    construction - a caller that only ever passes pre-phonemized segments never pays for it.
"""

import os
import re

from .. import hub
from ..standard_g2p import lang_codes as _lc
from ..standard_g2p import lang_group_inventory as LGI
from ..standard_g2p import phoneme_features as PF
from ..standard_g2p import phoneme_inventory_gold as PI
from ..standard_g2p import word_segmentation as _ws
from .base import Inventory

SIL_WORD = "<SIL>"

# Punctuation that marks a pause, for `punctuation_sil`. Western marks only count when they end
# a token, so "3.5", "e.g" and "o'clock" are left alone; the CJK ones are never word-internal.
_PAUSE_RE = re.compile(r"[,.;:!?…]+(?=\s|$)|[—–]|[，。！？；：、]+")
_EXPLICIT_SIL_RE = re.compile(re.escape(SIL_WORD))

# One loaded G2P per (weights directory, device): the model is ~1.2 GB, and two aligners for
# two language groups would otherwise each hold a copy of the same 100-language model.
_G2P_INSTANCES = {}


def std_inventory(lang_group):
    """The label space of one language group's `ph_head`, plus the 15 shared phoneme groups."""
    tokens = LGI.tokens(lang_group)
    id_to_label = dict(enumerate(tokens))
    return Inventory(
        name=f"std/{lang_group}", phoneme_set="std", lang_group=lang_group,
        id_to_label=id_to_label,
        group_id_to_label=dict(enumerate(PF.GROUP_NAMES)),
        # A local token is a gold token under another index, so its group is the gold one.
        phoneme_id_to_group_id={i: int(PF.GROUPS.get(tok, 0)) for i, tok in id_to_label.items()},
        sil_id=tokens.index(PI.SIL), sil_group=int(PF.GROUPS.get(PI.SIL, 0)),
        pad_id=tokens.index(PI.BLANK), pad_group=0,
        phonemes_key="local_ph", groups_key="gold_phg",
        fingerprint=LGI.gold_fingerprint())


class StdPhonemizer:
    """
    Text -> one language group's local tokens.

    Args:
        language: canonical BCP 47 code (`presets.normalize_std_lang` produces one).
        device: torch device for the G2P model.
        model_dir: directory already holding the G2P model. Default: $BFA_G2P_DIR, else it
            is downloaded from the Hub on first use (`hub.ensure_g2p_model`).
        cache_size: word -> IPA entries the G2P keeps. None = standard_g2p's default (500k,
            ~55 MB), 0 = unbounded.
        punctuation_sil: write a SIL where the text has pause punctuation. See the module
            docstring.
        verbose: 0 silent, 1 announces the one-off download.
    """

    def __init__(self, language, device="cpu", model_dir=None, cache_size=None,
                 punctuation_sil=False, verbose=1):
        self.device = device
        self.model_dir = model_dir
        self.cache_size = cache_size
        self.punctuation_sil = bool(punctuation_sil)
        self.verbose = verbose
        self.phonemes_ipa_key = "eipa"
        self._g2p = None
        self.set_backend(language)

    # ------------------------------------------------------------------ language

    def set_backend(self, language):
        """
        Switch language. The label space only changes when the language GROUP does, which is
        also when the caller needs a different checkpoint.
        """
        self.tag = _lc.bcp47_to_tag(language)
        self.language = _lc.tag_to_bcp47(self.tag)
        self.lang_group = LGI.lang_group_of(self.language)
        self.inventory = std_inventory(self.lang_group)
        # Same names the ph66 phonemizer carries, so either can be handed to old callers.
        self.phonemes_key = self.inventory.phonemes_key
        self.phoneme_groups_key = self.inventory.groups_key
        self.index_to_plabel = self.inventory.id_to_label
        self.index_to_glabel = self.inventory.group_id_to_label
        self.phoneme_id_to_group_id = self.inventory.phoneme_id_to_group_id
        # A missing segmenter package (pythainlp, fugashi, pycantonese) should fail here, once,
        # with its install line - not on the first sentence.
        if _ws.has_segmenter(self.tag):
            _ws.require_segmenter(self.tag)

    # ------------------------------------------------------------------ the G2P model

    @property
    def g2p(self):
        """The loaded G2P, fetched and built on first access."""
        if self._g2p is None:
            self._g2p = self._load_g2p()
        return self._g2p

    def _load_g2p(self):
        from ..standard_g2p import gold_g2p
        model_dir, tokenizer_dir = hub.ensure_g2p_model(self.model_dir, verbose=self.verbose)
        key = (os.path.abspath(model_dir), str(self.device))
        if key not in _G2P_INSTANCES:
            # goldG2P resolves its weights from module globals that point into the CharsiuG2P
            # repository (and fetches them itself when they are missing there); in an
            # installed package they are redirected to the copies `hub` has just made sure of.
            gold_g2p.G2P_MODEL_local_dir = model_dir
            gold_g2p.G2P_TOKENIZER_local_dir = tokenizer_dir
            kwargs = {} if self.cache_size is None else {"cache_size": int(self.cache_size)}
            _G2P_INSTANCES[key] = gold_g2p.goldG2P(device=self.device,
                                                   default_lang=self.language, **kwargs)
        return _G2P_INSTANCES[key]

    # ------------------------------------------------------------------ text

    def _split(self, text):
        """
        Text -> a list of items, each a word (str) or None for a silence.

        The text is cut at explicit `<SIL>` tokens, and at pause punctuation when
        `punctuation_sil` is on; each piece is word-segmented by standard_g2p, which is what
        handles the unspaced scripts (th, ja, zh, yue).
        """
        marked = _EXPLICIT_SIL_RE.sub("\x00", text)
        if self.punctuation_sil:
            marked = _PAUSE_RE.sub("\x00", marked)
        items = []
        for piece in marked.split("\x00"):
            items.extend(_ws.segment(piece, self.tag))
            items.append(None)
        items.pop()                                   # no silence after the last piece
        # A cut with no word on one side of it is a silence at the edge of the segment, or two
        # marks in a row. The aligner already pads an optional SIL onto both ends, so keep one
        # between words and drop the rest.
        out = []
        for item in items:
            if item is None and (not out or out[-1] is None):
                continue
            out.append(item)
        while out and out[-1] is None:
            out.pop()
        return out

    def phonemize_sentence(self, text):
        """
        Phonemize one sentence into the language group's local tokens.

        Returns a dict. One entry per TARGET TOKEN, all the same length:
            local_ph   token id in the language group's list (what the aligner decodes)
            gold_phg   phoneme group id (15 groups, shared by every language)
            gold_ph    the same token's id in the 270-token gold inventory
            eipa       the IPA phoneme the G2P wrote, before any backoff
            mipa       the token's own label (differs from eipa where a phoneme backed off)
            tone, stress, length   the phoneme's layers (0 = none / short)
            word_num   index into `words`
        Per sentence:
            words, text, lang, lang_group, phoneme_set
            n_folded       tokens outside the language group's list, now `<unk>`
            gold_unmapped  IPA segments with no gold mapping at all
        """
        items = self._split(text)
        words = [w for w in items if w is not None]
        d = self.g2p.phonemize_words(words, lang=self.language) if words else None
        layers = self._expand(d) if d else self._empty_layers()
        return self._assemble(text, items, layers, d.get("gold_unmapped", []) if d else [])

    def phonemize_gold(self, segment):
        """
        A segment that is ALREADY phonemized to gold indices - a `.gs.json` segment - into the
        same dict `phonemize_sentence` returns, without running the G2P.

        Needs `gold_ph`. `gold_ph_count`, `phonemes`, `tone`, `stress`, `length`, `word_num`
        and `words` are used when present; without a count (or the `phonemes` to recompute it
        from) the per-phoneme layers cannot be lined up with the tokens and are left empty.
        """
        gold_ph = [int(g) for g in segment["gold_ph"]]
        phonemes = segment.get("phonemes")
        count = segment.get("gold_ph_count")
        if count is None and phonemes is not None:
            count = LGI.gold_ph_count_of(phonemes)
        if count is None or sum(count) != len(gold_ph):
            count, phonemes = [1] * len(gold_ph), None
        n = len(count)

        def per_phoneme(key, default):
            values = segment.get(key)
            return list(values) if values is not None and len(values) == n else [default] * n

        d = {"gold_ph": gold_ph, "gold_ph_count": count,
             "gold_phg": segment.get("gold_phg"),
             "phonemes": list(phonemes) if phonemes is not None
             else [PI.TOKENS[g] for g in gold_ph],
             "tone": per_phoneme("tone", 0), "stress": per_phoneme("stress", 0),
             "length": per_phoneme("length", 0), "word_num": per_phoneme("word_num", 0)}
        words = list(segment.get("words") or [])
        if not words:
            words, d["word_num"] = [segment.get("text", "")], [0] * n
        layers = self._expand(d)
        out = self._assemble(segment.get("text", ""), words, layers,
                             segment.get("gold_unmapped", []))
        return out

    # ------------------------------------------------------------------ layers

    def _expand(self, d):
        """standard_g2p's per-phoneme output -> one entry per local token."""
        t = LGI.to_local(d["gold_ph"], self.language, d["gold_ph_count"])
        count = t["local_ph_count"]
        gold_phg = d.get("gold_phg")
        if gold_phg is None or len(gold_phg) != len(d["gold_ph"]):
            gold_phg = [int(PF.GROUPS.get(PI.TOKENS[g], 0)) for g in d["gold_ph"]]
        return {
            "local_ph": [int(i) for i in t["local_ph"]],
            "gold_ph": [int(g) for g in d["gold_ph"]],
            "gold_phg": [int(g) for g in gold_phg],
            "eipa": LGI.expand(d["phonemes"], count),
            "tone": LGI.expand(d["tone"], count),
            "stress": LGI.expand(d["stress"], count),
            "length": LGI.expand(d["length"], count),
            "word_num": LGI.expand(d["word_num"], count),
            "n_folded": int(t["n_folded"]),
        }

    @staticmethod
    def _empty_layers():
        return {"local_ph": [], "gold_ph": [], "gold_phg": [], "eipa": [], "tone": [],
                "stress": [], "length": [], "word_num": [], "n_folded": 0}

    def _assemble(self, text, items, layers, gold_unmapped):
        """
        Lay the tokens out in `items` order, giving each silence a token and a word of its own
        (`<SIL>`, as the ph66 phonemizer does) and renumbering `word_num` to match.
        """
        inv = self.inventory
        keys = ("local_ph", "gold_ph", "gold_phg", "eipa", "tone", "stress", "length")
        by_word = {}
        for pos, wi in enumerate(layers["word_num"]):
            by_word.setdefault(int(wi), []).append(pos)

        out = {k: [] for k in keys}
        out.update(word_num=[], words=[])
        sil = {"local_ph": inv.sil_id, "gold_ph": PI.TOKEN_INDEX[PI.SIL],
               "gold_phg": inv.sil_group, "eipa": PI.SIL, "tone": 0, "stress": 0, "length": 0}
        word_i = 0
        for item in items:
            new_index = len(out["words"])
            if item is None:
                out["words"].append(SIL_WORD)
                for k in keys:
                    out[k].append(sil[k])
                out["word_num"].append(new_index)
                continue
            out["words"].append(item)
            for pos in by_word.get(word_i, ()):
                for k in keys:
                    out[k].append(layers[k][pos])
                out["word_num"].append(new_index)
            word_i += 1

        out["mipa"] = [inv.id_to_label[i] for i in out["local_ph"]]
        out.update(text=text, lang=self.language, lang_group=self.lang_group,
                   phoneme_set="std", n_folded=layers["n_folded"],
                   gold_unmapped=list(gold_unmapped))
        return out
