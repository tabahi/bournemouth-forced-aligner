"""
TIMIT's and Buckeye's hand-written phone labels in the std phoneme set.

Both corpora are English, and in the std set English belongs to the `latin` language group, so
"std" here always means the latin token list - the 208 tokens `mbfa-latin` and
`mbfa-latin-benchmark` emit. No other mbfa model can be scored on these corpora, and the ids
written here mean nothing to one.

## Three stages, two of them standard_g2p's own

    corpus label --(the tables below)--> gold tokens --(PI.TOKEN_INDEX)--> gold ids
                 --(LGI.to_local, 'en')--> latin local ids

The last two are what labelled the mbfa training data and what the training-time TIMIT probe
uses (`bfa_models/p4mbfa/timit_probe_prep.py`); only the first is written here, because the
probe gets its phonemes from the G2P and these come from a human transcriber.

## How a label is written

**The way the std G2P writes that phoneme in English**, so that every id is one the latin
models were trained to emit on English speech:

  - diphthongs and affricates are TWO tokens (`ay` -> a ɪ, `ch` -> t ʃ). The latin list also
    has one-token forms (a͡ɪ, t͡ʃ), but they come from other languages' dictionaries and the
    G2P never writes them for English;
  - `l` is ɫ, `r` is ɹ, `er` is ɝ, and `ah` is ə - the G2P has no ʌ in English;
  - an allophone is written as its phoneme, as in the usual TIMIT folding: `ix` -> ɪ,
    `ux` -> u, `axr` -> ɝ, `hv` -> h, `nx` -> n, syllabic `el em en eng` -> ɫ m n ŋ, Buckeye's
    nasalised vowels (`aan`, `own`, ...) -> the oral vowel, and a stop's closure -> the stop
    (so a stop that was labelled closure + burst is its token twice);
  - the two sounds the G2P cannot write at all keep their own IPA token, which the latin
    list has: the flap `dx` -> ɾ and the glottal stop `q` / `tq` -> ʔ.

Silence is SIL. Anything that is sound but not the speaker's phones - Buckeye's noises,
laughter, the interviewer, excluded names - is `noise`, which in std is a real token and not a
padding slot. Buckeye's handful of typos are read from the word they sit in, so nothing is
`<unk>`.

None of this moves a boundary score: the scorer reads a segment's `text` and `boundaries` and
never a label. What a label does decide is whether its interval starts a boundary at all
(`is_speech`). evaluation/LABEL_MAPPING.md walks through both corpora.

A label outside its table raises: a new one has to be given a meaning, not a default.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bournemouth_aligner.standard_g2p import lang_group_inventory as LGI      # noqa: E402
from bournemouth_aligner.standard_g2p import phoneme_features as PF           # noqa: E402
from bournemouth_aligner.standard_g2p import phoneme_inventory_gold as PI     # noqa: E402

LANG = "en"
LANG_GROUP = "latin"

SIL, NOISE, UNK = PI.SIL, PI.NOISE, PI.UNK

# English phonemes as the std G2P writes them, keyed by the ARPAbet symbol TIMIT and Buckeye
# share. Checked against the G2P's output on both corpora: these 35 tokens are the whole of
# what it produces for English.
_PHONEMES = {
    # vowels
    "iy": "i", "ih": "ɪ", "eh": "ɛ", "ae": "æ", "aa": "ɑ", "ao": "ɔ", "uh": "ʊ", "uw": "u",
    "ah": "ə", "er": "ɝ",
    "ey": "e ɪ", "ay": "a ɪ", "oy": "ɔ ɪ", "aw": "a ʊ", "ow": "o ʊ",
    # stops, affricates, fricatives
    "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "g": "ɡ",
    "ch": "t ʃ", "jh": "d ʒ",
    "f": "f", "v": "v", "th": "θ", "dh": "ð", "s": "s", "z": "z", "sh": "ʃ", "zh": "ʒ",
    "hh": "h",
    # nasals, liquids, glides
    "m": "m", "n": "n", "ng": "ŋ", "l": "ɫ", "r": "ɹ", "w": "w", "y": "j",
    # the flap: no phoneme of its own in the G2P's English, so its own IPA token
    "dx": "ɾ",
}

# The 61 labels of TIMIT's .PHN files.
TIMIT_TO_STD = {
    **_PHONEMES,
    # reduced and fronted vowels
    "ax": "ə", "ax-h": "ə", "ix": "ɪ", "ux": "u", "axr": "ɝ",
    # closures: the stop they belong to
    "pcl": "p", "bcl": "b", "tcl": "t", "dcl": "d", "kcl": "k", "gcl": "ɡ",
    # syllabic consonants, the nasal flap, voiced h
    "el": "ɫ", "em": "m", "en": "n", "eng": "ŋ", "nx": "n", "hv": "h",
    "q": "ʔ",
    # silences: utterance edges, pauses, and the short epenthetic gap inside a word
    "h#": SIL, "pau": SIL, "epi": SIL,
}

# Buckeye's .phones labels, after `clean_buckeye_label`.
BUCKEYE_TO_STD = {
    **_PHONEMES,
    "el": "ɫ", "em": "m", "en": "n", "eng": "ŋ", "nx": "n",
    "tq": "ʔ", "q": "ʔ",
    # nasalised vowels: the oral vowel
    **{vowel + "n": _PHONEMES[vowel] for vowel in
       ("aa", "ae", "ah", "ao", "aw", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh",
        "uw")},
    # TIMIT symbols a few Buckeye labels use, and `h` for `hh`
    "ix": "ɪ", "h": "h", "hhn": "h",
    "SIL": SIL,
    # sound that is not the speaker's phones
    "VOCNOISE": NOISE, "NOISE": NOISE, "LAUGH": NOISE, "IVER": NOISE, "IVER-LAUGH": NOISE,
    "UNKNOWN": NOISE, "EXCLUDE": NOISE,
    # Typos (25 labels in the whole corpus), read from the word each one sits in - the
    # `.words` file gives the word and its dictionary phones beside what was transcribed:
    #   i   `w i th` (with), `b i k ah z` (because)       ih with a letter dropped
    #   e   `dh e r` (there)                              eh
    #   a   `a n` (on), `a` (I)                           aa
    #   an  `t w an nx iy` (twenty)                       a nasalised `a`
    #   x   `t eh x n` (ten), `y ih x n ow` (yknow)       a stray reduced vowel (TIMIT ax / ix)
    #   id  `r ah dx id d` (rotted)                       ih
    #   no  `no ow n` (don't)                             n
    #   j   `<CUTOFF-jih=just>`                           jh
    "i": "ɪ", "e": "ɛ", "a": "ɑ", "an": "ɑ", "x": "ə", "id": "ɪ", "no": "n", "j": "d ʒ",
}

# The diphthongs and affricates above, as (first token, second token): one hand-labelled phone
# that the aligner places as two. The scorer can count such a pair as one onset
# (`merge_two_token_phones` in compute_metrics.py). In the G2P's English `a`, `e` and `o`
# occur nowhere else.
TWO_TOKEN_PHONES = {tuple(written.split()) for written in _PHONEMES.values() if " " in written}

# Transcription markers: they delimit the labelled stretch and are not intervals of sound.
BUCKEYE_MARKERS = {"{B_TRANS}", "{E_TRANS}"}


def clean_buckeye_label(raw):
    """
    A `.phones` label field as the symbols it holds.

    The files are xwaves label files with `;` as the field separator, and about one label in
    twenty carries a second field (`iy; *`, `t; *q`) that marks the transcriber's doubt, not a
    different phone. `+1` is a stress mark on a vowel (`ih+1`, and before the nasalisation
    mark in `ah+1n`). Names the corpus redacted are written several ways (`<EXCLUDE-name>`,
    `<exclude-Name>`, `<EXCLUDE>`).

    Returns a list, because a few labels hold two phones for one interval (`ah n`).
    """
    label = raw.split(";", 1)[0].strip()
    if label in BUCKEYE_MARKERS:
        return [label]
    if "exclude" in label.lower():
        return ["EXCLUDE"]
    return [part.replace("+1", "") for part in label.split()]


class StdLabels:
    """
    One corpus's label table, resolved to latin local ids.

    Attributes:
        header: what an SRT written with these ids is stamped with - the same fields the
            training probe cache carries, so a reader can check them against its model.
        counts: how many times each corpus label was converted, for the summary.
    """

    def __init__(self, table, corpus):
        if LGI.lang_group_of(LANG) != LANG_GROUP:
            raise ValueError(f"{LANG!r} is in the {LGI.lang_group_of(LANG)!r} language group, "
                             f"not {LANG_GROUP!r}: the standard_g2p tables have changed")
        self.corpus = corpus
        self.tokens = list(LGI.tokens(LANG_GROUP))
        self.n_tokens = int(LGI.n_tokens(LANG_GROUP))
        self.fingerprint = LGI.gold_fingerprint()
        self.sil_id = self.tokens.index(SIL)
        self.noise_id = self.tokens.index(NOISE)
        self.unk_id = self.tokens.index(UNK)
        self.counts = {}
        self._local, self._groups = {}, {}
        for label, written in table.items():
            self._local[label], self._groups[label] = self._resolve(label, written.split())
        self.header = {"phoneme_set": f"std/{LANG_GROUP}", "lang": LANG,
                       "lang_group": LANG_GROUP, "n_tokens": self.n_tokens,
                       "gold_fingerprint": self.fingerprint}

    def _resolve(self, label, gold_tokens):
        """Gold tokens -> `(latin local ids, gold phoneme-group ids)`, refusing any fold."""
        missing = [t for t in gold_tokens if t not in PI.TOKEN_INDEX]
        if missing:
            raise ValueError(f"{self.corpus} label {label!r}: {missing} is not a gold token")
        gold = [PI.TOKEN_INDEX[t] for t in gold_tokens]
        local = LGI.to_local(gold, LANG)
        if local["lang_group"] != LANG_GROUP or local["n_local_ph"] != self.n_tokens:
            raise ValueError(f"to_local gave {local['lang_group']} / {local['n_local_ph']}, "
                             f"expected {LANG_GROUP} / {self.n_tokens}")
        # `<unk>` is in the table on purpose for the unreadable labels; any OTHER token that
        # comes back as `<unk>` is one the latin list does not have.
        if local["n_folded"] and gold_tokens != [UNK]:
            raise ValueError(f"{self.corpus} label {label!r} -> {gold_tokens} is not in the "
                             f"{LANG_GROUP} token list (it would fold to {UNK})")
        return list(local["local_ph"]), [int(PF.GROUPS[t]) for t in gold_tokens]

    def local_ph(self, label):
        """One corpus label -> its latin local ids (one, or two for a diphthong / affricate)."""
        if label not in self._local:
            raise KeyError(f"{self.corpus} label {label!r} has no entry in the std table "
                           f"(evaluation/dataset/std_labels.py)")
        self.counts[label] = self.counts.get(label, 0) + 1
        return list(self._local[label])

    def gold_phg(self, label):
        """The gold phoneme-group id of each token `local_ph(label)` returns."""
        return list(self._groups[label])

    def decode(self, local_ids):
        """Local ids -> their token strings."""
        return [self.tokens[i] for i in local_ids]

    def is_silence(self, label):
        return self._local[label] == [self.sil_id]

    def is_speech(self, label):
        """True for a phone the aligner places: anything but silence, noise and `<unk>`."""
        return not set(self._local[label]) & {self.sil_id, self.noise_id, self.unk_id}

    def summary(self):
        """One line per label that was seen: count and what it became."""
        lines = [f"{self.corpus} labels -> {self.header['phoneme_set']} "
                 f"({self.n_tokens} tokens, gold fingerprint {self.fingerprint}):"]
        for label, n in sorted(self.counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {label:<12} {n:>8}  ->  {' '.join(self.decode(self._local[label]))}")
        unseen = sorted(set(self._local) - set(self.counts))
        if unseen:
            lines.append(f"  never seen: {' '.join(unseen)}")
        return "\n".join(lines)


if __name__ == "__main__":
    for name, table in (("TIMIT", TIMIT_TO_STD), ("Buckeye", BUCKEYE_TO_STD)):
        labels = StdLabels(table, name)
        used = sorted({t for ids in labels._local.values() for t in labels.decode(ids)})
        print(f"{name}: {len(table)} labels -> {len(used)} of {labels.n_tokens} "
              f"{LANG_GROUP} tokens: {' '.join(used)}")
