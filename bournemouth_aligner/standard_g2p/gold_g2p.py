"""
G2P via CharsiuG2P (charsiu/g2p_multilingual_byT5_small_100).

The model emits an IPA *string* per word, not a list of phonemes. Turning that
string into countable units needs an IPA segmenter, because a single phoneme can
span several codepoints:

    t͡ʃ   -> tie bar joins two base letters into one affricate
    pʰ    -> modifier letter (Lm) rides on the preceding base
    aː    -> length mark (Lm) rides on the preceding base
    ẽ     -> combining diacritic (Mn) rides on the preceding base
    ˈkæt  -> stress mark is suprasegmental, not a phoneme
    saː˩˩˦ -> tone letters (Sk) form a contour, not a phoneme

`segment_ipa` handles all of these. Everything else here builds on it:
`decompose_ipa` splits the segments into index-aligned phoneme / tone /
stress / length layers, and the `goldG2P` class runs the model and maps the
result onto the gold inventory (phoneme_inventory_gold).
"""

import os
import json
import unicodedata
from collections import Counter, OrderedDict

try:                                  # imported as part of the package
    from . import lang_codes as _lc
    from . import phoneme_inventory_gold
    from . import word_segmentation as _ws
except ImportError:                   # imported with standard_g2p/ itself on sys.path
    import lang_codes as _lc          # (the scripts/ generators do this)
    import phoneme_inventory_gold
    import word_segmentation as _ws

# Same dual-style import as at the top of the module: this is
# deferred to here only because panphon-derived tables are large,
# not because the import context differs.
try:
    from . import phoneme_features
except ImportError:
    import phoneme_features

# The original model is 'charsiu/g2p_multilingual_byT5_small_100' on the HF hub;
# the bucket mirror is used for faster download. Weights are cached under the
# repo's tmp/ (gitignored), resolved from this file so the cwd does not matter.
G2P_MODEL_bucket_uri = "hf://buckets/Tabahi/g2p_multilingual_byT5_small_100-bucket"
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G2P_MODEL_local_dir = os.path.join(
    _REPO_ROOT, "tmp", "local_model_weights", "g2p_multilingual_byT5_small_100-bucket")
G2P_TOKENIZER_local_dir = os.path.join(G2P_MODEL_local_dir, "tokenizer")

# ---------------------------------------------------------------------------
# IPA character classes
# ---------------------------------------------------------------------------

TIE_BARS = {'͡', '͜'}                                  # ͡ (above), ͜ (below)
LENGTH_MARKS = {'ː', 'ˑ'}                              # ː ˑ
STRESS_MARKS = {'ˈ', 'ˌ', "'", '’', 'ˉ'}     # ˈ ˌ ' ’
TONE_LETTERS = {chr(c) for c in range(0x02e5, 0x02ea)}           # ˥ ˦ ˧ ˨ ˩
TONE_MISC = {'ꜛ', 'ꜜ', '↑', '↓', '↗',   # ꜛ ꜜ ↑ ↓ ↗ ↘
             '↘', '̋', '́', '̄', '̀',
             '̏', '̌', '̂'}
# Boundary / junk symbols that never carry phonetic content on their own.
BOUNDARIES = set('.|‖-‿#/[]()~ \t\n​‌‍')

# Tone diacritics double as vowel-quality marks in some languages, so they are
# only treated as tone when `tone_diacritics=True` is passed explicitly.
TONE_DIACRITICS = {'̋', '́', '̄', '̀', '̏',
                   '̌', '̂'}


def _is_modifier(ch):
    """Modifier letters (ʰ ʲ ʷ ˠ ˤ ⁿ ˡ ʼ) and combining marks ride on a base."""
    cat = unicodedata.category(ch)
    return cat in ('Mn', 'Me', 'Lm', 'Sk')


# ---------------------------------------------------------------------------
# Normalisation
#
# The CharsiuG2P training data is scraped from Wiktionary and is NOT uniformly
# IPA. Roughly 5% of all training tokens are transcription artifacts, and the
# model reproduces them at inference. Without this pass you end up allocating
# phoneme classes to what are really encoding accidents.
# ---------------------------------------------------------------------------

# Chao tone numerals (nan, and some zho varieties) — tone, not phonemes.
SUPERSCRIPT_TONE = set('⁰¹²³⁴⁵⁶⁷⁸⁹⁻')

# Same sound, different codepoint. Unify or you double-count.
CHAR_FIXES = {
    ':': 'ː',        # ASCII colon used as length (swe, yue)
    'g': 'ɡ',        # LATIN SMALL G -> SCRIPT G (the IPA one)
    'ʦ': 't͡s',       # deprecated ligatures (ltz, spa-latin, ger)
    'ʧ': 't͡ʃ',
    'ʣ': 'd͡z',
    'ʤ': 'd͡ʒ',
    'ʥ': 'd͡ʑ',
    'ʨ': 't͡ɕ',
    'ǝ': 'ə',        # TURNED E -> SCHWA (khm)
    'ł': 'ɫ',        # (kur)
    'X': 'x',        # (ltz)
    'ƹ': 'ʕ',        # (kur)
    # Greek look-alikes typed for IPA letters. NOTE beta, theta and chi are
    # genuine IPA symbols and must NOT be listed here.
    'ε': 'ɛ',        # GREEK SMALL EPSILON (fra-qu, grc)
    'α': 'ɑ', 'ο': 'o', 'ι': 'i', 'ν': 'v', 'ρ': 'r',
    # IPA writes the voiceless ring and the syllabic stroke ABOVE the letter
    # when it has a descender (ŋ̊, ŋ̍) and BELOW otherwise (n̥, n̩). They are the
    # same diacritic, so the corpus ends up with both spellings of one phoneme.
    '̊': '̥',   # ring above    -> ring below   (voiceless)
    '̍': '̩',   # line above    -> line below   (syllabic)
}

# Swedish (`swe`) is transcribed in SAMPA throughout rather than IPA
# ('PLANAVÄGEN -> plA:na%vE:gen'), and uzb/ger/swa/isl leak SAMPA capitals too
# ('Birinchi -> byɾˈynt͡Sy', where S is ʃ).
#
# IPA has no uppercase letters at all, so mapping capitals through SAMPA is
# unambiguous and safe to apply to every language.
SAMPA_UPPER = {
    'A': 'ɑ', 'E': 'ɛ', 'I': 'ɪ', 'O': 'ɔ', 'U': 'ʊ', 'Y': 'ʏ',
    'N': 'ŋ', 'S': 'ʃ', 'Z': 'ʒ', 'C': 'ɕ', 'T': 'θ', 'D': 'ð',
    'Q': 'ɒ', 'V': 'ʌ', 'J': 'ɲ', 'B': 'β', 'G': 'ɣ', 'H': 'ɥ',
    'L': 'ʎ', 'M': 'ɱ', 'R': 'ʀ', 'W': 'ʍ', 'X': 'x', 'K': 'ɬ',
    'F': 'ɸ', 'P': 'ʋ',
}
# Digits/brackets are SAMPA vowels, but only where the language uses SAMPA --
# elsewhere digits are tone. Applied for `swe` only.
SAMPA_SWE_EXTRA = {'}': 'ʉ', '2': 'ø', '9': 'œ', '0': 'ɵ', '@': 'ə', '{': 'æ'}
# In SAMPA a trailing backtick marks retroflexion: n` -> ɳ
SAMPA_RETROFLEX = {'n': 'ɳ', 't': 'ʈ', 'd': 'ɖ', 's': 'ʂ', 'l': 'ɭ', 'r': 'ɽ'}


def normalize_ipa(ipa, lang=None):
    """Repair the known encoding artifacts in CharsiuG2P output.

    Pass the language tag so language-specific schemes (Swedish SAMPA) are
    handled. Returns a plain IPA string, still unsegmented.
    """
    if not ipa:
        return ipa

    # Russian marks optional palatalisation as ⁽ʲ⁾; keep the ʲ, drop the parens.
    ipa = ipa.replace('⁽', '').replace('⁾', '')

    # SAMPA retroflex is written base + backtick: n` -> ɳ
    if '`' in ipa:
        out = []
        i = 0
        while i < len(ipa):
            ch, nxt = ipa[i], (ipa[i + 1] if i + 1 < len(ipa) else '')
            if nxt == '`' and ch.lower() in SAMPA_RETROFLEX:
                out.append(SAMPA_RETROFLEX[ch.lower()])
                i += 2
            else:
                out.append(ch)
                i += 1
        ipa = ''.join(out)

    if lang == 'swe':
        ipa = ''.join(SAMPA_SWE_EXTRA.get(ch, ch) for ch in ipa)

    ipa = ipa.replace('%', 'ˌ').replace('*', '').replace('"', 'ˈ')

    # Decompose first, so a precomposed letter (å = a + ring above) exposes its
    # combining mark to the diacritic fixes below.
    ipa = unicodedata.normalize('NFD', ipa)

    # Uppercase first (SAMPA), then the codepoint unifications.
    ipa = ''.join(SAMPA_UPPER.get(ch, ch) for ch in ipa)
    ipa = ''.join(CHAR_FIXES.get(ch, ch) for ch in ipa)

    # Wiktionary typos: a modifier typed twice ('qatˤˤala' in ara, 't͡͡s').
    # Collapsing them keeps tˤˤ as tˤ rather than losing the pharyngealisation
    # when the backoff later peels an unknown segment down to bare t.
    # Tone letters are Sk and so look like modifiers, but a repeated tone letter
    # is a real contour (Thai ˩˩˦ is low-low-rising) and must survive.
    out = []
    for ch in ipa:
        if (out and ch == out[-1] and (ch in TIE_BARS or _is_modifier(ch))
                and ch not in TONE_LETTERS and ch not in SUPERSCRIPT_TONE
                and ch not in TONE_MISC):
            continue
        out.append(ch)
    ipa = ''.join(out)

    # ʼ marks ejectives on consonants, but fra/fra-qu also use it on vowels as
    # an h-aspiré/liaison marker ('du haut de' -> dyʼodə), which is not a phone.
    out = []
    for ch in ipa:
        if ch == 'ʼ':
            # look back past any diacritics to find the base it rides on
            prev = next((c for c in reversed(out) if not _is_modifier(c)), '')
            if prev in IPA_VOWELS:
                continue
        out.append(ch)
    return ''.join(out)


def segment_ipa(ipa, keep_stress=False, keep_tone=False, keep_length=True,
                keep_diacritics=True, tone_diacritics=False, split_ties=False,
                normalize=True, lang=None):
    """Split an IPA string into a list of phoneme tokens.

    keep_stress      emit ˈ / ˌ as their own tokens instead of dropping them
    keep_tone        emit tone-letter contours (˧˩˦) as their own tokens
    keep_length      keep the ː / ˑ length mark attached to its vowel
    keep_diacritics  keep combining marks and modifier letters on the base
    tone_diacritics  also treat ́  ̀  ̄  ̌  ̂ as tone rather than vowel quality
    split_ties       break t͡ʃ into t + ʃ instead of keeping one affricate token
    normalize        repair encoding artifacts first (see `normalize_ipa`)
    lang             language tag, needed for Swedish SAMPA
    """
    if not ipa:
        return []

    if normalize:
        ipa = normalize_ipa(ipa, lang=lang)

    # NFD so precomposed forms (ẽ) and decomposed forms (e + U+0303) segment
    # identically. Tie bars and modifier letters are unaffected by NFD.
    ipa = unicodedata.normalize('NFD', ipa)

    tone_set = TONE_LETTERS | TONE_MISC | SUPERSCRIPT_TONE
    if not tone_diacritics:
        tone_set -= TONE_DIACRITICS

    tokens = []
    cur = ''
    pending_tie = False

    def flush():
        nonlocal cur, pending_tie
        if cur:
            tokens.append(unicodedata.normalize('NFC', cur))
        cur = ''
        pending_tie = False

    for ch in ipa:
        if ch in TIE_BARS:
            # A tie bar with nothing before it is stray; ignore it.
            if cur and not split_ties:
                cur += ch
                pending_tie = True
            elif cur and split_ties:
                pending_tie = True   # consume the next base as a separate token
            continue

        if ch in STRESS_MARKS:
            flush()
            if keep_stress:
                tokens.append('ˈ' if ch in ('ˈ',) else 'ˌ')
            continue

        if ch in tone_set:
            flush()
            if keep_tone:
                # Contours are runs of tone letters: ˧ + ˩ + ˦ -> one token.
                if tokens and tokens[-1] and tokens[-1][-1] in tone_set:
                    tokens[-1] += ch
                else:
                    tokens.append(ch)
            continue

        if ch in BOUNDARIES:
            flush()
            continue

        if ch in LENGTH_MARKS:
            if keep_length and cur:
                cur += ch
            continue

        if _is_modifier(ch):
            if keep_diacritics and cur:
                cur += ch
            continue

        # A base letter.
        if pending_tie:
            if split_ties:
                flush()
                cur = ch
            else:
                cur += ch
                pending_tie = False
        else:
            flush()
            cur = ch

    flush()
    return tokens


def strip_to_base(token):
    """Reduce a phoneme token to its bare base letter(s): pʰ -> p, aː -> a."""
    out = ''.join(ch for ch in unicodedata.normalize('NFD', token)
                  if not _is_modifier(ch) and ch not in TIE_BARS)
    return unicodedata.normalize('NFC', out)


# ---------------------------------------------------------------------------
# Layered decomposition
#
# Tone, stress and length are suprasegmental: they belong to a syllable, not to
# a segment. Folding them into the phoneme label multiplies the inventory
# (aː, á, âː, ... are all separate labels) for no acoustic gain. Modelling them
# as parallel output layers keeps the segment inventory small and lets each
# layer have its own head.
# ---------------------------------------------------------------------------

IPA_VOWELS = set('iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɞʌɔæɐaɶɑɒ')
SYLLABIC = '̩'   # combining vertical line below


def is_vowel(token):
    """True for vowels and for syllabic consonants (n̩, l̩) - i.e. anything that
    can be a syllable nucleus and therefore carry tone/stress."""
    nfd = unicodedata.normalize('NFD', token)
    if SYLLABIC in nfd:
        return True
    base = strip_to_base(token)
    return bool(base) and base[0] in IPA_VOWELS


def strip_length(token):
    """Remove length marks, returning (segment_without_length, length_level)."""
    nfd = unicodedata.normalize('NFD', token)
    n_long = nfd.count('ː')
    half = 'ˑ' in nfd
    seg = ''.join(ch for ch in nfd if ch not in LENGTH_MARKS)
    level = 2 if n_long else (1 if half else 0)
    return unicodedata.normalize('NFC', seg), level


def decompose_ipa(ipa, lang=None, tone_diacritics=False):
    """Split an IPA string into parallel, index-aligned layers.

    Returns a dict of equal-length lists, one entry per segment:
        segments  phoneme without length mark   ['pʰ', 'a', 's', 'a']
        length    0 short / 1 half / 2 long     [0, 2, 0, 2]
        tone      contour on that syllable      ['', '˧', '', '˩˩˦']
        stress    0 none / 1 primary / 2 sec.   [0, 0, 0, 0]

    Tone attaches to the most recent nucleus; stress to the next one.
    """
    toks = segment_ipa(ipa, lang=lang, keep_stress=True, keep_tone=True,
                       keep_length=True, keep_diacritics=True,
                       tone_diacritics=tone_diacritics)

    segments, length, tone, stress = [], [], [], []
    pending_stress = 0
    last_nucleus = -1

    for t in toks:
        if t in ('ˈ', 'ˌ'):
            pending_stress = 1 if t == 'ˈ' else 2
            continue
        if t and (t[0] in TONE_LETTERS or t[0] in SUPERSCRIPT_TONE
                  or t[0] in TONE_MISC):
            # Tone belongs to the syllable just closed. If that nucleus already
            # carries a tone we are in a new syllable whose nucleus is a
            # syllabic consonant written without the ̩ diacritic (bare m/n/ŋ,
            # common in yue/nan/zho) -- attach to the last segment instead, or
            # the contours of adjacent syllables silently concatenate.
            idx = last_nucleus
            if idx < 0 or tone[idx]:
                idx = len(segments) - 1
            if idx >= 0:
                tone[idx] = (tone[idx] + t) if tone[idx] else t
                last_nucleus = idx
            continue

        seg, lvl = strip_length(t)
        segments.append(seg)
        length.append(lvl)
        tone.append('')
        if is_vowel(seg):
            last_nucleus = len(segments) - 1
            stress.append(pending_stress)
            pending_stress = 0
        else:
            stress.append(0)

    # A stress mark sits before the onset, so push it onto the nucleus it opens.
    out = {'segments': segments, 'length': length, 'tone': tone, 'stress': stress}
    return out


_CHAO = {'˥': '5', '˦': '4', '˧': '3', '˨': '2', '˩': '1'}
_SUPER = {'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
          '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9', '⁻': '-'}


def normalize_tone(tone, keep_sandhi=False):
    """Put both tone encodings into one numeric Chao form.

    The data uses tone letters (˧˥, in tha/vie/yue/zho) and superscript Chao
    digits (²¹⁻⁵³, in nan) for the same thing. Left as-is you get two disjoint
    tone vocabularies. Returns e.g. '35', '214', '44-22'.

    keep_sandhi  keep Min Nan's underlying-to-surface notation ('44-22');
                 otherwise only the underlying tone is kept ('44').
    """
    if not tone:
        return ''
    out = ''.join(_CHAO.get(c, _SUPER.get(c, c)) for c in tone)
    if not keep_sandhi:
        out = out.split('-')[0]
    return out


# ---------------------------------------------------------------------------
# Language selection
#
# The ONLY language identifier this module accepts from a caller is a single
# BCP 47 code -- ISO 639-1 where a language has one, else ISO 639-3 (see
# lang_codes.BCP47_TO_TAG for the vocabulary: 'pt', 'pt-BR', 'en-GB',
# 'es-419', 'zh-Hant', 'hbs-Cyrl', ...). CharsiuG2P's own tag spelling
# ('eng-us', 'ger', 'zho-s', 'por-bz') is an internal implementation detail
# resolved by `resolve_tag` and never a valid `lang` argument -- this is what
# keeps a caller from having to know that the model spells German 'ger'
# (639-2/B) rather than 'deu' (639-3), or that Brazilian Portuguese is
# 'por-bz' rather than 'por-br'.
# ---------------------------------------------------------------------------

# Languages whose script has no spaces: the model is a WORD g2p, so text must
# be word-segmented or every "word" handed to the model is a whole clause and
# the output is garbage. `segment` does this where a backend exists
# (`has_segmenter`); see word_segmentation.py. Defined over ISO codes in
# lang_codes and expanded to tags here; see the README.
#
# Note this now also covers 'tts' (Isan, written in Thai script), which the
# hand-written version of this set missed.
NEEDS_WORD_SEGMENTATION = {t for t, i in _lc.TAG_TO_ISO.items()
                           if i in _lc.NEEDS_WORD_SEGMENTATION_ISO}


# The 100 tags the model was actually trained on. An unrecognised tag does not
# raise: byT5 is byte-level, so '<rw>: word' still decodes to something that
# looks like IPA but is not that language. Check `is_supported` before running a
# dataset rather than discovering it in the labels.
SUPPORTED_LANGS = frozenset("""
ady afr amh ang ara arg arm-e arm-w aze bak bel bos bul bur cat cze dan dut egy
eng-uk eng-us enm epo est eus fas fin fra fra-qu geo ger gle glg grc gre
hbs-cyrl hbs-latn hin hun ice ido ina ind isl ita jpn kaz khm kor kur lat-clas
lat-eccl lit ltz mac mlt mri msa nan nob ori pap pol por-bz por-po ron rus san
slk slo slv sme snd spa spa-latin spa-me sqi srp swa swe syc tam tat tgl tha
tts tuk tur uig ukr urd uzb vie-c vie-n vie-s wel-nw wel-sw yue zho-s zho-t
""".split())


# The BCP 47 side of the mapping lives in lang_codes, which imports nothing,
# so a dataset loader can label its clips with standard codes without pulling in
# transformers. Re-exported here: `from standard_g2p.gold_g2p import bcp47_to_tag`
# works just as well as importing from lang_codes directly.
UnsupportedLanguageError = _lc.UnsupportedLanguageError
TAG_TO_ISO, ISO_TO_TAG = _lc.TAG_TO_ISO, _lc.ISO_TO_TAG
ISO_DEFAULT_TAG, ISO_BRIDGE = _lc.ISO_DEFAULT_TAG, _lc.ISO_BRIDGE
SUPPORTED_ISO = _lc.SUPPORTED_ISO
BCP47_TO_TAG, TAG_TO_BCP47, BCP47_SUFFIXES = (
    _lc.BCP47_TO_TAG, _lc.TAG_TO_BCP47, _lc.BCP47_SUFFIXES)
TONAL_ISO, TONE_LETTER_ISO = _lc.TONAL_ISO, _lc.TONE_LETTER_ISO
TONE_DIACRITIC_ISO, PITCH_ACCENT_ISO = _lc.TONE_DIACRITIC_ISO, _lc.PITCH_ACCENT_ISO
NEEDS_WORD_SEGMENTATION_ISO = _lc.NEEDS_WORD_SEGMENTATION_ISO
tag_to_iso, bcp47_to_tag = _lc.tag_to_iso, _lc.bcp47_to_tag
tag_to_bcp47 = _lc.tag_to_bcp47
lang_is_supported = _lc.lang_is_supported
is_tonal, needs_word_segmentation = _lc.is_tonal, _lc.needs_word_segmentation
LANG_GROUPS = _lc.LANG_GROUPS
LANG_GROUP_MEMBERS = _lc.LANG_GROUP_MEMBERS
ISO_TO_LANG_GROUP, EXCLUDED_ISO, FAMILY_ISO = _lc.ISO_TO_LANG_GROUP, _lc.EXCLUDED_ISO, _lc.FAMILY_ISO
lang_group, is_excluded = _lc.lang_group, _lc.is_excluded
lang_group_members, family = _lc.lang_group_members, _lc.family

# SUPPORTED_LANGS above is what the model reports; TAG_TO_ISO is what the
# mapping tables were built against. If they ever drift, every ISO lookup for
# the affected language is wrong, so say so at import rather than at inference.
_drift = SUPPORTED_LANGS.symmetric_difference(TAG_TO_ISO)
if _drift:
    raise UnsupportedLanguageError(
        'lang_codes.TAG_TO_ISO and SUPPORTED_LANGS disagree on: %s' % sorted(_drift))


def resolve_tag(lang, default='eng'):
    """BCP 47 code -> CharsiuG2P tag. INTERNAL USE.

    Every public method on goldG2P takes `lang` (a single BCP 47 code,
    e.g. 'pt', 'pt-BR', 'zh-Hant') and calls this to get the tag it
    actually feeds the model. It is exposed at module level too, for callers
    that need the tag for a gating check (e.g. `resolve_tag(lang) in
    NEEDS_WORD_SEGMENTATION`) without wanting to load the model.

    `lang`/`default` empty or None falls back to `default`, which itself is
    resolved to a tag the same way -- this is the "clip has no lang" case,
    not a way to pass a tag directly.
    """
    if not lang:
        lang = default
    return _lc.bcp47_to_tag(lang)


def is_supported(lang):
    """True if `lang` (a BCP 47 code) is one of the model's 100 training
    languages. Unlike `resolve_tag`, an empty/None `lang` is False, not a
    fallback to some default language."""
    try:
        return _lc.bcp47_to_tag(lang) in SUPPORTED_LANGS
    except UnsupportedLanguageError:
        return False


def _tag_from_field(value):
    """Best-effort tag for a value read out of a stored file (an srt's
    'lang'/'language' field), which may be a BCP 47 code (current writers)
    or a raw CharsiuG2P tag (files written before this module's `lang`
    argument was made BCP-47-only). Returns None rather than raising, so
    callers can fall through to the next source. NEVER use this for a `lang`
    argument supplied by a caller -- those are BCP 47, strictly."""
    if not value:
        return None
    value = value.strip()
    if value in SUPPORTED_LANGS:
        return value
    try:
        return _lc.bcp47_to_tag(value)
    except UnsupportedLanguageError:
        return None


# Tone contours found across the corpus, in unified numeric form (see
# `normalize_tone`). Index 0 is "no tone", so an atonal language is all zeros
# and the tone head simply never fires.
TONE_VOCAB = ['', '35', '21', '33', '45', '51', '32', '214', '55', '24', '3',
              '312', '5', '13', '2', '212', '42', '44', '53', '22', '4', '114']
TONE_INDEX = {t: i for i, t in enumerate(TONE_VOCAB)}
N_TONES = len(TONE_VOCAB)

# Stress and length are small ordinal scales, not flags. Length in particular is
# TERNARY: the half-long mark ˑ is rare (well under 0.1% of segments) but real,
# so a binary head would quietly mislabel it. Collapse HALF into LONG yourself
# if you want two classes -- do not assume two.
STRESS_NONE, STRESS_PRIMARY, STRESS_SECONDARY = 0, 1, 2
N_STRESS = 3

LENGTH_SHORT, LENGTH_HALF, LENGTH_LONG = 0, 1, 2
N_LENGTHS = 3


# Text -> words lives in word_segmentation (stdlib only, like lang_codes).
# Re-exported so `from standard_g2p.gold_g2p import split_words` keeps working.
split_words, segment = _ws.split_words, _ws.segment
has_segmenter, require_segmenter = _ws.has_segmenter, _ws.require_segmenter


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class goldG2P:
    """Grapheme-to-phoneme over 100 languages, loaded once and reused.

    The model is ~300M parameters and takes a few seconds to load, so build one
    instance per process and call it per file:

        G = goldG2P(device='cuda:0', default_lang='en')
        for srt, gs in files:
            G.phonemize_srt(srt, gs, lang='en')
        G.print_stats()

    The word -> IPA cache is held on the instance and therefore persists across
    files. On real corpora word types repeat heavily, so this is most of the
    speed: only unseen word types ever reach the GPU.
    """

    # Word types grow sub-linearly with corpus size (Heaps' law), so the cache
    # tends to plateau near a language's vocabulary: ~126k types for English,
    # ~279k German, ~403k Russian in the CharsiuG2P dictionaries. At ~111 bytes
    # an entry that is 14-45 MB, which is fine.
    #
    # What breaks the plateau is unbounded token classes that appear in ASR
    # transcripts - numerals, IDs, URLs, timestamps, misrecognitions - and
    # agglutinative or compounding languages. Those have no vocabulary ceiling,
    # so the cache is bounded and evicts least-recently-used entries. Access is
    # Zipfian, so LRU keeps the words that actually repeat and the hit rate
    # barely moves.
    DEFAULT_CACHE_SIZE = 500_000          # ~55 MB

    # A batch's memory grows with words x its longest word, since every input
    # is padded to the longest. 2048 padded input bytes (128 words of up to 16
    # bytes, tag included) needs ~0.25 GB on top of the 1.2 GB of weights, so
    # the G2P still fits a 2 GB GPU; one 250-byte "word" in a batch of 128
    # would need ~2.6 GB.
    DEFAULT_BATCH_BYTES = 2048

    def __init__(self, device='cpu', default_lang='en',
                 batch_size=128, max_length=64, map_inventory=True,
                 tone_diacritics=False, cache_size=DEFAULT_CACHE_SIZE,
                 batch_bytes=DEFAULT_BATCH_BYTES):
        # Any torch device string: 'cpu', 'cuda', 'cuda:1', ...
        self.device = device
        if not (os.path.exists(os.path.join(G2P_MODEL_local_dir, 'pytorch_model.bin'))
                and os.path.exists(os.path.join(G2P_TOKENIZER_local_dir, 'tokenizer_config.json'))):
            from huggingface_hub import sync_bucket
            sync_bucket(G2P_MODEL_bucket_uri, G2P_MODEL_local_dir)
        # Imported here, not at module level, so the IPA utilities and the
        # scripts/ generators work without transformers installed.
        from transformers import T5ForConditionalGeneration, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(G2P_TOKENIZER_local_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(G2P_MODEL_local_dir)
        self.model.to(self.device)
        self.model.eval()

        # BCP 47 code, kept for reference/printing, and the tag it resolves
        # to, which is what every method actually falls back on when a clip
        # carries no language of its own.
        self.set_default_lang(default_lang)
        self.batch_size = batch_size
        self.batch_bytes = batch_bytes
        self.max_length = max_length
        self.tone_diacritics = tone_diacritics

        # Inventory mapping is optional so the class is still useful for raw
        # IPA work (inventory analysis, building a different mapping).
        self.map_inventory = map_inventory
        self._inv = self._feat = None
        if map_inventory:
            
            self._inv, self._feat = phoneme_inventory_gold, phoneme_features

        # OrderedDict so eviction is O(1) from the cold end.
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.stats = Counter()
        self.unmapped = Counter()


    def set_default_lang(self, default_lang):
        """Change the default language for clips that carry no language of
        their own. `default_lang` is a single BCP 47 code (e.g. 'pt',
        'pt-BR', 'zh-Hant')."""
        self.default_lang = default_lang
        self.default_tag = _lc.bcp47_to_tag(default_lang)

    def scope(self):
        """Cardinality of every indexed layer this instance emits.

        An index is meaningless without the size of the space it indexes, so
        every output carries these. A reader sizes its heads from them and
        asserts against the tables it loaded:

            n_tones     `tone` indexes TONE_VOCAB     (0 = no tone)
            n_stresses  `stress` is 0 none / 1 primary / 2 secondary
            n_lengths   `length` is 0 short / 1 half-long / 2 long
            n_gold_ph   `gold_ph` indexes phoneme_inventory_gold.TOKENS
            n_gold_phg  `gold_phg` indexes phoneme_features.GROUP_NAMES

        The gold keys are present only when `map_inventory` is set.
        """
        out = {'n_tones': N_TONES, 'n_stresses': N_STRESS, 'n_lengths': N_LENGTHS}
        if self.map_inventory:
            out['n_gold_ph'] = self._inv.N_TOKENS
            out['n_gold_phg'] = self._feat.N_GROUPS
        return out

    def lang_flags(self, tag):
        """What a reader needs to know about the language itself, so that a
        .gs.json can be interpreted without consulting lang_codes:

            tonal                    lexical tone, as written in the training
                                     dictionary (lang_codes.TONAL_ISO)
            tone_layer               the tone reaches `tone`. False for a
                                     non-tonal language, and for Burmese, whose
                                     tone is a vowel diacritic (unless
                                     `tone_diacritics` is set): `tone` is then
                                     all 0 by construction and should be masked,
                                     not learned as "no tone"
            needs_word_segmentation  the script does not put spaces between
                                     words (lang_codes.NEEDS_WORD_SEGMENTATION_ISO)
            word_segmenter           the segmenter that produced `words`, with
                                     package versions, or None. None together
                                     with needs_word_segmentation means the
                                     text was NOT segmented (nan, tts) and the
                                     "words" are whole clauses
        """
        return {'tonal': _lc.is_tonal(tag),
                'tone_layer': _lc.is_tonal(tag, diacritic_tone=self.tone_diacritics),
                'needs_word_segmentation': _lc.needs_word_segmentation(tag),
                'word_segmenter': _ws.segmenter_name(tag)}

    def _cache_put(self, key, value):
        self.cache[key] = value
        if self.cache_size and len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)      # drop least recently used
            self.stats['cache_evictions'] += 1

    def cache_clear(self):
        """Drop everything. Worth calling between languages on a big run: the
        per-language vocabularies do not share entries, so an English cache is
        dead weight while phonemizing Russian."""
        self.cache.clear()

    def _tag(self, lang):
        """BCP 47 code -> CharsiuG2P tag, falling back to this instance's
        default when `lang` is empty. The only place a raw tag is ever
        produced from a `lang` argument."""
        if not lang:
            return self.default_tag
        return _lc.bcp47_to_tag(lang)

    # -- inference ---------------------------------------------------------

    def _batches(self, prompts):
        """Indices of `prompts` in model batches: shortest first, so a batch
        pads little and stops generating early, and each batch is at most
        `batch_size` prompts and `batch_bytes` padded bytes (ByT5 reads one
        token per byte). A prompt longer than `batch_bytes` gets a batch of
        its own."""
        n_bytes = [len(p.encode('utf-8')) for p in prompts]
        order = sorted(range(len(prompts)), key=n_bytes.__getitem__)
        batch = []
        for i in order:
            # Sorted ascending, so prompt i is the longest of the batch.
            if batch and (len(batch) >= self.batch_size
                          or (len(batch) + 1) * n_bytes[i] > self.batch_bytes):
                yield batch
                batch = []
            batch.append(i)
        if batch:
            yield batch

    def _generate(self, words, tag):
        """Raw model call for a list of words, given an already-resolved tag."""
        import torch
        prompts = ['<%s>: %s' % (tag, w) for w in words]
        out = [None] * len(words)
        for idx in self._batches(prompts):
            enc = self.tokenizer([prompts[i] for i in idx], padding=True,
                                 add_special_tokens=False,
                                 return_tensors='pt').to(self.device)
            with torch.no_grad():
                preds = self.model.generate(**enc, num_beams=1,
                                            max_length=self.max_length)
            for i, ipa in zip(idx, self.tokenizer.batch_decode(
                    preds.tolist(), skip_special_tokens=True)):
                out[i] = ipa
        return out

    def g2p_words(self, words, lang=None):
        """Words -> raw IPA strings, straight from the model (no cache).

        `lang` is a single BCP 47 code (e.g. 'pt', 'pt-BR', 'zh-Hant').
        """
        if not words:
            return []
        return self._generate(words, self._tag(lang))

    def ipa_for(self, words, lang=None):
        """Words -> IPA, going through the cache. Unseen types are batched."""
        return self._ipa_for_tag(words, self._tag(lang))

    def _ipa_for_tag(self, words, tag):
        """Core of `ipa_for`, given an already-resolved tag."""
        todo, seen = [], set()
        for w in words:
            key = (tag, w)
            if key in self.cache:
                self.cache.move_to_end(key)     # mark recently used
                self.stats['cache_hits'] += 1
            elif key in seen:
                # A repeat within this same call: costs no model run either.
                # Checked separately because the cache is written after this
                # loop, so the first occurrence is not in it yet.
                self.stats['cache_hits'] += 1
            else:
                seen.add(key)
                todo.append(w)
        if todo:
            fresh = dict(zip(todo, self._generate(todo, tag)))
            for w, ipa in fresh.items():
                self._cache_put((tag, w), ipa)
            self.stats['cache_misses'] += len(todo)
            # Read through `fresh` as well: with a small cache_size a long
            # enough batch could evict its own earlier entries before we return.
            return [self.cache.get((tag, w)) or fresh.get(w, '') for w in words]
        return [self.cache.get((tag, w), '') for w in words]

    # -- layered output ----------------------------------------------------

    def _gold_map(self, phonemes):
        """Map standardized IPA segments onto the gold inventory (see
        phoneme_inventory_gold). Returns (gold_ph, gold_phg, gold_ph_count,
        gold_unmapped):

            gold_ph        inventory index per mapped unit (a segment that
                            backs off to several units contributes several)
            gold_phg       phoneme group index (phoneme_features.GROUPS),
                           aligned with gold_ph
            gold_ph_count  per segment of `phonemes`, how many gold_ph
                           entries it became; aligned with `phonemes`, sums
                           to len(gold_ph)
            gold_unmapped  sorted list of distinct segments that had no
                           mapping (direct or backoff) -- i.e. still <unk>

        gold_ph is not index-aligned with `phonemes`: a segment can expand to
        several gold units (37 BACKOFF rules split one, e.g. ʈ͡ʂ -> ʈ ʂ).
        gold_ph_count is what re-aligns them, and so the tone / stress /
        length layers with gold_ph and local_ph (lang_group_inventory.expand).
        """
        gold_ph, gold_phg, count, unmapped = [], [], [], set()
        for seg in phonemes:
            mapped = self._inv.map_phoneme(seg)
            if mapped == [self._inv.UNK]:
                unmapped.add(seg)
                self.unmapped[seg] += 1
            for m in mapped:
                gold_ph.append(self._inv.TOKEN_INDEX[m])
                gold_phg.append(self._feat.GROUPS.get(m, 0))
            count.append(len(mapped))
        return gold_ph, gold_phg, count, sorted(unmapped)

    def _phonemize_words_tag(self, words, tag):
        """Core of `phonemize_words`, given an already-resolved tag. Shared
        with `phonemize_srt`, which resolves its tag once per file rather
        than once per segment."""
        ipas = self._ipa_for_tag(words, tag)

        phonemes, tone, stress, length, word_num = [], [], [], [], []
        for wi, raw in enumerate(ipas):
            if not raw:
                continue
            d = decompose_ipa(raw, lang=tag, tone_diacritics=self.tone_diacritics)
            for seg, tn, st, ln in zip(d['segments'], d['tone'],
                                       d['stress'], d['length']):
                phonemes.append(seg)
                tone.append(TONE_INDEX.get(normalize_tone(tn), 0))
                stress.append(st)
                length.append(ln)
                word_num.append(wi)

        self.stats['words'] += len(words)
        self.stats['phonemes'] += len(phonemes)
        out = {'words': list(words), 'ipa': ipas, 'phonemes': phonemes,
               'tone': tone, 'stress': stress, 'length': length,
               'word_num': word_num}
        if self.map_inventory:
            (out['gold_ph'], out['gold_phg'], out['gold_ph_count'],
             out['gold_unmapped']) = self._gold_map(phonemes)
        return out

    def phonemize_words(self, words, lang=None):
        """Words -> standardized IPA phonemes, the aligned layers, and (when
        `map_inventory`) the gold-inventory mapping.

        `lang` is a single BCP 47 code (e.g. 'pt', 'pt-BR', 'zh-Hant').

        Returns a dict. Scope (see `scope`):
            lang      the BCP 47 code used (the instance default if not given)
            g2p_lang  the CharsiuG2P tag it resolved to
            tonal, tone_layer, needs_word_segmentation, word_segmenter
                      (see `lang_flags`)
            n_tones, n_stresses, n_lengths, n_gold_ph, n_gold_phg
        One entry per phoneme, all the same length:
            phonemes  standardized IPA segment, as produced by CharsiuG2P
                      (encoding artifacts fixed, e.g. Swedish SAMPA -> IPA;
                      NOT yet mapped onto any fixed inventory)
            tone      tone index into TONE_VOCAB (0 = none)
            stress    0 none / 1 primary / 2 secondary
            length    0 short / 1 half-long / 2 long
            word_num  index into `words` telling which word each phoneme is from
        Not per-phoneme:
            words     the words actually phonemized
            ipa       raw IPA string per word, kept for audit
        Only when `map_inventory` is set (see `_gold_map`):
            gold_ph, gold_phg  one entry per gold token; NOT the length of
                      `phonemes` when a phoneme backs off to several tokens
            gold_ph_count      per phoneme, how many gold tokens it became
                      (aligned with `phonemes`; sums to len(gold_ph))
            gold_unmapped
        """
        tag = self._tag(lang)
        return {'lang': lang or self.default_lang, 'g2p_lang': tag,
                **self.lang_flags(tag), **self.scope(),
                **self._phonemize_words_tag(words, tag)}

    def phonemize_sentence(self, text, lang=None):
        """Free text -> the same layered dict as `phonemize_words`. The text
        is split with `segment`, so unspaced scripts (th, km, my, ja, zh, yue)
        are word-segmented; see word_segmentation.py."""
        return self.phonemize_words(segment(text, self._tag(lang)), lang=lang)

    # -- file level --------------------------------------------------------

    # A clip that starts or ends with real silence should be labelled as such,
    # or the model learns to map silence onto whatever phoneme happens to be at
    # the edge. Thresholds carried over from the previous pipeline.
    TRIM_START_THRESHOLD = 0.25
    TRIM_END_THRESHOLD = 0.30

    # The layers that must stay index-aligned with `phonemes`. Everything
    # downstream assumes it, so adding a layer means adding it here AND at
    # both sites that build these arrays (phonemize_words, and the trim/SIL
    # insertion below). `gold_ph`/`gold_phg` are NOT in this set: mapping a
    # segment can expand it to several gold units, so they are not
    # index-aligned with `phonemes`. `gold_ph_count` is per phoneme but exists
    # only with map_inventory, so it is checked separately.
    PER_PHONEME_KEYS = ('phonemes', 'tone', 'stress', 'length', 'word_num')
    PER_WORD_KEYS = ('words', 'ipa')

    def phonemize_srt(self, srt_path, gs_out_path, lang=None,
                      sil_from_trim=True, audio_path=None):
        """Read a Whisper/stable-ts JSON and write the phoneme JSON beside it.

        `lang` (a single BCP 47 code, e.g. 'pt', 'pt-BR', 'zh-Hant') is the
        caller's language selection and takes priority when given. Otherwise
        the srt's own `lang` field is used if present (current writers), else
        the legacy `lang_iso`/`language` fields, which may hold a BCP 47 code
        or a raw CharsiuG2P tag from before this module's `lang` argument was
        BCP-47-only -- `_tag_from_field` accepts either, for reading old files
        only. `self.default_tag` is the last resort.

        Input (Whisper/stable-ts style; `words` optional, `trim` optional):
            {"segments": [{"start","end","text",
                           "words":[{"word","start","end","probability"}]}],
             "lang": "en", "audio_path": "..."}

        Output: one entry per segment, with the phoneme layers as parallel
        arrays. `phonemes` holds standardized IPA strings straight from
        CharsiuG2P (Swedish SAMPA etc. fixed up, but not mapped onto any
        fixed inventory yet). Every per-phoneme array has the same length as
        `phonemes`. When `map_inventory` is set, each segment is additionally
        mapped onto the gold inventory (see `_gold_map`); `gold_ph`/`gold_phg`
        are NOT index-aligned with `phonemes`, since a segment can expand to
        several gold units; `gold_ph_count` (per phoneme) says how many.

            {"audio_path", "lang", "g2p_lang",
             "tonal", "tone_layer", "needs_word_segmentation", "word_segmenter",
             "n_tones", "n_stresses", "n_lengths", "n_gold_ph", "n_gold_phg",
             "segments": [{"start","end","text","words",
                           "phonemes","tone","stress","length","word_num",
                           "gold_ph","gold_phg","gold_ph_count","gold_unmapped"}]}

        `lang` in the output is the BCP 47 code (the only thing the rest of
        the pipeline should ever read back); `g2p_lang` is CharsiuG2P's own
        tag, kept only for callers that need to talk to that library
        directly.

        Returns True if a file was written.
        """
        with open(srt_path, 'r', encoding='utf-8') as f:
            srt = json.load(f)

        if lang:
            tag = self._tag(lang)
        else:
            tag = (_tag_from_field(srt.get('lang'))
                   or _tag_from_field(srt.get('lang_iso'))
                   or _tag_from_field(srt.get('language'))
                   or self.default_tag)
        # Synthesised SRTs (MSWC builds one per clip from the directory name)
        # carry neither 'language' nor 'audio_path', so fall back to what the
        # caller knows rather than writing empty strings into every file.
        out = {'audio_path': audio_path or srt.get('audio_path', ''),
               'lang': lang or srt.get('lang') or srt.get('lang_iso') or srt.get('language') or '',
               'g2p_lang': tag,
               **self.lang_flags(tag),
               **self.scope(),
               'segments': []}

        unspaced = _lc.needs_word_segmentation(tag)
        for seg in srt.get('segments', []):
            # Prefer the word list, which segments better than raw text
            # (e.g. across punctuation Whisper already resolved).
            if seg.get('words') and not unspaced:
                raw_words = [w.get('word', '').strip() for w in seg['words']]
                words = [p for w in raw_words for p in split_words(w)]
            elif seg.get('words'):
                # Whisper's "words" in an unspaced script are tokenizer pieces
                # cut wherever the BPE vocabulary cuts. They can be part of a
                # word or run across two, so rebuild the text and segment that.
                # The raw pieces carry their own leading spaces, so joining them
                # unstripped keeps the phrase breaks. (Not yet checked against
                # real Whisper output in these scripts: the FLEURS srts have no
                # `words`.)
                words = segment(''.join(w.get('word', '') for w in seg['words']), tag)
            else:
                words = segment(seg.get('text', ''), tag)

            d = self._phonemize_words_tag(words, tag)

            # MSWC-style SRTs carry 'trim': [leading_silence, trailing_silence].
            # SIL comes from the module, not self._inv, which is None when
            # map_inventory is off. The gold arrays get SIL added directly
            # rather than remapping the whole segment, which would count
            # every unmapped segment in it twice in self.unmapped.
            trim = seg.get('trim')
            if sil_from_trim and trim and d['phonemes']:
                sil = phoneme_inventory_gold.SIL
                if self.map_inventory:
                    sil_ph = self._inv.TOKEN_INDEX[sil]
                    sil_phg = self._feat.GROUPS.get(sil, 0)
                if trim[0] > self.TRIM_START_THRESHOLD:
                    d['phonemes'].insert(0, sil)
                    d['tone'].insert(0, 0)
                    d['stress'].insert(0, 0)
                    d['length'].insert(0, 0)
                    d['word_num'].insert(0, d['word_num'][0])
                    if self.map_inventory:
                        d['gold_ph'].insert(0, sil_ph)
                        d['gold_phg'].insert(0, sil_phg)
                        d['gold_ph_count'].insert(0, 1)
                if len(trim) > 1 and trim[1] > self.TRIM_END_THRESHOLD:
                    d['phonemes'].append(sil)
                    d['tone'].append(0)
                    d['stress'].append(0)
                    d['length'].append(0)
                    d['word_num'].append(d['word_num'][-1])
                    if self.map_inventory:
                        d['gold_ph'].append(sil_ph)
                        d['gold_phg'].append(sil_phg)
                        d['gold_ph_count'].append(1)

            # Misalignment here is silent corruption that only shows up as bad
            # training targets much later, so refuse to write it out.
            n = len(d['phonemes'])
            off = {k: len(d[k]) for k in self.PER_PHONEME_KEYS if len(d[k]) != n}
            if off:
                raise ValueError(
                    'layer misalignment in %s (phonemes=%d, mismatched: %s)'
                    % (srt_path, n, off))
            if len(d['words']) != len(d['ipa']):
                raise ValueError('per-word misalignment in %s' % srt_path)
            if d['word_num'] and max(d['word_num']) >= len(d['words']):
                raise ValueError('word_num out of range in %s' % srt_path)
            if self.map_inventory and (
                    len(d['gold_ph_count']) != n
                    or sum(d['gold_ph_count']) != len(d['gold_ph'])
                    or len(d['gold_phg']) != len(d['gold_ph'])):
                raise ValueError('gold misalignment in %s' % srt_path)

            seg_out = {
                'start': seg.get('start'),
                'end': seg.get('end'),
                'text': seg.get('text', ''),
                'words': d['words'],
                'phonemes': d['phonemes'],
                'tone': d['tone'],
                'stress': d['stress'],
                'length': d['length'],
                'word_num': d['word_num'],
            }
            if self.map_inventory:
                seg_out['gold_ph'] = d['gold_ph']
                seg_out['gold_phg'] = d['gold_phg']
                seg_out['gold_ph_count'] = d['gold_ph_count']
                seg_out['gold_unmapped'] = d['gold_unmapped']
            out['segments'].append(seg_out)

        os.makedirs(os.path.dirname(os.path.abspath(gs_out_path)), exist_ok=True)
        with open(gs_out_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False)
        self.stats['files'] += 1
        return True

    # -- reporting ---------------------------------------------------------

    def print_stats(self):
        # misses == word types actually sent to the model; hits == occurrences
        # served without one. hits + misses == total word occurrences.
        hits, misses = self.stats['cache_hits'], self.stats['cache_misses']
        total = hits + misses
        rate = (100.0 * hits / total) if total else 0.0
        print('goldG2P: files=%d words=%d phonemes=%d'
              % (self.stats['files'], self.stats['words'], self.stats['phonemes']))
        ev = self.stats['cache_evictions']
        print('  cache: %d/%s types held | %d hits / %d model lookups '
              '(%.1f%% hit rate)%s'
              % (len(self.cache), self.cache_size or 'inf', hits, misses, rate,
                 ' | %d evicted' % ev if ev else ''))
        if self.unmapped:
            print('  unmapped segments (%d types): %s'
                  % (len(self.unmapped),
                     ' '.join('%s(%d)' % (s, n)
                              for s, n in self.unmapped.most_common(20))))
