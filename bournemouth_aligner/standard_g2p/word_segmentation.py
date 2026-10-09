"""
Text -> the words CharsiuG2P is fed.

The model is a WORD g2p: one word in, one IPA string out. For a space-delimited
script `split_words` is all that is needed. For a script that does not put
spaces between words (lang_codes.NEEDS_WORD_SEGMENTATION_ISO) it hands the
model whole clauses, and the output is IPA-shaped noise, so those languages go
through a segmenter instead. `segment(text, lang)` picks the right path;
everything that turns text into words should call it.

    lang       backend                                package
    th         newmm + TCC repair + ๆ expansion       pythainlp
    km         CRF segmenter + ៗ expansion            khmer-nltk
    my         syllable-based segmenter               pyidaungsu
    ja         UniDic morphemes, fed as katakana      fugashi, unidic-lite
    zh         longest match on dicts/zho-{s,t}.tsv   (none)
    yue        HKCanCor-trained segmenter             pycantonese
    nan, tts   none -- see has_segmenter()

Only the standard library is imported here, so a data loader can use
`split_words` without the segmenter packages. Each segmenter is imported the
first time its language is segmented.

Each language has exactly ONE backend. "Use whichever package happens to be
installed" would give the same corpus different labels on different machines.

The backends were chosen by running the model, not by counting how many
segmented words are keys of dicts/<tag>.tsv. That count rewards splitting
into the model's training words, and for Thai, Khmer and Burmese those splits
change the pronunciation: the whole word is read correctly, its pieces are
not (see each section below). The test sets were FLEURS dev for Thai, and a
dozen everyday and news sentences per language, written for this purpose,
for the rest.
"""

import os
import unicodedata

try:                                  # imported as part of the package
    from . import lang_codes as _lc
except ImportError:                   # imported with standard_g2p/ itself on sys.path
    import lang_codes as _lc

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DICTS_DIR = os.path.join(_REPO_ROOT, 'dicts')


# ---------------------------------------------------------------------------
# Space-delimited scripts
# ---------------------------------------------------------------------------

# Characters that join a word besides letters and digits: apostrophes and the
# internal hyphen, and four Burmese symbols that Unicode files as punctuation
# (Po) but that are words read aloud, each with its own dicts/bur.tsv entry:
# ၌ n̥aɪʔ (locative), ၍ jwḛ (conjunctive), ၎ ləɡáʊɴ, ၏ ʔḭ (genitive). ၏ is
# common in formal text (မြန်မာနိုင်ငံ၏ "of Myanmar"); treated as punctuation,
# it was silently deleted. They occur in no other script.
_WORD_EXTRA = frozenset("'’-၌၍၎၏")

# Combining marks are word characters. This is not a detail: Python's `\w`
# follows str.isalnum(), which is False for every category Mn/Mc mark, so a
# regex built on `\w` treats Brahmic vowel signs and viramas as SEPARATORS.
# The previous pattern here, r"[^\w'’̀-ͯ-]+", whitelisted only the
# Latin/Greek/Cyrillic combining block, so Tamil புஷ்ஷின் came apart into
# ['ப','ஷ','ஷ','ன'] -- four bare consonants with their vowels deleted, not one
# word. Sixteen FLEURS languages were affected (every Brahmic script plus Thai,
# Khmer, Lao and Burmese), at up to 11x the true token count.
#
# Expressing "any mark" as a character class needs 310 ranges, so this tests the
# Unicode category directly. It costs roughly 3x the regex (11us vs 4us per
# sentence), which is nothing next to the model call it feeds.


def _is_word_char(ch):
    return (ch.isalnum() or ch in _WORD_EXTRA
            or unicodedata.category(ch) in ('Mn', 'Mc', 'Me'))


def split_words(sentence):
    """Whitespace/punctuation word split. Not sufficient on its own for the
    languages in NEEDS_WORD_SEGMENTATION_ISO; use `segment` for those."""
    out, cur = [], []
    for ch in sentence.strip().lower():
        if _is_word_char(ch):
            cur.append(ch)
        elif cur:
            out.append(''.join(cur))
            cur = []
    if cur:
        out.append(''.join(cur))
    return out


# ---------------------------------------------------------------------------
# Shared repairs
# ---------------------------------------------------------------------------

def _expand_repeat(mark):
    """Thai ๆ and Khmer ៗ repeat the previous word: ต่าง ๆ is read ต่าง ต่าง.
    No dictionary entry contains either mark, so it is expanded rather than
    handed to the model (which reads it as a letter name)."""
    def post(words):
        out = []
        for w in words:
            n = w.count(mark)
            w = w.replace(mark, '')
            if w:
                out.append(w)
            # A leading mark has nothing to repeat and is dropped.
            if out:
                out.extend([out[-1]] * n)
        return out
    return post


# Khmer coeng and the Burmese virama join the next consonant into a stack, so
# a boundary right after one is inside a syllable.
_STACKERS = frozenset('្္')


def _merge_broken_syllables(tokens):
    """Glue a token that starts with a combining mark, or follows a stacker,
    onto the previous one. A segmenter that cuts there has cut a syllable in
    two, and the model reads the pieces as letter names."""
    out = []
    for t in tokens:
        if out and (unicodedata.category(t[0]) in ('Mn', 'Mc')
                    or out[-1][-1] in _STACKERS):
            out[-1] += t
        else:
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# Thai
#
# pythainlp's `newmm`: maximal matching over pythainlp's own ~62k-word
# dictionary, constrained to Thai Character Cluster (TCC) boundaries. Measured
# on the 439 FLEURS th_th dev sentences (0.2 ms per sentence after the first
# call, which loads the dictionary):
#
#   segmenter              tokens  keys of dicts/tha.tsv  boundaries inside a TCC
#   ICU BreakIterator       13244        91.2%                  36
#   newmm                   10636        78.6%                   9
#   newmm on tha.tsv keys   12407        94.2%                  21
#
# The coverage column is misleading. tha.tsv has only 13.6k entries, so newmm
# emits compounds the model never saw (ออกเสียง, หลีกเลี่ยง), and restricting
# it to tha.tsv splits them back into training words. We ran the model on the
# 448 compound types that the restriction splits. For 94 of them (21%) the
# parts do not concatenate to the whole, and the whole is the correct one:
#
#   พุทธศาสนา  whole pʰut̚.tʰa.saːt̚.sa.naː  parts pʰut̚ | saːt̚.sa.naː  (linking vowel lost)
#   บุช        whole but̚                   parts buʔ | t͡ɕʰɔː           (ช read as a letter name)
#
# ICU makes the same kind of cut and also cuts through syllables
# (เป็นก|ลุ่ม for เป็น|กลุ่ม). So newmm is used unrestricted, with two repairs:
#
# - A boundary inside a TCC is always wrong. newmm's nine such boundaries all
#   come before a thanthakhat (ไน|ต์, เซน|ไท|น์ in loanwords), and a lone ต์
#   comes back from the model as a pronounced letter. Such tokens are merged
#   back together.
# - ๆ (mai yamok) is expanded (see `_expand_repeat`). It was the most frequent
#   out-of-dictionary token (75).
# ---------------------------------------------------------------------------

def _load_thai(tag):
    from pythainlp.tokenize import word_tokenize
    from pythainlp.tokenize.tcc import tcc_pos

    def seg(chunk):
        toks = word_tokenize(chunk, engine='newmm', keep_whitespace=False)
        ok = tcc_pos(chunk)              # offsets at which a TCC ends
        out, pos = [], 0
        for t in toks:
            if out and pos not in ok:
                out[-1] += t
            else:
                out.append(t)
            pos += len(t)
        return out

    return seg


# ---------------------------------------------------------------------------
# Khmer
#
# khmer-nltk's CRF segmenter (0.4 ms per sentence). dicts/khm.tsv has only
# 3.3k entries, so 16% of its tokens were not keys. Splitting those into
# dictionary words is harmful, as for Thai. 12 of 14 such words changed, and
# the pieces were mostly single letters read as letter names:
#
#   រដ្ឋាភិបាល  whole roattʰaapʰiʔbaal  parts roat aa pʰɔɔ iʔ bɑɑ aa lɔɔ
#
# So its tokens are used whole. ៗ (lek too) is expanded, and a token starting
# inside a syllable is glued back (none did in the test set).
# ---------------------------------------------------------------------------

def _load_khmer(tag):
    from khmernltk import word_tokenize

    def seg(chunk):
        return _merge_broken_syllables(word_tokenize(chunk, return_tokens=True))

    return seg


# ---------------------------------------------------------------------------
# Burmese
#
# pyidaungsu's word tokenizer (0.8 ms per sentence). Its words are often longer
# than dicts/bur.tsv entries (4.6k), and for 10 of 16 such words the whole
# differs from its dictionary parts. The whole is right: Burmese voices the
# onset of a syllable joined to the one before it, and reduces a bound first
# syllable to ə. Both only happen inside the unit the model is given:
#
#   ထမင်းစား  whole tʰəmɪ́ɴzá   parts tʰəmɪ́ɴ sá     (voicing, s -> z)
#   ကောင်းတယ်  whole káʊɴdɛ̀     parts káʊɴ tɛ̀       (voicing, t -> d)
#   အသစ်       whole ʔəθɪʔ     parts ʔa̰ θɪʔ        (reduction)
#
# So its tokens are used whole. The suffix symbols ၏ ၍ ၌ come out as tokens of
# their own, and the model reads them alone as noise (၏ kʰɛ̝, ၍ kwḛ, ၌ la̰).
# They are joined to the preceding word, as in their dictionary entries
# (သွား၍, ကြိုဆိုပါ၏): သွား၍ gives the correct θwájwḛ, and နိုင်ငံ၏ gives
# nàɪɴŋàɴjɛ̰, with ၏ read as the colloquial genitive jɛ̰. ၎ begins a word and
# is left alone.
#
# Burmese remains in lang_codes.EXCLUDED_ISO: its tone is written with
# diacritics, so the tone layer stays empty.
# ---------------------------------------------------------------------------

_MY_SUFFIXES = frozenset('၏၍၌')


def _load_burmese(tag):
    from pyidaungsu import tokenize

    def seg(chunk):
        out = []
        for t in _merge_broken_syllables(tokenize(chunk, form='word')):
            if out and t in _MY_SUFFIXES:
                out[-1] += t
            else:
                out.append(t)
        return out

    return seg


# ---------------------------------------------------------------------------
# Japanese
#
# fugashi with unidic-lite (0.2 ms per sentence), always that dictionary: the
# full UniDic segments and reads differently, and fugashi would otherwise pick
# whichever is installed.
#
# The model is NOT given the surface form. It is given UniDic's `pron` field,
# the pronunciation in katakana, so `words` holds katakana for Japanese.
# Kanji readings depend on context, and the model sees one word at a time.
# UniDic chooses the reading from the whole sentence. In 12 test sentences
# (120 tokens) the model's IPA differed for 34 tokens:
#
#   は (topic particle)  surface ha        pron ɰᵝa     -- every は in the set
#   へ (particle)        surface he        pron e
#   明日                 surface akaçi     pron asɯ
#   昨日                 surface sakɯdʑitsɯ pron kinoː
#   雨                   surface ama       pron ame
#   東京                 surface toɯkjoɯ   pron toːkjoː  (spelling vs. long vowel)
#
# `kana` (the reading as spelled) is not used: it keeps ハ for the particle.
# The cost is that UniDic-lite reads 日本 as ニッポン (nipːoɴ), not the more
# common nihoɴ.
#
# UniDic cuts a geminate at the sokuon (持っ|て), and a token ending in ッ
# comes back with a literal ッ in the IPA. Such a token is joined to the next
# one: モッテ -> motːe. A token UniDic does not know (Latin, digits) has no
# `pron` and is passed through as written.
# ---------------------------------------------------------------------------

def _load_japanese(tag):
    import fugashi
    import unidic_lite
    tagger = fugashi.Tagger('-d "%s" -r "%s"' % (
        unidic_lite.DICDIR, os.path.join(unidic_lite.DICDIR, 'mecabrc')))

    def seg(chunk):
        out = []
        # fugashi reuses its node objects on the next parse, so read each
        # feature inside this loop and keep only strings.
        for node in tagger(chunk):
            pron = node.feature.pron
            if not pron or pron == '*':
                pron = node.surface
            if out and out[-1][-1:] in ('ッ', 'っ'):
                out[-1] += pron
            else:
                out.append(pron)
        return out

    return seg


# ---------------------------------------------------------------------------
# Cantonese
#
# pycantonese's segmenter, trained on HKCanCor (0.04 ms per sentence). 91% of
# its tokens are keys of dicts/yue.tsv. The rest (香港, 本書, 喺那裡等) gave the
# same IPA whole as split into dictionary words, 7 of 7, so they are left
# as they are.
# ---------------------------------------------------------------------------

def _load_cantonese(tag):
    import pycantonese

    def seg(chunk):
        return pycantonese.segment(chunk)

    return seg


# ---------------------------------------------------------------------------
# Mandarin
#
# Longest match over the model's own training dictionary, dicts/zho-s.tsv or
# dicts/zho-t.tsv according to the tag. No package is needed. Per-character
# input, the other package-free option, loses readings that depend on the
# word (12 test sentences):
#
#   音乐会  longest match ɪn˥˥ɥœ˥˩xweɪ˥˩   per character ɪn˥˥ lɤ˥˩ xweɪ˥˩  (乐 read as in 快乐)
#   我们    longest match uɔ˨˩˦mən˧         per character uɔ˨˩˦ mən˧˥     (neutral tone lost)
#
# The segmentation is the one with the fewest words, with a character that
# starts no entry as a word of its own. That fails where a phrase happens to
# spell a longer entry across a real boundary, where a statistical segmenter (jieba) could do better,
# but every word it emits is one the model was trained on. Runs of non-Han
# characters (5G, iPhone) are kept whole.
# Neither option fixes 了, whose dictionary reading is liǎo, not the particle le.
# ---------------------------------------------------------------------------

_MAX_ENTRY = 8          # longer dictionary keys (a few idioms) are never matched


def _is_han(ch):
    return 'CJK' in unicodedata.name(ch, '')


def _longest_match(chunk, keys):
    """Fewest-word segmentation of `chunk` into `keys`, O(len * _MAX_ENTRY)."""
    n = len(chunk)
    best = [None] * (n + 1)          # best[j] = (words, start of last word)
    best[0] = (0, 0)
    for i in range(n):
        if best[i] is None:
            continue
        for j in range(i + 1, min(n, i + _MAX_ENTRY) + 1):
            if j == i + 1 or chunk[i:j] in keys:
                cand = best[i][0] + 1
                if best[j] is None or cand < best[j][0]:
                    best[j] = (cand, i)
    out, j = [], n
    while j > 0:
        i = best[j][1]
        out.append(chunk[i:j])
        j = i
    return out[::-1]


def _load_dict_keys(tag):
    path = os.path.join(DICTS_DIR, '%s.tsv' % tag)
    with open(path, encoding='utf-8') as f:
        return frozenset(line.split('\t', 1)[0] for line in f)


def _load_mandarin(tag):
    keys = _load_dict_keys(tag)

    def seg(chunk):
        out, run = [], ''
        for ch in chunk:
            if run and _is_han(ch) != _is_han(run[-1]):
                out.extend(_longest_match(run, keys) if _is_han(run[-1]) else [run])
                run = ''
            run += ch
        if run:
            out.extend(_longest_match(run, keys) if _is_han(run[-1]) else [run])
        return out

    return seg


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

# ISO 639-3 -> (pip packages, loader). The loader takes the CharsiuG2P tag
# (Mandarin needs it to pick zho-s or zho-t), imports what it needs, and
# returns a function that segments one space- and punctuation-free chunk.
_BACKENDS = {
    'tha': ('pythainlp', _load_thai),
    'khm': ('khmer-nltk', _load_khmer),
    'mya': ('pyidaungsu', _load_burmese),
    'jpn': ('fugashi unidic-lite', _load_japanese),
    'yue': ('pycantonese', _load_cantonese),
    'zho': (None, _load_mandarin),
}
# Applied to the whole word list after segmentation.
_POSTPROCESS = {
    'tha': _expand_repeat('ๆ'),
    'khm': _expand_repeat('ៗ'),
}
_loaded = {}            # tag -> chunk segmenter


def _to_tag(lang):
    return lang if lang in _lc.TAG_TO_ISO else _lc.bcp47_to_tag(lang)


def has_segmenter(lang):
    """True if `segment` gives real words for `lang` (a BCP 47 code or a
    CharsiuG2P tag): the script is space-delimited, or a backend is
    registered. The backend's package may still be missing; call
    `require_segmenter` to find out."""
    iso = _lc.normalize_iso(lang)
    return iso not in _lc.NEEDS_WORD_SEGMENTATION_ISO or iso in _BACKENDS


def segmenter_name(lang):
    """Which segmenter `segment` uses for `lang`, with package versions, or
    None for a space-delimited language (and for one with no backend, which
    falls back to split_words). Written into every output, because the words
    and so the labels of these languages depend on the segmenter's own
    dictionary and change with its version."""
    iso = _lc.normalize_iso(lang)
    if iso not in _lc.NEEDS_WORD_SEGMENTATION_ISO or iso not in _BACKENDS:
        return None
    packages, _ = _BACKENDS[iso]
    if packages is None:                              # Mandarin
        return 'longest match on dicts/%s.tsv' % _to_tag(lang)
    from importlib import metadata
    out = []
    for p in packages.split():
        try:
            out.append('%s %s' % (p, metadata.version(p)))
        except metadata.PackageNotFoundError:
            out.append('%s (not installed)' % p)
    return ', '.join(out)


def require_segmenter(lang):
    """Import `lang`'s segmenter now, so a missing package fails once, before
    a run, not once per clip. No-op for a space-delimited language."""
    iso = _lc.normalize_iso(lang)
    if iso not in _lc.NEEDS_WORD_SEGMENTATION_ISO:
        return
    tag = _to_tag(lang)
    if tag in _loaded:
        return
    if iso not in _BACKENDS:
        raise NotImplementedError(
            'no word segmenter for %r (ISO %r); see has_segmenter()' % (lang, iso))
    packages, loader = _BACKENDS[iso]       # packages is None for Mandarin
    try:
        _loaded[tag] = loader(tag)
    except ImportError as ex:
        raise ImportError('word segmentation for %r needs %s: pip install %s'
                          % (lang, packages, packages)) from ex


def segment(text, lang):
    """Free text -> the list of words to phonemize.

    `lang` is a BCP 47 code or a CharsiuG2P tag. Space-delimited languages get
    exactly `split_words(text)`. A language that needs segmentation but has no
    backend also gets `split_words`, i.e. whole clauses, which is the old
    behaviour; gate on `has_segmenter` to avoid that. For Japanese the words
    are katakana readings, not the surface text (see above).
    """
    iso = _lc.normalize_iso(lang)
    if iso not in _lc.NEEDS_WORD_SEGMENTATION_ISO or iso not in _BACKENDS:
        return split_words(text)
    tag = _to_tag(lang)
    require_segmenter(tag)
    seg = _loaded[tag]
    # split_words first, so punctuation and phrase spaces are handled the same
    # way as everywhere else; the backend only sees unbroken runs of script.
    words = [p for chunk in split_words(text)
             for w in seg(chunk) for p in split_words(w)]
    post = _POSTPROCESS.get(iso)
    return post(words) if post else words
