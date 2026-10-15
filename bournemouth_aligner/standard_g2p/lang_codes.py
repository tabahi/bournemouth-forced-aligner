"""
BCP 47 language code <-> CharsiuG2P tag.

CharsiuG2P names its 100 languages with a mix of ISO 639-2/B spellings ('ger'
not 'deu', 'dut' not 'nld', 'cze' not 'ces') and its own region/era suffixes
('eng-us', 'por-bz', 'lat-clas'). Everything OUTSIDE this module -- dataset
loaders, g2p_task.py, the .gs.json 'lang' field -- speaks a single, real BCP 47
code instead: ISO 639-1 where a language has one ('en', 'pt', 'zh'), ISO 639-3
otherwise ('ckb', 'hbs'), optionally followed by a region ('pt-BR'), script
('zh-Hant') or, where no registered subtag fits, a short informal label
('vi-c'). This module is the join between that and CharsiuG2P's own tag
spelling, so a loader never needs to know the model calls German 'ger' or
Brazilian Portuguese 'por-bz'.

It deliberately has no heavy imports: dataset loaders import it to label their
clips, and none of them should pull in transformers to do that. `gold_g2p`
re-exports everything here, so `from standard_g2p.gold_g2p import bcp47_to_tag`
works as well as importing from this module directly.

The two directions are not symmetric:

    tag_to_bcp47('eng-us')   -> 'en'        bare code for the default tag
    tag_to_bcp47('eng-uk')   -> 'en-GB'     explicit code for a non-default one
    bcp47_to_tag('en')       -> 'eng-us'    one of two tags, chosen by ISO_DEFAULT_TAG

Fourteen ISO codes have more than one tag (regional and historical variants), so
the reverse direction needs a stated preference rather than whichever entry a
dict happened to see first.

Nothing here falls back to a default. `bcp47_to_tag` raises, because the failure
is silent otherwise: byT5 is byte-level, so an invented tag like '<lav>: vards'
still decodes to something IPA-shaped -- it is just not Latvian, and nothing
downstream can tell the difference.
"""


class UnsupportedLanguageError(ValueError):
    """A language could not be mapped onto a CharsiuG2P tag."""


# The 100 tags charsiu/g2p_multilingual_byT5_small_100 was trained on, each with
# its ISO 639-3 code. This is the authoritative spelling of the tag set;
# gold_g2p.SUPPORTED_LANGS is checked against it at import.
TAG_TO_ISO = {
    'ady': 'ady',       # Adyghe
    'afr': 'afr',       # Afrikaans
    'amh': 'amh',       # Amharic
    'ang': 'ang',       # Old English
    'ara': 'ara',       # Arabic
    'arg': 'arg',       # Aragonese
    'arm-e': 'hye',     # Armenian, Eastern
    'arm-w': 'hye',     # Armenian, Western
    'aze': 'aze',       # Azerbaijani
    'bak': 'bak',       # Bashkir
    'bel': 'bel',       # Belarusian
    'bos': 'bos',       # Bosnian
    'bul': 'bul',       # Bulgarian
    'bur': 'mya',       # Burmese
    'cat': 'cat',       # Catalan
    'cze': 'ces',       # Czech
    'dan': 'dan',       # Danish
    'dut': 'nld',       # Dutch
    'egy': 'egy',       # Egyptian (ancient) -- NOT Hebrew, despite the look of the code
    'eng-uk': 'eng',    # English, UK
    'eng-us': 'eng',    # English, US
    'enm': 'enm',       # Middle English
    'epo': 'epo',       # Esperanto
    'est': 'est',       # Estonian
    'eus': 'eus',       # Basque
    'fas': 'fas',       # Persian
    'fin': 'fin',       # Finnish
    'fra': 'fra',       # French
    'fra-qu': 'fra',    # French, Quebec
    'geo': 'kat',       # Georgian
    'ger': 'deu',       # German
    'gle': 'gle',       # Irish
    'glg': 'glg',       # Galician
    'grc': 'grc',       # Greek, Ancient
    'gre': 'ell',       # Greek, Modern
    'hbs-cyrl': 'hbs',  # Serbo-Croatian, Cyrillic
    'hbs-latn': 'hbs',  # Serbo-Croatian, Latin
    'hin': 'hin',       # Hindi
    'hun': 'hun',       # Hungarian
    'ice': 'isl',       # Icelandic (639-2/B spelling)
    'ido': 'ido',       # Ido
    'ina': 'ina',       # Interlingua
    'ind': 'ind',       # Indonesian
    'isl': 'isl',       # Icelandic (639-3 spelling; duplicate of 'ice')
    'ita': 'ita',       # Italian
    'jpn': 'jpn',       # Japanese
    'kaz': 'kaz',       # Kazakh
    'khm': 'khm',       # Khmer
    'kor': 'kor',       # Korean
    'kur': 'kur',       # Kurdish
    'lat-clas': 'lat',  # Latin, Classical
    'lat-eccl': 'lat',  # Latin, Ecclesiastical
    'lit': 'lit',       # Lithuanian
    'ltz': 'ltz',       # Luxembourgish
    'mac': 'mkd',       # Macedonian
    'mlt': 'mlt',       # Maltese
    'mri': 'mri',       # Maori
    'msa': 'msa',       # Malay
    'nan': 'nan',       # Chinese, Min Nan
    'nob': 'nob',       # Norwegian Bokmal
    'ori': 'ori',       # Odia
    'pap': 'pap',       # Papiamento
    'pol': 'pol',       # Polish
    'por-bz': 'por',    # Portuguese, Brazil
    'por-po': 'por',    # Portuguese, Portugal
    'ron': 'ron',       # Romanian
    'rus': 'rus',       # Russian
    'san': 'san',       # Sanskrit
    'slk': 'slk',       # Slovak
    'slo': 'slk',       # Slovak (639-2/B spelling; duplicate of 'slk')
    'slv': 'slv',       # Slovenian
    'sme': 'sme',       # Sami, Northern
    'snd': 'snd',       # Sindhi
    'spa': 'spa',       # Spanish
    'spa-latin': 'spa',  # Spanish, Latin America
    'spa-me': 'spa',    # Spanish, Mexico
    'sqi': 'sqi',       # Albanian
    'srp': 'srp',       # Serbian
    'swa': 'swa',       # Swahili
    'swe': 'swe',       # Swedish
    'syc': 'syc',       # Syriac, Classical
    'tam': 'tam',       # Tamil
    'tat': 'tat',       # Tatar
    'tgl': 'tgl',       # Tagalog
    'tha': 'tha',       # Thai
    'tts': 'tts',       # Isan / Thai, Northeastern
    'tuk': 'tuk',       # Turkmen
    'tur': 'tur',       # Turkish
    'uig': 'uig',       # Uyghur
    'ukr': 'ukr',       # Ukrainian
    'urd': 'urd',       # Urdu
    'uzb': 'uzb',       # Uzbek
    'vie-c': 'vie',     # Vietnamese, Central
    'vie-n': 'vie',     # Vietnamese, Northern
    'vie-s': 'vie',     # Vietnamese, Southern
    'wel-nw': 'cym',    # Welsh, North
    'wel-sw': 'cym',    # Welsh, South
    'yue': 'yue',       # Chinese, Cantonese
    'zho-s': 'zho',     # Chinese, simplified script
    'zho-t': 'zho',     # Chinese, traditional script
}

SUPPORTED_TAGS = frozenset(TAG_TO_ISO)


# The tag to use for an ISO code that several tags claim. Without this the
# reverse mapping would be decided by dict ordering, and 'por' would resolve to
# Brazilian or European Portuguese depending on where a line was inserted.
ISO_DEFAULT_TAG = {
    'cym': 'wel-nw',    # North Welsh over South
    'eng': 'eng-us',
    'fra': 'fra',       # France over Quebec
    'hbs': 'hbs-latn',  # Latin script over Cyrillic
    'hye': 'arm-e',     # Eastern Armenian over Western
    'isl': 'ice',       # both spellings are trained tags
    'lat': 'lat-clas',
    'por': 'por-po',    # European Portuguese; use 'pt-BR' for Brazil
    'slk': 'slk',
    'spa': 'spa',       # Castilian; use 'es-419' or 'es-MX' explicitly
    'vie': 'vie-n',     # Northern/Hanoi is the standard dialect
    'zho': 'zho-s',     # simplified script
}

# ISO codes with no identically-named tag, bridged to the tag that covers them.
# These are the cases where the model's language inventory is coarser than
# ISO 639-3: a macrolanguage tag standing in for one of its members, or one
# standardised variety standing in for another.
ISO_BRIDGE = {
    'cmn': 'zho-s',     # Mandarin -> Chinese, simplified script
    'ckb': 'kur',       # Sorani -> Kurdish macrolanguage
    'fil': 'tgl',       # Filipino is standardised Tagalog
    'hrv': 'hbs-latn',  # Croatian -> Serbo-Croatian, Latin
    'nno': 'nob',       # Nynorsk -> Bokmal (the only Norwegian the model has)
    'nor': 'nob',       # Norwegian macrolanguage -> Bokmal
    'swh': 'swa',       # Coastal Swahili -> Swahili macrolanguage
    'zsm': 'msa',       # Standard Malay -> Malay macrolanguage
}

ISO_TO_TAG = {}
for _tag, _iso in TAG_TO_ISO.items():
    ISO_TO_TAG.setdefault(_iso, _tag)
ISO_TO_TAG.update(ISO_DEFAULT_TAG)
ISO_TO_TAG.update(ISO_BRIDGE)

SUPPORTED_ISO = frozenset(ISO_TO_TAG)


def tag_to_iso(tag):
    """CharsiuG2P tag -> ISO 639-3. Raises on a tag the model does not have."""
    try:
        return TAG_TO_ISO[tag]
    except KeyError:
        raise UnsupportedLanguageError(
            f'{tag!r} is not one of the {len(TAG_TO_ISO)} CharsiuG2P tags') from None


# ---------------------------------------------------------------------------
# ISO 639-1
#
# Strict BCP 47 prefers the 2-letter ISO 639-1 subtag over the 3-letter
# ISO 639-3 one whenever a language has one ('en', not 'eng'); 639-3 is only
# the fallback for languages 639-1 never covered (Sorani Kurdish 'ckb',
# Cebuano 'ceb', Mandarin specifically as opposed to the 'zh' macrolanguage,
# ...). This table covers every language this module names -- CharsiuG2P's
# and the wider set the FLEURS loader labels -- that HAS a 639-1 code; a
# language with no entry here keeps its 639-3 spelling as the external code.
# ---------------------------------------------------------------------------

ISO_639_1 = {
    'afr': 'af', 'amh': 'am', 'ara': 'ar', 'asm': 'as', 'aze': 'az', 'bak': 'ba',
    'bel': 'be', 'bul': 'bg', 'ben': 'bn', 'bos': 'bs',
    'cat': 'ca', 'ces': 'cs', 'cym': 'cy', 'dan': 'da', 'deu': 'de', 'ell': 'el',
    'eng': 'en', 'epo': 'eo', 'spa': 'es', 'est': 'et',
    'eus': 'eu', 'fas': 'fa', 'ful': 'ff', 'fin': 'fi', 'fra': 'fr', 'gle': 'ga',
    'glg': 'gl', 'guj': 'gu', 'hau': 'ha', 'heb': 'he',
    'hin': 'hi', 'hrv': 'hr', 'hun': 'hu', 'hye': 'hy', 'ina': 'ia', 'ind': 'id',
    'ibo': 'ig', 'ido': 'io', 'isl': 'is', 'ita': 'it',
    'jpn': 'ja', 'jav': 'jv', 'kat': 'ka', 'kaz': 'kk', 'khm': 'km', 'kan': 'kn',
    'kor': 'ko', 'kur': 'ku', 'kir': 'ky', 'lat': 'la',
    'ltz': 'lb', 'lug': 'lg', 'lin': 'ln', 'lao': 'lo', 'lit': 'lt', 'lav': 'lv', 'mri': 'mi',
    'mkd': 'mk', 'mal': 'ml', 'mon': 'mn', 'mar': 'mr',
    'msa': 'ms', 'mlt': 'mt', 'mya': 'my', 'nob': 'nb', 'nep': 'ne', 'nld': 'nl',
    'nno': 'nn', 'nor': 'no', 'nya': 'ny', 'oci': 'oc',
    'orm': 'om', 'ori': 'or', 'pan': 'pa', 'pol': 'pl', 'pus': 'ps', 'por': 'pt',
    'ron': 'ro', 'rus': 'ru', 'san': 'sa', 'snd': 'sd',
    'sme': 'se', 'slk': 'sk', 'slv': 'sl', 'sna': 'sn', 'som': 'so', 'sqi': 'sq',
    'srp': 'sr', 'swe': 'sv', 'swa': 'sw', 'tam': 'ta',
    'tel': 'te', 'tgk': 'tg', 'tha': 'th', 'tuk': 'tk', 'tgl': 'tl', 'tur': 'tr',
    'tat': 'tt', 'uig': 'ug', 'ukr': 'uk', 'urd': 'ur',
    'uzb': 'uz', 'vie': 'vi', 'wol': 'wo', 'xho': 'xh', 'yor': 'yo', 'zho': 'zh',
    'zul': 'zu',
}
# Deliberately absent (639-3 stays the external code): ady, ang, arg, ast,
# ceb, ckb, cmn, egy, enm, fil, grc, hbs, kam, kea, luo, nan, nso, pap, syc,
# tts, umb, yue -- and the bridge sources 'swh'/'zsm', whose macrolanguages
# 'swa'/'msa' already have 'sw'/'ms' above.

# Reverse, for parsing a bare code back to the internal ISO 639-3: a caller
# may type either the 639-1 code ('en') or the 639-3 one ('eng') and get the
# same answer.
ISO3_OF_PRIMARY = {iso1: iso3 for iso3, iso1 in ISO_639_1.items()}


# ---------------------------------------------------------------------------
# BCP 47 codes
#
# A handful of languages cover more than one CharsiuG2P tag: regional (por,
# spa, eng), script (hbs, zho), or historical/dialectal (lat, hye, vie, cym,
# fra) variants of the same language. `bcp47_to_tag` picks the default in
# ISO_DEFAULT_TAG for a bare code ('pt'); to reach any other tag, suffix the
# code BCP 47-style with a region ('pt-BR'), script ('zh-Hant') or, where no
# registered subtag applies, a short informal label ('vi-c'). The outside
# world always passes ONE string -- never a CharsiuG2P tag, never a
# (code, variant) pair -- and the language part is 639-1 where one exists
# ('en-GB'), 639-3 otherwise ('hbs-Cyrl').
# ---------------------------------------------------------------------------

# Non-default codes. The bare code alone (see ISO_DEFAULT_TAG) always reaches
# the default tag; these are the extra spellings needed for anything else.
BCP47_TO_TAG = {
    'en-GB': 'eng-uk',
    'fr-CA': 'fra-qu',        # Quebec French; bare 'fr' is France
    'hbs-Cyrl': 'hbs-cyrl',   # 'hbs' has no ISO 639-1 code
    'hy-west': 'arm-w',       # Western Armenian
    'la-eccl': 'lat-eccl',    # Ecclesiastical Latin
    'pt-BR': 'por-bz',        # Brazilian Portuguese
    'es-419': 'spa-latin',    # Latin American Spanish
    'es-MX': 'spa-me',        # Mexican Spanish
    'vi-c': 'vie-c',          # Central Vietnamese
    'vi-s': 'vie-s',          # Southern Vietnamese
    'cy-sw': 'wel-sw',        # South Welsh
    'zh-Hant': 'zho-t',       # Traditional script
}

# Explicit spellings of a language's DEFAULT tag. Accepted on lookup, so
# 'en-US' works as well as bare 'en' -- but never produced by tag_to_bcp47,
# which prefers the bare code for a default tag.
BCP47_DEFAULT_ALIASES = {
    'en-US': 'eng-us',
    'hbs-Latn': 'hbs-latn',
    'hy-east': 'arm-e',
    'la-clas': 'lat-clas',
    'pt-PT': 'por-po',
    'vi-n': 'vie-n',
    'cy-nw': 'wel-nw',
    'zh-Hans': 'zho-s',
}

_BCP47_LOOKUP = {k.lower(): v for k, v in {**BCP47_TO_TAG, **BCP47_DEFAULT_ALIASES}.items()}

# Reverse: tag -> canonical BCP 47 code. A default tag is not a key here --
# tag_to_bcp47 falls back to the bare code for it.
TAG_TO_BCP47 = {tag: code for code, tag in BCP47_TO_TAG.items()}

# Language subtags that have at least one non-default suffix, for error
# messages.
BCP47_SUFFIXES = {}
for _code in BCP47_TO_TAG:
    _lang, _suffix = _code.split('-', 1)
    BCP47_SUFFIXES.setdefault(_lang.lower(), set()).add(_suffix)
del _code, _lang, _suffix


def bcp47_to_tag(lang):
    """BCP 47 code ('pt', 'pt-BR', 'zh-Hant', ...) -> CharsiuG2P tag.

    The language subtag is ISO 639-1 where one exists ('en', 'pt', 'zh'),
    else ISO 639-3 ('ckb', 'hbs'); an optional suffix -- a region ('BR'),
    script ('Hant') or informal dialect label ('c') -- picks a non-default
    tag (see BCP47_TO_TAG for the vocabulary). A bare code resolves to its
    entry in ISO_DEFAULT_TAG (Castilian Spanish, Continental Portuguese, US
    English, ...). Raises if unsupported.
    """
    if not lang:
        raise UnsupportedLanguageError('no language code given')
    lang = lang.strip()
    key = lang.lower()
    if '-' in key:
        tag = _BCP47_LOOKUP.get(key)
        if tag is None:
            base = key.split('-', 1)[0]
            known = sorted(BCP47_SUFFIXES.get(base, ()))
            raise UnsupportedLanguageError(
                f'{lang!r} is not a known BCP 47 code. Known suffixes for '
                f'{base!r}: {known or "none"}')
        return tag
    iso3 = ISO3_OF_PRIMARY.get(key, key)
    tag = ISO_TO_TAG.get(iso3)
    if tag is None:
        raise UnsupportedLanguageError(
            f'{lang!r} has no CharsiuG2P tag: the model was not trained on '
            f'this language, and an invented tag would return IPA for some '
            f'other language rather than failing.')
    return tag


def tag_to_bcp47(tag):
    """CharsiuG2P tag -> canonical BCP 47 code: the bare code (639-1 where
    the language has one, else 639-3) for the default tag of its language,
    else the code that names it explicitly."""
    iso3 = tag_to_iso(tag)
    primary = ISO_639_1.get(iso3, iso3)
    return TAG_TO_BCP47.get(tag, primary)


def lang_is_supported(lang):
    """True if `lang` (a BCP 47 code) maps onto a trained tag."""
    try:
        bcp47_to_tag(lang)
        return True
    except UnsupportedLanguageError:
        return False


# ---------------------------------------------------------------------------
# Tone
#
# Measured over the CharsiuG2P training dictionaries themselves
# (CharsiuG2P/dicts/<tag>.tsv), not taken from a typology reference, because
# what matters downstream is whether the model *emits* tone for a language --
# not whether linguists call it tonal.
# ---------------------------------------------------------------------------

# Tone written with modifier tone letters (U+02E5..U+02E9, e.g. saː˩˩˦) or, for
# Min Nan, superscript Chao digits (kʰuan²¹⁻⁵³). 100% of the entries in each of
# these dictionaries carry a mark, and `decompose_ipa` routes them to the `tone`
# layer, so for these languages the tone head actually fires.
TONE_LETTER_ISO = frozenset({
    'nan',   # Chinese, Min Nan   (44,588 entries, Chao digits)
    'tha',   # Thai               (13,608)
    'vie',   # Vietnamese         (70,901)
    'yue',   # Chinese, Cantonese (56,189)
    'zho',   # Chinese, Mandarin  (93,018 across zho-s + zho-t)
})

# Tonal, but the corpus writes the tone as a combining diacritic on the vowel
# (Burmese ka̰kṵθàɴpʰəjá: creaky, low, high). 83.9% of the 4,631 'bur' entries
# carry one. By default `decompose_ipa` treats these as vowel quality and keeps
# them in the segment, so the tone layer stays empty and the inventory gains
# 'à'/'á'-style classes instead. Pass tone_diacritics=True to route them to the
# tone layer -- and note that doing so also reinterprets the languages in
# PITCH_ACCENT_ISO below, which is usually not what you want.
TONE_DIACRITIC_ISO = frozenset({
    'mya',   # Burmese
})

TONAL_ISO = TONE_LETTER_ISO | TONE_DIACRITIC_ISO

# NOT tone. These mark pitch accent or stress with the same acute/grave/
# circumflex/caron diacritics, which is why they are listed explicitly: a naive
# "has a combining accent, must be tonal" rule sweeps them up.
#   hbs 99.4%, slv 99.1%  -- rising/falling pitch accent
#   san 30.8%, grc 27.3%  -- Vedic / Ancient Greek pitch accent
#   kur 62.9%             -- stress, not pitch at all
PITCH_ACCENT_ISO = frozenset({'grc', 'hbs', 'san', 'slv'})


# ---------------------------------------------------------------------------
# Word segmentation
#
# The model is a WORD g2p: it takes one word and returns one IPA string. A
# script that does not delimit words with spaces therefore has to be segmented
# upstream, or every "word" handed to the model is a whole clause and the
# output is IPA-shaped noise. See the README for the options.
# ---------------------------------------------------------------------------

NEEDS_WORD_SEGMENTATION_ISO = frozenset({
    'jpn',   # Japanese -- no spaces at all
    'khm',   # Khmer    -- spaces separate phrases, not words
    'mya',   # Burmese  -- spaces separate phrases, not words
    'nan',   # Min Nan  -- Han script
    'tha',   # Thai     -- spaces separate phrases, not words
    'tts',   # Isan     -- Thai script; see the caveat below
    'yue',   # Cantonese
    'zho',   # Chinese
})

# Separately worth knowing: the 'tts' (Isan) dictionary is not IPA at all. Its
# entries are a romanisation ('กกไม้' -> 'gok.mai'), so the model reproduces
# that at inference. Segmenting Isan correctly will still not get you IPA.
BROKEN_DICT_ISO = frozenset({'tts'})


def normalize_iso(lang):
    """Accept a CharsiuG2P tag or a BCP 47 code (639-1 or 639-3, bare or
    suffixed), return the ISO 639-3 code of the tag that covers it.

    Use this before testing membership of any set in this module. The sets are
    keyed by the ISO 639-3 code of the tag that covers a language, so Mandarin
    is 'zho' in them even though 'cmn' is what the FLEURS loader emits and
    'zh' is the BCP 47 code a caller would normally pass:

        'cmn' in LANG_GROUP_MEMBERS['cjk']              # False - wrong
        normalize_iso('zh') in LANG_GROUP_MEMBERS['cjk']   # True

    The lang_group() / family() / is_tonal() helpers all normalise for
    you; only direct set membership needs this.

    Bridged codes normalise to the ISO 639-3 code of the tag that covers them,
    so 'cmn' and 'zh' both answer 'zho', and 'hr' answers 'hbs'. Without this
    a dataset that labels Mandarin 'cmn' would be reported as neither tonal
    nor in need of segmentation. A BCP 47 suffix ('pt-BR') is dropped before
    bridging, since it never changes which family/language group/tone answer
    applies.
    """
    if not lang:
        return ''
    lang = lang.strip()
    if lang in TAG_TO_ISO:
        return TAG_TO_ISO[lang]
    base = lang.lower().split('-', 1)[0]
    iso3 = ISO3_OF_PRIMARY.get(base, base)
    if iso3 in ISO_BRIDGE:
        return TAG_TO_ISO[ISO_BRIDGE[iso3]]
    return iso3


def is_tonal(lang, diacritic_tone=True):
    """True if `lang` (tag or ISO 639-3) has lexical tone.

    diacritic_tone=False restricts the answer to the languages whose tone
    actually reaches the `tone` layer under the default settings, i.e. it
    excludes Burmese.
    """
    iso = normalize_iso(lang)
    return iso in (TONAL_ISO if diacritic_tone else TONE_LETTER_ISO)


def needs_word_segmentation(lang):
    """True if `lang` (tag or ISO 639-3) must be word-segmented before g2p."""
    return normalize_iso(lang) in NEEDS_WORD_SEGMENTATION_ISO


# ---------------------------------------------------------------------------
# Language groups
#
# Languages are sorted into language groups by script, not by language
# family. Small scripts that need no special processing are merged by kind
# (other_alphabetic, abjad, brahmic); scripts that need their own processing
# (word segmentation) keep a language group of their own. One exception to
# script: Vietnamese is written in Latin script but is tonal, and would be the
# only language in 'latin' whose tone layer is filled, so it has a language
# group of its own ('vietnamese'). It added only 3 tokens to latin's list
# (ɤ̆ ŋ͡m k͡p); the reason is tone, not the phoneme inventory. Each language
# group has its own token list (lang_group_inventory), and downstream tasks are expected to
# use those local tokens. "Language group" is never shortened to "group" here:
# the phoneme groups of phoneme_features (gold_phg) are the other meaning.
#
# Grouping by family was measured and rejected. `scripts/measure_lang_groups.py`
# extracts every language's phoneme inventory from the CharsiuG2P dictionaries
# and clusters them; the numbers are in mappings/lang_stats.md. In short: phoneme
# inventories do not recover families, measured in the 270-token gold space
# the local tokens are drawn from. Jaccard over which phonemes a language
# uses puts 67 of 86 languages in one cluster at k=8, because the shared IPA core
# dominates. Jensen-Shannon over how often each phoneme is used splits more
# evenly but no more meaningfully -- at k=8 its largest cluster is `ar arg el
# eo es et eu gl grc hbs ia io is it ja ku mi mt pap ro se sk sv sw tts`, none
# of its 8 clusters is a single family, and a language's nearest neighbour
# shares its sub-family for only 34 of 86. The two most coherent clusters are
# East Asian (tone and script) and mostly-Turkic (7 of 10).
#
# Writing system, by contrast, separates cleanly and predicts what the text
# processing has to branch on (word segmentation). Membership below is
# generated from the measured dominant script of each dictionary's keys, except
# for 'vietnamese' (see above).
# ---------------------------------------------------------------------------

LANG_GROUPS = {
    'latin': 'Latin script, space-delimited, no tone. The default path.',
    'vietnamese': 'Latin script, space-delimited, tonal (tone letters). Kept out '
                  'of latin so that latin holds no tonal language.',
    'cyrillic': 'Cyrillic script, space-delimited, no tone.',
    'other_alphabetic': 'Greek/Armenian/Georgian/Hangul/Ethiopic. Space-delimited '
                        'alphabets and abugidas that need no special handling.',
    'abjad': 'Arabic script and Syriac. Orthography underspecifies vowels, so '
             'the grapheme sequence carries less than the phoneme sequence.',
    'brahmic': 'Indic abugidas. Inherent vowel and conjunct consonants, so '
               'character count and phoneme count diverge.',
    'cjk': 'Han script. Needs word segmentation AND carries lexical tone.',
    'thai_khmer': 'Thai and Khmer script. Needs word segmentation; Thai is tonal.',
    'japanese': 'Mixed kanji/kana. Needs word segmentation; no tone.',
    'burmese': 'Myanmar script. Needs word segmentation; tone written as '
               'diacritics, so it does not reach the tone layer.',
}

LANG_GROUP_MEMBERS = {
    'latin': {
        'afr', 'ang', 'arg', 'aze', 'bos', 'cat', 'ces', 'cym',
        'dan', 'deu', 'egy', 'eng', 'enm', 'epo', 'est', 'eus',
        'fin', 'fra', 'gle', 'glg', 'hun', 'ido', 'ina', 'ind',
        'isl', 'ita', 'lat', 'lit', 'ltz', 'mlt', 'mri', 'msa',
        'nld', 'nob', 'pap', 'pol', 'por', 'ron', 'slk', 'slv',
        'sme', 'spa', 'sqi', 'swa', 'swe', 'tgl', 'tuk', 'tur',
        'uzb',
    },
    'vietnamese': {
        'vie',
    },
    'cyrillic': {
        'ady', 'bak', 'bel', 'bul', 'hbs', 'kaz', 'mkd', 'rus',
        'srp', 'tat', 'ukr',
    },
    'other_alphabetic': {
        'amh', 'ell', 'grc', 'hye', 'kat', 'kor',
    },
    'abjad': {
        'ara', 'fas', 'kur', 'snd', 'syc', 'uig', 'urd',
    },
    'brahmic': {
        'hin', 'ori', 'san', 'tam',
    },
    'cjk': {
        'nan', 'yue', 'zho',
    },
    'thai_khmer': {
        'khm', 'tha', 'tts',
    },
    'japanese': {
        'jpn',
    },
    'burmese': {
        'mya',
    },
}
# 'hbs' lands in cyrillic because Serbo-Croatian has a tag in each script
# (hbs-cyrl, hbs-latn) and the ISO row takes the majority. Either path works
# for it -- it is the one language here where script is a property of the tag
# rather than of the language.

ISO_TO_LANG_GROUP = {iso: g for g, members in LANG_GROUP_MEMBERS.items()
                     for iso in members}


# Languages left out for now: they did not shape their language group's token
# list. This is deliberately ORTHOGONAL to the language group: a language keeps
# its language group, and comes back into use by being removed from here once
# its blocker is fixed, without the language groups changing. See the README.
EXCLUDED_ISO = {
    'khm': 'left out for now to focus on the major languages (it is '
           'word-segmented, with khmer-nltk)',
    'mya': 'left out for now to focus on the major languages; also, its tone '
           'is written as diacritics, so the tone layer stays empty (it is '
           'word-segmented, with pyidaungsu)',
    'nan': 'no maintained segmenter, and 70% of its entries are two-character '
           'words carrying tone sandhi that per-character input cannot produce',
    'tts': 'the dictionary is a romanisation, not IPA, so the model returns '
           'romanisation at inference; segmentation will not fix it',
}


def lang_group(lang):
    """CharsiuG2P tag or ISO 639-3 -> language group name. Raises if unknown."""
    iso = normalize_iso(lang)
    try:
        return ISO_TO_LANG_GROUP[iso]
    except KeyError:
        raise UnsupportedLanguageError(
            f'{lang!r} (ISO {iso!r}) is not in any language group; it is not '
            f'one of the {len(ISO_TO_LANG_GROUP)} languages CharsiuG2P covers') from None


def is_excluded(lang):
    """True if `lang` is currently on the exclusion list (see EXCLUDED_ISO)."""
    return normalize_iso(lang) in EXCLUDED_ISO


def lang_group_members(lang_group, include_excluded=False):
    """The ISO codes in a language group, excluded ones dropped by default."""
    try:
        members = LANG_GROUP_MEMBERS[lang_group]
    except KeyError:
        raise UnsupportedLanguageError(
            f'{lang_group!r} is not a language group. Known: {sorted(LANG_GROUPS)}') from None
    if include_excluded:
        return sorted(members)
    return sorted(m for m in members if m not in EXCLUDED_ISO)


# ---------------------------------------------------------------------------
# Language family
#
# Genetic classification, kept deliberately separate from the language groups
# above: it describes what a language IS, not how the pipeline handles it, and
# the two disagree constantly (Vietnamese is Austroasiatic but processed on the
# Latin path; Maltese is Semitic but written in Latin script; Afrikaans is
# Germanic but spoken in South Africa).
#
# This is metadata for corpus balance and per-family error reporting. It is NOT
# a proxy for phonological similarity -- see the clustering result above.
# ---------------------------------------------------------------------------

FAMILY_ISO = {
    # Indo-European
    **{i: 'indo-european:germanic' for i in
       ('afr', 'ang', 'dan', 'deu', 'eng', 'enm', 'isl', 'ltz', 'nld', 'nob', 'swe')},
    **{i: 'indo-european:romance' for i in
       ('arg', 'cat', 'fra', 'glg', 'ita', 'lat', 'por', 'ron', 'spa')},
    **{i: 'indo-european:slavic' for i in
       ('bel', 'bos', 'bul', 'ces', 'hbs', 'mkd', 'pol', 'rus', 'slk', 'slv', 'srp', 'ukr')},
    **{i: 'indo-european:indo-iranian' for i in
       ('fas', 'hin', 'kur', 'ori', 'san', 'snd', 'urd')},
    **{i: 'indo-european:hellenic' for i in ('ell', 'grc')},
    **{i: 'indo-european:celtic' for i in ('cym', 'gle')},
    'lit': 'indo-european:baltic',
    'hye': 'indo-european:armenian',
    'sqi': 'indo-european:albanian',
    # Afro-Asiatic
    **{i: 'afro-asiatic:semitic' for i in ('amh', 'ara', 'mlt', 'syc')},
    'egy': 'afro-asiatic:egyptian',
    # Other families
    **{i: 'turkic' for i in ('aze', 'bak', 'kaz', 'tat', 'tuk', 'tur', 'uig', 'uzb')},
    **{i: 'uralic' for i in ('est', 'fin', 'hun', 'sme')},
    **{i: 'sino-tibetan' for i in ('mya', 'nan', 'yue', 'zho')},
    **{i: 'austronesian' for i in ('ind', 'mri', 'msa', 'tgl')},
    **{i: 'austroasiatic' for i in ('khm', 'vie')},
    **{i: 'tai-kadai' for i in ('tha', 'tts')},
    **{i: 'constructed' for i in ('epo', 'ido', 'ina')},
    'ady': 'northwest-caucasian',
    'eus': 'isolate',
    'jpn': 'japonic',
    'kat': 'kartvelian',
    'kor': 'koreanic',
    'pap': 'creole:iberian',
    'swa': 'niger-congo:bantu',
    'tam': 'dravidian',
}


def family(lang):
    """CharsiuG2P tag or ISO 639-3 -> genetic family. Raises if unknown."""
    iso = normalize_iso(lang)
    try:
        return FAMILY_ISO[iso]
    except KeyError:
        raise UnsupportedLanguageError(
            f'{lang!r} (ISO {iso!r}) has no family entry') from None


# Every language has to land in exactly one language group and one family, or
# a caller that iterates the language groups silently processes fewer languages than it thinks.
_covered = set(ISO_TO_LANG_GROUP)
_expected = set(TAG_TO_ISO.values())
if _covered != _expected:
    raise UnsupportedLanguageError(
        'language groups do not cover the tag set: missing %s, extra %s'
        % (sorted(_expected - _covered), sorted(_covered - _expected)))
if set(FAMILY_ISO) != _expected:
    raise UnsupportedLanguageError(
        'FAMILY_ISO does not cover the tag set: missing %s, extra %s'
        % (sorted(_expected - set(FAMILY_ISO)), sorted(set(FAMILY_ISO) - _expected)))
_dupes = [i for i in _expected
          if sum(i in m for m in LANG_GROUP_MEMBERS.values()) != 1]
if _dupes:
    raise UnsupportedLanguageError('languages in more than one language group: %s' % sorted(_dupes))
if set(EXCLUDED_ISO) - _expected:
    raise UnsupportedLanguageError(
        'EXCLUDED_ISO names languages that do not exist: %s'
        % sorted(set(EXCLUDED_ISO) - _expected))
del _covered, _expected, _dupes
