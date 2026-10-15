"""
Phoneme set + language -> model.

BFA runs one of two stacks, chosen by `phoneme_set`:

    "std"   (default)  standard_g2p phonemes  ->  a p4mbfa model, ONE PER LANGUAGE GROUP
    "ph66"             espeak-ng + ph66       ->  a p3cupe model: English, or English + 7

The pairing is fixed: a p4mbfa head emits one language group's token list and a p3cupe head
emits ph66, so neither can decode the other's phoneme sequences. The names below are the ONE
place a model name appears. `core.py` never special-cases a name, so renaming a model is an
edit here and nowhere else.

Every name is a Hugging Face repository of the `Tabahi` account (`hub.resolve_model`
downloads its config.json + model.safetensors on first use), published from the models
repository by `p4mbfa/export_hf.py`; `p4mbfa/hf_release.yaml` there records which training
run each one is and why it was chosen.

### std / p4mbfa

A language belongs to exactly one language group (by script; `standard_g2p.lang_codes`), and a
group with no entry in `P4MBFA_MODELS` cannot be aligned with the std set yet -
`get_std_preset` raises and names `phoneme_set="ph66"` as the way through.

    latin     mbfa-latin      FLEURS, 36 languages, trained at the most noise (0.05)
    cyrillic  mbfa-cyrillic   FLEURS: be bg hr kk mk ru sr uk
    cjk       mbfa-cjk        FLEURS: cmn yue
    japanese  mbfa-japanese   FLEURS: ja

`mbfa-latin-benchmark` is a second latin model, selectable by name only: the best TIMIT score
of every latin run, trained at less noise (0.02). TIMIT is clean studio speech and favours it;
`mbfa-latin` is the one meant for real recordings and so the one the presets use.

### ph66 / p3cupe

    CUPE-3-en        English, LibriSpeech 500 h           every English ph66 preset
    CUPE-3-mls7      + German, Spanish, French, Italian,  every other ph66 preset
                     Dutch, Polish, Portuguese (MLS, 100 h each)
    CUPE-3-en-base   English, LibriSpeech 500 h           by name only

`CUPE-3-mls7` is `CUPE-3-en-base` trained one more epoch on the seven MLS languages, so it is
the only p3cupe model that has heard anything but English - including the ph66 phonemes
English never uses. That makes it the model for every non-English ph66 preset: espeak-ng will
phonemize any language it knows into ph66, and for a language outside those seven
`get_preset` says the model was not trained on it. It is not the English choice: the MLS
epoch cost about two points of TIMIT f1@25 (76.0 -> 74.2).

`CUPE-3-en` and `CUPE-3-en-base` are two English models from the same lineage that score
within noise of each other on TIMIT (75.6 and 76.0 f1@25 on the held-out half).
`CUPE-3-en` is the preset; the other is selectable by name, like `mbfa-latin-benchmark`.
"""

from .standard_g2p import lang_codes as _lc

PHONEME_SETS = ("std", "ph66")
DEFAULT_PHONEME_SET = "std"
# The model family each phoneme set is decoded by, and back.
FAMILY_OF_SET = {"std": "p4mbfa", "ph66": "p3cupe"}
SET_OF_FAMILY = {family: name for name, family in FAMILY_OF_SET.items()}

DEFAULT_LANG = "en-us"

# --- p4mbfa models, by language group -------------------------------------------------------
P4MBFA_MODELS = {
    "latin": "mbfa-latin",
    "cyrillic": "mbfa-cyrillic",
    "cjk": "mbfa-cjk",
    "japanese": "mbfa-japanese",
    # TODO: vietnamese, other_alphabetic, abjad, brahmic and thai_khmer have training configs
    # in bfa_models/p4mbfa but no model yet. `burmese` cannot be trained until standard_g2p
    # gives it an inventory (it holds only the four special tokens).
}
# The latin model that scores best on TIMIT. Not a preset - pass it as `model_name`.
MBFA_LATIN_BENCHMARK = "mbfa-latin-benchmark"

# --- p3cupe models --------------------------------------------------------------------------
P3CUPE_EN = "CUPE-3-en"
# Seven MLS languages on top of English: the ph66 model for everything that is not English.
P3CUPE_MLS7 = "CUPE-3-mls7"
# The root of the English lineage. Not a preset - pass it as `model_name`.
P3CUPE_EN_BASE = "CUPE-3-en-base"

# Every published model, with the phoneme set it decodes.
MODELS = {**{name: "std" for name in (*P4MBFA_MODELS.values(), MBFA_LATIN_BENCHMARK)},
          **{name: "ph66" for name in (P3CUPE_EN, P3CUPE_MLS7, P3CUPE_EN_BASE)}}

# espeak-ng language codes the ph66 mapper covers. Tonal languages (cmn, yue, vi, th) and
# distant families (ja, ko, Bantu) are out of scope for this mapper, not merely untrained.
ENGLISH_LANGS = {
    "en-us", "en", "en-gb", "en-029", "en-gb-x-gbclan", "en-gb-x-rp", "en-gb-scotland",
    "en-gb-x-gbcwmd",
}

SUPPORTED_LANGS = ENGLISH_LANGS | {
    # Germanic / Romance / Slavic and their neighbours
    "de", "nl", "af", "lb", "da", "sv", "nb", "is",
    "fr", "fr-be", "fr-ch", "es", "es-419", "it", "pt", "pt-br", "ca", "an", "ro", "la",
    "pl", "cs", "sk", "sl", "hr", "bs", "sr", "mk", "bg", "ru", "ru-lv", "uk", "be",
    # Celtic, Baltic, Uralic
    "ga", "gd", "cy", "lv", "lt", "et", "hu", "fi",
    # Indo-Iranian, Hellenic, Armenian, Albanian
    "hi", "bn", "ur", "pa", "gu", "mr", "ne", "as", "or", "si", "kok", "sd",
    "fa", "fa-latn", "ku", "el", "hy", "hyw", "sq",
    # Turkic, Semitic, Dravidian, Kartvelian, isolates, constructed
    "tr", "az", "kk", "ky", "uz", "tt", "tk", "ba",
    "ar", "he", "am", "mt", "ta", "te", "kn", "ml", "ka", "eu", "qu",
    "id", "ms", "eo", "ia", "io", "lfn",
}

# The languages `P3CUPE_MLS7` was trained on besides English (base language of the code).
MLS7_LANGS = {"de", "es", "fr", "it", "nl", "pl", "pt"}

# espeak-ng and locale spellings that are not BCP 47 codes standard_g2p knows, so one `lang`
# works in both phoneme sets. Everything else is resolved by `lang_codes.bcp47_to_tag`, which
# is case-insensitive, so 'en-us', 'pt-br' and 'en-gb' need no entry here.
_STD_LANG_ALIASES = {
    "en-029": "en", "en-gb-x-rp": "en-GB", "en-gb-scotland": "en-GB",
    "en-gb-x-gbclan": "en-GB", "en-gb-x-gbcwmd": "en-GB",
    "fr-fr": "fr", "fr-be": "fr", "fr-ch": "fr", "ru-lv": "ru",
    "hyw": "hy-west", "an": "arg", "zh-cn": "zh", "zh-tw": "zh-Hant", "sv-se": "sv",
    "ga-ie": "ga",
}


class UnsupportedLanguageError(ValueError):
    """The requested language cannot be aligned with the requested phoneme set."""


def resolve_phoneme_set(phoneme_set=None):
    """The phoneme set to use: `phoneme_set` checked against `PHONEME_SETS`, None = the default."""
    if phoneme_set is None:
        phoneme_set = DEFAULT_PHONEME_SET
    if phoneme_set not in PHONEME_SETS:
        raise ValueError(f"unknown phoneme_set {phoneme_set!r}; expected one of {PHONEME_SETS}")
    return phoneme_set


def effective_lang(preset, lang):
    """
    The language a `(preset, lang)` pair means.

    `lang` is only overridden while it is still the default, so an explicit `lang=` always wins
    over the preset's language.
    """
    if preset and (not lang or lang == DEFAULT_LANG):
        return preset
    return lang or DEFAULT_LANG


def normalize_std_lang(lang):
    """
    Any language code BFA accepts -> `(canonical BCP 47 code, note)` for the std phoneme set.

    Accepts BCP 47 in any case ('en-us', 'pt-br', 'zh-Hant'), ISO 639-3 ('cmn', 'deu'), and
    the espeak-ng spellings the ph66 set takes ('fr-be', 'hyw'). `note` is None, or a
    sentence saying a regional variant was replaced by the nearest one the G2P has.

    Raises:
        UnsupportedLanguageError: the G2P was not trained on this language.
    """
    asked = (lang or "").strip()
    code, note = _STD_LANG_ALIASES.get(asked.lower(), asked), None
    try:
        tag = _lc.bcp47_to_tag(code)
    except _lc.UnsupportedLanguageError:
        base = code.split("-", 1)[0]
        try:
            tag = _lc.bcp47_to_tag(base) if base != code else None
        except _lc.UnsupportedLanguageError:
            tag = None
        if tag is None:
            raise UnsupportedLanguageError(
                f"'{asked}' is not covered by the 'std' phoneme set: the G2P model was not "
                f"trained on it. Use phoneme_set='ph66' (espeak-ng + p3cupe) for this "
                f"language.") from None
        note = (f"'{asked}' has no variant of its own in the std phoneme set; using "
                f"'{_lc.tag_to_bcp47(tag)}'")
    return _lc.tag_to_bcp47(tag), note


def std_lang_group(lang):
    """Canonical std language code -> its language group ('latin', 'cjk', ...)."""
    return _lc.lang_group(lang)


def get_std_preset(lang):
    """
    Resolve a language to `(model_name, canonical_lang, lang_group, note)` for the std set.

    Raises:
        UnsupportedLanguageError: the language is outside standard_g2p, is on its exclusion
            list, or its language group has no p4mbfa model yet.
    """
    code, note = normalize_std_lang(lang)
    group = std_lang_group(code)
    if _lc.is_excluded(code):
        raise UnsupportedLanguageError(
            f"'{lang}' is excluded from the 'std' phoneme set for now "
            f"({_lc.EXCLUDED_ISO[_lc.normalize_iso(code)]}). Use phoneme_set='ph66' if "
            f"espeak-ng covers it.")
    model_name = P4MBFA_MODELS.get(group)
    if model_name is None:
        raise UnsupportedLanguageError(
            f"'{lang}' belongs to the '{group}' language group, which has no p4mbfa "
            f"model yet (available: {', '.join(sorted(P4MBFA_MODELS))}). Pass "
            f"ckpt_path=... if you have one, or use phoneme_set='ph66' (espeak-ng + p3cupe).")
    return model_name, code, group, note


def std_languages():
    """`{lang_group: [BCP 47 codes]}` for the groups that have a model."""
    out = {}
    for group in P4MBFA_MODELS:
        tags = [t for t, iso in _lc.TAG_TO_ISO.items()
                if _lc.ISO_TO_LANG_GROUP.get(iso) == group and iso not in _lc.EXCLUDED_ISO]
        out[group] = sorted({_lc.tag_to_bcp47(t) for t in tags})
    return out


def ph66_model(lang):
    """The p3cupe model for an espeak-ng language code: English or everything else."""
    return P3CUPE_EN if (not lang or lang.lower() in ENGLISH_LANGS) else P3CUPE_MLS7


def get_preset(preset="en-us", lang='en-us'):
    """
    Resolve a ph66 language preset to `(model_name, lang, note)`.

    `lang` is only overridden when it is still the default, so an explicit `lang=` always wins
    over the preset's language. English presets get `P3CUPE_EN`; every other preset gets
    `P3CUPE_MLS7`. `note` is None, or a warning for the caller to show: the language is not
    one of the seven that model was trained on, or not one the ph66 mapper covers at all.
    """
    if preset and lang == 'en-us':
        lang = preset

    model_name, note = ph66_model(preset), None
    if model_name == P3CUPE_MLS7 and preset.lower().split("-", 1)[0] not in MLS7_LANGS:
        note = (f"the ph66 model ({P3CUPE_MLS7}) was not trained on '{preset}' (it knows "
                f"English and {', '.join(sorted(MLS7_LANGS))}); its phonemes are aligned by "
                f"their nearest sounds in those. The default phoneme set (phoneme_set='std') "
                f"has models trained on more languages.")
        if preset not in SUPPORTED_LANGS:
            note += (f" '{preset}' is also outside the ph66 mapper's language list: espeak-ng "
                     f"will still phonemize it, but tonal languages and distant families are "
                     f"not covered.")
    return model_name, lang, note
