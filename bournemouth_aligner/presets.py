"""
Language preset -> checkpoint mapping.

The names below are the ONE place a checkpoint filename appears. `core.py` never special-cases
a name, so renaming a checkpoint is an edit here and nowhere else.

## Status

The p3cupe checkpoints are **not published yet**. Until they are,
`core.PhonemeTimestampAligner.download_model` looks for the file on disk first (see
`MODEL_SEARCH_DIRS` there) and only then asks HuggingFace, so a local
`models/p3cupe/<name>` makes `preset=` work with no other setup.

    P3CUPE_EN   English, LibriSpeech 360 h
    P3CUPE_EU   MLS, 7 European languages, English EXCLUDED
    P3CUPE_UNIVERSAL   not trained yet - see below

`P3CUPE_EU` is the fallback for every non-English preset because it is the only multilingual
p3cupe checkpoint that exists. That is a placeholder, not a recommendation: it covers 7
European languages and nothing else, so a Hindi or Turkish preset currently resolves to a
model that never saw the language. Expect a universal checkpoint to take that role.
"""

# --- p3cupe checkpoints ---------------------------------------------------------------------
P3CUPE_EN = "m_pa07b_final_e5_libri360h.pt"
P3CUPE_EU = "m_pa07a_final_e4_MLS_7h.pt"

# TODO: a universal checkpoint (the MSWC-style coverage the preset table below promises) is not
# trained yet. Point this at it and update `get_preset`'s fallback when it exists.
P3CUPE_UNIVERSAL = None

DEFAULT_MODEL = P3CUPE_EN

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


def get_preset(preset="en-us", lang='en-us'):
    """
    Resolve a language preset to `(model_name, lang)`.

    `lang` is only overridden when it is still the default, so an explicit `lang=` always wins
    over the preset's language.
    """
    if preset and lang == 'en-us':
        lang = preset

    if preset in ENGLISH_LANGS or not preset:
        return P3CUPE_EN, lang

    if preset not in SUPPORTED_LANGS:
        print(f"Warning: '{preset}' is not in the ph66 mapper's supported language list. "
              f"Phonemization will still run through espeak-ng, but expect degraded "
              f"alignment - tonal languages and distant families are not covered.")

    # Everything non-English falls back to the only multilingual checkpoint there is. It was
    # trained on 7 European languages, so for anything further afield this is a placeholder
    # until a universal checkpoint exists.
    return P3CUPE_UNIVERSAL or P3CUPE_EU, lang
