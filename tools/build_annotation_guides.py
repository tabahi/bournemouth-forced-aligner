"""
Build the annotator guides for the web interface: what each phoneme token and each tone means.

    python tools/build_annotation_guides.py --fleurs DIR [--download] [--device cuda]

Two steps:

1. collect: every sentence of the FLEURS dev split (text only) of each trained language is run
   through the same G2P the aligner uses. Per token this counts how often each language uses it,
   which IPA the G2P wrote before it landed on the token (back-off), and the most frequent words
   that contain it. Per tone, the same for the tonal languages. Saved to --cache (json), so the
   second step can be rerun without the G2P.
2. render: the hand-written descriptions in this file + those counts ->
       docs/phoneme-guides/<group>.md           web_interface/static/guides/phonemes-<group>.json
       docs/phoneme-guides/tones.md             web_interface/static/guides/tones.json
       web_interface/static/guides/index.json   (which guides exist)

--download fetches the FLEURS dev .tsv files (about 250 KB each) from the Hugging Face Hub into
--fleurs first.
"""

import argparse
import collections
import csv
import json
import os
import re
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

csv.field_size_limit(1 << 24)

# Language group -> {BFA language code: FLEURS config}: the trained languages that FLEURS has.
GROUP_FLEURS = {
    "latin": {
        "af": "af_za", "az": "az_az", "bs": "bs_ba", "ca": "ca_es", "cs": "cs_cz", "cy": "cy_gb",
        "da": "da_dk", "de": "de_de", "en": "en_us", "es-419": "es_419", "et": "et_ee",
        "fi": "fi_fi", "fr": "fr_fr", "ga": "ga_ie", "gl": "gl_es", "hu": "hu_hu", "id": "id_id",
        "is": "is_is", "it": "it_it", "lb": "lb_lu", "lt": "lt_lt", "mi": "mi_nz", "ms": "ms_my",
        "mt": "mt_mt", "nb": "nb_no", "nl": "nl_nl", "pl": "pl_pl", "pt-BR": "pt_br",
        "ro": "ro_ro", "sk": "sk_sk", "sl": "sl_si", "sv": "sv_se", "sw": "sw_ke", "tl": "fil_ph",
        "tr": "tr_tr", "uz": "uz_uz",
    },
    "cyrillic": {"be": "be_by", "bg": "bg_bg", "hbs": "hr_hr", "kk": "kk_kz", "mk": "mk_mk",
                 "ru": "ru_ru", "sr": "sr_rs", "uk": "uk_ua"},
    "cjk": {"zh": "cmn_hans_cn", "yue": "yue_hant_hk"},
    "japanese": {"ja": "ja_jp"},
    "vietnamese": {"vi": "vi_vn"},
    "thai_khmer": {"th": "th_th"},
    "abjad": {"ar": "ar_eg", "ku": "ckb_iq", "fa": "fa_ir", "sd": "sd_in", "ur": "ur_pk"},
    "brahmic": {"hi": "hi_in", "or": "or_in", "ta": "ta_in"},
    "other_alphabetic": {"am": "am_et", "el": "el_gr", "hy": "hy_am", "ka": "ka_ge", "ko": "ko_kr"},
}
TONAL = {"zh", "yue", "vi", "th"}
UNSPACED = {"zh", "yue", "th", "ja"}   # read the raw transcript; the normalised one is cut per character

GROUP_MODELS = {
    "latin": "mbfa-latin", "cyrillic": "mbfa-cyrillic", "cjk": "mbfa-cjk", "japanese": "mbfa-japanese",
    "vietnamese": "mbfa-vietnamese", "thai_khmer": "mbfa-thai", "abjad": "mbfa-abjad",
    "brahmic": "mbfa-brahmic", "other_alphabetic": "mbfa-other-alphabetic",
}
GROUP_TITLES = {
    "latin": "latin group", "cyrillic": "cyrillic group", "cjk": "Chinese (cjk group)",
    "japanese": "Japanese", "vietnamese": "Vietnamese", "thai_khmer": "Thai",
    "abjad": "abjad group (Arabic script)", "brahmic": "brahmic group (Indic scripts)",
    "other_alphabetic": "other alphabets (Greek, Armenian, Georgian, Hangul, Ethiopic)",
}

LANG_NAMES = {
    "af": "Afrikaans", "az": "Azerbaijani", "bs": "Bosnian", "ca": "Catalan", "cs": "Czech",
    "cy": "Welsh", "da": "Danish", "de": "German", "en": "English", "es-419": "Spanish",
    "et": "Estonian", "fi": "Finnish", "fr": "French", "ga": "Irish", "gl": "Galician",
    "hu": "Hungarian", "id": "Indonesian", "is": "Icelandic", "it": "Italian",
    "lb": "Luxembourgish", "lt": "Lithuanian", "mi": "Māori", "ms": "Malay", "mt": "Maltese",
    "nb": "Norwegian", "nl": "Dutch", "pl": "Polish", "pt-BR": "Portuguese", "ro": "Romanian",
    "sk": "Slovak", "sl": "Slovenian", "sv": "Swedish", "sw": "Swahili", "tl": "Tagalog",
    "tr": "Turkish", "uz": "Uzbek", "zh": "Mandarin", "yue": "Cantonese", "vi": "Vietnamese",
    "th": "Thai", "be": "Belarusian", "bg": "Bulgarian", "hbs": "Croatian", "kk": "Kazakh",
    "mk": "Macedonian", "ru": "Russian", "sr": "Serbian", "uk": "Ukrainian", "ja": "Japanese",
    "ar": "Arabic", "ku": "Kurdish (Sorani)", "fa": "Persian", "sd": "Sindhi", "ur": "Urdu",
    "hi": "Hindi", "or": "Odia", "ta": "Tamil", "am": "Amharic", "el": "Greek", "hy": "Armenian",
    "ka": "Georgian", "ko": "Korean",
}

HUB = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/{}/dev.tsv"
DOCS = os.path.join(ROOT, "docs", "phoneme-guides")
GUIDES = os.path.join(ROOT, "web_interface", "static", "guides")
N_EXAMPLES = 4          # example words kept per token and language
SKIP = ("SIL", "noise", "<unk>", "<blank>")


# ---------------------------------------------------------------------------------------------
# 1. collect
# ---------------------------------------------------------------------------------------------

def all_languages():
    return [(g, lang, code) for g, langs in GROUP_FLEURS.items() for lang, code in langs.items()]


def download(fleurs_dir):
    os.makedirs(fleurs_dir, exist_ok=True)
    for _, _, code in all_languages():
        path = os.path.join(fleurs_dir, code + ".tsv")
        if not os.path.exists(path):
            print("downloading", code)
            urllib.request.urlretrieve(HUB.format(code), path)


def read_sentences(path, raw):
    out = []
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) < 4:
                continue
            text = row[2] if raw else row[3]
            # Digits are not expanded by the G2P: drop the words that hold them.
            text = " ".join(w for w in text.split() if not re.search(r"\d", w))
            if text.strip():
                out.append(text)
    return out


def top_words(table):
    # The most frequent words; among equally frequent ones, the shorter.
    ranked = sorted(table.items(), key=lambda kv: (-kv[1][0], len(kv[1][1]), kv[0]))
    return [[w, c, seq] for w, (c, seq) in ranked[:N_EXAMPLES]]


def collect(fleurs_dir, device):
    from bournemouth_aligner.phonemizers.std import StdPhonemizer
    from bournemouth_aligner.standard_g2p.gold_g2p import TONE_VOCAB

    groups = {}
    tones, tone_words = {}, {}           # lang -> {tone: count}, {tone: {word: [count, pairs]}}
    ph = None
    for group, lang, code in all_languages():
        G = groups.setdefault(group, {"lang_tokens": collections.Counter(), "tokens": {},
                                      "eipa": {}, "words": {}})
        sents = read_sentences(os.path.join(fleurs_dir, code + ".tsv"), raw=lang in UNSPACED)
        if ph is None:
            ph = StdPhonemizer(lang, device=device, verbose=0, punctuation_sil=False)
        else:
            ph.set_backend(lang)
        if ph.lang_group != group:
            raise ValueError(f"{lang} is in group {ph.lang_group}, not {group}")
        ph.prefetch(sents)
        inv = ph.inventory
        print(f"{lang:7s} {group:16s} {len(sents)} sentences", flush=True)
        for text in sents:
            d = ph.phonemize_sentence(text)
            by_word = collections.defaultdict(list)
            for i, wi in enumerate(d["word_num"]):
                by_word[wi].append(i)
            for wi, idx in by_word.items():
                word = d["words"][wi]
                seq = [inv.id_to_label[d["local_ph"][i]] for i in idx]
                for i, tok in zip(idx, seq):
                    if tok in SKIP:
                        continue
                    G["lang_tokens"][lang] += 1
                    G["tokens"].setdefault(tok, collections.Counter())[lang] += 1
                    G["eipa"].setdefault(tok, collections.Counter())[d["eipa"][i]] += 1
                    w = G["words"].setdefault(tok, {}).setdefault(lang, {})
                    w.setdefault(word, [0, seq])[0] += 1
                if lang in TONAL:
                    pairs = [[tok, TONE_VOCAB[d["tone"][i]]] for i, tok in zip(idx, seq)]
                    for i in idx:
                        t = TONE_VOCAB[d["tone"][i]]
                        if not t:
                            continue
                        tones.setdefault(lang, collections.Counter())[t] += 1
                        w = tone_words.setdefault(lang, {}).setdefault(t, {})
                        w.setdefault(word, [0, pairs])[0] += 1

    return {
        "source": "FLEURS dev transcripts, through bournemouth_aligner's std G2P",
        "groups": {g: {
            "lang_tokens": dict(G["lang_tokens"]),
            "tokens": {t: dict(c) for t, c in G["tokens"].items()},
            "eipa": {t: dict(c) for t, c in G["eipa"].items()},
            "words": {t: {l: top_words(w) for l, w in per.items()} for t, per in G["words"].items()},
        } for g, G in groups.items()},
        "tones": {l: dict(c) for l, c in tones.items()},
        "tone_words": {l: {t: top_words(w) for t, w in per.items()} for l, per in tone_words.items()},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fleurs", help="directory of FLEURS dev .tsv files, named <config>.tsv")
    ap.add_argument("--download", action="store_true", help="fetch the .tsv files into --fleurs first")
    ap.add_argument("--cache", default=os.path.join(ROOT, "tmp", "annotation_guides_stats.json"))
    ap.add_argument("--recollect", action="store_true", help="run the G2P again even if --cache exists")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.recollect or not os.path.exists(args.cache):
        if not args.fleurs:
            ap.error("--fleurs is needed to collect the counts")
        if args.download:
            download(args.fleurs)
        stats = collect(args.fleurs, args.device)
        os.makedirs(os.path.dirname(args.cache), exist_ok=True)
        with open(args.cache, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False)
    with open(args.cache, encoding="utf-8") as f:
        stats = json.load(f)
    render(stats)


# ---------------------------------------------------------------------------------------------
# 2. descriptions (hand-written)
# ---------------------------------------------------------------------------------------------

# Base symbol -> (class, IPA name, what it sounds like). A token is a base symbol plus diacritics.
V, DIPH, STOP, AFFR, FRIC, NAS, LAT, RHO, GLIDE = (
    "vowel", "diphthong", "stop", "affricate", "fricative", "nasal", "lateral", "rhotic", "glide")
BASE = {
    # vowels, close to open
    "i": (V, "close front unrounded vowel", "ee as in “see”"),
    "y": (V, "close front rounded vowel", "ee said with rounded lips (French “tu”, German “über”)"),
    "ɨ": (V, "close central unrounded vowel", "between ee and oo, lips spread (Polish “y”, Welsh “u”)"),
    "ʉ": (V, "close central rounded vowel", "a rounded vowel between ee and oo (Norwegian and Swedish “du”)"),
    "ɯ": (V, "close back unrounded vowel", "oo said with spread lips (Turkish “ı”)"),
    "u": (V, "close back rounded vowel", "oo as in “food”"),
    "ɪ": (V, "near-close front unrounded vowel", "i as in “sit”"),
    "ʏ": (V, "near-close front rounded vowel", "i as in “sit” said with rounded lips (German “hübsch”)"),
    "ʊ": (V, "near-close back rounded vowel", "oo as in “book”"),
    "e": (V, "close-mid front unrounded vowel", "e as in Spanish or Italian “de”; the start of English “say”"),
    "ø": (V, "close-mid front rounded vowel", "e said with rounded lips (German “schön”, French “peu”)"),
    "ɘ": (V, "close-mid central unrounded vowel", "a schwa-like vowel, a little higher than ə"),
    "ɵ": (V, "close-mid central rounded vowel", "a rounded vowel between e and o (Swedish short u)"),
    "ɤ": (V, "close-mid back unrounded vowel", "o said with spread lips (Estonian “õ”)"),
    "o": (V, "close-mid back rounded vowel", "o as in Spanish or Italian “no”"),
    "ə": (V, "mid central vowel (schwa)", "the weak vowel of “about”, “sofa”"),
    "ɛ": (V, "open-mid front unrounded vowel", "e as in “bed”"),
    "œ": (V, "open-mid front rounded vowel", "e as in “bed” said with rounded lips (French “neuf”, German “können”)"),
    "ɜ": (V, "open-mid central unrounded vowel", "like the vowel of British “bird”"),
    "ʌ": (V, "open-mid back unrounded vowel", "u as in “cup”"),
    "ɔ": (V, "open-mid back rounded vowel", "o as in British “thought”"),
    "æ": (V, "near-open front unrounded vowel", "a as in “cat”"),
    "ɐ": (V, "near-open central vowel", "a short, relaxed “uh” (Portuguese “cama”, German “-er” endings)"),
    "a": (V, "open front unrounded vowel", "a as in Spanish “casa”"),
    "ɑ": (V, "open back unrounded vowel", "a as in “father”"),
    "ɒ": (V, "open back rounded vowel", "a rounded “ah” (British “lot”, Hungarian “a”)"),
    "ɝ": (V, "r-coloured mid central vowel", "the vowel of American “bird”, “first”"),
    "ɚ": (V, "r-coloured schwa", "a weak vowel with an r colour (American “butter”, Mandarin “er”)"),
    # stops
    "p": (STOP, "voiceless bilabial stop", "p as in “pen”"),
    "b": (STOP, "voiced bilabial stop", "b as in “bed”"),
    "t": (STOP, "voiceless alveolar stop", "t as in “ten”"),
    "d": (STOP, "voiced alveolar stop", "d as in “dog”"),
    "ʈ": (STOP, "voiceless retroflex stop", "t with the tongue tip curled back (Swedish and Norwegian “rt”)"),
    "ɖ": (STOP, "voiced retroflex stop", "d with the tongue tip curled back (Swedish and Norwegian “rd”)"),
    "c": (STOP, "voiceless palatal stop", "a k made against the hard palate, close to “ty” (Hungarian “ty”, Czech “ť”)"),
    "ɟ": (STOP, "voiced palatal stop", "a g made against the hard palate, close to “dy” (Hungarian “gy”, Czech “ď”)"),
    "k": (STOP, "voiceless velar stop", "k as in “key”"),
    "ɡ": (STOP, "voiced velar stop", "g as in “go”"),
    "q": (STOP, "voiceless uvular stop", "a k made further back, at the uvula (Uzbek “q”)"),
    "ʔ": (STOP, "glottal stop", "the catch in the throat in “uh-oh”"),
    # nasals
    "m": (NAS, "bilabial nasal", "m"),
    "ɱ": (NAS, "labiodental nasal", "m made with the lower lip against the teeth, before f or v (“comfort”)"),
    "n": (NAS, "alveolar nasal", "n"),
    "ɳ": (NAS, "retroflex nasal", "n with the tongue tip curled back"),
    "ɲ": (NAS, "palatal nasal", "ny, as Spanish “ñ”"),
    "ŋ": (NAS, "velar nasal", "ng as in “sing”"),
    "ɴ": (NAS, "uvular nasal", "an n made far back, at the uvula (Japanese syllable-final ん)"),
    # rhotics
    "r": (RHO, "alveolar trill", "a rolled r"),
    "ʀ": (RHO, "uvular trill", "an r rolled at the uvula (Luxembourgish r)"),
    "ɾ": (RHO, "alveolar tap", "one quick flap of the tongue, as in Spanish “pero”"),
    "ɽ": (RHO, "retroflex flap", "a flap with the tongue tip curled back"),
    "ɹ": (RHO, "alveolar approximant", "English r"),
    "ɻ": (RHO, "retroflex approximant", "an English-like r with the tongue tip curled back"),
    # fricatives
    "ɸ": (FRIC, "voiceless bilabial fricative", "blowing through nearly closed lips"),
    "β": (FRIC, "voiced bilabial fricative", "a soft b where the lips do not close (Spanish b between vowels)"),
    "f": (FRIC, "voiceless labiodental fricative", "f"),
    "v": (FRIC, "voiced labiodental fricative", "v"),
    "θ": (FRIC, "voiceless dental fricative", "th as in “thin”"),
    "ð": (FRIC, "voiced dental fricative", "th as in “this”"),
    "s": (FRIC, "voiceless alveolar fricative", "s"),
    "z": (FRIC, "voiced alveolar fricative", "z"),
    "ʃ": (FRIC, "voiceless postalveolar fricative", "sh as in “ship”"),
    "ʒ": (FRIC, "voiced postalveolar fricative", "s as in “measure”"),
    "ʂ": (FRIC, "voiceless retroflex fricative", "a hard sh with the tongue tip back (Polish “sz”)"),
    "ʐ": (FRIC, "voiced retroflex fricative", "a hard zh (Polish “ż”, “rz”)"),
    "ɕ": (FRIC, "voiceless alveolo-palatal fricative", "a soft sh, with the tongue body raised (Polish “ś”)"),
    "ʑ": (FRIC, "voiced alveolo-palatal fricative", "a soft zh (Polish “ź”)"),
    "ç": (FRIC, "voiceless palatal fricative", "the ch of German “ich”; like a strong h before ee"),
    "ʝ": (FRIC, "voiced palatal fricative", "a y with friction"),
    "x": (FRIC, "voiceless velar fricative", "ch as in Scottish “loch”, German “Bach”"),
    "ɣ": (FRIC, "voiced velar fricative", "a g where the tongue does not close (Dutch g, Spanish g between vowels)"),
    "χ": (FRIC, "voiceless uvular fricative", "a rougher ch made at the uvula (Welsh “ch”)"),
    "ʁ": (FRIC, "voiced uvular fricative", "the r of French and German"),
    "ħ": (FRIC, "voiceless pharyngeal fricative", "a strong h squeezed in the throat (Arabic ح)"),
    "ʕ": (FRIC, "voiced pharyngeal fricative", "a voiced throat sound (Arabic ع)"),
    "h": (FRIC, "voiceless glottal fricative", "h as in “hat”"),
    "ɦ": (FRIC, "voiced glottal fricative", "a breathy, voiced h (Dutch and Czech h)"),
    "ɬ": (FRIC, "voiceless alveolar lateral fricative", "the Welsh “ll”: an l blown without voice"),
    # glides
    "ʋ": (GLIDE, "labiodental approximant", "a soft v without friction (Dutch w, Slovenian v)"),
    "j": (GLIDE, "palatal approximant", "y as in “yes”"),
    "ɥ": (GLIDE, "labial-palatal approximant", "y said with rounded lips (French “huit”)"),
    "ɰ": (GLIDE, "velar approximant", "a g so weak the tongue never closes"),
    "w": (GLIDE, "labial-velar approximant", "w as in “we”"),
    "ʍ": (GLIDE, "voiceless labial-velar approximant", "a breathy w (“wh” in some accents)"),
    # laterals
    "l": (LAT, "alveolar lateral approximant", "l as in “leaf”"),
    "ɫ": (LAT, "velarised alveolar lateral", "the dark l of “full”"),
    "ɭ": (LAT, "retroflex lateral approximant", "l with the tongue tip curled back"),
    "ʎ": (LAT, "palatal lateral approximant", "an l made with the tongue body, close to “ly” (Italian “gli”)"),
}

# Multi-symbol units that are one token.
UNITS = {
    "t͡s": (AFFR, "voiceless alveolar affricate", "ts as in “cats”"),
    "d͡z": (AFFR, "voiced alveolar affricate", "ds as in “beds”"),
    "t͡ʃ": (AFFR, "voiceless postalveolar affricate", "ch as in “church”"),
    "d͡ʒ": (AFFR, "voiced postalveolar affricate", "j as in “judge”"),
    "t͡ɕ": (AFFR, "voiceless alveolo-palatal affricate", "a soft ch (Polish “ć”, Bosnian “ć”)"),
    "d͡ʑ": (AFFR, "voiced alveolo-palatal affricate", "a soft j (Polish “dź”)"),
    "t͡ʂ": (AFFR, "voiceless retroflex affricate", "a hard ch (Polish “cz”)"),
    "d͡ʐ": (AFFR, "voiced retroflex affricate", "a hard j (Polish “dż”)"),
    "ŋ͡m": (NAS, "labial-velar nasal", "ng and m at once, lips closed (Vietnamese final -ng after o, u)"),
    "k͡p": (STOP, "labial-velar stop", "k and p at once, lips closed (Vietnamese final -c after o, u)"),
    "t͡ʃʼ": (AFFR, "ejective postalveolar affricate", "ch said with a pop from the throat, no breath"),
    "a͡ɪ": (DIPH, "diphthong a to ɪ", "eye"),
    "a͡ʊ": (DIPH, "diphthong a to ʊ", "ow as in “now”"),
    "e͡ɪ": (DIPH, "diphthong e to ɪ", "ay as in “say”"),
    "o͡ʊ": (DIPH, "diphthong o to ʊ", "o as in American “go”"),
    "o͡ɪ": (DIPH, "diphthong o to ɪ", "oy as in “boy”"),
    "e͡ʊ": (DIPH, "diphthong e to ʊ", "e gliding to oo"),
}

# Diacritics, in the order they are named. (mark, name, explanation)
DIACRITICS = [
    ("̃", "nasalised", "air also flows through the nose (French “an”, Portuguese “não”)"),
    ("̯", "non-syllabic", "the weak, glide-like half of a diphthong; it is not a syllable on its own"),
    ("̩", "syllabic", "forms a syllable with no vowel, as the n of “button”"),
    ("̥", "voiceless", "said without vocal-fold vibration, as if whispered"),
    ("̊", "voiceless", "said without vocal-fold vibration, as if whispered"),
    ("̪", "dental", "the tongue tip touches the upper teeth"),
    ("̠", "retracted", "made a little further back in the mouth"),
    ("̝", "raised", "a little closer (tongue higher) than the plain symbol"),
    ("̞", "lowered", "a little more open; a lowered fricative is a smooth approximant with no friction"),
    ("̺", "apical", "made with the very tip of the tongue"),
    ("̻", "laminal", "made with the blade of the tongue, just behind the tip"),
    ("̆", "extra-short", "a very short version of the vowel"),
    ("́", "accent ´ (high)", "the G2P's dictionary marks a pitch accent or stress here"),
    ("̀", "accent ` (low)", "the G2P's dictionary marks a pitch accent or stress here"),
    ("̂", "accent ˆ (falling)", "the G2P's dictionary marks a falling pitch accent"),
    ("̌", "accent ˇ (rising)", "the G2P's dictionary marks a rising pitch accent"),
    ("ʲ", "palatalised", "said with the tongue body raised toward the palate: a y-like colour (“soft” consonants)"),
    ("ʰ", "aspirated", "followed by a puff of breath before the vowel starts (English “top” has one, “stop” does not)"),
    ("ʷ", "labialised", "said with rounded lips"),
    ("ʼ", "ejective", "a pop made by the throat, with no breath from the lungs"),
    ("ʱ", "breathy-voiced", "voiced, then released with a breathy, murmured puff (Hindi “bh”, “gh”)"),
    ("ˤ", "pharyngealised (emphatic)", "said with the tongue root pulled back: a darker, “heavy” sound (Arabic ط, ص)"),
    ("\u031a", "unreleased", "a final stop with no audible burst: the closure is held and the sound ends silently"),
    ("\u0308", "centralised", "said closer to the centre of the mouth than the plain vowel"),
]
DIACRITIC_NAMES = {m: (n, e) for m, n, e in DIACRITICS}

SPECIAL = {
    "SIL": ("silence", "A pause: no speech. Use it for a pause the speaker makes inside a segment."),
    "noise": ("noise", "A non-speech sound: a breath, cough, laugh or click."),
    "<unk>": ("unknown", "The G2P wrote a sound this model has no token for. Keep its time span; the label "
              "cannot say what it is."),
}

CLASS_TITLES = [
    (V, "Vowels"), (DIPH, "Diphthongs"), (GLIDE, "Glides and non-syllabic vowels"), (STOP, "Stops"),
    (AFFR, "Affricates"), (FRIC, "Fricatives"), (NAS, "Nasals"), (LAT, "Laterals"), (RHO, "Rhotics (r sounds)"),
]

BOUNDARY_TIPS = [
    ("Stops (p t k b d g …)", "A stop starts where the closure starts: energy drops to a near-silent gap, "
     "and the formants of the vowel before it end. It ends after its burst (and after its aspiration, for "
     "ʰ tokens), where the next sound begins."),
    ("Vowels", "Start at the first regular glottal pulse with clear formants; end where the formants fade or "
     "the next consonant's closure or noise begins."),
    ("Fricatives (s ʃ f x …)", "The span of the noise: start and end where the high-frequency noise starts "
     "and stops."),
    ("Nasals (m n ŋ …)", "A steady, low murmur with weak upper formants. The edges are usually sudden changes "
     "in energy."),
    ("Glides, laterals, r (j w l ɹ …)", "Edges are smooth formant transitions: put the boundary at the "
     "middle of the transition."),
    ("Diphthongs (a͡ɪ …)", "One token for both parts: from the start of the first vowel to the end of the "
     "second. Do not split it."),
    ("Pauses", "Leave a short pause between words as a gap, or insert SIL where the speaker clearly stops."),
]


def split_token(tok):
    """A token -> (base unit, [diacritic marks in order])."""
    for unit in sorted(UNITS, key=len, reverse=True):
        if tok.startswith(unit):
            return unit, list(tok[len(unit):])
    if tok[0] in BASE:                     # ç is a base symbol of its own, not c + cedilla
        import unicodedata
        return tok[0], list(unicodedata.normalize("NFD", tok[1:]))
    import unicodedata
    nfd = unicodedata.normalize("NFD", tok)
    return nfd[0], list(nfd[1:])


def describe(tok):
    """A token -> (class, name, plain explanation)."""
    if tok in SPECIAL:
        name, plain = SPECIAL[tok]
        return "special", name, plain
    base, marks = split_token(tok)
    if base in UNITS:
        cls, name, plain = UNITS[base]
    elif base in BASE:
        cls, name, plain = BASE[base]
    else:
        raise KeyError(f"no description for {tok!r} (base {base!r})")
    mods = [DIACRITIC_NAMES[m] for m in marks if m in DIACRITIC_NAMES]
    unknown = [m for m in marks if m not in DIACRITIC_NAMES]
    if unknown:
        raise KeyError(f"unknown diacritic in {tok!r}: {[hex(ord(m)) for m in unknown]}")
    if cls == V and "̯" in marks:
        cls = GLIDE
    if mods:
        name = ", ".join([name] + [n for n, _ in mods])
        plain = f"{plain}; " + "; ".join(f"{n}: {e}" for n, e in mods)
    return cls, name, plain


# ---------------------------------------------------------------------------------------------
# Tones (hand-written; checked against the G2P's output in the collected counts)
# ---------------------------------------------------------------------------------------------

CHAO_LETTER = {"5": "˥", "4": "˦", "3": "˧", "2": "˨", "1": "˩"}
CONTOURS = {
    "55": "high level", "44": "high-mid level", "33": "mid level", "22": "low level",
    "5": "high (one level)", "4": "high-mid (one level)",
    "3": "mid (one level)", "2": "low (one level)",
    "35": "rising, mid to high", "45": "high rising", "24": "rising, low-mid to high-mid",
    "13": "low rising", "114": "low, then rising", "51": "falling, high to low",
    "53": "high falling", "42": "falling, high-mid to low", "21": "low falling", "32": "mid falling",
    "214": "dipping: falls low, then rises", "312": "dipping: mid, low, then mid", "212": "low dipping",
}

TONE_LANGS = {
    "zh": {
        "name": "Mandarin",
        "intro": "Four tones and a neutral tone. The G2P gives each character its dictionary reading, "
                 "so tone sandhi is not applied (a 3rd tone before another 3rd tone is still 214, "
                 "though it is said as 35) and particles may carry their full tone.",
        "tones": {
            "55": ("Tone 1, high level", "ā; yīnpíng 阴平"),
            "35": ("Tone 2, rising", "á; yángpíng 阳平"),
            "214": ("Tone 3, dipping", "ǎ; shǎngshēng 上声. Before another syllable it is often only low (21)."),
            "51": ("Tone 4, falling", "à; qùshēng 去声"),
            "3": ("Neutral tone", "qīngshēng 轻声: short and unstressed; its pitch follows the tone before it"),
            "5": ("(rare)", "only in spelled-out letters and foreign words"),
        },
    },
    "yue": {
        "name": "Cantonese",
        "intro": "Six tones, numbered 1–6 in Jyutping. Level tones are written with one digit. The "
                 "checked (entering) tones of syllables ending in p, t, k share the number of the level "
                 "tone at the same height.",
        "tones": {
            "5": ("Tone 1, high level", "Jyutping 1 (and the high entering tone, 7)"),
            "35": ("Tone 2, high rising", "Jyutping 2"),
            "3": ("Tone 3, mid level", "Jyutping 3 (and the mid entering tone, 8)"),
            "21": ("Tone 4, low falling", "Jyutping 4"),
            "13": ("Tone 5, low rising", "Jyutping 5"),
            "2": ("Tone 6, low level", "Jyutping 6 (and the low entering tone, 9)"),
        },
    },
    "vi": {
        "name": "Vietnamese (Northern)",
        "intro": "Six tones, shown by the accent marks of the spelling. Sắc is split in two: syllables "
                 "ending in p, t, c, ch take a short high rise (45). Counts are from Northern "
                 "Vietnamese (vi); vi-c and vi-s may number some tones differently.",
        "tones": {
            "33": ("ngang, level", "no mark: a"),
            "32": ("huyền, low falling", "grave: à"),
            "24": ("sắc, rising", "acute: á"),
            "45": ("sắc, high rising (checked)", "acute on a syllable ending in p, t, c, ch: các, nhất"),
            "21": ("nặng, low, glottalised", "dot below: ạ"),
            "312": ("hỏi, dipping", "hook above: ả"),
            "35": ("ngã, broken rising", "tilde: ã; often with a glottal break in the middle"),
            "3": ("(rare)", "only in foreign words"),
        },
    },
    "th": {
        "name": "Thai",
        "intro": "Five tones. The tone of a syllable follows from its consonant class, vowel length, "
                 "ending and tone mark; the G2P works it out.",
        "tones": {
            "3": ("mid tone", "siang saman"),
            "21": ("low tone", "siang ek"),
            "51": ("falling tone", "siang tho"),
            "45": ("high tone", "siang tri"),
            "114": ("rising tone", "siang chattawa: starts low, then rises"),
            "33": ("(rare)", "only in foreign words"),
            "35": ("(rare)", "only in foreign words"),
        },
    },
}


# ---------------------------------------------------------------------------------------------
# 3. render
# ---------------------------------------------------------------------------------------------

GROUP_NOTES = {
    "latin": "Accent marks on vowels (á, à, â, ǎ …) are pitch accents from the dictionaries of Slovenian "
             "and Lithuanian. They are separate tokens, but their boundaries are placed like those of the "
             "plain vowel.",
    "cyrillic": "Soft (palatalised) consonants are tokens of their own: tʲ is not t. Place their boundaries "
                "like those of the plain consonant; the soft colour belongs to the consonant. Croatian "
                "(`hbs`, Latin script) is aligned with this model too, and its accent marks (â, ǎ, ô …) are "
                "pitch accents: separate tokens, placed like the plain vowel.",
    "cjk": "Tone is a separate layer, not part of these tokens: see the [tone guide](tones.md). The "
           "unaspirated onsets written b d g z j zh in pinyin (b d g z in Jyutping) are the tokens "
           "b d ɡ d͡z d͡ʑ d͡ʐ, and each affricate is **one token**: 在 = d͡z a ɪ, 这 = d͡ʐ ɤ. An aspirated "
           "affricate is **split into two tokens**, with the aspiration on its second half "
           "(其 = t ɕʰ i, 此 = t sʰ i): give each part its own span. Diphthongs are split too (在 = a ɪ). "
           "The aspirated stops pʰ tʰ kʰ are the sounds written p t k in pinyin and Jyutping. Plain p t k "
           "are Cantonese syllable-final stops, which have no audible release, and foreign words.",
    "japanese": "Japanese has no tone layer in this model. A long vowel is one token (its length is a "
                "separate layer): ドーブツ = d o b ɯ t s ɯ. Affricates are **split into two tokens** "
                "(ツ = t s, ジ = d ʑ): give the stop and the fricative part their own spans. The "
                "syllable-final ン is ɴ.",
    "vietnamese": "Tone is a separate layer, not part of these tokens: see the [tone guide](tones.md). "
                  "k͡p and ŋ͡m are the finals -c and -ng after rounded vowels: the lips close as well. A ʔ "
                  "at the end of a syllable (bị = b i ʔ) is the glottal closure of the nặng tone.",
    "thai_khmer": "Tone is a separate layer, not part of these tokens: see the [tone guide](tones.md). Final "
                  "stops are unreleased (p̚ t̚ k̚): the token ends with the silent closure. A short vowel at "
                  "the end of a syllable is closed by ʔ (และ = l ɛ ʔ).",
    "abjad": "The script leaves most vowels out, so the G2P supplies them: listen for them rather than "
             "trusting the spelling. Emphatic consonants (tˤ sˤ) are tokens of their own. Sindhi and Urdu "
             "aspirated and breathy stops (kʰ, bʰ, dʰ …) are single tokens; the puff after the release "
             "belongs to the stop.",
    "brahmic": "Aspirated (ʰ) and breathy-voiced (ʱ) stops are tokens of their own; the puff or murmur "
               "after the release belongs to the stop, not to the vowel. bʰ and bʱ both occur for the same "
               "letter; treat them alike. Dental (t̪ d̪ n̪) and retroflex (ʈ ɖ ɳ) consonants are different "
               "tokens; plain t and d almost never occur.",
    "other_alphabetic": "Ejectives (ʼ, Georgian) and aspirated stops (ʰ, Armenian, Georgian, Korean) are "
                        "tokens of their own; the burst and the puff after it belong to the stop. Korean "
                        "tense consonants (ㄲ ㄸ ㅃ ㅆ ㅉ) have no token of their own: they are written as the "
                        "plain one (k t p s t͡ɕ).",
}
# Tokens whose meaning in a group is not what their IPA marks say.
TOKEN_OVERRIDES = {
    "abjad": {
        "î": ("long close front vowel (Kurdish î)", "ee as in “see”, long. The G2P writes Kurdish î as in Kurdish Latin spelling; not a pitch accent"),
        "ê": ("long close-mid front vowel (Kurdish ê)", "e as in Spanish “de”, long. Kurdish ê as in Kurdish Latin spelling; not a pitch accent"),
        "û": ("long close back vowel (Kurdish û)", "oo as in “food”, long. Kurdish û as in Kurdish Latin spelling; not a pitch accent"),
        "ř": ("alveolar trill (Kurdish ڕ)", "a strongly rolled r, the Kurdish ڕ (written ř in Kurdish Latin spelling)"),
    },
}
GROUP_RARE = {
    "latin": "they come from untrained languages (Basque, Albanian, Latin …) or are very rare.",
    "cyrillic": "they come from untrained languages (Adyghe, Bashkir, Tatar) or are very rare.",
    "abjad": "they come from untrained languages (Uyghur, Syriac) or are very rare.",
    "brahmic": "they come from an untrained language (Sanskrit) or are very rare.",
    "other_alphabetic": "they come from an untrained language (Ancient Greek) or are very rare.",
}
for _g in GROUP_MODELS:
    GROUP_RARE.setdefault(_g, "they are very rare in this language.")


def lang_label(code):
    return LANG_NAMES.get(code, code)


def token_rows(G, group):
    from bournemouth_aligner.phonemizers.std import std_inventory
    inv = std_inventory(group)
    stats = G
    lt = stats["lang_tokens"]
    rows = []
    for tok in inv.tokens:
        if tok == "<blank>":
            continue
        cls, name, plain = describe(tok)
        if tok in TOKEN_OVERRIDES.get(group, {}):
            name, plain = TOKEN_OVERRIDES[group][tok]
        counts = stats["tokens"].get(tok, {})
        shares = sorted(((l, 100.0 * n / lt[l]) for l, n in counts.items()), key=lambda x: -x[1])
        # Example words: English first when English uses the token, then the heaviest users.
        langs = [l for l, _ in shares]
        if "en" in langs and dict(shares)["en"] >= 0.3:
            langs = ["en"] + [l for l in langs if l != "en"]
        examples = []
        for l in langs[:3]:
            w = stats["words"][tok][l][0]
            examples.append([l, w[0], " ".join(w[2])])
        also = sorted(((ipa, n) for ipa, n in stats["eipa"].get(tok, {}).items() if ipa != tok),
                      key=lambda x: -x[1])
        total = sum(counts.values())
        also = [ipa for ipa, n in also if n >= max(20, 0.01 * total)]
        rows.append({
            "token": tok, "class": cls, "name": name, "plain": plain, "count": total,
            "langs": [[l, round(p, 2)] for l, p in shares if p >= 0.01],
            "examples": examples, "also": also,
        })
    return inv, rows


def md_escape(s):
    return s.replace("|", "\\|")


def render_phonemes(stats, group):
    G = stats["groups"][group]
    inv, rows = token_rows(G, group)
    n_langs = len(G["lang_tokens"])
    model = GROUP_MODELS[group]
    # json for the UI
    tokens = {}
    for r in rows:
        plain = r["plain"]
        if r["also"]:
            plain += f". Also used for {', '.join(r['also'])}, which this model has no token of its own for"
        if r["count"] == 0 and r["class"] != "special":
            plain += ". Rare: not in the test sentences of any trained language"
        tokens[r["token"]] = {
            "name": r["name"], "class": r["class"], "plain": plain,
            "examples": [[l, w, s] for l, w, s in r["examples"]],
            "langs": r["langs"][:8], "also": r["also"], "count": r["count"],
        }
    os.makedirs(GUIDES, exist_ok=True)
    with open(os.path.join(GUIDES, f"phonemes-{group}.json"), "w", encoding="utf-8") as f:
        json.dump({"group": group, "model": model, "source": stats["source"],
                   "note": re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", GROUP_NOTES[group]).replace("**", ""),
                   "lang_names": {l: lang_label(l) for l in G["lang_tokens"]},
                   "tokens": tokens}, f, ensure_ascii=False, indent=0)

    # markdown
    out = [
        f"# Phoneme guide: {GROUP_TITLES[group]}",
        "",
        "<!-- Generated by tools/build_annotation_guides.py. Edit that file, not this one. -->",
        "",
        f"Every phoneme token of `{model}` ({len(rows) + 1} tokens: {len(rows) - len(SPECIAL)} phonemes, "
        f"{len(SPECIAL)} special tokens and the model's internal blank). The web interface shows the same "
        "text when you hover a phoneme.",
        "",
        "How to read the tables:",
        "",
        "- **Sounds like** is an approximation. The example words are the real thing: they are the most "
        "frequent words containing the token in the FLEURS test sentences (dev split) of "
        f"{n_langs} language{'s' if n_langs > 1 else ''} ({', '.join(lang_label(l) for l in G['lang_tokens'])}), "
        "written in the G2P's own tokens, which is exactly what the aligner sees.",
        "- **Used most in** gives the languages that use the token most, as a share of all the phonemes "
        "of that language.",
        "- **Also used for**: IPA the G2P wrote that has no token of its own in this model, so it was mapped "
        "to this one.",
        f"- **Rare** tokens never appear in those sentences: {GROUP_RARE[group]}",
        "",
        GROUP_NOTES[group],
        "",
        "## Where boundaries go",
        "",
        "Suggested conventions. Agree on them before annotating, so that every annotator places "
        "boundaries the same way.",
        "",
    ]
    for head, text in BOUNDARY_TIPS:
        out.append(f"- **{head}.** {text}")
    out += ["", "## Diacritics", "", "Marks added to a base symbol change it as follows.", "",
            "| mark | name | meaning |", "|---|---|---|"]
    seen = set()
    for m, n, e in DIACRITICS:
        if n in seen:
            continue
        seen.add(n)
        shown = ("◌" + m) if len(m) == 1 and 0x300 <= ord(m) < 0x370 else m
        out.append(f"| {shown} | {n} | {e} |")
    out += ["", "## Special tokens", "", "| token | meaning |", "|---|---|"]
    for t, (n, e) in SPECIAL.items():
        out.append(f"| `{t}` | **{n}**. {e} |")

    for cls, title in CLASS_TITLES:
        sel = [r for r in rows if r["class"] == cls]
        if not sel:
            continue
        out += ["", f"## {title}", "",
                "| token | name | sounds like | examples | used most in |", "|---|---|---|---|---|"]
        for r in sel:
            name = r["name"]
            if r["also"]:
                name += f". *Also used for {', '.join(r['also'])}.*"
            ex = "; ".join(f"{lang_label(l)} *{w}* /{s}/" for l, w, s in r["examples"])
            used = ", ".join(f"{lang_label(l)} {p:.2g}%" for l, p in r["langs"][:4])
            if r["count"] == 0:
                ex, used = "", "**rare**"
            elif r["count"] < 20:
                used = (used + " " if used else "") + f"(rare: {r['count']}×)"
            plain = (TOKEN_OVERRIDES.get(group, {}).get(r["token"]) or describe(r["token"])[1:])[1]
            out.append("| " + " | ".join(md_escape(x) for x in (f"**{r['token']}**", name, plain, ex, used)) + " |")
    out.append("")
    os.makedirs(DOCS, exist_ok=True)
    with open(os.path.join(DOCS, f"{group}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def letters(t):
    return "".join(CHAO_LETTER[c] for c in t)


def render_tones(stats):
    from bournemouth_aligner.standard_g2p.gold_g2p import TONE_VOCAB
    contours = {}
    for t in TONE_VOCAB[1:]:
        explain = ("one pitch level" if len(t) == 1 else
                   f"starts at {t[0]}, ends at {t[-1]}" if len(t) == 2 else
                   f"starts at {t[0]}, turns at {t[1]}, ends at {t[2]}")
        contours[t] = {"shape": CONTOURS[t], "letters": letters(t), "explain": explain + " (1 = lowest, 5 = highest)"}
    langs = {}
    for code, L in TONE_LANGS.items():
        counts = stats["tones"].get(code, {})
        total = sum(counts.values()) or 1
        tones = {}
        for t, (name, desc) in L["tones"].items():
            ex = stats["tone_words"].get(code, {}).get(t, [])
            tones[t] = {
                "name": name, "desc": desc, "share": round(100.0 * counts.get(t, 0) / total, 1),
                "examples": [[w, " ".join(f"{a}{b}" for a, b in seq)] for w, _, seq in ex[:3]],
            }
        missing = set(counts) - set(tones)
        if missing:
            raise KeyError(f"{code}: tones seen but not described: {sorted(missing)}")
        # "order": JSON objects with number-like keys come back sorted numerically in JavaScript.
        langs[code] = {"name": L["name"], "intro": L["intro"], "order": list(L["tones"]), "tones": tones}
    os.makedirs(GUIDES, exist_ok=True)
    with open(os.path.join(GUIDES, "tones.json"), "w", encoding="utf-8") as f:
        json.dump({"vocab": TONE_VOCAB, "contours": contours, "langs": langs}, f, ensure_ascii=False, indent=0)

    out = [
        "# Tone guide",
        "",
        "<!-- Generated by tools/build_annotation_guides.py. Edit that file, not this one. -->",
        "",
        "Mandarin, Cantonese, Vietnamese and Thai are the tonal languages with a model. Tone is a "
        "separate layer: the vowel /a/ with tone 35 and with tone 21 is the same phoneme token, and "
        "**tone never moves a boundary**. It sits on the syllable's vowel (on the last token when the vowel "
        "is split in two), and consonants carry none.",
        "",
        "The tone comes from the G2P's dictionary, not from the audio: it is the word's lexical tone, "
        "with tone sandhi not applied, so the speaker may say something different. Change it only when you "
        "are sure.",
        "",
        "## Reading the numbers",
        "",
        "Tones are written as Chao numbers: pitch on a scale from **1 (lowest)** to **5 (highest)** in the "
        "speaker's range. Two digits are where the pitch starts and ends, three digits add the turning "
        "point. One digit is a level pitch, used for short (checked) syllables and for level tones written "
        "compactly. The same number can be a different tone in another language: **read the table of the "
        "language you are annotating.**",
        "",
        "## Mandarin, Cantonese, Vietnamese, Thai",
    ]
    for code, L in langs.items():
        out += ["", f"### {L['name']} (`{code}`)", "", L["intro"], "",
                "| number | tone | pitch | notes | share | examples |", "|---|---|---|---|---:|---|"]
        for t, e in L["tones"].items():
            ex = "; ".join(f"{w} /{s}/" for w, s in e["examples"])
            out.append("| " + " | ".join(md_escape(x) for x in (
                f"**{t}** {letters(t)}", e["name"], CONTOURS[t], e["desc"], f"{e['share']}%", ex)) + " |")
    out += ["", "## All 22 tone numbers", "",
            "The tone head of every tonal model has these 22 classes (0 is “no tone”).", "",
            "| number | letters | pitch |", "|---|---|---|"]
    for t, c in contours.items():
        out.append(f"| {t} | {c['letters']} | {c['shape']} |")
    out += ["", "Shares and examples: FLEURS test sentences (dev split) through the same G2P "
            "(Mandarin 403 sentences, Cantonese 359, Vietnamese 361, Thai 439).", ""]
    with open(os.path.join(DOCS, "tones.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def render(stats):
    for group in stats["groups"]:
        render_phonemes(stats, group)
    render_tones(stats)
    # Which guides exist, so the page does not ask for ones that do not.
    phonemes = sorted(f[len("phonemes-"):-len(".json")] for f in os.listdir(GUIDES)
                      if f.startswith("phonemes-") and f.endswith(".json"))
    with open(os.path.join(GUIDES, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"phonemes": phonemes, "tones": True}, f)
    print("wrote", os.path.relpath(DOCS, ROOT), "and", os.path.relpath(GUIDES, ROOT))


if __name__ == "__main__":
    main()
