# Language support

[Main README](../README.md) · [Developer guide](README.md) · [About the models](about-models.md) · [Evaluation](evaluation.md) · **Language support**

This page covers the default `std` phoneme set and its `mbfa-*` models: which languages they align, the code to
pass for each, and what changes for tonal and unspaced languages. The espeak-ng `ph66` set is
summarised [at the end](#the-ph66-set).

## How a language picks its model

```text
lang code ──► canonical BCP 47 code ──► language group ──► mbfa model
  "pt-br"          "pt-BR"                  latin            mbfa-latin
  "cmn"            "zh"                     cjk              mbfa-cjk
  "hr"             "hbs"                    cyrillic         mbfa-cyrillic
```

Languages are grouped by **writing system**, not by language family. Each language group has its own
token list (the model's phoneme head) and its own model. Within a group, switching language only
changes the G2P's language:
`aligner.phonemizer.set_backend(language="fr")`. Across groups, the model has to change too:
`aligner.load_model(preset="ru")`.

Codes are accepted in any case and in several spellings: BCP 47 (`pt-BR`), ISO 639-3
(`deu`, `cmn`), and the espeak-ng spellings the ph66 set uses (`en-us`, `fr-be`, `en-gb-x-rp`). A
regional variant the G2P does not have falls back to its base language with a warning.
`aligner.lang` holds the canonical code.

A language outside the nine groups below raises `UnsupportedLanguageError`, and the message names
`phoneme_set="ph66"` as the alternative. BFA never falls back to the other phoneme set silently.

`balign --list-presets` prints the live list.

## Supported: 82 languages in 9 language groups

**Trained** means the language was in the model's FLEURS training data; the hours are the
FLEURS training audio for that language. `mbfa-latin` and the models of the newer groups
(vietnamese, thai_khmer, abjad, brahmic, other_alphabetic) used all of it. `mbfa-cyrillic` and `mbfa-cjk` used a
62% sample. A language the G2P covers but the model never heard is still aligned, through
the phonemes it shares with the trained ones. Expect it to be less accurate.

### `latin` → `mbfa-latin` (208 tokens)

The default group. Latin script, words separated by spaces, no tone. Two more latin models cover the
same languages and are selected by name: `mbfa-latin-benchmark` (best on clean speech) and
`mbfa-latin-noEnglish` (the same recipe trained without English, for transfer tests).

| code | language | trained (h) |
|---|---|---:|
| `af` | Afrikaans | 3.2 |
| `az` | Azerbaijani | 8.5 |
| `bs` | Bosnian | 9.5 |
| `ca` | Catalan | 7.0 |
| `cs` | Czech | 8.2 |
| `cy` | Welsh (North); `cy-sw` South | 6.8 |
| `da` | Danish | 7.3 |
| `de` | German | 8.8 |
| `en` | English (US); `en-GB` British | 7.1 |
| `es` | Spanish (Castilian); `es-419` Latin American, `es-MX` Mexican | 8.5 (`es-419`) |
| `et` | Estonian | 7.0 |
| `fi` | Finnish | 8.2 |
| `fr` | French; `fr-CA` Quebec | 9.7 |
| `ga` | Irish | 7.3 |
| `gl` | Galician | 6.5 |
| `hu` | Hungarian | 9.0 |
| `id` | Indonesian | 8.1 |
| `is` | Icelandic | 2.7 |
| `it` | Italian | 8.8 |
| `lb` | Luxembourgish | 7.3 |
| `lt` | Lithuanian | 8.9 |
| `mi` | Māori | 4.3 |
| `ms` | Malay | 8.3 |
| `mt` | Maltese | 9.1 |
| `nb` | Norwegian Bokmål (`no` and `nn` resolve here) | 8.8 |
| `nl` | Dutch | 7.6 |
| `pl` | Polish | 8.8 |
| `pt` | Portuguese (European); `pt-BR` Brazilian | 8.7 (`pt-BR`) |
| `ro` | Romanian | 9.5 |
| `sk` | Slovak | 5.7 |
| `sl` | Slovenian | 7.3 |
| `sv` | Swedish | 7.3 |
| `sw` | Swahili | 6.8 |
| `tl` | Tagalog / Filipino (`fil`) | 6.0 |
| `tr` | Turkish | 7.7 |
| `uz` | Uzbek | 9.1 |
| `arg` (`an`) | Aragonese | — |
| `eo` | Esperanto | — |
| `eu` | Basque | — |
| `ia` | Interlingua | — |
| `io` | Ido | — |
| `la` | Latin (classical); `la-eccl` ecclesiastical | — |
| `pap` | Papiamento | — |
| `se` | Northern Sami | — |
| `sq` | Albanian | — |
| `tk` | Turkmen | — |
| `ang` | Old English | — |
| `enm` | Middle English | — |
| `egy` | Egyptian (transliteration) | — |

### `cyrillic` → `mbfa-cyrillic` (127 tokens)

Cyrillic script, words separated by spaces, no tone.

| code | language | trained (h) |
|---|---|---:|
| `be` | Belarusian | 7.7 |
| `bg` | Bulgarian | 8.9 |
| `hbs` | Serbo-Croatian, Latin script (`hr` resolves here); `hbs-Cyrl` Cyrillic | 9.3 (`hr`) |
| `kk` | Kazakh | 8.5 |
| `mk` | Macedonian | 6.5 |
| `ru` | Russian | 7.6 |
| `sr` | Serbian | 8.3 |
| `uk` | Ukrainian | 8.5 |
| `ady` | Adyghe | — |
| `ba` | Bashkir | — |
| `tt` | Tatar | — |

**Croatian is in this group** even though it is written in Latin script. The G2P has one
Serbo-Croatian language with a tag for each script, and the language takes the group of the majority. So
`BFA(preset="hr")` loads `mbfa-cyrillic`, which was trained on Croatian. Bosnian (`bs`) is its
own G2P language and stays in `latin`.

### `cjk` → `mbfa-cjk` (48 tokens)

Han script. These languages need word segmentation, and they carry lexical tone.

| code | language | trained (h) | segmenter |
|---|---|---:|---|
| `zh` | Mandarin, simplified script (`cmn`, `zh-cn` resolve here) | 9.0 | built in (longest match on the bundled dictionary keys) |
| `zh-Hant` | Mandarin, traditional script (`zh-tw`) | (as `zh`) | built in |
| `yue` | Cantonese | 5.9 | `pycantonese`: `pip install "bournemouth-forced-aligner[yue]"` |

### `japanese` → `mbfa-japanese` (28 tokens)

| code | language | trained (h) | segmenter |
|---|---|---:|---|
| `ja` | Japanese | 7.0 | `fugashi` + `unidic-lite`: `pip install "bournemouth-forced-aligner[ja]"` |

`BFA(preset="ja")` raises `ImportError` at construction until the segmenter is installed.

### `vietnamese` → `mbfa-vietnamese` (45 tokens)

Latin script with tone letters. Kept out of `latin` because it is tonal.

| code | language | trained (h) |
|---|---|---:|
| `vi` | Vietnamese (Northern); `vi-c` Central, `vi-s` Southern | 8.7 |

### `thai_khmer` → `mbfa-thai` (38 tokens)

Thai script. Needs word segmentation, and carries lexical tone. Khmer shares the token list but is
excluded (see below), so the model was trained on Thai only.

| code | language | trained (h) | segmenter |
|---|---|---:|---|
| `th` | Thai | 7.8 | `pythainlp`: `pip install "bournemouth-forced-aligner[th]"` |

### `abjad` → `mbfa-abjad` (90 tokens)

Arabic script and Syriac. The orthography leaves most vowels out, so the G2P supplies them.

| code | language | trained (h) |
|---|---|---:|
| `ar` | Arabic | 5.8 |
| `ku` | Kurdish (Central, `ckb`) | 9.0 |
| `fa` | Persian | 8.2 |
| `sd` | Sindhi | 8.8 |
| `ur` | Urdu | 6.3 |
| `ug` | Uyghur | — |
| `syc` | Syriac | — |

### `brahmic` → `mbfa-brahmic` (96 tokens)

Indic abugidas (Devanagari, Odia, Tamil scripts).

| code | language | trained (h) |
|---|---|---:|
| `hi` | Hindi | 6.2 |
| `or` | Odia | 3.2 |
| `ta` | Tamil | 7.4 |
| `sa` | Sanskrit | — |

### `other_alphabetic` → `mbfa-other-alphabetic` (104 tokens)

The remaining alphabets: Greek, Armenian, Georgian, Hangul and Ethiopic. Words are separated by
spaces, and no language in the group has tone.

| code | language | trained (h) |
|---|---|---:|
| `am` | Amharic | 9.0 |
| `el` | Greek | 9.6 |
| `hy` | Armenian (Eastern); `hy-west` Western | 9.3 |
| `ka` | Georgian | 4.5 |
| `ko` | Korean | 7.4 |
| `grc` | Ancient Greek | — |

The models of these five groups were warm-started from the latin model's trunk and selected on
FLEURS validation loss; no boundary labels exist for them, so they have no boundary evaluation yet.

## Not supported yet

| language group | languages | why not yet |
|---|---|---|
| excluded | `km` Khmer, `my` Burmese, `nan` Min Nan, `tts` Isan | the G2P output is not usable yet (see below) |

The excluded languages have these problems: Khmer folds 20% of its tokens to `<unk>`. Burmese writes tone as vowel
diacritics, and its group has no phoneme tokens at all. Min Nan has no maintained segmenter, and most of its
dictionary carries tone sandhi. Isan's dictionary is a romanisation, not IPA.

For any of these, `phoneme_set="ph66"` runs if espeak-ng knows the language, with the caveats
[below](#the-ph66-set).

## Nuances

### No text normalisation

The G2P reads one dictionary word at a time. **Digits, symbols and abbreviations are not
expanded**: write "twenty five", not "25", and "doctor", not "Dr.". espeak-ng (the ph66 set) does
expand them.

### Silence and punctuation

By default no silence is written at punctuation, because the mbfa models were trained on
sequences without one. `punctuation_sil=True` writes a `SIL` at every pause mark (`, . ; : ! ? …`, dashes
and the CJK marks `，。！？；：、`). The aligner can skip any `SIL` at no cost, so this only gives the path
a place to put a pause. An explicit `<SIL>` in the text is always honoured.

### Tone

Mandarin, Cantonese, Vietnamese and Thai are the tonal languages with a model. Tone is a **separate layer**, not
part of the phoneme: the vowel /a/ with tone 35 and with tone 21 is the same token, so the phoneme
inventory stays small and the phonemes are shared with other languages.

- Each phone carries a `tone` index into a fixed vocabulary of 22 Chao contours
  (`'' 35 21 33 45 51 32 214 55 24 3 312 5 13 2 212 42 44 53 22 4 114`). 0 means no tone. The tone sits
  on the syllable's vowel, on the last token when the vowel is split into two
  (`知道` → `ʈ ʂ ɚ⁵⁵ t ɑ ʊ⁵¹`), and consonants carry 0.
- The tone comes **from the G2P's dictionary pronunciation**, not from the audio. It is the
  word's lexical tone, with tone sandhi not applied, so a speaker's realised contour can differ
  from it. The tone's *timing* is the timing of the vowel it sits on. The tonal models have a
  trained tone head, but BFA does not read it yet.
- Atonal languages have 0 everywhere. Burmese tone is written as vowel diacritics, so it would
  not reach this layer (Burmese is excluded anyway).
- **Pitch accent is not tone.** Serbo-Croatian and Slovenian mark rising and falling accents with the
  same diacritics, and the G2P keeps them as part of the vowel. Accented vowels are
  therefore separate tokens in those languages (`dobar dan` → `d ô b a r d â n`), and `tone`
  stays 0.

### Stress and length

Two more layers, one value per phone, filled for every language where the G2P writes them:

| layer | values |
|---|---|
| `stress` | 0 none, 1 primary (`ˈ`), 2 secondary (`ˌ`), on the syllable nucleus |
| `length` | 0 short, 1 half-long (`ˑ`), 2 long (`ː`) |

Japanese long vowels and other length contrasts are written this way: `aː` is the token `a`
with `length` 2. Length has three values, not two, because the half-long mark is rare but real.

### Two tokens for one phoneme

The token lists are small, so a phoneme that is not in its group's list backs off: to its
nearest neighbour (`phoneme_label` then differs from `ipa_label`), to **two** tokens, or as a last resort to
`<unk>`. Affricates and diphthongs are the usual two-token cases: Mandarin `ʈ͡ʂ` → `ʈ ʂ`, English
/aɪ/ → `a ɪ`, /tʃ/ → `t ʃ`. Both tokens get their own timestamp, and both carry the phoneme's
tone, stress, length and word. If you need one interval per phoneme, merge them by
`target_seq_idx` and word. (This is also why the std models place more onsets than a corpus
labelled phone by phone: see [Evaluation](evaluation.md#why-the-std-models-score-lower).)

`phonemize_sentence(text)["n_folded"]` counts the tokens that fell back to `<unk>`.

### Word segmentation

The G2P takes one word in and gives one IPA string out, so a script that does not put spaces between
words must be segmented first. BFA does this automatically:

| language | segmenter | install |
|---|---|---|
| Mandarin (`zh`, `zh-Hant`) | longest match on the bundled dictionary keys | nothing |
| Cantonese (`yue`) | pycantonese (HKCanCor) | `[yue]` |
| Japanese (`ja`) | fugashi with UniDic morphemes, fed to the G2P as katakana | `[ja]` |
| Thai (`th`) | pythainlp `newmm` | `[th]` |

Each language has exactly one segmenter, so the same text always produces the same words on
every machine. A missing segmenter package raises once, when the language is set, and the error includes the install
line.

### Already-phonemized input

A segment that carries `gold_ph` (a standard_g2p `.gs.json` segment) is aligned from those
phonemes and is not phonemized again, so the G2P model is never loaded for it:

```python
gs = json.load(open("clip.gs.json"))
result = aligner.process_segments([{"segments": gs["segments"]}], [audio])
```

## The ph66 set

`phoneme_set="ph66"` phonemizes with espeak-ng (`pip install "bournemouth-forced-aligner[ph66]"`
plus the system `espeak-ng` package) into one 66-phoneme inventory shared by every language. The
language picks one of two models:

- English codes (`en-us`, `en`, `en-gb`, `en-029`, `en-gb-x-rp`, `en-gb-scotland`,
  `en-gb-x-gbclan`, `en-gb-x-gbcwmd`) → `CUPE-3-en`, the most accurate English model BFA has.
- Every other code → `CUPE-3-mls7`, trained on English plus German, Spanish, French, Italian,
  Dutch, Polish and Portuguese. For any other language, its phonemes are aligned by their nearest
  sounds in those, with a warning.

espeak-ng can phonemize 80+ languages into ph66. These are the codes the ph66 mapper covers:

```text
Germanic   de nl af lb da sv nb is
Romance    fr fr-be fr-ch es es-419 it pt pt-br ca an ro la
Slavic     pl cs sk sl hr bs sr mk bg ru ru-lv uk be
Celtic     ga gd cy        Baltic  lv lt        Uralic  et hu fi
Indic      hi bn ur pa gu mr ne as or si kok sd
Iranian    fa fa-latn ku   Hellenic el          Armenian hy hyw     Albanian sq
Turkic     tr az kk ky uz tt tk ba
Semitic    ar he am mt     Dravidian ta te kn ml
Other      ka eu qu id ms eo ia io lfn
```

ph66 has no tone, so it does not cover tonal languages (Chinese, Vietnamese, Thai, Burmese) or
Japanese and Korean. That is a limit of the inventory, and a different model would not fix it. Use the std set for
Mandarin, Cantonese and Japanese. The full espeak-ng code list is in
[espeak-ng's docs](https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md).

ph66 output uses the keys `ph66` / `pg16` where std uses `local_ph` / `gold_phg`, and it has no
`tone`, `stress` or `length`. A compound phoneme such as /ɑːɹ/ is split into two entries, and the second one
has `ipa_label: "-"`.
