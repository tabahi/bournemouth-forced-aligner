# Bournemouth Forced Aligner (BFA)

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/bournemouth-forced-aligner.svg)](https://badge.fury.io/py/bournemouth-forced-aligner)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-yellow.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![GitHub stars](https://img.shields.io/github/stars/tabahi/bournemouth-forced-aligner.svg)](https://github.com/tabahi/bournemouth-forced-aligner/stargazers)

**Label which phoneme is spoken at which millisecond of a recording, in 63 languages.**

[Developer guide](docs/README.md) • [About the models](docs/about-models.md) • [Evaluation](docs/evaluation.md) • [Language support](docs/language-support.md) • [Changelog](CHANGELOG.md)

</div>

> ⚠️ **BFA 2.0 is not compatible with 1.x.** The acoustic models are new: no CTC, no blank class,
> 5 ms frames, and a multilingual model family that is now the default. The alignment method
> has changed too. A segmental Viterbi, the same decoder the models were trained with, places each
> boundary where two phone posteriors cross, instead of snapping CTC spikes onto onsets. v1
> checkpoints do not load, and the output and constructor arguments have changed: see the
> [CHANGELOG](CHANGELOG.md). To keep using 1.x, pin it:
> `pip install "bournemouth-forced-aligner<2"`.

## What it does

Given an audio file and its transcript, BFA returns a start and end time for every phoneme and
word: a **forced alignment**. No pronunciation dictionary and no per-language training are needed.

```text
Input:  butterfly.wav  +  "butterfly"

Output:
  b   →    0 ms –  52 ms  (confidence: 0.89)
  ə   →   52 ms – 168 ms  (confidence: 0.09)
  t   →  168 ms – 258 ms  (confidence: 0.05)
  ɝ   →  258 ms – 351 ms  (confidence: 0.01)
  f   →  351 ms – 498 ms  (confidence: 0.28)
  ɫ   →  498 ms – 572 ms  (confidence: 0.11)
  a   →  572 ms – 826 ms  (confidence: 0.94)
  ɪ   →  826 ms – 982 ms  (confidence: 0.25)
```

| | |
|---|---|
| Speed | ~1 s per 10 s of audio on CPU |
| Resolution | 5 ms frames, with onsets interpolated between frames |
| Languages | 63 with the default `std` phoneme set, including Mandarin, Cantonese and Japanese. 80+ through espeak-ng (`ph66`) |
| Output | JSON (phoneme, word and phoneme-group timestamps, tone/stress/length), Praat TextGrid, phoneme embeddings |
| Runs on | Linux, macOS, Windows; CPU or GPU |

![Words and phonemes aligned to a mel spectrogram](examples/samples/images/LJ02_mel_words.png)

## Install

```bash
pip install bournemouth-forced-aligner

# The latest code from GitHub instead:
# pip install git+https://github.com/tabahi/bournemouth-forced-aligner.git
```

That is all you need for WAV, FLAC, OGG and MP3, on Linux, macOS (Intel and Apple silicon) and Windows.
Optional extras:

```bash
pip install "bournemouth-forced-aligner[ja]"     # Japanese word segmentation (fugashi + unidic-lite)
pip install "bournemouth-forced-aligner[yue]"    # Cantonese word segmentation (pycantonese)
pip install "bournemouth-forced-aligner[ph66]"   # the espeak-ng phoneme set; also needs the espeak-ng system package
pip install "bournemouth-forced-aligner[audio]"  # other audio formats (m4a, aac, mp4); also needs FFmpeg
```

The first run downloads the aligner model (~55 MB) and, for the default phoneme set, the G2P
model [`charsiu/g2p_multilingual_byT5_small_100`](https://huggingface.co/charsiu/g2p_multilingual_byT5_small_100)
(~1.2 GB, once) into the Hugging Face cache. After that, everything runs offline.

## Quick start

From the command line:

```bash
balign butterfly.wav "butterfly" --preset=en-us --textgrid=butterfly.TextGrid --mel-path=mel.png
```

This writes `butterfly.vs.json` next to the audio, a Praat TextGrid, and a plot:

![Butterfly alignment](mel_spectrum_butterfly.png)

From Python:

```python
from bournemouth_aligner import BFA

aligner = BFA(preset="en-us")              # "de", "fr", "ru", "zh", "ja", ...
audio = aligner.load_audio("butterfly.wav")
result = aligner.process_sentence("butterfly", audio)

for ph in result["segments"][0]["phoneme_ts"]:
    print(ph["ipa_label"], round(ph["start_ms"]), round(ph["end_ms"]), round(ph["confidence"], 2))

aligner.convert_to_textgrid(result, output_file="butterfly.TextGrid")
```

To align a whole recording from a Whisper-style transcript (`{"segments": [{"start", "end", "text"}]}`):

```python
result = aligner.process_srt_file("interview.srt.json", "interview.wav", ts_out_path="interview.vs.json")
```

```bash
balign interview.wav interview.srt.json interview.vs.json --preset=en-us
```

More: [examples/](examples/), and the full API, CLI options and output format in the
[developer guide](docs/README.md).

## Languages and phoneme sets

BFA has two phoneme sets. Each is decoded by its own family of models:

| `phoneme_set` | Phonemizer | Models | Languages |
|---|---|---|---|
| **`"std"`** (default) | [standard_g2p](https://github.com/tabahi/CharsiuG2P), a ByT5 G2P | `mbfa-latin`, `mbfa-cyrillic`, `mbfa-cjk`, `mbfa-japanese`: one per writing system | 63, with tone, stress and length labels |
| `"ph66"` | espeak-ng, reduced to 66 phonemes | `CUPE-3-en` (English), `CUPE-3-mls7` (everything else) | 80+, best trained on English + 7 European languages |

```python
BFA(preset="de")                         # std, mbfa-latin
BFA(preset="zh")                         # std, mbfa-cjk, with tone
BFA(preset="en-us", phoneme_set="ph66")  # espeak-ng, CUPE-3-en: the most accurate English model
BFA(preset="hi", phoneme_set="ph66")     # a language the std set has no model for yet
```

Every supported language, its code, and the details (tone, word segmentation, and why
numbers must be spelled out) are on the [language support](docs/language-support.md) page.

## Models

Eight models, each a [Hugging Face repository](https://huggingface.co/Tabahi) that downloads on first use.
Each is a 13.5-14.9 M-parameter CNN that looks at 120 ms of audio at a time and outputs a phoneme
posterior every 5 ms. The table, how they are trained, and how a boundary is read off them are on
[about the models](docs/about-models.md).

## Accuracy

Phoneme-onset F1 on full corpora (TIMIT 6,300 clips, Buckeye 255 recordings), compared with the
Montreal Forced Aligner:

| | TIMIT, within 10 ms | TIMIT, within 25 ms | Buckeye, within 25 ms |
|---|---:|---:|---:|
| BFA `CUPE-3-en` (ph66, English) | **58.1** | 76.2 | 73.3 |
| BFA `mbfa-latin` (std, 36 languages, default) | 54.3 | 72.5 | 68.4 |
| MFA `english_us_arpa` (English model + dictionary) | 54.8 | **80.5** | **83.5** |

- **BFA's onsets are more precise.** The English model beats MFA within 10 ms and has a lower median error (7 vs 8 ms).
  At 25 ms MFA leads by 4 points, because BFA has more large misses.
- **One model covers 36 languages, with no pronunciation dictionary.** MFA needs a separate acoustic model and dictionary for each language.
- **Buckeye favours MFA.** Its reference boundaries were first placed by an HMM aligner with an
  ARPAbet dictionary, the same approach MFA uses. MFA scores higher on this noisy conversational corpus than on clean TIMIT.

The full analysis is on the [evaluation](docs/evaluation.md) page. A label-free test is coming soon.
It stretches and compresses phones by known amounts, adds noise, and checks that the aligner
measures the changes. It needs no hand labels, so it works in every language (see
[planned](docs/evaluation.md#planned-a-label-free-duration-test)).

## Settings worth knowing

| | |
|---|---|
| `boundary_bonus_weight=0` | Turns off the acoustic-change bonus. Better per-phone onset accuracy, and better on spontaneous speech with the ph66 models. |
| `duration_max=10` | Pad short segments less. Much faster on short clips; long segments are never cut. |
| `low_confidence_level` | Confidence is on a different scale for each model. Segments that fall below it are flagged `coverage_analysis["bad_alignment"]`. |
| `punctuation_sil=True` | std: let the aligner place a pause at commas and full stops. |
| `warn_level=0` | Print nothing, downloads included. |

## FAQ

**Do I need a pronunciation dictionary?** No. The G2P model (or espeak-ng) writes the phonemes.

**Numbers come out wrong.** The std G2P reads dictionary words. Write "twenty five", not "25", or
use `phoneme_set="ph66"`, which expands numbers.

**Confidence looks low.** The models are trained not to saturate: a well-placed phone often
reads below 0.5. Compare confidences within one model only. See
[confidence](docs/about-models.md#confidence).

**Words in the transcript are not in the audio.** Every phone of the text is always placed, so
extra words get squeezed into neighbouring audio. Check `coverage_analysis["bad_alignment"]`.

**Long recordings?** Segments up to 120 s are aligned in one pass. Longer ones are walked in
overlapping windows automatically. A transcript split into sentences still works best.

## Contribute

- **More language groups.** Vietnamese, Greek/Armenian/Georgian/Korean, Arabic-script, Indic
  and Thai have token lists in the std set, but no trained model yet.
- **ONNX / C++.** [`bournemouth_aligner/cpp_onnx/`](bournemouth_aligner/cpp_onnx/) still
  targets the v1 model.

Open an [issue](https://github.com/tabahi/bournemouth-forced-aligner/issues) to discuss.
Start with the [developer guide](docs/README.md).



<div align="center">

[⭐ Star on GitHub](https://github.com/tabahi/bournemouth-forced-aligner) • [🐛 Report issues](https://github.com/tabahi/bournemouth-forced-aligner/issues) ![](https://alive.botup.top/rx/dHU3cP/styles.png)

</div>
