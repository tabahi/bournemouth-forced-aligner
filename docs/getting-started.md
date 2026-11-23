# Getting started

## 1. Install

You need [Python](https://www.python.org/downloads/) 3.10 or newer. Then, in a terminal
(Command Prompt or PowerShell on Windows):

```bash
pip install bournemouth-forced-aligner
```

This works on Windows, macOS and Linux, and reads WAV, FLAC, OGG and MP3 files.

## 2. Align one recording

```bash
balign recording.wav "the words spoken in the recording" --preset=en-us --textgrid=recording.TextGrid
```

- `recording.wav` is your audio file.
- The text in quotes is what is said in it, written out in ordinary spelling.
- `--preset=en-us` is the language: American English here. See [step 3](#3-other-languages) for others.
- `--textgrid=recording.TextGrid` writes a TextGrid you can open in [Praat](https://www.fon.hum.uva.nl/praat/),
  with a `phonemes` tier and a `words` tier.

BFA also writes `recording.vs.json` next to the audio, with the same times and more detail.

The first run downloads the models, about 1.3 GB in all, once. After that BFA works offline.

To try it on a recording that comes with BFA, download
[butterfly.wav](../examples/samples/audio/109867__timkahn__butterfly.wav) and run:

```bash
balign butterfly.wav "butterfly" --preset=en-us --textgrid=butterfly.TextGrid
```

```text
  b   →    0 ms –  52 ms  (confidence: 0.89)
  ə   →   52 ms – 168 ms  (confidence: 0.09)
  t   →  168 ms – 258 ms  (confidence: 0.05)
  ɝ   →  258 ms – 351 ms  (confidence: 0.01)
  f   →  351 ms – 498 ms  (confidence: 0.28)
  ɫ   →  498 ms – 572 ms  (confidence: 0.11)
  a   →  572 ms – 826 ms  (confidence: 0.94)
  ɪ   →  826 ms – 982 ms  (confidence: 0.25)
```

Phonemes are written in IPA. The confidence says how sure the model is of each phoneme. Compare
it between phonemes of one recording, not as an absolute score: a correctly placed phoneme often
reads below 0.5 ([why](about-models.md#confidence)).

## 3. Other languages

Change `--preset`:

```bash
balign aufnahme.wav "guten Morgen" --preset=de --textgrid=aufnahme.TextGrid
```

| | | | |
|---|---|---|---|
| `en-us` English (US) | `en-GB` English (UK) | `de` German | `fr` French |
| `es` Spanish | `pt` Portuguese | `it` Italian | `nl` Dutch |
| `ru` Russian | `uk` Ukrainian | `pl` Polish | `tr` Turkish |
| `zh` Mandarin | `yue` Cantonese | `ja` Japanese | `ko` Korean |
| `vi` Vietnamese | `th` Thai | `ar` Arabic | `fa` Persian |
| `hi` Hindi | `ur` Urdu | `el` Greek | `sw` Swahili |

That is a sample. All 82 languages, and what to know about each (tone, word segmentation), are
on the [language support](language-support.md) page, or run `balign --list-presets`.
Japanese, Cantonese and Thai need one extra install step, listed [below](#optional-extras).

## 4. A long recording

For a recording with many sentences, such as an interview, put the transcript in a JSON file.
Give each stretch of speech its start and end time in seconds:

```json
{"segments": [
  {"start": 0.0, "end": 4.2, "text": "so where did you grow up"},
  {"start": 4.6, "end": 9.8, "text": "in a small town near the coast"}
]}
```

```bash
balign interview.wav interview.json --preset=en-us --textgrid=interview.TextGrid
```

The times don't need to be exact, but each stretch must contain all of its words (a little
silence either side is fine). BFA finds the phonemes inside each stretch.
[Whisper](https://github.com/openai/whisper) and most speech-to-text tools write this format
(with extra fields, which BFA ignores).

## 5. From Python

```python
from bournemouth_aligner import BFA

aligner = BFA(preset="en-us")
audio = aligner.load_audio("butterfly.wav")
result = aligner.process_sentence("butterfly", audio)

for ph in result["segments"][0]["phoneme_ts"]:
    print(ph["ipa_label"], round(ph["start_ms"]), round(ph["end_ms"]), round(ph["confidence"], 2))

aligner.convert_to_textgrid(result, output_file="butterfly.TextGrid")
```

For a transcript file: `aligner.process_srt_file("interview.json", "interview.wav", ts_out_path="interview.vs.json")`.
More in [examples/](../examples/) and the [developer guide](developer-guide.md), which also explains
every field of the JSON output.

### Many files at once

Aligning all your files in one call is much faster than one call per file. With a GPU, also set
`n_jobs=4`: the alignment search then runs in 4 worker processes while the GPU works on the next
batch. The output is the same.

```python
import json
from bournemouth_aligner import BFA

if __name__ == "__main__":                    # needed when n_jobs > 1
    aligner = BFA(preset="en-us", n_jobs=4)
    names = ["interview1", "interview2"]
    transcripts = [json.load(open(n + ".json")) for n in names]
    audio = [aligner.load_audio(n + ".wav") for n in names]

    results = aligner.process_segments(transcripts, audio)    # one call for every file
    for name, result in zip(names, results):
        aligner.convert_to_textgrid(result, output_file=name + ".TextGrid")
```

## Tips for good results

- **The text must match the audio.** Every phoneme of the text is placed somewhere, so a word that
  isn't spoken gets squeezed into its neighbours. Segments that went badly are flagged in the
  JSON as `coverage_analysis["bad_alignment"]`.
- **Write numbers as words.** "twenty five", not "25". The same goes for abbreviations and symbols.
- **Shorter stretches align better.** A stretch of up to 2 minutes is aligned in one pass, and
  longer ones are handled automatically, but a transcript split into sentences works best.
- **The phonemes come from a dictionary-style pronunciation of the text**, not from what the
  speaker actually said. Reduced or dialectal pronunciations are aligned against the standard form.
- **Conversational vs read speech.** The default setting is tuned for conversational speech. For
  each phoneme's own onset, rather than the boundaries as a whole, try `--boundary-bonus-weight=0`.

## Optional extras

```bash
pip install "bournemouth-forced-aligner[ja]"     # Japanese word segmentation (fugashi + unidic-lite)
pip install "bournemouth-forced-aligner[yue]"    # Cantonese word segmentation (pycantonese)
pip install "bournemouth-forced-aligner[th]"     # Thai word segmentation (pythainlp)
pip install "bournemouth-forced-aligner[ph66]"   # the espeak-ng phoneme set; also needs the espeak-ng program
pip install "bournemouth-forced-aligner[audio]"  # other audio formats (m4a, aac, mp4); also needs FFmpeg
```

The latest code from GitHub: `pip install git+https://github.com/speech-industry/bournemouth-forced-aligner.git`.

## Phoneme sets

BFA has two phoneme sets, each with its own models:

| `phoneme_set` | Pronunciations from | Models | Languages |
|---|---|---|---|
| **`std`** (default) | [standard_g2p](https://github.com/tabahi/CharsiuG2P), a neural G2P model | one per writing system: `mbfa-latin`, `mbfa-cyrillic`, `mbfa-cjk`, `mbfa-japanese`, `mbfa-vietnamese`, `mbfa-thai`, `mbfa-abjad`, `mbfa-brahmic`, `mbfa-other-alphabetic` | 82, with tone, stress and length labels |
| `ph66` | espeak-ng, reduced to 66 phonemes | `CUPE-3-en` (English), `CUPE-3-mls7` (everything else) | 80+, trained on English and 7 European languages |

```bash
balign recording.wav "hello world" --preset=en-us --phoneme-set=ph66   # the most accurate English model
balign recording.wav "<Bengali text>" --preset=bn --phoneme-set=ph66   # Bengali: only ph66 has it so far
```

## Settings worth knowing

| Python / command line | |
|---|---|
| `boundary_bonus_weight` / `--boundary-bonus-weight` | How strongly boundaries are pulled towards sudden changes in the sound. The default, 15, is set for conversational speech. 0 gives better onsets for individual phonemes. |
| `duration_max=10` | Up to 2.6× faster on short clips, same result. Long segments are never cut. |
| `n_jobs=4` | Run the alignment search in 4 worker processes when one call aligns many segments (`process_segments`, `process_srt_file`). With a GPU: 2.4× faster on long recordings. Little effect on a CPU. Same output; your script needs `if __name__ == "__main__":`. |
| `low_confidence_level` | The confidence below which a phoneme counts as doubtful when flagging bad segments. |
| `punctuation_sil=False` | Stop writing a pause at commas and full stops (on by default; default phoneme set only). Without it, a pause is swallowed by the phone next to it. |
| `decoder={"word_sil": False}` | Offer a pause only at punctuation, as 2.0 did. Since 2.1 a pause may go between any two words, and breath and clicks around the speech are labelled `noise` instead of being stretched into a phone. |
| `fill_nonspeech=False` | Leave what no phone covers as gaps instead of `SIL` / `noise` entries (marked `"inserted": true`). |
| `warn_level=0` | Print nothing, downloads included. |

All options: `balign --help`, and the [developer guide](developer-guide.md).

## Upgrading from 1.x

BFA 2.0 is not compatible with 1.x. The acoustic models are new: no CTC, no blank class, 5 ms
frames, and a multilingual model family that is now the default. The alignment method has
changed too: a segmental Viterbi, the same decoder the models were trained with, places each
boundary where two phoneme posteriors cross, instead of snapping CTC spikes onto onsets. v1
models do not load, and the output and constructor arguments have changed: see the
[CHANGELOG](../CHANGELOG.md). To keep using 1.x, pin it: `pip install "bournemouth-forced-aligner<2"`.
