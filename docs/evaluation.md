# Evaluation: how accurate are BFA's boundaries?

[Main README](../README.md) · [Developer guide](README.md) · [About the models](about-models.md) · **Evaluation** · [Language support](language-support.md)

This page reports how close BFA v2's phoneme onsets come to hand-labelled ones on two English
corpora, compared with the Montreal Forced Aligner (MFA). The harness that produced these numbers is
described in [`evaluation/README.md`](../evaluation/README.md). The raw scores are in
[`evaluation/results/`](../evaluation/results/), one CSV per corpus.

## Summary

- **One model per writing system, no dictionary.** `mbfa-latin` covers 36 trained languages
  (49 through its G2P) with one acoustic model and no pronunciation dictionary, and the std set
  has nine such models (82 languages). MFA needs a separate acoustic model and pronunciation
  dictionary for each language. Both corpora here are English, so neither measures this.
- **BFA's onsets are more precise.** On TIMIT the three CUPE-3 models beat both MFA models at
  10 ms (58.0-58.7 vs 55.2 and 56.5 F1), with a lower median error (7 ms vs 8 ms). Measured as
  distance from each hand-labelled boundary to the nearest predicted onset, all five BFA models
  are ahead of MFA up to 20 ms and level with `english_us_arpa` at 25 ms.
- **On TIMIT, MFA leads by 2-4 points at 25 ms** (78.6-80.9 vs 76.8 for `CUPE-3-en`). The gap
  comes from BFA's heavier tail of large misses and from how many onsets it places, not from
  where it places them.
- **Buckeye favours MFA by construction.** Its reference boundaries were first placed by an HMM
  forced aligner and then corrected by hand, and its phone sequences were picked from that aligner's ARPAbet
  dictionary. MFA is an HMM aligner, which is why it scores *higher* on noisy conversational
  Buckeye (84.3-85.3) than on clean TIMIT (78.6-80.9). See [Why Buckeye favours MFA](#why-buckeye-favours-mfa).
  TIMIT, labelled by hand, is the fairer comparison.
- **For English, the ph66 models beat the std ones.** `CUPE-3-en` is the best BFA model
  on Buckeye and `CUPE-3-en-base` on TIMIT. The multilingual `mbfa-latin` trails
  them by 4.2-5.8 points. Part of that gap comes from the scoring, which does not merge std's
  two-token diphthongs and affricates back into one phone (see [Why the std models score lower](#why-the-std-models-score-lower)).

## Setup

| | TIMIT | Buckeye |
|---|---|---|
| speech | read sentences, studio | conversational interviews |
| scored | all 6,300 clips (TRAIN + TEST) | all 255 recordings (40 speakers), cut into 33,670 segments of ~2.3 s |
| hand-labelled onsets | 223,864 | 840,116 |

Every model was run on the full corpus at BFA's decoder defaults (`presets.DECODER_DEFAULTS`):
minimum phone duration 6 frames, soft duration 10 frames, contrast window 33 frames, and onset
bonus 15 for every model. Each method was asked to align only the
words of the transcript. Two settings differ from the library defaults, and both are the
harness's choice:

- **Frame-grid onsets** (`sub_frame_onsets: false`): onsets sit on the 5 ms frame grid rather
  than at the interpolated posterior crossing that the library uses by default.
- **Batches of 16 segments**, each padded to 30 s (`duration_max: 30`). Scores move slightly with
  batch composition, because the class prior used by the decoder is taken over the batch.

MFA 3.4.3 ran with two pretrained English setups: the `english_us_arpa` acoustic model and
dictionary, and the `english_mfa` acoustic model with the `english_us_mfa` dictionary, each
speaker's segments in their own folder (per-speaker adaptation). On Buckeye they returned no
phones for a few segments, so they were scored on 838,709 and 839,016 boundaries rather than
840,116.

### How a boundary is scored

For every segment, the scorer takes the onsets of every predicted speech phone and the hand-labelled
onsets. It matches the two lists one-to-one in time order with an edit-distance alignment, and
counts a hit where a matched pair is within the tolerance. Recall, precision and F1 follow
from the hit count. No phoneme label is compared: a correctly timed boundary between the wrong
phones still counts. [`evaluation/LABEL_MAPPING.md`](../evaluation/LABEL_MAPPING.md) explains how
each corpus's labels become reference boundaries.

## Results

![Boundary F1 at 10, 25 and 50 ms for each method on TIMIT and Buckeye](../evaluation/plots/summary_f1.png)

### TIMIT

| method | phoneme set | F1 @10 | F1 @25 | F1 @50 | recall @25 | precision @25 | median error | predicted / labelled |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `CUPE-3-en` | ph66 | **58.7** | 76.8 | 86.8 | 74.1 | 79.6 | **7 ms** | 0.93 |
| `CUPE-3-en-base` | ph66 | 58.5 | 77.2 | 87.1 | 74.5 | 80.0 | **7 ms** | 0.93 |
| `CUPE-3-mls7` | ph66 | 58.0 | 76.0 | 86.4 | 73.4 | 78.8 | **7 ms** | 0.93 |
| `mbfa-latin` | std | 53.7 | 72.6 | 84.6 | 72.2 | 73.0 | 8 ms | 0.99 |
| `mbfa-latin-benchmark` | std | 55.4 | 74.2 | 85.4 | 73.9 | 74.6 | 8 ms | 0.99 |
| `mbfa-latin-noEnglish` | std | 53.6 | 72.1 | 84.0 | 71.7 | 72.4 | 8 ms | 0.99 |
| MFA `english_us_arpa` | ARPAbet | 55.2 | **80.9** | **90.4** | **77.0** | **85.1** | 8 ms | 0.90 |
| MFA `english_mfa` | IPA | 56.5 | 78.6 | 89.1 | 73.8 | 84.0 | 8 ms | 0.88 |

### Buckeye

| method | phoneme set | F1 @10 | F1 @25 | F1 @50 | recall @25 | precision @25 | median error | predicted / labelled |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `CUPE-3-en` | ph66 | 51.0 | 74.2 | 86.6 | 78.7 | 70.3 | 9 ms | 1.12 |
| `CUPE-3-en-base` | ph66 | 50.2 | 73.7 | 86.3 | 78.1 | 69.8 | 9 ms | 1.12 |
| `CUPE-3-mls7` | ph66 | 48.9 | 71.4 | 84.4 | 75.7 | 67.6 | 10 ms | 1.12 |
| `mbfa-latin` | std | 47.1 | 68.4 | 80.6 | 75.4 | 62.6 | 9 ms | 1.20 |
| `mbfa-latin-benchmark` | std | 47.9 | 69.4 | 81.3 | 76.5 | 63.5 | 9 ms | 1.20 |
| `mbfa-latin-noEnglish` | std | 46.9 | 68.1 | 80.4 | 75.1 | 62.3 | 9 ms | 1.20 |
| MFA `english_us_arpa` | ARPAbet | 60.6 | 84.3 | 92.1 | **87.3** | 81.5 | 8 ms | 1.07 |
| MFA `english_mfa` | IPA | **63.6** | **85.3** | **93.2** | 85.9 | **84.7** | **7 ms** | 1.01 |

*Median error* is over matched boundaries. *Predicted / labelled* is the number of predicted
onsets over the number of hand-labelled ones: above 1 means the method placed boundaries the
transcribers did not mark.

### How close is the nearest onset?

F1 is strict: each predicted onset may be matched to one hand-labelled boundary only, so an
extra onset costs precision, and a missing one costs recall. The figure below is the lenient
view. For each hand-labelled boundary, it takes the distance to the nearest predicted onset,
whether or not that onset is also the nearest one to another boundary.

![Share of hand-labelled boundaries with a predicted onset within t ms, per method](../evaluation/plots/summary_distance.png)

On TIMIT the five BFA models are ahead of both MFA models up to 20 ms (`CUPE-3-en`: 62.1% within
10 ms, against 55.0% and 55.4%), level with `english_us_arpa` at 25 ms, and within about a point
of it after that. On Buckeye, MFA is ahead by 5-8 points between 10 and 30 ms. Read together
with the tables: BFA puts its onsets where the boundaries are at least as well as MFA does on
read speech. Where it loses F1 is in how many onsets it places, against how many were labelled.

## Analysis

### Read speech: BFA is more precise, MFA has fewer large misses

On TIMIT, BFA's errors are smaller when it is right (F1 @10 58.7 vs 55.2-56.5, median 7 vs 8 ms).
MFA pulls ahead from 25 ms on, because BFA's tail is heavier: its mean absolute error is
14.1 ms (`CUPE-3-en`) against MFA's 12.6-12.7 ms.

TIMIT also labels most stops as two intervals, a closure and a release (`tcl t`). Both
aligners place one onset per stop from the text, so one TIMIT boundary in ten (22,624 of
225,282 speech-phone onsets) is a release that neither can match one-to-one. This caps recall
for every method alike. The onset bonus pulls stop onsets towards the burst. This scores higher
on boundary F1, and lower when each phone's own onset is what you need: see
[About the models](about-models.md#the-onset-bonus).

### Why Buckeye favours MFA

Buckeye's phone labels were not drawn from scratch. According to the
[corpus documentation](https://buckeyecorpus.osu.edu/php/corpusTran.php), the audio and
transcripts went first through the Entropic Aligner, an HMM forced aligner using phone models trained on TIMIT.
Research assistants picked the best-fitting pronunciation from that aligner's dictionary of variants. They then corrected
labels "placed at the wrong locations", using waveform and spectrogram displays. A boundary
the transcriber found acceptable kept the position the HMM aligner gave it.

That gives MFA three advantages on this corpus that have nothing to do with the audio:

1. **Same kind of aligner.** MFA is a GMM-HMM aligner. Where Buckeye's boundaries are the
   original aligner's output, they follow HMM placement conventions that MFA reproduces.
2. **Same pronunciations.** Buckeye's pronunciations were chosen from an ARPAbet variant
   dictionary, and MFA's dictionaries list variants too. BFA writes one pronunciation per word
   with a G2P (std) or espeak-ng (ph66), so its phone sequences agree less often with the
   reference's. This shows in the predicted/labelled ratio, 1.01-1.07 for MFA and 1.12-1.20 for
   BFA, and in BFA's 11-14 point precision deficit.
3. **The benchmark it is developed against.** MFA's own
   [English benchmarks](https://mfa-models.readthedocs.io/en/latest/benchmarks/english_alignments.html)
   are Buckeye and TIMIT.

Our own numbers are consistent with this. Buckeye's audio is far harder than TIMIT's
(conversational, interview recordings), and every BFA model scores 2.6-5.4 points lower on it
than on TIMIT. Both MFA models score *higher*, by 3.4 and 6.7 points. MFA also gains most at the
tightest tolerance (F1 @10 +5.4 and +7.1), which is what reproducing the reference's own
placement looks like. The phone set is not the reason: the IPA `english_mfa` gains more than the
ARPAbet model.

TIMIT was segmented by hand, and it is the better test of where an aligner puts boundaries.

Two further things hold BFA back on Buckeye, and both can be changed:

- **Reduced speech.** Conversational English drops and merges phones ("probably" → "prolly").
  Forced alignment places every phone of the text, so a phone the speaker skipped becomes an
  extra onset squeezed between real ones. A phonemizer that wrote reduced pronunciations would
  close part of the gap. A better acoustic model would not.
- **The onset bonus.** On the full corpus, turning the bonus off (`boundary_bonus_weight=0`)
  raises the CUPE-3 models' Buckeye F1 @25 by 1.6-3.2 points over their default of 15
  (`CUPE-3-en` 76.4 against 74.2), and costs them 2.0-3.7 points on TIMIT. On Buckeye the std
  models move by under 1 point either way (`mbfa-latin-benchmark` 70.0 without it, 69.4 at its
  default of 15).

`coverage_analysis["bad_alignment"]` flags the worst segments: under 1% of Buckeye segments for
the two English CUPE-3 models, and 4-8% for `CUPE-3-mls7` and the mbfa models.

### Why the std models score lower

The std models are multilingual (36 trained languages for `mbfa-latin`) and score 4.2-5.8 points below the
English-only CUPE-3 models. Two separate things are in that gap:

1. **Diphthongs and affricates are two tokens in std.** The std phoneme set writes /aɪ/ as
   `a ɪ` and /tʃ/ as `t ʃ`, so the std models place an onset inside every diphthong and
   affricate. ph66 has one token for each, as the corpora do. The extra onsets are the
   difference between std's predicted/labelled ratio and ph66's (0.99 vs 0.93 on TIMIT, 1.20 vs 1.12 on
   Buckeye). The scorer can merge them (`merge_two_token_phones: true` in `evaluate.yaml`), but
   it barely moves F1 (TIMIT 72.6 → 72.5, Buckeye 68.4 → 68.1 for `mbfa-latin`), because TIMIT
   itself splits stops and affricates and the second token often lands on the release. These
   tables leave it off, so every method is scored on the onsets it actually emits.
2. **Training data and noise.** `CUPE-3-en` heard 500 h of English. `mbfa-latin` heard 273 h
   across 36 languages, of which English is a small share, and was trained with more added
   noise (for robustness on real recordings). `mbfa-latin-benchmark`, trained with less noise,
   is 1.0-1.6 points ahead on the two corpora. `mbfa-latin-noEnglish`, trained with no English
   at all, is only 0.5 points behind `mbfa-latin` on TIMIT (72.1 vs 72.6 F1 @25) and 0.3
   on Buckeye (68.1 vs 68.4), so English data are a small part of that gap.

For English alone, use the ph66 set (`BFA(preset="en-us", phoneme_set="ph66")`). For anything
else, or for tone, stress and length labels, use std.

### Per phone class

Mean absolute error (ms) of matched onsets, by manner class. These columns are filled the same way
for every method (`mn_*` in the CSVs).

| | vowels | voiceless stops | voiced stops | voiceless fricatives | voiced fricatives | nasals | laterals | rhotics | glides |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **TIMIT** `CUPE-3-en` | 12.9 | 11.8 | 17.1 | 11.5 | 15.0 | 14.1 | 20.6 | 17.7 | 21.0 |
| TIMIT `mbfa-latin` | 16.6 | 12.7 | 15.9 | 12.9 | 16.6 | 14.2 | 19.5 | 19.3 | 17.4 |
| TIMIT MFA `english_us_arpa` | 11.3 | 11.1 | 14.6 | 13.4 | 12.7 | 10.8 | 17.6 | 18.7 | 16.7 |
| **Buckeye** `CUPE-3-en` | 15.5 | 15.3 | 12.9 | 12.5 | 14.0 | 13.5 | 22.2 | 22.2 | 17.8 |
| Buckeye `mbfa-latin` | 16.5 | 17.4 | 15.1 | 14.6 | 15.1 | 14.7 | 19.7 | 22.5 | 17.0 |
| Buckeye MFA `english_us_arpa` | 10.6 | 11.2 | 11.9 | 11.4 | 10.8 | 10.6 | 16.7 | 16.6 | 14.3 |

The hardest boundaries are the same for everyone: laterals, rhotics and glides, whose onsets
are gradual transitions with no acoustic edge. On TIMIT `CUPE-3-en` beats MFA on voiceless
fricatives and rhotics, and is within a millisecond on voiceless stops. Its largest deficits
are on nasals, voiced stops and voiced fricatives (2-3 ms each).

### Confidence

Mean per-phone confidence over each corpus (the model's peak posterior inside each phone's
segment):

| | `CUPE-3-en` | `CUPE-3-en-base` | `CUPE-3-mls7` | `mbfa-latin` | `mbfa-latin-benchmark` |
|---|---:|---:|---:|---:|---:|
| TIMIT | 0.54 | 0.40 | 0.26 | 0.25 | 0.17 |
| Buckeye | 0.43 | 0.32 | 0.21 | 0.20 | 0.16 |

The models are trained towards a target with deliberate entropy, so these numbers are far
below 1.0 even where the alignment is right, and they are on a different scale for each model. Compare
confidences within one model, never across models. See
[About the models](about-models.md#confidence).

### Cost

Measured on CPU (16 torch threads, one process, batches of 16), including phonemization:

| | per TIMIT clip (3.1 s) | full Buckeye (21.5 h of speech) |
|---|---:|---:|
| CUPE-3 models | ~1.0 s | 7.6-8.0 h |
| mbfa models | ~1.1-1.2 s | ~6.2 h (`mbfa-latin-benchmark`) |

The TIMIT figure is dominated by padding every clip to `duration_max` (30 s). With
`duration_max=10` the same clips take about a third of the time, with identical onsets.

## Per-run error histograms

Distance from each hand-labelled boundary to the nearest predicted onset (0-100 ms), one plot
per run. These are written by `compute_metrics.py`.

<details>
<summary>TIMIT (7 plots)</summary>

| | |
|---|---|
| ![](../evaluation/plots/timit__CUPE-3-en__b15_d6_s10_grid_cf33.png) | ![](../evaluation/plots/timit__CUPE-3-en-base__b15_d6_s10_grid_cf33.png) |
| ![](../evaluation/plots/timit__CUPE-3-mls7__b15_d6_s10_grid_cf33.png) | ![](../evaluation/plots/timit__mbfa-latin__b15_d6_s10_grid_cf33.png) |
| ![](../evaluation/plots/timit__mbfa-latin-benchmark__b15_d6_s10_grid_cf33.png) | ![](../evaluation/plots/timit__mfa-english_us_arpa-spk.png) |
| ![](../evaluation/plots/timit__mfa-english_mfa-spk.png) |  |

</details>

<details>
<summary>Buckeye (7 plots)</summary>

| | |
|---|---|
| ![](../evaluation/plots/buckeye__CUPE-3-en__b15_d6_s10_grid_cf33.png) | ![](../evaluation/plots/buckeye__CUPE-3-en-base__b15_d6_s10_grid_cf33.png) |
| ![](../evaluation/plots/buckeye__CUPE-3-mls7__b15_d6_s10_grid_cf33.png) | ![](../evaluation/plots/buckeye__mbfa-latin__b15_d6_s10_grid_cf33.png) |
| ![](../evaluation/plots/buckeye__mbfa-latin-benchmark__b15_d6_s10_grid_cf33.png) | ![](../evaluation/plots/buckeye__mfa-english_us_arpa-spk.png) |
| ![](../evaluation/plots/buckeye__mfa-english_mfa-spk.png) |  |

</details>

## What is not measured here

- **Languages other than English.** Neither corpus has non-English boundary labels, so
  the eight non-latin models (cyrillic, cjk, japanese, vietnamese, thai, abjad, brahmic,
  other_alphabetic) were selected on validation loss over their own alignments, and have no
  boundary score. `CUPE-3-mls7` is scored here on English only:
  the score shows how much English accuracy survived its multilingual epoch, not how well it aligns
  German or French.
- **Sub-frame onsets**, the library default. These runs use the 5 ms frame grid.

## Planned: a label-free duration test

Both corpora above score an aligner against human labels, and those labels carry their own
conventions. Buckeye's were seeded by an HMM aligner, and TIMIT splits a stop into closure and release.
Neither exists outside English. We will add a more objective test that needs no reference
labels:

1. **Align the original audio** with BFA as it is.
2. **Distort the audio.** Stretch or compress chosen phones by known amounts, and add noise.
3. **Align the distorted audio**, and check that the change in each phone's aligned duration
   matches the change that was applied.

The ground truth is the distortion itself, so it is known exactly and does not depend on any
transcriber's or aligner's conventions. The test also works in **any language** the aligner
supports, which gives the non-English models a boundary score for the first time.

## Reproducing

```bash
python evaluation/dataset/timit_parser.py       # ground truth, once
python evaluation/dataset/buckeye_parser.py
python evaluation/process_alignments.py         # align (the experiments in evaluate.yaml)
python evaluation/compute_metrics.py            # score into evaluation/results/results_<dataset>.csv
python evaluation/plot_summary.py               # the two comparison figures above
```

See [`evaluation/README.md`](../evaluation/README.md) for the configuration, run naming, caching
and parallel runs.
