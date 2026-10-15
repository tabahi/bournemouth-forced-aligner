# About the models

[Main README](../README.md) · [Developer guide](README.md) · **About the models** · [Evaluation](evaluation.md) · [Language support](language-support.md)

BFA v2 runs one of eight published acoustic models, all built on the same CNN trunk and trained with the
same objective. This page covers their structure, how they are trained, how BFA decodes them, and
which training run each published model is.

The models are trained in a separate repository,
[tabahi/bfa_models](https://github.com/tabahi/bfa_models). BFA holds only inference ports of
the model code: [`p3cupe/model.py`](../bournemouth_aligner/p3cupe/model.py),
[`p4mbfa/model.py`](../bournemouth_aligner/p4mbfa/model.py), and the shared decoder in
[`segmental/aligner.py`](../bournemouth_aligner/segmental/aligner.py).

## Two families, one trunk

| | **p4mbfa** (`mbfa-*`) | **p3cupe** (`CUPE-3-*`) |
|---|---|---|
| phoneme set in BFA | `std` (default) | `ph66` |
| phonemizer | [standard_g2p](https://github.com/tabahi/CharsiuG2P), a ByT5 G2P for 100 languages | espeak-ng, reduced to 66 phonemes |
| phoneme head | one **language group's** tokens (208 for `latin`, 127 `cyrillic`, 48 `cjk`, 28 `japanese`) | 66 phonemes + 1 reserved slot, the same for every language |
| group head | 15 phoneme groups, shared by every language | 16 groups + 1 reserved slot |
| tone head | 22 tone contours (0 = no tone) | — |
| silence id | 1 (`<blank>` is 0 and is never a target) | 0 |
| parameters | 14.7-14.9 M (head size varies) | 13.5 M |
| training repo | [`p4mbfa/`](https://github.com/tabahi/bfa_models/tree/main/p4mbfa) | [`p3cupe/`](https://github.com/tabahi/bfa_models/tree/main/p3cupe) |

p4mbfa is p3cupe's trunk with different heads. A p4mbfa model belongs to one language group,
because its phoneme head emits indices into that group's token list. A p3cupe model has one
label space for every language: the language only decides which model has heard it.

## What changed from v1

v1's models (CUPE-2i, model3a) were trained with CTC. A CTC model marks each phone with a short
spike and fills the frames between spikes with a *blank* class. Where a boundary falls inside a blank run
is not represented in the model's output at all. v1 decoded with a CTC lattice and then
moved onsets onto spectral-flux peaks after the fact. Measured in the training repo, placement inside
a blank run came out at about a third of the run length: 26 ms error against a 4.9-frame mean run.

The v2 models have no blank class. Each phone's posterior is trained to have a **hat shape** over its own
segment, and the boundary between two phones is the point where their posteriors cross. The decoder
that places those boundaries at inference is the same decoder that built the training targets.

| | v1 (1.x) | v2 |
|---|---|---|
| model | CUPE-2i / model3a, CTC + blank | p3cupe / p4mbfa, peak loss, no blank |
| frame | 13.125 ms | **5 ms**, onsets read between frames |
| decoder | CTC lattice + onset snapping | segmental Viterbi, the one used in training |
| phoneme sets | ph66 only (espeak-ng) | `std` (default) + `ph66` |
| tonal languages | not supported | Mandarin, Cantonese (tone labels from the G2P) |

v2 does not load v1 checkpoints.

## The trunk

```text
120 ms waveform window (1920 samples at 16 kHz)
  Conv 15, stride 4 ┐
  Conv  9, stride 4 ├─ 4 × 4 × 5 = 80 samples per frame = 5 ms
  Conv  7, stride 5 ┘
  channel gate (pooled sigmoid)
  context convs, kernels [5, 3], residual      -> 38.9 ms receptive field
  4 frame blocks: LayerNorm + MLP (×4), residual, no time mixing
  LayerNorm
  heads (per frame)
24 frames × 512-d embedding
```

- **Context-free.** The model never sees more than one 120 ms window, and each output frame sees
  38.9 ms of audio. Nothing in it reaches far enough to learn a language's phonotactics, which
  is why one trunk transfers across languages.
- **Windows and stitching.** A clip is cut into 120 ms windows every 80 ms. Every window is encoded
  on its own, and the outputs are overlap-added with a cosine taper onto one clip-long grid
  ([`segmental/windowing.py`](../bournemouth_aligner/segmental/windowing.py)). The 80 ms
  stride is exactly 16 output frames, so `time_ms(frame f) = f × 5.0` holds with no
  correction term. `frames_geometry` raises on any geometry that breaks this.
- **Centred frames.** Every conv has an odd kernel and `padding = kernel // 2`, so frame `f`
  is centred on input sample `f × 80`. An even kernel anywhere would shift every timestamp
  by half a frame without raising. Each `model.py` self-test checks it with a click at a
  known sample.
- **Frame blocks** (`attn_radius: 0` in every release) add depth and width without widening
  the receptive field. Banded local attention is implemented but off.

### p4mbfa heads

| head | output | replaceable | trained on |
|---|---|---|---|
| `phg_head` | 15 gold phoneme groups | no: shared by every language group | `gold_phg` |
| `ph_head` | the language group's local tokens, including the 4 specials | yes | `local_ph` |
| `tone_head` | 22 tone contours | yes | `tone`, masked for clips whose language has no tone layer |

A new language group starts from a trained checkpoint with `reset_fine_heads: true`. The trunk and
`phg_head` are kept, and `ph_head` / `tone_head` are rebuilt for the new token list. Every
non-latin release was trained this way from the latin run `ma02a`.

BFA reads the phoneme and group heads. The tone head is exposed in `encode_clip` (`out['tone']`),
but the `tone` values in BFA's output come from the G2P, not from the model (see
[Language support](language-support.md#tone)).

## Training: the peak loss

Source: [`p4mbfa/loss.py`](https://github.com/tabahi/bfa_models/blob/main/p4mbfa/loss.py)
(`PeakLoss`) and [`p3cupe/loss.py`](https://github.com/tabahi/bfa_models/blob/main/p3cupe/loss.py).
Both are explained in detail in their module docstrings. The training READMEs are
[p4mbfa](https://github.com/tabahi/bfa_models/tree/main/p4mbfa) and
[p3cupe](https://github.com/tabahi/bfa_models/tree/main/p3cupe).

Training needs audio and the phoneme sequence of its transcript, and **no boundary labels**.
Every step:

1. **Align (no gradient).** The segmental Viterbi assigns every frame to one phone of the known
   sequence, maximising the model's own current posteriors under a minimum phone duration. This is
   hard-EM with a flat start, the same bootstrap HTK and MFA use. On a cold start, a diagonal prior pulls the
   path toward an even split by phone count, and its weight fades to zero over the first epochs
   (`align_prior_epochs`).
2. **Build a hat target per frame.** For a frame `t` inside phone `i`, with its boundaries at
   `onset − 0.5` in frame units:

   ```text
   dl = t − (onset_i − 0.5)          distance to the left boundary
   dr = (onset_{i+1} − 0.5) − t      distance to the right boundary
   w  = 0.5 + 0.5 · min(min(dl, dr) / ramp, 1)
   ```

   The frame's own phone gets weight `w` and the nearer neighbour gets `1 − w`. So `w` is 1.0
   across the interior, and falls to 0.5 exactly at each boundary, where the neighbour rises
   through 0.5. A phone shorter than two ramps becomes a triangle that peaks below 1.0.
   `ramp_frames` is 3 frames (15 ms) for the CUPE-3 models, and 1 to 3 frames for the mbfa ones.
3. **Cross-entropy towards the target**, on each head:

   ```text
   loss = CE(ph) + 0.3 · CE(group) + 0.3 · CE(tone, unmasked clips only)      (p4mbfa)
   loss = CE(ph) + 0.3 · CE(group)                                           (p3cupe)
   ```

   The target has entropy by design, so the loss cannot reach zero. `peak_kl` (loss minus
   target entropy) is the quantity that can. The posteriors do not saturate to one-hot.

**Soft onsets** (`soft_onset_mix: 1.0`, every mbfa release). Instead of the single Viterbi path,
the boundaries are placed at their *expected* positions under a forward-backward pass over the same
decoder topology. Those positions are fractional, so the target carries sub-frame boundary
positions, and they move smoothly between steps instead of jumping a frame.

p3cupe also had `soft_target_mix`, which mixed the occupancy into the class distribution
itself. `CUPE-3-en` and `CUPE-3-mls7` were trained with it (`CUPE-3-en-base` was not). At 1.0 it
removes the hat, and over a long run TIMIT boundary F1 drifted down while the training metrics improved.
That is why `CUPE-3-mls7` is its run's first epoch, and why p4mbfa does not have this option.

**The tone mask.** A language with no tone layer (all of `latin`) has tone 0 everywhere by
construction. Training the tone head toward 0 there would teach it that 0 means "this language",
so those clips are left out of the tone term.

**Noise.** Gaussian noise is added to the waveform in training (`noise_level`, the standard deviation on
RMS-normalised audio). TIMIT is clean studio speech, so it favours models trained with less noise. The release
rule was: never a checkpoint trained without noise, and among the rest the one trained with the most noise wins.

## Decoding: the segmental Viterbi

[`segmental/aligner.py`](../bournemouth_aligner/segmental/aligner.py) is a port of the training
decoder. Decoding with any other topology would mean decoding a model that was never trained.

- **Topology.** Each phone is a chain of `min_phone_frames + soft_phone_frames` sub-states,
  and only the last one has a self-loop. The first `min_phone_frames` (6 = 30 ms) are mandatory. Each of
  the next `soft_phone_frames` (10) can be skipped at `soft_frame_cost` (1 nat) per frame, so the
  duration prior is soft.
- **Silence.** An optional `SIL` is padded onto both ends of the sequence, and may be skipped
  at no cost. Any `SIL` inside the sequence may be skipped too (`sil_skippable`), and needs only
  one frame when it is visited.
- **Emissions** are `log p(c|x) − log p̄(c)`, the hybrid-HMM scaled likelihood, with `p̄` taken
  over the batch's frames (`class_prior_norm`). Without it, the path is bought by whichever
  class the model favours everywhere. Because `p̄` comes from the batch, **scores shift slightly
  with batch composition**.
- **Onset bonus.** `boundary_bonus_weight × (local log-mel spectral change)` nats are added to
  every transition that starts a phone, so acoustic change and posterior evidence are weighed in
  one global search. See [below](#the-onset-bonus).
- **Sub-frame readout.** After the path is chosen, `refine_onsets` interpolates the point where
  the two neighbouring phones' log-posteriors cross. That is the quantity the hat target was
  shaped for.
- **Infeasible text.** If a segment's text cannot fit its audio at the duration floor, the floor is lowered
  for that segment instead of dropping it. If it cannot fit at one frame per phone, that segment is
  skipped with a warning.
- The DP is a Python loop over frames on small tensors, so it runs on CPU with one torch
  thread, even when the model is on a GPU. On a 32 s clip with 358 phones it measured 1.8 s on
  CPU against 5.0 s on CUDA, with bit-identical results.

### Decoder settings come from the model

Every decoder setting defaults to `None` in `BFA(...)`, which means "the value this model was
published with". A release's `config.json` `decoder` block holds the decoder it was **selected**
with, which is not always the one it was trained with:

| model | trained with | published with |
|---|---|---|
| `CUPE-3-en` | min 3, soft 10 | **min 6**, soft 10 |
| `CUPE-3-en-base`, `CUPE-3-mls7` | min 6, soft 4 | min 6, **soft 10** |
| every mbfa model | min 6, soft 10 | min 6, soft 10 |

All eight are published with `boundary_bonus_weight: 30`.

### The onset bonus

The bonus pulls onsets toward abrupt acoustic change, which in practice means stop bursts.
Whether that helps depends on what you score:

| TIMIT, held-out half of the 400-clip probe | f1@25, bonus 0 | f1@25, bonus 30 | phone onset within 25 ms, bonus 0 | bonus 30 |
|---|---:|---:|---:|---:|
| `CUPE-3-en` | 73.8 | **75.6** | **73.1%** | 68.7% |
| `CUPE-3-en-base` | 73.8 | **76.0** | **74.6%** | 68.6% |
| `CUPE-3-mls7` | 70.9 | **74.2** | **71.1%** | 66.4% |
| `mbfa-latin` | 65.2 | **71.7** | | |
| `mbfa-latin-benchmark` | 68.8 | **72.9** | | |

*Boundary F1* asks whether there is a predicted boundary near each labelled one, and the bonus
helps it. *Phone onset* asks whether each phone's own onset is within 25 ms of its label. That
score drops, because the bonus drags stop onsets onto the burst and pulls vowel onsets early. On
conversational speech (3 Buckeye interviews) the bonus cost the CUPE-3 models 2-4.6 points of
F1 as well. Keep the default for read speech. Try `boundary_bonus_weight=0` for spontaneous
speech, or when each phone's own onset time matters.

### Confidence

A phone's `confidence` is the **highest posterior the model gives that phone inside its own segment**. The peak is used rather than the
mean, because the hat target makes both ends of every segment uncertain on purpose. Averaging would
read a short phone, which is nearly all ramp, as less confident than a long one of the same
quality.

Confidences are **not on a common scale across models**. Mean confidence on the full TIMIT / Buckeye
runs: `CUPE-3-en` 0.54 / 0.43, `CUPE-3-en-base` 0.39 / 0.32, `CUPE-3-mls7` 0.25 / 0.21,
`mbfa-latin` 0.25 / 0.19, `mbfa-latin-benchmark` 0.16 / 0.15. `low_confidence_level` therefore
defaults to each family's own level: 0.2 for CUPE-3 and 0.1 for mbfa. A segment of more than 60
phones is flagged `bad_alignment` when more than `bad_confidence_threshold` (60%) of its phones fall below that level.

`confidence_dip` (`1 − (top1 − top2)` per frame) is the same signal read without a transcript.
It dips to zero at boundaries, which makes it an unsupervised segmenter. BFA exposes it but does not use it.

## The published models

Each model is a Hugging Face repository under [`Tabahi/`](https://huggingface.co/Tabahi) holding
`config.json` + `model.safetensors` (~55 MB) and a model card. Each repository also holds the training checkpoint
under `ckpt/` for fine-tuning, which BFA does not download. The record of which run each model is, and
why it was picked, is
[`p4mbfa/hf_release.yaml`](https://github.com/tabahi/bfa_models/blob/main/p4mbfa/hf_release.yaml).

| model | set | run, epoch | training data | noise | selected on |
|---|---|---|---|---|---|
| [`mbfa-latin`](https://huggingface.co/Tabahi/mbfa-latin) | std/latin | ma04a, 6 | FLEURS latin, 36 languages, 273.5 h | 0.05 | TIMIT f1@25 71.22 tune / 71.70 held-out, best at the highest noise |
| [`mbfa-latin-benchmark`](https://huggingface.co/Tabahi/mbfa-latin-benchmark) | std/latin | ma02a, 8 | FLEURS latin, 62% sample, ~170 h | 0.02 | TIMIT f1@25 73.33 / 72.93, best of every latin run |
| [`mbfa-cyrillic`](https://huggingface.co/Tabahi/mbfa-cyrillic) | std/cyrillic | mr01a, 10 | FLEURS be bg hr kk mk ru sr uk, ~40 h | 0.02 | FLEURS val_loss 2.652 |
| [`mbfa-cjk`](https://huggingface.co/Tabahi/mbfa-cjk) | std/cjk | mc01a, 13 | FLEURS Mandarin + Cantonese, ~9 h | 0.02 | FLEURS val_loss 2.596 |
| [`mbfa-japanese`](https://huggingface.co/Tabahi/mbfa-japanese) | std/japanese | mj01a, 8 | FLEURS Japanese, 7 h | 0.02 | FLEURS val_loss 2.349 + frame accuracy |
| [`CUPE-3-en`](https://huggingface.co/Tabahi/CUPE-3-en) | ph66 | pa07f, 0 | LibriSpeech, ~500 h | 0.01 | TIMIT f1@25 76.10 / 75.58 |
| [`CUPE-3-en-base`](https://huggingface.co/Tabahi/CUPE-3-en-base) | ph66 | pa01d, 3 | LibriSpeech, ~500 h | 0.01 | TIMIT f1@25 76.20 / 76.00 |
| [`CUPE-3-mls7`](https://huggingface.co/Tabahi/CUPE-3-mls7) | ph66 | pa07a, 0 | `CUPE-3-en-base` + MLS de es fr it nl pl pt, 100 h each | 0.01 | TIMIT f1@25 74.79 / 74.15 |

TIMIT numbers here are from the training repo's 400-clip probe, on two halves: one picked the
checkpoint, the other is held out. The full-corpus scores are on the [Evaluation](evaluation.md) page.

- **Which latin model.** `mbfa-latin` was trained with the most noise and is the one meant for real
  recordings, which is why every latin preset uses it. `mbfa-latin-benchmark` scores higher on clean
  speech and is selected by name only.
- **Which English ph66 model.** `CUPE-3-en` and `CUPE-3-en-base` are from the same lineage
  (pa01d → pa07a → … → pa07f) and score within noise of each other. `CUPE-3-en` is the
  preset. `CUPE-3-en-base` is selected by name.
- **`CUPE-3-mls7`** is `CUPE-3-en-base` trained for one more epoch on seven MLS languages, with no English.
  It is the only p3cupe model that has heard anything but English, so every non-English ph66 preset
  uses it. The epoch cost about two points of English TIMIT F1.
- **Groups without a model.** The non-latin groups were selected on validation loss over the model's own
  alignments, because no boundary labels exist outside English. Training configs exist for `vietnamese`,
  `other_alphabetic`, `abjad`, `brahmic` and `thai_khmer`, but no model has been trained yet.

### Release format

```text
config.json         format ("p3cupe" | "p4mbfa"), format_version (1)
                    model_config   tensor shapes (trunk, heads, lang_group, n_tokens)
                    geometry       sample_rate, window_size_ms, window_stride_ms
                    decoder        the settings above
                    labels         the token list (p4mbfa) or phoneme list (p3cupe) the heads emit
                    training       run, epoch, source checkpoint, selection metrics
model.safetensors   the weights
```

[`segmental/extractor.py`](../bournemouth_aligner/segmental/extractor.py) loads either a release
directory or a training checkpoint, and reads the family from `model_config`. A p4mbfa
release carries its token list and the gold-inventory fingerprint, and
`MbfaExtractor.check_inventory` refuses to load the model if this installation's standard_g2p tables differ. A
regenerated token list of the same length would otherwise relabel every phoneme silently. A p3cupe release is
checked against the ph66 phoneme list in the same way.

Releases are built by
[`p4mbfa/export_hf.py`](https://github.com/tabahi/bfa_models/blob/main/p4mbfa/export_hf.py)
(and `p3cupe/export_hf.py` for the CUPE-3 models).

## Training your own

See the [bfa_models README](https://github.com/tabahi/bfa_models). In short:

```bash
python p4mbfa/smoke_test.py                                # CPU, synthetic data
python p4mbfa/train_mbfa.py p4mbfa/mbfa_hp_<group>         # one yaml per language group
python p4mbfa/export_hf.py export <checkpoint> <out_dir>   # config.json + model.safetensors
```

Then load the model in BFA with `BFA(ckpt_path="<out_dir>", lang="...")`. `ckpt_path` also accepts
a raw training checkpoint, but only a release carries the token list that the inventory check
compares against.
