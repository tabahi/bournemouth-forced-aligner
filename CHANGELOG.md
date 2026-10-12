# Changelog

## [2.0.0] - Work in progress

A new major version: v2 does not load v1 checkpoints, and BFA's model, decoder and boundary
readout are all different. v1.x is the model2i/model3a line and ends at 1.1.5.

### Added — a second phoneme set and model family, now the default

BFA has two stacks behind one aligner, chosen by `phoneme_set`:

| `phoneme_set` | phonemizer | model family | label space |
|---|---|---|---|
| `"std"` (**new default**) | standard_g2p (ByT5 G2P, vendored) | **p4mbfa** | one token list per language group |
| `"ph66"` | espeak-ng + ph66 mapper | p3cupe | 66 phonemes, every language |

p4mbfa is p3cupe's trunk with three heads: `ph_head` (one LANGUAGE GROUP's tokens — 208 for
`latin`, 127 `cyrillic`, 48 `cjk`, 28 `japanese`), `phg_head` (15 phoneme groups, shared) and
`tone_head` (22 tones, shared). Its training decoder is the one BFA already had, with silence
at index 1 instead of 0, so the aligner itself did not change.

**New packages**

- `bournemouth_aligner/standard_g2p/` — the runtime of
  [tabahi/CharsiuG2P](https://github.com/tabahi/CharsiuG2P)'s `standard_g2p`, vendored
  verbatim (MIT). `tools/sync_standard_g2p.py` is its only writer and `VENDORED.md` records
  the commit. Only the Mandarin dictionary *keys* are shipped (600 KB instead of 196 MB).
- `bournemouth_aligner/p4mbfa/` — `model.py` (inference port of `MbfaCNN`) and `extractor.py`.
- `bournemouth_aligner/segmental/` — `aligner.py` and `windowing.py` moved here from `p3cupe/`
  (which still re-exports them), plus the clip-level forward pass both families share.
  `encode_clip` now returns a dict (`ph`, `group`, `tone`, `emb`, `flux`), not a tuple.
- `bournemouth_aligner/phonemizers/` — `std.py`, `ph66.py`, and the `Inventory` that `core.py`
  reads label ids from instead of the ph66 phonemizer's attributes.
- `bournemouth_aligner/hub.py` — every download. Models and the G2P come from the Hugging
  Face Hub through `huggingface_hub`'s own cache; nothing is written inside the installed
  package.

**New constructor parameters:** `phoneme_set`, `ckpt_path`, `g2p_model_dir`, `g2p_device`,
`g2p_cache_size`, `punctuation_sil`. They replace `mapper` and `cupe_ckpt_path`, which are
removed (see "Removed — everything kept for older versions"). CLI: `--phoneme-set`, `--ckpt-path`, `--g2p-device`,
`--punctuation-sil`; `--list-presets` prints both tables.

**Published models.** Every model is a Hugging Face repository of two files, `config.json` +
`model.safetensors`, exported from the training repository
([bfa_models](https://github.com/tabahi/bfa_models), `p4mbfa/export_hf.py`; its
`hf_release.yaml` records which run each one is and why it was chosen). BFA downloads those
two files on first use and reads the architecture, frame geometry, decoder settings and label
space from `config.json`.

| model | set | what it is |
|---|---|---|
| [`mbfa-latin`](https://huggingface.co/Tabahi/mbfa-latin) | std / latin | **the default.** FLEURS, 36 languages, 273 h, the most training noise (run `ma04a`, epoch 6) |
| [`mbfa-latin-benchmark`](https://huggingface.co/Tabahi/mbfa-latin-benchmark) | std / latin | best latin model on TIMIT, less training noise (`ma02a`, epoch 8). By name only |
| [`mbfa-cyrillic`](https://huggingface.co/Tabahi/mbfa-cyrillic) | std / cyrillic | FLEURS, 8 languages (`mr01a`, epoch 10) |
| [`mbfa-cjk`](https://huggingface.co/Tabahi/mbfa-cjk) | std / cjk | FLEURS Mandarin + Cantonese (`mc01a`, epoch 13) |
| [`mbfa-japanese`](https://huggingface.co/Tabahi/mbfa-japanese) | std / japanese | FLEURS Japanese (`mj01a`, epoch 8) |
| [`CUPE-3-en`](https://huggingface.co/Tabahi/CUPE-3-en) | ph66 | LibriSpeech 500 h, English (`pa07f`, epoch 0). Every English ph66 preset |
| [`CUPE-3-mls7`](https://huggingface.co/Tabahi/CUPE-3-mls7) | ph66 | `CUPE-3-en-base` + one epoch of MLS: de es fr it nl pl pt, 100 h each (`pa07a`, epoch 0). Every other ph66 preset |
| [`CUPE-3-en-base`](https://huggingface.co/Tabahi/CUPE-3-en-base) | ph66 | LibriSpeech 500 h, English, the root of the p3cupe lineage (`pa01d`, epoch 3). By name only |

These eight are the only models v2 knows: the model2i / model3a checkpoints of v1 do not load,
and no preset points at a local training checkpoint.

`model_name=` / `--model` takes one of these names, or any repository id in the same layout.
`ckpt_path=` / `--ckpt-path` takes a model on disk: a release directory, or a training
checkpoint file. `$BFA_MODEL_DIR/<name>/` and `./models/<name>/` are looked in before the Hub.

**How a model is chosen.** A model decides the phoneme set by itself (p3cupe is ph66, p4mbfa
is std); naming one set and passing a model of the other raises. In the std set the language
picks its language group and the group picks the model. A language whose group has no model,
or that standard_g2p does not cover, **raises with a message naming `phoneme_set="ph66"`** —
there is no silent fallback, because the two sets' ids and output keys differ.

**Label check.** `ph_head` emits indices into a token list that lives in standard_g2p, not in
the weights; a regenerated list of the same size would silently relabel every phoneme. A
released model therefore carries its token list and gold fingerprint in `config.json`, and
`MbfaExtractor.check_inventory` refuses a mismatch with this installation's tables. A released
p3cupe model's phoneme list is checked against the ph66 mapper the same way.

**The G2P model** is
[`charsiu/g2p_multilingual_byT5_small_100`](https://huggingface.co/charsiu/g2p_multilingual_byT5_small_100)
(~1.2 GB) with the tokenizer of `google/byt5-small`, downloaded by `huggingface_hub` on the
first call that phonemizes text. `$BFA_G2P_DIR` / `g2p_model_dir=` point at an existing copy.

### Changed — defaults, output and dependencies (breaking)

- **`PhonemeTimestampAligner(preset="en-us")` is now the std set.** Pass `phoneme_set="ph66"`
  for the previous behaviour. Alignments from an explicit p3cupe `ckpt_path` are
  unchanged: verified identical (timestamps, confidences, pooled embeddings, TextGrid) on the
  LJSpeech samples before and after the refactor.
- **Language codes.** The std set takes BCP 47 codes and accepts every spelling BFA took
  before (`en-us`, `en-gb`, `pt-br`, `cmn`, `fr-be`), resolving them to canonical ones
  (`en`, `en-GB`, `pt-BR`, `zh`, `fr`). `aligner.lang` holds the canonical code.
- **Results carry a header:** `{"phoneme_set": "std/latin", "lang": "en", "segments": [...]}`.
  A `phoneme_id` means nothing without its inventory. An empty result is no longer exactly
  `{"segments": []}`.
- **std segments** use `local_ph` / `gold_phg` where ph66 has `ph66` / `pg16`, and add `tone`,
  `stress`, `length` per segment and per `phoneme_ts` entry. `words_ts[].ph66` is keyed by
  the set's phoneme key (`local_ph` in std; still `ph66` in ph66). Groups are the 15 gold
  groups, not pg16.
- **No silence at punctuation in the std set** unless `punctuation_sil=True`, and **no
  number/abbreviation expansion** — the G2P reads dictionary words. Spell numbers out.
- **Padding id.** Ragged batches are padded with the inventory's `pad_id` (`<blank>` in std,
  the unused `noise` slot in ph66). `aligner.noise_class` / `noise_group` are replaced by
  `pad_class` / `pad_group`; in std `noise` is a real label.
- **Dependencies.** `transformers` and `safetensors` are now required. `phonemizer` moved to the `[ph66]` extra
  and is imported lazily, so the default install needs no espeak-ng. Word segmenters are
  extras: `[th]`, `[ja]`, `[yue]`. `bournemouth_aligner.Phonemizer_ph66` is imported on first
  access.
- **`load_model(preset=...)`** now changes the language too when none was given explicitly;
  before, `PhonemeTimestampAligner(preset="de").load_model(preset="fr")` loaded the French
  preset's model and kept phonemizing German.

### Changed — evaluation

- Scoring was already inventory-free (SRT `boundaries` against predicted onsets). What tied it
  to ph66 is fixed: silence was `phoneme_id == 0`, which is `<blank>` in std; the scorer now
  reads each alignment file's `phoneme_set`.
- Per-group errors are reported in each set's own groups (`err_*` for ph66's 16, `std_*` for
  std's 15) and in **manner classes both can fill** (`mn_*`), which is what a ph66 row and a
  std row are compared on. The CSV gains a `phoneme_set` column.
- `process_alignments.py`: `MODELS` lists the eight released models and nothing else, each
  twice — by `model_name` (downloaded from the Hub) and as `<name>-local`, the release
  directory `export_hf.py` builds before a push. The manifest records family, phoneme set,
  language group, gold fingerprint and the training run a model is, read from the model. A
  p4mbfa model of another language group than the corpus is refused before the sweep starts
  instead of failing once per file. `process_alignments.py N K` runs every N-th experiment of
  the grid starting at the K-th, so N shells cover it in parallel.
- `compute_metrics.py` scores each `.bfN` on the corpus its own manifest names, so one sweep
  over TIMIT and Buckeye is scored in one pass.
- **The corpus parsers write the hand labels in the std phoneme set.** `timit_parser.py` and
  `buckeye_parser.py` used to index TIMIT's and Buckeye's phone labels in ph66. They now write
  latin local ids - the label space of `mbfa-latin` / `mbfa-latin-benchmark`, the only mbfa
  models English belongs to - through the new `evaluation/dataset/std_labels.py`: a table per
  corpus, then standard_g2p's own gold and `to_local` stages, the path the training probe
  uses. A label is written the way the std G2P writes that phoneme in English (diphthongs and
  affricates are two tokens, `l` is ɫ, `ah` is ə, allophones fold to their phoneme; the flap
  and the glottal stop keep ɾ and ʔ). Every SRT is stamped with `labels`: phoneme set,
  language group, token count and gold fingerprint. An unknown label raises.
  On the 400 TIMIT probe clips the hand labels written this way agree with the G2P's own
  sequence on 76.7% of tokens; the rest is what a transcriber hears and a dictionary does not
  (unreleased stops, flaps, reduced vowels, glottal stops).
- **Fixed: Buckeye phone labels were one interval late.** A `.phones` row's time is where its
  label ENDS; the parser read it as the start, so every label sat on the interval after its
  own (the interviewer's turns had a median length of 115 ms, flaps 80 ms; read correctly
  they are 3 s and 25 ms). The boundary TIMES, which are all the scorer reads, were right
  and are unchanged: start, end and text are identical for all 12,182 segments, and the
  boundaries for all but 2, which lose a stray end-of-transcription mark 1 ms past their last
  phone. TIMIT's boundaries are identical for all 6,300 clips.
- The parsers list the corpus in sorted order and number TIMIT's speakers in sorted order.
  Both depended on directory order or on Python's per-process set order before, so a
  regenerated paths list drew a different `subset` and carried different speaker ids.
- Paths lists are named `<dataset>_paths_all.json`; the old `*_ph66.json` names are still read.
- **New `evaluation/dataset/probe_cache.py`** builds the boundary-probe cache the p4mbfa
  training repository reads, for TIMIT and now Buckeye, from this repository's SRT trees.
  Its TIMIT output is identical, clip for clip, to `bfa_models/tmp/data/timit_probe_400_latin.json`.
  Buckeye entries are segments of long files (`start` / `end`), which the training probe now
  cuts out before aligning.

### Measured — std / p4mbfa

**Parity with training.** BFA with the `ma04a` epoch-6 weights (now `mbfa-latin`) on the
training repository's 400 TIMIT probe clips, in the probe's own batches of 16, frame-grid
onsets: f1@10/25/50 = 52.79 / 71.22 / 82.91 on the tune half and 54.00 / 71.70 / 83.41 on the
held-out half — identical to the training probe run in fp32 on the same checkpoint. The
vendored phonemizer reproduces the cache's target sequence for all 400 clips.

**The score depends on batch composition.** `class_prior_normalise` takes the class prior over
the batch's frames, so the same clips aligned one at a time score 71.76 / 72.55 at f1@25
instead of 71.22 / 71.70. This is the decoder as trained and applies to both families; it
means two runs are only comparable at the same batch size.

**The five English-capable models side by side**, same files and scorer
(`evaluation/process_alignments.py` + `compute_metrics.py`), frame-grid onsets, batch size 16,
with the onset bonus off and at its published weight of 30. Small samples, to show the
pipeline end to end — the model cards carry the 400-clip TIMIT probe numbers:

| f1@10 / 25 / 50 | TIMIT, 100 files, bonus 0 | bonus 30 | Buckeye, 3 interviews, bonus 0 | bonus 30 |
|---|---|---|---|---|
| ph66, `CUPE-3-en` | 54.3 / 74.8 / 85.8 | 57.3 / **75.8** / 85.7 | 46.0 / **72.0** / 83.9 | 43.2 / 68.9 / 82.6 |
| ph66, `CUPE-3-en-base` | 47.4 / 74.1 / 86.1 | 58.0 / **76.7** / 86.3 | 44.5 / **72.4** / 83.8 | 42.2 / 67.8 / 81.9 |
| ph66, `CUPE-3-mls7` | 49.7 / 72.1 / 85.0 | 56.7 / **74.9** / 85.1 | 42.2 / **67.3** / 80.7 | 40.6 / 65.2 / 80.0 |
| std, `mbfa-latin` | 40.0 / 67.2 / 83.9 | 54.1 / **73.1** / 84.1 | 37.3 / 63.2 / 77.1 | 39.3 / **63.6** / 77.5 |
| std, `mbfa-latin-benchmark` | 40.7 / 71.0 / 84.8 | 56.2 / **74.1** / 85.1 | 38.8 / **64.1** / 77.7 | 38.8 / 63.2 / 76.9 |

The bonus helps every model on TIMIT (read speech, where a stop's closure and burst are
labelled as two boundaries) and costs the three p3cupe models 2 - 4.6 points of f1@25 on
Buckeye (conversational); the mbfa models are indifferent there. The published default stays
30; `boundary_bonus_weight=0` is the setting to try on spontaneous speech with a ph66 model.

### Changed — the ph66 stack runs on p3cupe, and only on p3cupe

The acoustic model, the decoder and the boundary readout have all been replaced. p3cupe is a
CNN-only phoneme model whose output **is** the boundary detector: each phone's posterior is
trained to hold 1.0 across the interior of its own segment and fall to exactly 0.5 at each of
its two boundaries, where the neighbour rises through 0.5, so a boundary is the *crossing* of
two posteriors rather than something to be recovered afterwards. The training package it was
ported from is `p3cupe/` in [bfa_models](https://github.com/tabahi/bfa_models), whose
`README.md` and `memory.md` are the design record.

**New package `bournemouth_aligner/p3cupe/`** — inference only:

- `model.py` — the trunk. 13.5M params, 38.9 ms receptive field, **5 ms frames** (down from
  13.125 ms), no transformer, no boundary head, no blank class.
- `extractor.py` — checkpoint loading and the windowed forward pass over a whole clip. The
  checkpoint is the source of truth for both the tensor shapes and the frame geometry.
- `aligner.py` — the segmental Viterbi, ported from the training package so that what is
  decoded at inference is the same decoder that built the training targets.
- `windowing.py` — 120 ms windows at an 80 ms stride, which is a whole 16 output frames, so
  the stitched grid is the model's own conv grid and `time_ms(frame f) = f * 5.0` exactly.

**Removed.** `bournemouth_aligner/cupe2i/` (model2i, model2j, model3a), `forced_alignment.py`
(the CTC lattice) and `segmental.py` (the blank-based segmental decoder). The model2i and
model3a checkpoints no longer load — `CupeExtractor` raises with a message saying so.

**Removed constructor parameters**, all of which only had meaning for the CTC lattice or the
onset snapper: `boost_targets`, `enforce_minimum`, `silence_anchors`, `ignore_noise`,
`extend_soft_boundaries`, `boundary_softness`, `ensure_completeness`, `enforce_all_targets`,
`decoder`, `boundary_snap_source`, `boundary_snap_radius`, `boundary_snap_radius_ms`,
`boundary_snap_offset`. `segmental_max_seconds` is renamed `segment_max_seconds`.
`set_boundary_snap()` and `extract_timestamps_from_segment_simplified()` are gone.

**Note on `enforce_all_targets=False`.** The old non-forced mode — skipping up to 3 whole
words the audio did not contain — belonged to the blank-based decoder and has no counterpart
in the p3cupe aligner. BFA >= 2.0 always performs forced alignment. Unreliable transcripts are
now *reported* (`coverage_analysis["bad_alignment"]`) rather than absorbed.

**New constructor parameters:** `boundary_bonus_weight`, `sub_frame_onsets`,
`min_phone_frames`, `soft_phone_frames`, `low_confidence_level`. Each decoder setting defaults
to `None`, meaning "whatever the checkpoint was trained with", because its posteriors were
shaped by targets this same decoder built.

### Measured

Every number for the released models is in "Measured — std / p4mbfa" above; the onset bonus
defaults to the value a model is published with (30 for all eight). The earlier comparisons
in this section were made on a v1 checkpoint and on training runs that are not released, and
have been removed with them.

### Changed — confidence

A phoneme's `confidence` is now **the peak posterior the model puts on it inside its own
segment**, not an average seeded from the segment's first frame. The hat-shaped target makes
the frames at both ends of a segment uncertain by design, so the old measure systematically
under-read short phones, which are nearly all ramp.

**Confidence values are not comparable with BFA <= 1.2's.** Two separate changes, measured
independently on `30s.wav`:

| | mean confidence | below 0.5 |
|---|---|---|
| model3a, old measure | 0.728 | 24% |
| model3a, new measure | 0.794 | 18% |
| p3cupe, new measure | 0.319 | ~68% |

So the measure accounts for +0.07 and the model for the rest: this loss deliberately does not
saturate (its target distribution has entropy), so a well-aligned phone reads far below 1.0.
`low_confidence_level` (new) replaces the hard-coded 0.5 cutoff in the bad-segment check.
(The p3cupe row is an unreleased training checkpoint.)

**The released models are not on one confidence scale**, so `low_confidence_level` defaults to
`None` = the loaded model family's own level: **0.2 for the CUPE-3 (p3cupe) models, 0.1 for
the mbfa (p4mbfa) ones**. Mean confidence on the samples above (TIMIT / Buckeye, published
settings): `CUPE-3-en` 0.53 / 0.43, `CUPE-3-en-base` 0.39 / 0.32, `CUPE-3-mls7` 0.26 / 0.21,
`mbfa-latin` 0.25 / 0.20, `mbfa-latin-benchmark` 0.17 / 0.15.

### Removed — everything kept for older versions

2.0 is a clean break, so nothing is carried for code written against 1.x or the 2.0 nightlies:

- The constructor's `mapper=` and `cupe_ckpt_path=` (`phoneme_set=` and `ckpt_path=` are the
  only spellings), `aligner.selected_mapper`, and `presets.DEFAULT_MODEL`.
- **`is_estimated`** in every `phoneme_ts` / `group_ts` entry, which could only be `false`
  since forced alignment places every target it reports. The low-level timestamp tuples are
  7 long: `(id, start_frame, end_frame, target_idx, confidence, start_ms, end_ms)`.
- The aligner functions re-exported from `bournemouth_aligner.p3cupe` (they live in
  `bournemouth_aligner.segmental`), and `Phonemizer.break_words` (use `break_words_alpha`).
- Decoder defaults for p3cupe checkpoints that do not carry every decoder setting: both
  families now refuse such a file, as p4mbfa already did. The released models carry them all.
- `presets.get_preset` returns `(model_name, lang, note)` and prints nothing.
- Evaluation: the `*_paths_all_ph66.json` list names, the `cupe_ckpt_path` spelling in
  `MODELS` and manifests, `drop_estimated`, and the duplicate `detected_*` / `predicted_*`
  result keys.

### Changed — downloads and console output

- **The cache comes first.** A model or the G2P that is already in the huggingface_hub cache
  is used without contacting the Hub at all. Before, every load asked the Hub whether the
  repository had changed and drew a "Fetching 2 files" bar even when nothing was downloaded.
  A repository updated on the Hub is therefore not picked up until it is deleted from the
  cache or requested with `revision=`.
- **`warn_level=0` prints nothing**, including on a first download. Level 1 adds one line per
  download and real warnings; level 2 adds progress bars and the informational output.
  Silenced at levels 0 and 1: huggingface_hub's progress bars, transformers' loading bar and
  its tied-weights notice for the G2P checkpoint, and the espeak backend lines. The ph66
  preset warning and the unknown-phoneme warning follow `warn_level` instead of printing
  unconditionally.

### Fixed

- **Word timestamps were paired positionally with the phonemizer's `word_num`.** They are now
  matched by `target_seq_idx`. Any phoneme the aligner declined — a `SIL` written for a comma
  the speaker did not pause at is the normal case — used to shift every later word onto the
  wrong phonemes.
- **A batch died as a whole when one segment's text outran its audio.** The batch is now
  retried one segment at a time, so only the segment that genuinely cannot fit is dropped, with
  a warning naming it.
- **The chunker ran ahead of its audio.** A window is now trusted only as far into its own
  audio as `TRUST_WINDOW_FRACTION` (75%), as a cap on top of the derailment detector rather
  than a fallback when it finds nothing. This decoder cannot cram — it enforces a minimum phone
  duration and reports infeasibility instead — so an over-budgeted window places its surplus
  phonemes early rather than announcing itself, and a count-based trim kept almost all of them.
  Measured on `30s.wav` forced through the chunker at 10 s windows:

  | | last phoneme ends | mean confidence | flagged |
  |---|---|---|---|
  | count trim only | 30.31 s | 0.217 | yes |
  | time cap | 31.87 s | 0.317 | no |
  | whole-segment decode | 31.88 s | 0.319 | no |

- **`total_phonemes_aligned` double-counted**, adding the group timestamps to the phoneme
  total. `total_phonemes_extra` and `total_phonemes_aligned_easily` are removed: with forced
  alignment each target is placed at most once and nothing is estimated after the fact.
- **The Viterbi's first call took 30-120 s on a loaded multi-core box.** It now pins torch's
  intra-op pool to one thread for the duration of the DP: the per-frame tensors are far below
  the size that pays for a thread fan-out, so the threads only contended. Measured on a 32 s
  clip with 358 phones: 1.70 s at one thread against 34-117 s on the first call at 32.
- The DP runs on CPU even when the model is on a GPU — 1.8 s against 5.0 s for the same clip,
  bit-identical, because a python loop over frames on small tensors is launch-bound.

### Changed — chunking

Chunking is now a **resource guard only**. The segmental decoder does not collapse on long
audio, so `_needs_chunking` triggers on length alone (`segment_max_seconds`, 120 s) and no
longer on phoneme count. `chunking.stride4_max_phonemes()` and `DEFAULT_MAX_PHONEMES`'s
derivation from the CTC lattice are gone; the bound that actually keeps a window feasible is
now applied in `core.align_segment_chunked` and depends on the loaded checkpoint's duration
floor.

### Changed — presets and model discovery

`presets.py` holds the model names, and is the only place one appears. In the ph66 set the
language picks between two p3cupe models: an English preset resolves to `CUPE-3-en`, every
other one to `CUPE-3-mls7`, which adds German, Spanish, French, Italian, Dutch, Polish and
Portuguese to English; a preset outside those eight prints a warning that the model was not
trained on that language. An explicit `lang=` / `--lang` picks the model the same way a preset
does. `CUPE-3-en-base` and `mbfa-latin-benchmark` are selectable by `model_name` only. The
earlier local-file presets and the `P3CUPE_UNIVERSAL` placeholder are gone; a model on disk
loads with `ckpt_path=`.

`download_model` and every preset go through `hub.resolve_model`: `$BFA_MODEL_DIR/<name>/`,
then `./models/<name>/`, then the model's Hugging Face repository, with an error that names
all three when they fail.

### Added — the evaluation harness (`evaluation/`, was `scoring/`)

Sweep a corpus once per configuration, then score every run into a labelled CSV row.

```bash
python evaluation/process_alignments.py     # align, once per grid point
python evaluation/compute_metrics.py        # score into evaluation/results/*.csv
```

`evaluation/layout.py` defines where everything goes, and BOTH scripts import it - the writer
and the reader agreeing by coincidence is how a sweep ends up scoring an empty set and
reporting every run as "never aligned":

    tmp/alignments/<dataset>/<bfN>/<clip>.json   alignments, and nothing else
    tmp/manifests/                               what produced each run, plus logs
    tmp/plots/                                   error histograms
    evaluation/results/*.csv                     the scores - committed, unlike the rest
    tmp/data/<dataset>_data/                     the corpus: SRTs and MFA's .mfa.json only

Alignments moved out of the corpus tree so a sweep is disposable: `rm -rf tmp/alignments`
drops every run without touching the corpus, and `rm -rf tmp/alignments/timit/bf3` drops one.
Clip files are keyed by `layout.clip_stem` (the SRT path relative to the corpus root:
`all_MABC0_SX61`, `s01_s0103a`), because TIMIT basenames are not unique across speakers and
Buckeye's `clip_id` is a positional index that would rename every clip when the paths list is
regenerated. `assert_unique_stems` checks the whole list on load rather than trusting the rule.

The sweep grid is now the p3cupe decoder axes - `boundary_bonus_weight`, `min_phone_frames`,
`soft_phone_frames`, `sub_frame_onsets` - each accepting `None` for "the checkpoint's trained
value". `_check_for_duplicates` resolves those against the checkpoint's own hyperparameters and
REFUSES a grid whose rows collapse to one run, which `[None, 30.0]` on a checkpoint trained at
30 does. Manifests record `decoder_effective` so a row says what ran, not what was requested.



## [1.2.0] — never released, superseded by 2.0.0

> The work below targeted the model2i/model3a checkpoints and the blank-based
> `segmental.py` decoder, both removed in 2.0.0. Kept as the record of why the segmental
> topology (minimum phone duration, skippable silences, an onset bonus from acoustic change)
> was adopted — p3cupe's aligner inherits all three.

### Added — segmental decoder (`decoder="segmental"`, now the default)

New module `bournemouth_aligner/segmental.py`. Every frame is assigned to a phoneme or a
pause, each phoneme lasts at least 2 frames (a 3rd is skippable at a small cost), and every
transition into a phoneme is rewarded by the contrast of the boundary tap's spectral flux at
that frame. Posteriors and acoustic change are traded off **inside one Viterbi pass**, where
the CTC path commits to its spikes first and snaps each onset greedily afterwards.

**Why.** The CTC decoder places the CTC *spike*, not the onset (signed onset error +5 ms),
and when its early placements go wrong the rest of a long segment follows them. Onset F1 at
25 ms on Buckeye (model3a `en-us` checkpoint, scorer identical to
`UnsupervisedPhones/modules/aligner/alignment_tests.py`, see `tests/test_buckeye.py`):

| sample | `ctc` | `segmental` |
|---|---|---|
| 200 segments, offline decode (tuning set) | 59.0 | **68.2** |
| 40 segments, full pipeline | 58.1 | **69.2** |
| 50 held-out segments, full pipeline | 58.9 | **69.7** |
| `s0103a`, 30 segments | 54.9 | **65.5** |
| TIMIT test, 150 random utterances (not tuned on) | 59.6 | **67.5** |

f1@10 goes 32.3 -> 42.9 and f1@50 76.9 -> 83.2 on the held-out set; signed onset error median
goes to 0; recall@25 is flat across position within the segment. Still short of MFA's 78.5 on
full Buckeye. How each piece contributed (100 segments): phone HMM alone 48.1, + flux-contrast
onset bonus 64.7, + 2-frame minimum duration 68.4. Without a flux signal the topology scores
55.5, below CTC, so the decoder falls back to `ctc` when `boundary_snap_source="none"` and for
sub-state checkpoints. Tried and rejected: group-head posteriors as extra emission (-0.1),
pre-transformer flux added to spectral_stream (+0.0), mel flux from the waveform (+1 to +1.7
f1@25 but trades away f1@10 and depends on sub-frame alignment), sub-frame onset refinement
(+6 f1@10, -0.4 f1@25), constant onset shifts, pauses taking the blank mass (-9 to -16:
CUPE's blank fires inside speech, not in pauses).

**Long segments.** It does not collapse, so chunking is no longer needed with it: on 40
segments whole-segment decoding scores 69.2 vs 68.3 chunked, and on `30s.wav` one-shot mean
confidence is 0.728 (0.721 / 0.735 by half) vs 0.661 chunked, where the one-shot `ctc` decoder
stops placing phonemes at 19.2 s of 32 s. With the segmental decoder `chunk_audio` only engages
above `segmental_max_seconds` (new parameter, default 120 s), as a memory guard: on
`s0102a`, a single 225 s segment, whole-segment decoding scores 67.3, the same decoder through
the chunker 54.7, and the chunked `ctc` decoder 45.3 (with recall@25 of 0.0 in its last
decile). Raise the limit when memory allows.

**Memory.** CUPE inference now runs in batches of 800 windows (~64 s of audio) instead of one
forward pass over the whole segment - outputs are identical, since windows are classified
independently - and the decoder computes emissions per frame and stores 1-byte backpointers.
Peak for one segment decoded whole, before -> after: 120 s GPU 1116 -> 658 MB; 225 s GPU
1984 -> 766 MB and extra RAM 1281 -> ~470 MB. No slowdown.

**Defaults re-checked with the new decoder** (full pipeline, two Buckeye samples, f1@25):
the automatic snap source (`spectral_stream` on model3a) 69.2 / 69.7, `pre_transformer`
66.2 / 64.7, `boundary_head` 55.5 / 55.6, `embeddings` 56.2 / 54.4. `duration_max` (10/30/60)
and `extend_soft_boundaries` leave onsets unchanged. No default changes.

**`enforce_all_targets=False`** now has a decoder that follows acoustic evidence: whole words
(up to 3 in a row, cost 6 each) may be skipped anywhere including the segment edges, and pause
states carry a filler emission so untranscribed speech falls between words. On `s0103a` with
2 foreign words glued onto each end of every transcript: forced 62.7, non-forced **63.8**
(f1@50 76.4 -> 79.1); with 2 interior words deleted: 63.2 / 62.9; clean: 65.5 / 65.4. The
forced default is unchanged in meaning: every target is aligned.

- `decoder` constructor parameter and `--decoder` CLI flag; `--enforce-all-targets /
  --no-enforce-all-targets` CLI flag.
- `extract_timestamps_from_segment_batch(word_sequences=...)`: word index per phoneme, used by
  the segmental decoder for inter-word pauses and word skips. `process_segments` passes the
  phonemizer's `word_num` through, including on the chunked path.
- With the segmental decoder, group timestamps are derived from the phoneme segmentation
  instead of a second Viterbi over group posteriors: consistent by construction, and the
  whole pipeline runs 2-3x faster (124 s -> 42 s on 40 segments).
- `align_segment_whole()`: decode one segment on its own unpadded audio.
- `tests/test_segmental.py` (13 tests on synthetic posteriors, no model needed) and
  `tests/test_buckeye.py` (random-segment Buckeye benchmark with position deciles and
  `--corrupt extra_ends|missing` for unreliable transcripts).

### Fixed
- **Segments longer than `duration_max` were silently truncated** when they were not
  chunked (`chunk_audio=False`, or under the chunking thresholds): the batched path pads or
  cuts every segment to `duration_max`. They are now decoded on their own audio.



## [1.1.5] - 2026-06-05
- Default `boundary_softness=3` is changed from 7 to 3 because at 3 it performs the best on TIMIT and buckeye dataset. 
- Added warnings and parameter `self.warn_level` for verbose level. Use `PhonemeTimestampAligner.warn_level=3` for high verbosity.
- Fixed small segments alignment. Previously one-word or half-words ("umm or oh") raised error of not enough phonemes for the given duration. That is patched up here: https://github.com/tabahi/bournemouth-forced-aligner/commit/e58757b5ff9c3cb71303067a8050b06d01f80097#diff-c530cc1a536c371d38e233b3192f5f7935f97342d6460e039edc8107fbdf0348



## [1.1.4] - 2026-02-28
- `batch_size=16` parameter in `process_srt_file()` and `process_segments()`. Reduce it to prevent out-of-memory errors when processing large audio segments.

## [1.1.3] - 2026-02-28
- Fixed the bug: The viterbi algorithm didn't really do "forced" alignment before since v1.1.0. Now it does as long as `enforce_all_targets=True` is set.
- Added another layer for completeness. `ensure_completeness=True` (default: False). This is a post-processing hack for completeness. It tries to insert missing phonemes in their most likely positions based on the surrounding aligned phonemes and the original target sequence.
- [torchcodec](https://github.com/meta-pytorch/torchcodec) dependency.  

## [1.1.2] - 2026-02-23
- Added `load_model()` so that the models can be loaded after `__init__`, while the class `PhonemeTimestampAligner(lang ='en-us') can be initialized without specifying a model.`
- Added more CLI functionality. Direct text input. Generate mel-spectrum directly from blaign cli command.
- Readme reorganization.
- ONNX porting still in progress.

## [1.1.0] - 2026-02-13

This version breaks compatibility with the previous versions (<=0.1.7)

### Changed
- `silence_anchors` now works correctly. It detects silence regions in the audio and matches them to SIL tokens (from punctuation) in the target sequence, splitting long segments into smaller chunks for more accurate Viterbi decoding. Set `silence_anchors` to control the sliding window size for silence detection (default: 10). Set to 0 to disable.
- Output JSON key `phoneme_idx` renamed to `phoneme_id` in `phoneme_ts` entries.
- Output JSON key `group_idx` renamed to `group_id` in `group_ts` entries.
- `output_frames_key` parameter options updated: `phoneme_idx` → `phoneme_id`, `group_idx` → `group_id`.

### Added
- `ipa_label` field in `phoneme_ts` and `group_ts` output entries — contains the original espeak-ng IPA phoneme, before mapping to the reduced 66-class set. Useful when you need the full espeak IPA dictionary after alignment.
- `is_estimated` field in `phoneme_ts` and `group_ts` entries — indicates whether a phoneme was directly aligned by Viterbi (`false`) or inserted by the coverage enforcement post-processing (`true`). Downstream tasks can use this to filter out forced insertions.
- `coverage_analysis` per-segment output with `target_count`, `aligned_count`, `missing_count`, `extra_count`, `coverage_ratio`, `missing_phonemes`, and `extra_phonemes`.
- `process_segments()` now supports true batch processing: accepts a list of clips (each with multiple segments) and their corresponding audio waveforms. Embeddings are returned as nested lists `[clip][segment]`.
- Enabled MPS device support for Apple Silicon.

### Fixed
- Batch processing: sequence lengths are now computed before padding, preventing the decoder from aligning padding tokens as real phonemes.
- Batch processing: coverage enforcement (`ensure_target_coverage`) is now padding-aware, ignoring padded positions.

## [0.1.7] - 2025-10-17
- Fixed the rhotics and compound phoneme mappings in [ph66_mapper](bournemouth_aligner/ipamappers/ph66_mapper.py).
- CPU only mapping during checkpoint load in model2i.py
- Batch processing by @JulianaFrancis

## [0.1.6] - 2025-09-30
- `preser` parameter for automatically selecting language suitable model.


## [0.1.5] - 2025-09-10
- `silence_anchors` parameter for spliting search path at long silences.
- `offset_ms=0` parameter in `framewise_assortment` for same audio, multiple segments start offset corrected.
- `ignore_noise=False` parameter to view predicted noise.

## [0.1.4] - 2025-09-02
- `framewise_assortment` updated.
- timestamps_dict won't include frames and compressed frames.
- Use method `framewise_assortment(aligned_ts)` to get the frames assortment as needed.
- Added `extract_mel_spectrum` adapted from https://github.com/NVIDIA/BigVGAN
- Fixed `do_groups=False`.


## [0.1.2] - 2025-08-29
- `enforce_all_targets=False` option in _init_.
- Compressed frames output only option.

## [0.1.1] - 2025-08-24
- Renamed `process_transcription` to `process_sentence`
- Framewise assortment of phoneme_idx with variable output framerate.
- `phoneme_id_to_label`, `phoneme_label_to_id` and `group_id_to_label`, `group_label_to_id` and `phoneme_id_to_group_id`.

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-08-19

### Added
- Initial release of Bournemouth Forced Aligner
- Phoneme-level timestamp extraction using CUPE models
- Command-line interface (`balign`)
- Embedding extraction capabilities
- Enhanced forced alignment with confidence scoring
- Word-level alignment from phoneme timestamps
- Automatic model downloading from HuggingFace Hub
- Comprehensive documentation and examples
- (Planned support for multiple languages via phonemizer), currently available pretrained CUPE model is only for English.

### Features
- **Core Alignment**: PhonemeTimestampAligner class for programmatic use
- **CLI Tool**: `balign` command for batch processing
- **Embeddings**: Extract phoneme embeddings for downstream tasks
- **Confidence Scores**: Alignment quality metrics
- **Debug Mode**: Detailed processing information

### Dependencies
- torch>=1.9.0
- torchaudio>=0.9.0
- huggingface_hub>=0.8.0
- numpy>=1.19.0
- click>=8.0.0
- phonemizer>=3.3.0


[1.1.5]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v1.1.5
[1.1.3]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v1.1.3
[1.1.2]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v1.1.2
[1.1.0]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v1.1.0
[0.1.7]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v0.1.7
[0.1.6]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v0.1.6
[0.1.5]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v0.1.5
[0.1.4]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v0.1.3
[0.1.2]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v0.1.2
[0.1.1]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v0.1.1
[0.1.0]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v0.1.0
