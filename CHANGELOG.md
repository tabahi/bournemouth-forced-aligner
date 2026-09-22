# Changelog

## [2.0.0] - Work in progress

A new major version: v2 does not load v1 checkpoints, and BFA's model, decoder and boundary
readout are all different. v1.x is the model2i/model3a line and ends at 1.1.5.

### Changed — BFA now runs on p3cupe, and only on p3cupe

The acoustic model, the decoder and the boundary readout have all been replaced. p3cupe is a
CNN-only phoneme model whose output **is** the boundary detector: each phone's posterior is
trained to hold 1.0 across the interior of its own segment and fall to exactly 0.5 at each of
its two boundaries, where the neighbour rises through 0.5, so a boundary is the *crossing* of
two posteriors rather than something to be recovered afterwards. The training package it was
ported from lives at `bournemouth_aligner/model3p/`, whose `README.md` and `memory.md` are the
design record.

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

**Buckeye, 40 random segments per seed, onset f1 at 25 ms** (same scorer and seeds as before,
`tests/test_buckeye.py`). Measured against `pa07a`, which was trained on MLS across 7 European
languages with English EXCLUDED - so it scores below the English model2i/model3a checkpoint it
replaces, exactly as `model3p/memory.md` records on TIMIT (per-phone f1@25 75.4 -> 63.8 over
the 4 MLS epochs). **This is a model-availability result, not an architecture result.**

An English checkpoint (`m_pa07b_final_e5_libri360h.pt`, LibriSpeech 360 h) now exists and has
not been scored here - that is the one to compare English numbers against.

| seed 2 | model3a `ue03c` | p3cupe `pa07a` (bonus 0) | p3cupe `pa07a` (bonus 30) |
|---|---|---|---|
| f1@10 | **44.1** | 39.1 | 45.1 |
| f1@25 | **69.8** | 60.8 | 63.2 |
| f1@50 | **82.8** | 74.6 | 76.5 |
| median abs onset error | 11 ms | 12 ms | 9 ms |

**The onset bonus.** `boundary_bonus_weight` now defaults to the checkpoint's own trained
value (30), not to 0. Measured on Buckeye, 40 segments x 3 seeds, weight 0 -> 30:

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| f1@25 | 60.2 -> **64.6** | 59.8 -> **61.5** | 60.8 -> **63.2** |
| f1@10 | 36.2 -> **43.8** | 34.5 -> **40.1** | 39.1 -> **45.1** |

It also holds on TIMIT's boundary scorer (100 files: f1@25 66.3 -> 72.2, median abs error
18.9 -> 15.7 ms) - though that scorer counts TIMIT's separate closure and burst labels, which
`model3p/memory.md` says the bonus wins on for the wrong reason, so it is corroboration of the
existing record rather than independent evidence.

It runs AGAINST that record's TIMIT finding that the bonus costs ~7 points per phone onset,
measured on the English `pa01d` checkpoint. The likely difference is the checkpoint: a model
with weak English posteriors leans on acoustic change for placement more than a strong one
does. **Re-measure on `pa07b`.**

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
`low_confidence_level` (new, default 0.25) replaces the hard-coded 0.5 cutoff in the
bad-segment check, and is calibrated against this one checkpoint.

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

### Changed — presets and checkpoint discovery

`presets.py` now holds the p3cupe checkpoints. English presets get
`m_pa07b_final_e5_libri360h.pt` (LibriSpeech 360 h); everything else falls back to
`m_pa07a_final_e4_MLS_7h.pt` (MLS, 7 European languages). `P3CUPE_UNIVERSAL` is a declared
placeholder: that checkpoint does not exist yet, so a preset for a language outside those 7
currently loads a model that has never heard it, and the preset table says so.

Because nothing is published yet, `download_model` looks on disk first — `$BFA_MODEL_DIR`,
then `./models/p3cupe/`, then `~/.cache/bournemouth_aligner/` — and only then asks
HuggingFace, with an error that names the search path when both fail.

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
