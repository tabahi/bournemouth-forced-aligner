# Changelog

## [Unreleased]

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

### Changed — onset snapping defaults, from `tmp/results/conclusions.md`

`conclusions.md` §10 added the missing `none` controls on **both** corpora, which is the
first time the snap sources could be read as gains over not snapping rather than as a
head-to-head ordering. Two defaults move as a result.

**1. The snap radius is now expressed in FRAMES.**

`DEFAULT_SNAP_RADIUS_FRAMES` / `DEFAULT_SNAP_RADIUS_FRAMES_FALLBACK` replace the `_MS`
pair (kept as derived aliases). The snapper searches `sig[f-r : f+r+1]` — a whole number of
frames — so milliseconds were only ever a way of naming a frame count, and two different ms
settings routinely meant the same search: `52.5` and `55.0` are both 4 frames at a
210-sample stride. The eval CSVs show the cost of leaving the unit implicit — `cfg_snap_radius`
holds `52.5` on some rows and `2`/`4`/`6` on others, ms and frames in one column.

- `boundary_snap_radius` (frames) is now the primary parameter; `boundary_snap_radius_ms`
  still works, still wins when `boundary_snap_radius` is None, and is still the right tool
  for comparing models with *different* strides, where one frame count is two durations.
- `boundary_snap_radius_ms` is now also an **output**: it reports the duration the resolved
  frame count actually spans, not whatever was requested.
- `set_boundary_snap(radius_frames=...)` takes precedence over a previously-set ms value
  instead of being silently outranked by it. Its signature now leads with `radius_frames`.
- Default is unchanged in effect: **4 frames** (52.5 ms) for the flux taps, 2 for
  `boundary_head`. 4 is the only radius sweep on a full corpus (`results_TIMIT_lev.csv`,
  `en` model2i, spectral_stream: r=2 → 0.6522, **r=4 → 0.6897**, r=6 → 0.6788).
  *Not settled:* every model3a full-corpus row was taken at r=2, and the 20-file Buckeye
  sweeps prefer r=1. There is no full-Buckeye radius sweep for any checkpoint.

**2. `boundary_snap_source='auto'` now branches on the checkpoint ARCHITECTURE**, not on
whether a boundary head exists — which is what §10 recommends in as many words.

| | before | now | basis |
|---|---|---|---|
| model3a | `boundary_head` if trained, else `pre_transformer` | **`spectral_stream`** | best real source on both corpora, above control on both (+0.055 TIMIT, +0.026 Buckeye) |
| model2i / model2j | `pre_transformer` | `pre_transformer` *(unchanged)* | `spectral_stream` is best on TIMIT for model2i too (+0.106) but **below** control on Buckeye (0.5174 vs 0.5437); `pre_transformer` is above on both |
| trained `boundary_head` | selected automatically | **never selected automatically** | the only real source that loses to its own control: −0.0168 on 6300 TIMIT files, −0.028 on the 400-clip probe |

The trained head is not broken — it detects boundaries about as well as spectral flux (frame
AUC 0.886 vs 0.893 against ground truth). But its training targets come from the model's own
Viterbi output, so what it learned best is the *decoder's* placement (AUC 0.864 against its
own decoder, vs 0.70–0.75 for the flux taps), and snapping to it reproduces the decoder's
lateness instead of correcting it — signed-error median +6, the same as no snapping, where
the flux taps pull it to +3/+4. Requesting `'boundary_head'` explicitly still works and now
warns with the relevant caveat for trained and untrained heads separately.

`AUTO_SNAP_FALLBACK` is kept as an alias of `AUTO_SNAP_MODEL2I`.

**Not changed, and still open:** on Buckeye the single best-scoring setting in the CSVs is a
plain −1-frame shift (what an *untrained*, constant head degenerates into): +0.047, ahead of
`spectral_stream`'s +0.026. The decoder runs about a frame late and Buckeye punishes that
more than TIMIT does. A global shift is not reachable through the current API — `offset` only
applies after a search and `radius=0` disables snapping — and is worth adding as a
first-class option.

### Added — adaptive chunking for long audio (`chunk_audio`, on by default)

Long segments are no longer decoded in one shot. Past `chunk_seconds` (10 s) of audio or
`chunk_max_phonemes` (190) targets, BFA now walks the segment in overlapping windows and
finds where each window's alignment stops being trustworthy.

**Why.** One-shot Viterbi does not degrade gracefully on long clips, it collapses — it
packs every phoneme into the first part of the clip and emits the rest at one frame
(13.125 ms) apiece with confidence 0.000. `silence_anchors` was the previous mitigation,
and it works *when the transcript's punctuation happens to sit where the speaker actually
paused*. When it doesn't, there is nothing to anchor to and the collapse is total. Strip
the commas and full stops from a transcript and one-shot alignment drops to ~0.02 mean
confidence over ~23 % of the audio at **every** `silence_anchors` value.

Chunking cuts where the alignment demonstrably fails instead, so it does not depend on
punctuation at all.

**How.** Each window is handed a deliberately *over-provisioned* phoneme budget, sized
from its speech seconds at `chunk_phonemes_per_second` (15 /s). The surplus phonemes cram
into the window's final frames at one frame apiece, and that cram marks exactly where the
window's usable alignment ends. Everything before it is kept, and the next window restarts
at the end of the last kept phoneme — so the windows overlap by however much was
discarded, and no audio between two kept phonemes is skipped. Confidence collapse is used
as a second, corroborating signal: the cut is walked back over the near-zero-confidence
drift that precedes the cram, and a long run of weak confidence can trigger a cut on its
own when durations still look plausible.

**Measured** on `30s.wav` (32 s, 356 phonemes) and a 36.5 s LJSpeech concatenation (402
phonemes), `en_libri1000_ua01c` model2i. Mean confidence, with audio coverage in
parentheses:

| clip | `silence_anchors` | one-shot | chunked |
|---|---|---|---|
| 30s.wav, punctuated | 0 | 0.021 (22 %) | **0.399** (100 %) |
| | 3 | 0.049 (57 %) | **0.399** (100 %) |
| | 10 *(default)* | 0.402 (100 %) | **0.399** (100 %) |
| 30s.wav, punctuation stripped | 0 / 3 / 10 | 0.011 (24 %) | **0.408** (100 %) |
| LJSpeech ×3, punctuated | 0 | 0.016 (21 %) | **0.417** (99 %) |
| | 3 | 0.072 (99 %) | **0.417** (99 %) |
| | 10 *(default)* | 0.380 (99 %) | **0.417** (99 %) |
| LJSpeech ×3, punctuation stripped | 0 / 3 / 10 | 0.019 (23 %) | **0.385** (100 %) |

The read: at its default `silence_anchors=10`, one-shot alignment is *fine* on a
well-punctuated clip whose commas match the speaker's pauses (0.402, 0.380) and falls off a
cliff otherwise — a 20× swing in confidence driven by whether the transcript happens to
have the right punctuation. Chunked alignment sits at 0.385–0.417 across every row, and is
identical across `silence_anchors` values. The gain is **robustness**, not a higher ceiling
on the easy case: on the one row where one-shot does well it is 0.402 against chunked's
0.399.

Every chunked row aligns 100 % of the target phonemes, covers ~100 % of the audio, and none
sets `bad_alignment` — including the hardest case here, the same sentence three times over
with no punctuation to distinguish the repetitions.

Also on `30s.wav`: phonemes emitted at ≤1 frame drop from 84 % to 14 %, and SIL tokens
landing inside a real (VAD-detected) pause go from 0/6 to 7/9.

- `chunk_audio`, `chunk_seconds`, `chunk_max_phonemes`, `chunk_phonemes_per_second` and
  `chunk_vad` constructor parameters, with matching `--chunk-audio/--no-chunk-audio`,
  `--chunk-seconds`, `--chunk-max-phonemes`, `--chunk-pps` and `--chunk-vad/--no-chunk-vad`
  CLI flags.
- `set_chunking()` to retune a live aligner (mirrors `set_boundary_snap()`), including the
  cut-detection thresholds `derail_frames`, `derail_run`, `backoff`, `conf_floor` and
  `weak_run`.
- `detect_speech_segments()` — a plain energy VAD, exposed as a public utility. Inside BFA
  it does two narrow jobs: measure speech seconds per window so a pause-heavy window is not
  over-budgeted at 15 phonemes/s, and let a cut land in a pause rather than mid-word. It is
  **not** `silence_anchors`: it never splits the target sequence and never touches the
  posteriors.
- `align_segment_chunked()` — the walk itself, callable directly.
- New module `bournemouth_aligner/chunking.py` and `tests/test_chunking.py`.

### Changed
- Segments shorter than the chunking thresholds take the existing single-shot path
  unchanged, so short-clip results are unaffected (verified bit-identical).
- **`silence_anchors` no longer applies inside a chunk window.** It exists to split a long
  segment at transcript punctuation — the job the chunker now does on evidence — and a 10 s
  window does not need splitting. Leaving it on only cost accuracy: on `30s.wav` chunked,
  `silence_anchors=3` scored 0.330 and tripped `bad_alignment` where 0 and 10 both scored
  0.403. Chunked results are now independent of the setting. It still does its original job
  when `chunk_audio=False`, and is unchanged for segments below the chunking threshold.
- The bad-alignment warnings are kept, and now name the relevant knob: whether the segment
  was chunked decides whether the advice is to retune the chunker or to enable it. The hard
  `raise` on a head-vs-tail confidence collapse now fires only when both mitigations are
  off (`chunk_audio=False` and `silence_anchors=0`), and the head-vs-tail warning is
  reported at `warn_level>=1` instead of `>=2`.

### Measured against ground truth (Buckeye `s0103a`, 2267 reference phone boundaries)

The first evaluation of chunking against real boundaries rather than confidence proxies.
Median absolute onset error, by position within the segment:

| decile (0 = start) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| single-shot | 18 | 19 | 19 | 18 | **1582** | **7918** | 7277 | 6117 | 14840 | **29245** |
| chunked | 17 | 20 | 14 | 19 | 18 | 14 | 13 | 18 | 19 | 21 |

| | single-shot | chunked |
|---|---|---|
| median boundary error | 190.4 ms | **16.8 ms** |
| within 25 ms | 31.9 % | **63.2 %** |
| within 50 ms | 41.9 % | **81.4 %** |

Two things this settles. The single-shot decoder is **accurate for the first ~40 % of a
long segment and then detonates** — it does not degrade, and the failure is not a bad start
propagating (no segment in this file had a bad first decile). And chunking removes the
position dependence entirely: 13–21 ms flat across all ten deciles.

`chunk_phonemes_per_second` was swept against the same ground truth:

| phonemes/s | 12 | 15 | 18 | 20 |
|---|---|---|---|---|
| median error | 16.4 ms | 16.8 ms | 17.8 ms | 17.8 ms |
| within 25 ms | 64.6 % | 63.2 % | 60.3 % | 59.9 % |

Monotonic — **less over-provisioning aligns better**. `enforce_all_targets` means the forced
path must emit every target inside the window, so surplus targets take frames from the
phonemes that are genuinely there and distort the whole window, not just its tail. The
default stays at 15: 12 was marginally better on this one conversational file, which is not
enough to move a default that must also serve read speech. The cram still has to be large
enough to *see*, since it is the cut signal.

### Changed
- **`chunk_max_phonemes` is now derived, not chosen**: `(frames_in_window − 1) // 4` = **190**
  for a 10 s window, the largest budget that keeps the CTC lattice at stride 4. New helper
  `chunking.stride4_max_phonemes(chunk_seconds)`, and `PhonemeTimestampAligner` warns on
  construction and on `set_chunking()` when the effective budget overruns it — an 8 s window
  holds only 152, and nothing previously flagged a shortened window crossing the line.
  Scope stated honestly in the code: this is a guard against a stride drop that no tested
  configuration actually hit (cap 190 and 210 gave identical results at 20 /s), not the
  explanation for the rate measurements above.

### Fixed
- **`chunk_seconds` now means what it says.** The routing test carried 10% slack on each
  side, so a nominal `chunk_seconds=10` really meant 11 s: a 10.2 s segment took the
  single-shot path and collapsed while the setting that governs it appeared satisfied. Same
  for `chunk_max_phonemes`. A segment of exactly `chunk_seconds` still takes the cheaper
  single-shot path.
- **Segments between `chunk_seconds` and 1.25 × `chunk_seconds` were silently un-chunked.**
  The rule that extends a window to the end of the segment fired whenever less than a
  quarter of a window remained — but that measures the wrong quantity, because the next
  window starts at the derailment CUT, not at the window's end. A 12 s segment therefore saw
  "2 s left, under 2.5 s, extend" and became one 12 s window: exactly the single-shot decode
  chunking exists to avoid. It now extends only when the remainder could not be aligned at
  all, and an 11.8 s segment splits into two real windows (0–10 s, 8.85–11.8 s) as intended.
- The bad-alignment warning told users to "enable `chunk_audio=True`" when it was already
  enabled and the segment had simply fallen under the threshold. It now distinguishes
  chunked / under-threshold / disabled, names the actual thresholds, and reports the
  segment's phoneme rate when it exceeds `chunk_phonemes_per_second` — which usually means
  the transcript carries text that is not spoken in that segment.
- The chunker applied its derailment cut to the **final** window too, which deleted the
  discarded tail instead of deferring it — there is no next window to re-align it. On
  `30s.wav` this dropped the last three phonemes ("find", at 31.4–31.9 s); on a harder case
  it dropped 54 of 402. The last window now keeps everything and reports the derailment
  without acting on it.
- `process_segments()` emitted `batch_size` placeholder results when a batch slice raised
  `ValueError`, rather than one per segment actually in that slice. A short final slice
  therefore shifted every later segment's timestamps onto the wrong segment.
- Chunked segments report coverage counters once per segment. Per-window counting would
  otherwise tally each over-provisioned tail as "missed" and count re-aligned phonemes
  twice (356 targets were reported as 920).


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
