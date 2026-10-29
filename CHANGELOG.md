# Changelog

## [2.0.1] - 2026-10-29

### Added
- **`n_jobs`** constructor argument: above 1, `process_segments` (and `process_srt_file`,
  `process_sentences_batch`) runs each batch's alignment search in worker processes while the
  network handles the next batch. The output is identical to `n_jobs=1`. With a GPU, 10 Buckeye
  interviews align 2.4× faster at `n_jobs=4`; on a CPU it changes little. Scripts that set it
  need `if __name__ == "__main__":`. `close()` stops the workers.
- Timings for v2.0.1 against MFA 3.4.3 in [docs/evaluation.md](docs/evaluation.md#cost).

### Changed
- **Faster G2P for the std set.** Each `process_segments` call runs all of its new words through
  the G2P model together (up to 128 words per batch) instead of once per sentence: 4-21× less
  G2P time, identical phonemes.
- **Decoder defaults** (`presets.DECODER_DEFAULTS`), from a decode-time sweep on TIMIT and
  Buckeye: onset bonus `boundary_bonus_weight` 15 (was 30) and contrast window 33 frames (was
  17), the same for every model. 15 is chosen for conversational speech. The model files are
  unchanged; pass `boundary_bonus_weight=30, boundary_contrast_frames=17` for the 2.0.0
  behaviour.
- README rewritten as a getting-started guide.

### Added
- **Five more language groups**: `mbfa-vietnamese` (vi), `mbfa-thai` (th), `mbfa-abjad` (ar, ckb, fa,
  sd, ur; also aligns ug, syc), `mbfa-brahmic` (hi, or, ta; also aligns sa) and
  `mbfa-other-alphabetic` (am, el, hy, ka, ko; also aligns grc). The std set now covers 82
  languages in 9 groups. Thai needs `pip install "bournemouth-forced-aligner[th]"`.
- `mbfa-latin-noEnglish`, selectable by name: the `mbfa-latin` recipe trained without English.
- `boundary_contrast_frames` argument and `--boundary-contrast-frames` CLI option.

### Fixed
- `balign --version` reported 2.0.0.
- `examples/run_simplipied_pipeline.py` read the wrong fields of the timestamp tuples, and
  `examples/read_embeddings.py` needed a file that no example wrote.
- The sample outputs in `examples/samples/` were still in the 1.x format.

## [2.0.0] - 2026-10-15

**Not compatible with 1.x.** New acoustic models and a different alignment method. v1
checkpoints do not load, and the output format and constructor arguments have changed. Details are in
[docs/](docs/README.md).

### Added
- **`std` phoneme set, now the default.** It phonemizes with a ByT5 G2P (vendored standard_g2p) and aligns with
  p4mbfa models, one per language group: `mbfa-latin`, `mbfa-latin-benchmark`,
  `mbfa-cyrillic`, `mbfa-cjk`, `mbfa-japanese`. It covers 63 languages, including Mandarin, Cantonese and
  Japanese, with per-phone tone, stress and length.
- **New ph66 models**: `CUPE-3-en`, `CUPE-3-en-base`, `CUPE-3-mls7`.
- Models download from the Hugging Face Hub (`Tabahi/`), cache first. Local models load with
  `ckpt_path=`, `$BFA_MODEL_DIR` or `./models/`.
- New arguments: `phoneme_set`, `ckpt_path`, `g2p_model_dir`, `g2p_device`, `g2p_cache_size`,
  `punctuation_sil`, `boundary_bonus_weight`, `sub_frame_onsets`, `min_phone_frames`,
  `soft_phone_frames`, `low_confidence_level`.
- Install extras: `[ja]`, `[yue]`, `[th]` (word segmenters), `[ph66]` (espeak-ng), `[audio]`
  (formats other than WAV/FLAC/OGG/MP3).
- An evaluation harness (`evaluation/`, configured by `evaluate.yaml`), with MFA as a baseline. Full TIMIT
  and Buckeye results are in [docs/evaluation.md](docs/evaluation.md).
- Documentation in `docs/`.

### Changed
- Acoustic models with no CTC and no blank class, at 5 ms frames (was 13.125 ms). A boundary is read where two
  phone posteriors cross, interpolated between frames.
- One decoder: a segmental Viterbi, the same one that built the models' training targets.
- Decoder settings default to the values each model was published with.
- Results carry a `phoneme_set` / `lang` header. std segments use `local_ph` / `gold_phg` and
  add `tone`, `stress`, `length`.
- `confidence` is the peak posterior inside each phone. Its scale differs per model, so
  `low_confidence_level` defaults to each model family's own level.
- Audio is read with soundfile, so WAV, FLAC, OGG and MP3 need no FFmpeg and no matching torchcodec build.
- Long audio is decoded in one pass up to 120 s (`segment_max_seconds`, renamed from
  `segmental_max_seconds`), and chunked beyond that.
- `warn_level=0` prints nothing, downloads included.

### Removed
- v1 models (model2i, model3a) and the CTC decoder.
- Arguments `mapper`, `cupe_ckpt_path`, `decoder`, `silence_anchors`, `boost_targets`,
  `enforce_minimum`, `enforce_all_targets`, `ensure_completeness`, `ignore_noise`,
  `extend_soft_boundaries`, `boundary_softness`, `boundary_snap_*`.
- `is_estimated` in the output, `extract_timestamps_from_segment_simplified()`, and `set_boundary_snap()`.
- Non-forced alignment. Every phone of the transcript is placed, and doubtful segments are
  flagged in `coverage_analysis["bad_alignment"]`.

### Fixed
- Word timestamps are matched to phones by `target_seq_idx`, so a skipped pause no longer shifts the words after it.
- One segment whose text cannot fit its audio no longer fails the whole batch.
- The long-audio chunker no longer runs ahead of the audio.
- The first alignment no longer stalls for tens of seconds on many-core machines.
- `load_model(preset=...)` switches the language too.

## [1.2.0] - never released

Development of the segmental decoder on the v1 models. That work became the basis of 2.0.0.

## [1.1.5] - 2026-06-05
- Default `boundary_softness=3` is changed from 7 to 3 because at 3 it performs the best on TIMIT and buckeye dataset. 
- Added warnings and parameter `self.warn_level` for verbose level. Use `BFA.warn_level=3` for high verbosity.
- Fixed small segments alignment. Previously one-word or half-words ("umm or oh") raised error of not enough phonemes for the given duration. That is patched up here: https://github.com/tabahi/bournemouth-forced-aligner/commit/e58757b5ff9c3cb71303067a8050b06d01f80097#diff-c530cc1a536c371d38e233b3192f5f7935f97342d6460e039edc8107fbdf0348



## [1.1.4] - 2026-02-28
- `batch_size=16` parameter in `process_srt_file()` and `process_segments()`. Reduce it to prevent out-of-memory errors when processing large audio segments.

## [1.1.3] - 2026-02-28
- Fixed the bug: The viterbi algorithm didn't really do "forced" alignment before since v1.1.0. Now it does as long as `enforce_all_targets=True` is set.
- Added another layer for completeness. `ensure_completeness=True` (default: False). This is a post-processing hack for completeness. It tries to insert missing phonemes in their most likely positions based on the surrounding aligned phonemes and the original target sequence.
- [torchcodec](https://github.com/meta-pytorch/torchcodec) dependency.  

## [1.1.2] - 2026-02-23
- Added `load_model()` so that the models can be loaded after `__init__`, while the class `BFA(lang ='en-us') can be initialized without specifying a model.`
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
- `ipa_label` field in `phoneme_ts` and `group_ts` output entries - contains the original espeak-ng IPA phoneme, before mapping to the reduced 66-class set. Useful when you need the full espeak IPA dictionary after alignment.
- `is_estimated` field in `phoneme_ts` and `group_ts` entries - indicates whether a phoneme was directly aligned by Viterbi (`false`) or inserted by the coverage enforcement post-processing (`true`). Downstream tasks can use this to filter out forced insertions.
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
- **Core Alignment**: BFA class for programmatic use
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


[2.0.0]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v2.0.0
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
