# Changelog

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

[1.1.2]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v1.1.2
[1.1.0]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v1.1.0
[0.1.7]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v0.1.7
[0.1.6]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v0.1.6
[0.1.5]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v0.1.5
[0.1.4]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v0.1.3
[0.1.2]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v0.1.2
[0.1.1]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v0.1.1
[0.1.0]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v0.1.0
