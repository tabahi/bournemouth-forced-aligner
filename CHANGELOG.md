# Changelog

## [0.1.8] - UNRELEASED
- `"ipa_label"` key in `["segments"]["phoneme_ts"]` and in `["segments"]["words_ts"]`in the output dictionary denotes the original IPA phonemes from espeak-ng, instead of the reduced mapped IPA dictionary. It's helpful if you want to use the full epeak IPA dictionary after alignment.
- Enabled mps device support for Apple silicon

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

[0.1.7]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v0.1.7
[0.1.6]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v0.1.6
[0.1.5]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v0.1.5
[0.1.4]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v0.1.3
[0.1.2]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v0.1.2
[0.1.1]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v0.1.1
[0.1.0]: https://github.com/tabahi/bournemouth-forced-aligner/releases/tag/v0.1.0
