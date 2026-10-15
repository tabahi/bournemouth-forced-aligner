# BFA developer guide

[Main README](../README.md) · **Developer guide** · [About the models](about-models.md) · [Evaluation](evaluation.md) · [Language support](language-support.md)

This guide is for working on BFA's code. It covers how a call flows through the
package, the rules the code depends on, the full API and CLI reference, the output format, and
how to test and extend the library. For using BFA, the [main README](../README.md) is enough.

> **v2 breaks compatibility with v1.** The acoustic models (p3cupe / p4mbfa instead of
> CUPE-2i / model3a), the decoder (a segmental Viterbi instead of a CTC lattice), the default phoneme
> set (`std`), the output format and many constructor arguments have all changed. v1 checkpoints do not
> load. The `main` branch holds the last v1 release, and v2 is developed on `v2.0-nightly`. The
> [CHANGELOG](../CHANGELOG.md) lists every removed and renamed name.

## Contents

- [Repository layout](#repository-layout)
- [How an alignment runs](#how-an-alignment-runs)
- [Rules the code depends on](#rules-the-code-depends-on)
- [API reference](#api-reference)
- [Output format](#output-format)
- [Command line](#command-line)
- [Long audio and chunking](#long-audio-and-chunking)
- [Models on disk and downloads](#models-on-disk-and-downloads)
- [Testing](#testing)
- [Extending BFA](#extending-bfa)
- [Not ported to v2](#not-ported-to-v2)

## Repository layout

```text
bournemouth_aligner/
  core.py              BFA: the public class. Model loading, batching, routing, output assembly
  presets.py           phoneme set + language -> model name. The ONLY place model names appear
  hub.py               every download (models, G2P), through huggingface_hub's cache
  extractors.py        load_extractor(): a model on disk -> CupeExtractor | MbfaExtractor
  chunking.py          VAD, phoneme budgets and derailment detection for segments > 120 s
  utils.py             confidences, frame -> ms, embedding pooling, TextGrid, mel plot
  cli.py               the `balign` command
  segmental/           what both model families share
    windowing.py         120 ms windows / 80 ms stride, cosine-taper stitching, frame geometry
    extractor.py         checkpoint/release loading, BaseExtractor.encode_clip (forward pass)
    aligner.py           the segmental Viterbi, sub-frame onset readout (port of training loss.py)
  p3cupe/              ph66 model family: model.py (CupeCNN), extractor.py
  p4mbfa/              std model family:  model.py (MbfaCNN, 3 heads), extractor.py (+ inventory check)
  phonemizers/         one per phoneme set, behind one interface
    base.py              Inventory: a head's label space (ids, groups, SIL, pad, output keys)
    std.py               standard_g2p: text -> words -> IPA -> gold -> language-group tokens
    ph66.py              espeak-ng + the ph66 mapper (imported lazily)
  ipamappers/          the ph66 mapper and espeak-ng phonemizer
  standard_g2p/        VENDORED runtime of tabahi/CharsiuG2P's standard_g2p. Never edit by hand
  helpers/mel_spec.py  vocoder-compatible mel spectrogram
  cpp_onnx/            v1 ONNX/C++ port, not updated for v2 (see the end of this page)
evaluation/            the TIMIT/Buckeye benchmark harness (see evaluation/README.md)
tests/                 pytest suite + tests/test_buckeye.py (a quick experiment script)
tools/sync_standard_g2p.py   the only writer of bournemouth_aligner/standard_g2p/
examples/              runnable scripts, one per entry point
docs/                  this documentation
```

## How an alignment runs

```text
BFA(preset="de")
 └─ load_model ─ presets.get_std_preset("de") -> ("mbfa-latin", "de", "latin")
                 hub.resolve_model -> $BFA_MODEL_DIR, ./models, HF cache, Hub
                 load_extractor -> MbfaExtractor (family read from config.json)
                 make_phonemizer("std", "de") -> StdPhonemizer, Inventory "std/latin"
                 extractor.check_inventory(inventory)      # refuses a mismatched token list
                 _resolve_decoder()                        # model's decoder + overrides

process_segments(srt_data, audio_wavs)
 ├─ phonemize every segment        text -> {local_ph, gold_phg, eipa, tone, ..., word_num}
 ├─ chop_wav                       cut, downmix, RMS-normalise, pad to duration_max
 ├─ route by length                > segment_max_seconds  -> align_segment_chunked
 │                                 > duration_max         -> align_segment_whole (unpadded)
 │                                 otherwise              -> batches of batch_size
 ├─ extract_timestamps_from_segment_batch
 │    encode_clip                  windows -> trunk -> heads -> stitched [B, T, C] logits (+ flux)
 │    log_softmax, move to CPU
 │    segmental.align_batch        Viterbi path -> onsets -> refine_onsets (sub-frame)
 │    groups                       take the PHONEME segmentation (no second DP)
 │    _calculate_confidences       peak posterior inside each phone's own segment
 │    convert_to_ms                frame * 5.0 ms + segment start
 └─ post_process_segment           phoneme_ts / group_ts / words_ts, coverage_analysis
```

`process_sentence`, `process_sentences_batch` and `process_srt_file` are thin wrappers around
`process_segments`.

Three details matter when you change this path:

- **A batch is retried one segment at a time** (`_align_slice`) if any of its segments cannot be
  aligned, so one bad segment does not cost the whole batch.
- **The DP always runs on CPU**, with torch pinned to one thread for its duration
  (`segmental.aligner._single_threaded`). The first call at 32 threads on a loaded machine
  took 34-117 s. At one thread it takes 1.7 s.
- **Groups are a relabelling of the phonemes.** `group_ts` reuses the phoneme boundaries. Only the group
  confidences come from the group head.

## Rules the code depends on

Each of these has a comment where it is enforced. Breaking one usually fails silently, which is why
they are listed here.

1. **`time_ms(frame f) = f × 5.0`, exactly.** The window stride (80 ms) must be a whole number of
   output frames. `windowing.frames_geometry` raises otherwise. Every conv kernel must be odd,
   or every timestamp shifts by half a frame. The `model.py` self-tests check this with a click at a known sample.
2. **The decoder is the training decoder.** `segmental/aligner.py` is a port of
   `bfa_models/<family>/loss.py`'s Viterbi. Change it there first, or both stop matching.
3. **Decoder settings come from the model.** Constructor arguments default to `None`, meaning the
   model's `config.json` `decoder` values. A new decoder knob must follow the same rule
   (`segmental.extractor.DECODER_KEYS`).
4. **SIL is not a constant.** It is id 0 in ph66 and 1 in std. It is always passed as `sil_class` from
   the `Inventory`. Ragged batches are padded with `Inventory.pad_id` (`<blank>` in std, the unused
   `noise` slot in ph66). Never pad with std's `noise`, which is a real label there.
5. **A model must match its label space.** `check_inventory` compares a release's stored
   token list (p4mbfa) or phoneme list (p3cupe) with this installation's tables. A mismatch
   would decode without error and return the wrong phonemes.
6. **Model names live only in `presets.py`.** `core.py` never tests a model name.
7. **The cache comes first.** `hub.py` loads anything already in the huggingface_hub cache
   without contacting the Hub. `warn_level=0` prints nothing at all, downloads included.
8. **No compatibility shims.** v2 is a clean release. When renaming or removing something,
   delete the old spelling outright and update the tests and docs. Do not keep aliases.
9. **The vendored `standard_g2p/` is a snapshot.** It is changed only by
   `tools/sync_standard_g2p.py`. Changing the token tables invalidates every released mbfa model
   (`check_inventory` will refuse them).

## API reference

### Constructor

```python
BFA(
    preset="en-us",               # language code; picks the language and, through it, the model
    phoneme_set=None,             # "std" (default) | "ph66"; None = default, or what the model is
    model_name=None,              # a published model by name, or any HF repo id in the same layout
    ckpt_path=None,               # a model on disk: release directory or training checkpoint
    lang="en-us",                 # language code; overrides the preset's language
    g2p_model_dir=None,           # std: existing copy of the G2P model (default $BFA_G2P_DIR / HF cache)
    g2p_device=None,              # std: device for the G2P (default: `device`)
    g2p_cache_size=None,          # std: word -> IPA entries remembered (default 500k; 0 = unbounded)
    punctuation_sil=False,        # std: write a SIL at pause punctuation
    duration_max=30,              # padding width (s) of the batched path; longer segments run alone
    device="auto",                # "auto" | "cpu" | "cuda" | "mps"
    boundary_bonus_weight=None,   # onset bonus; None = the model's value (30); 0 = off
    sub_frame_onsets=True,        # interpolate onsets at the posterior crossing
    min_phone_frames=None,        # duration floor in 5 ms frames; None = the model's (6)
    soft_phone_frames=None,       # soft duration frames above the floor; None = the model's (10)
    bad_confidence_threshold=0.6, # share of low-confidence phones that flags a segment
    low_confidence_level=None,    # None = the family's level (0.2 CUPE-3, 0.1 mbfa)
    segment_max_seconds=120,      # decode whole up to this length, chunk beyond it
    chunk_audio=True,             # chunk segments longer than segment_max_seconds
    chunk_seconds=10.0,           # chunk window length
    chunk_max_phonemes=200,       # cap on a window's phoneme budget
    chunk_phonemes_per_second=15, # budget rate per second of SPEECH
    chunk_vad=True,               # energy VAD for speech seconds and cut placement
    warn_level=1,                 # 0 silent, 1 warnings, 2 + progress bars, geometry, decoder
)
```

**Model priority:** `ckpt_path` → `model_name` → `preset`. A model decides the phoneme set by itself
(p3cupe → ph66, p4mbfa → std). Asking for one set and passing a model of the other raises. With
`preset=None` and no model, only phonemization works until `load_model()` is called.

| parameter | notes |
|---|---|
| `preset`, `lang` | `lang` is overridden by `preset` only while it is still the default. std takes BCP 47 / ISO 639-3 / espeak spellings ([language support](language-support.md)). ph66 takes espeak-ng codes. |
| `model_name` | `mbfa-latin`, `mbfa-latin-benchmark`, `mbfa-cyrillic`, `mbfa-cjk`, `mbfa-japanese`, `CUPE-3-en`, `CUPE-3-en-base`, `CUPE-3-mls7`, or a repo id. |
| `ckpt_path` | A p4mbfa model also fixes the language group, so `lang` must belong to that group. |
| `duration_max` | Memory against batch size. Short segments are padded to this width, which dominates CPU time on short clips (TIMIT: ~1.0 s/clip at 30, ~0.3 s at 10, identical onsets). It never truncates. |
| `boundary_bonus_weight` | Helps boundary F1 on read speech, and hurts per-phone onset accuracy and spontaneous speech with ph66 models. See [the onset bonus](about-models.md#the-onset-bonus). |
| `min_phone_frames` | If a segment cannot fit its text at this floor, it gets a shorter floor automatically. |
| `low_confidence_level` | Confidence scales differ per model ([confidence](about-models.md#confidence)). |
| `segment_max_seconds` | Whole-segment decoding is more accurate than chunking. The cost is ~1.8 s of Viterbi per 32 s of audio. |

### Methods

| method | what it does |
|---|---|
| `load_audio(path_or_tensor, sr=None)` | Load and resample to 16 kHz. RMS normalisation happens later, per segment. |
| `process_sentence(text, wav, extract_embeddings=False, do_groups=False, debug=False)` | One sentence. Returns `{"phoneme_set", "lang", "segments": [one]}`. |
| `process_sentences_batch(texts, wavs, ...)` | One result dict per (text, wav). |
| `process_segments(srt_data, wavs, extract_embeddings=False, do_groups=False, batch_size=16, debug=False)` | Many clips × many timed segments. The main entry point. With embeddings, returns `(results, phoneme_embs, group_embs)`. |
| `process_srt_file(srt_path, audio_path, ts_out_path=None, extract_embeddings=False, vspt_path=None, do_groups=False, batch_size=16, debug=False)` | Whisper-style JSON transcript + audio file → result (written to `ts_out_path`). |
| `phonemize_sentence(text)` | The phonemizer's output. See below. |
| `extract_timestamps_from_segment_batch(wavs, wav_lens, phoneme_sequences, ...)` | The low-level call: padded waveforms + id sequences → 7-tuples `(id, start_frame, end_frame, target_idx, confidence, start_ms, end_ms)`. |
| `load_model(preset=..., model_name=None, ckpt_path=None)` | Switch model, and the language if none was given explicitly. |
| `phonemizer.set_backend(language=...)` | Switch language within the current language group (std) or espeak voice (ph66). |
| `set_chunking(enabled=, seconds=, max_phonemes=, phonemes_per_second=, vad=, segment_max_seconds=)` | Retune chunking on a live aligner. |
| `detect_speech_segments(wav)` | The energy VAD: `[(start_s, end_s), ...]`. |
| `convert_to_textgrid(result, output_file=None, include_confidence=False)` | Praat TextGrid with phoneme, group and word tiers. Returns the string when `output_file` is None. |
| `plot_mel_alignment(result, wav, save_path, ...)` | Mel spectrogram with the alignment drawn on it. |
| `extract_mel_spectrum(wav, sr, vocoder_config)` | HiFi-GAN / BigVGAN-compatible mel, `(frames, mels)`. |
| `framewise_assortment(aligned_ts, total_frames, frames_per_second, gap_contraction=5, select_key="phoneme_id")` | Timestamps → one label per mel frame, for TTS duration targets. |
| `compress_frames(frames)` / `decompress_frames(rle)` | Run-length encoding of frame labels. |
| `download_model(model_name, local_dir=None)` | Resolve a model to a local path. |

`aligner.phonemes_key` / `aligner.phoneme_groups_key` give the per-set key names (`local_ph` /
`gold_phg` in std, `ph66` / `pg16` in ph66).

### `phonemize_sentence`

```python
r = BFA().phonemize_sentence("butterfly")          # std / latin
r["eipa"]      # ['b', 'ə', 't', 'ɝ', 'f', 'ɫ', 'a', 'ɪ']   IPA phonemes as the G2P wrote them
r["mipa"]      # same: the token each one mapped to (differs only on backoff)
r["local_ph"]  # [22, 21, 9, 97, 19, 43, 4, 17]             ids in the 'latin' token list
r["gold_phg"]  # [5, 2, 4, 1, 7, 10, 3, 1]                  15 phoneme groups
r["stress"]    # [0, 1, 0, 0, 0, 0, 2, 0]                   also "tone", "length"
r["words"], r["word_num"], r["n_folded"]                    # n_folded: tokens folded to <unk>

r = BFA(phoneme_set="ph66").phonemize_sentence("butterfly")
r["eipa"]      # ['b', 'ʌ', 'ɾ', 'ɚ', 'f', 'l', 'aɪ']       espeak-ng IPA
r["ph66"]      # [29, 10, 58, 9, 43, 56, 23]
r["pg16"]      # [7, 2, 14, 2, 8, 13, 5]
```

## Output format

```json
{
  "phoneme_set": "std/latin",
  "lang": "en",
  "segments": [{
    "start": 0.0, "end": 1.26, "text": "butterfly",
    "local_ph": [...], "gold_phg": [...], "ipa": [...], "word_num": [...], "words": ["butterfly"],
    "tone": [...], "stress": [...], "length": [...],
    "phoneme_ts": [
      {"phoneme_id": 22, "phoneme_label": "b", "ipa_label": "b",
       "start_ms": 0.0, "end_ms": 52.5, "confidence": 0.89,
       "target_seq_idx": 0, "index": 0, "tone": 0, "stress": 0, "length": 0}
    ],
    "group_ts": [{"group_id": 5, "group_label": "stop_voiced", "start_ms": 0.0, "end_ms": 52.5, "confidence": 0.46, "target_seq_idx": 0, "index": 0}],
    "words_ts": [{"word": "butterfly", "start_ms": 0.0, "end_ms": 982.5, "confidence": 0.33, "local_ph": [...], "ipa": [...]}],
    "coverage_analysis": {"target_count": 8, "aligned_count": 8, "missing_count": 0, "extra_count": 0,
                          "coverage_ratio": 1.0, "missing_phonemes": [], "extra_phonemes": [],
                          "bad_alignment": false}
  }]
}
```

- `phoneme_set` names the inventory every id belongs to (`"ph66"` or `"std/<group>"`). The same
  number is a different phoneme in each.
- ph66 segments use `ph66` / `pg16` instead of `local_ph` / `gold_phg`, have no `tone` /
  `stress` / `length`, and `words_ts[].ph66` instead of `words_ts[].local_ph`.
- Phones tile the segment with no gaps: each phone ends where the next placed phone starts.
- `target_seq_idx` is the phone's index in the phonemized sequence. A position that is missing from
  `phoneme_ts` is a `SIL` the path skipped (the speaker did not pause there). Words are matched to phones by
  `target_seq_idx`, not by position.
- `group_ts` is filled only with `do_groups=True`.
- `coverage_analysis["bad_alignment"]` is set when more than `bad_confidence_threshold` of a
  segment's phones (over 60 phones) fall below `low_confidence_level`, or when confidence is
  healthy at the start of a segment and collapses at the end (an alignment that ran away from its audio).
- `phoneme_label` is the model's token, and `ipa_label` is what the phonemizer wrote. ph66 splits a
  compound phoneme into two entries, and the second has `ipa_label: "-"`.

## Command line

```text
balign AUDIO TEXT_OR_SRT [OUTPUT] [options]
```

`TEXT_OR_SRT` is a quoted sentence, or the path of a Whisper-style JSON
(`{"segments": [{"start", "end", "text"}]}`), which is detected when the file exists. `OUTPUT` defaults to
`<audio>.vs.json`.

| option | default | |
|---|---|---|
| `--preset LANG` | `en-us` | language; picks the model |
| `--phoneme-set std\|ph66` | `std` | with `--model` / `--ckpt-path`: what that model is |
| `--lang CODE` | from preset | override the language only |
| `--model NAME` | from preset | published model or HF repo id |
| `--ckpt-path PATH` | — | release directory or training checkpoint |
| `--device auto\|cpu\|cuda` | `auto` | |
| `--g2p-device DEVICE` | `--device` | std only |
| `--punctuation-sil / --no-punctuation-sil` | off | std only |
| `--duration-max FLOAT` | **60** | padding width of the batched path (the Python default is 30) |
| `--boundary-bonus-weight FLOAT` | model's | `0` = off |
| `--sub-frame-onsets / --frame-grid-onsets` | sub-frame | |
| `--min-phone-frames INT` | model's | |
| `--segment-max-seconds FLOAT` | 120 | |
| `--chunk-audio / --no-chunk-audio` | on | |
| `--chunk-seconds`, `--chunk-max-phonemes`, `--chunk-pps` | 10, 200, 15 | |
| `--chunk-vad / --no-chunk-vad` | on | |
| `--textgrid PATH` | — | also write a Praat TextGrid |
| `--mel-path PATH` | — | `.png` plot or `.pt` tensor of the mel + alignment |
| `--embeddings PATH` | — | pooled per-phoneme embeddings (`.pt`) |
| `--debug` | off | per-segment and per-window progress |
| `--list-presets` | | print both sets' languages and exit |
| `--version` | | |

## Long audio and chunking

Segments up to `segment_max_seconds` (120 s) are decoded in one pass. The decoder has no band
constraint and does not drift on long audio: on `30s.wav` (32 s, 356 phones) the last phone ends at
31.88 s of 31.88 s. Chunking exists only to bound Viterbi time and memory.

Beyond 120 s, `align_segment_chunked` walks overlapping windows (`chunk_seconds`). Each window
gets a phoneme budget from its **speech** seconds (`chunk_vad`) at `chunk_phonemes_per_second`, capped by
`chunk_max_phonemes` and by what its frames can hold at the duration floor. A window is trusted
only up to `TRUST_WINDOW_FRACTION` (75%) of its audio and up to the first derailment
(`chunking.find_derailment`: collapsing durations or confidence). The rest is handed to the next window,
whose cut is snapped into a pause. This decoder cannot cram, so an over-budgeted window places its surplus
early rather than failing. The time cap is what stops that:

| `30s.wav` forced through 10 s windows | last phone ends | mean confidence | flagged |
|---|---|---|---|
| count trim only | 30.31 s | 0.217 | yes |
| time cap (current) | 31.87 s | 0.317 | no |
| whole-segment decode | 31.88 s | 0.319 | no |

## Models on disk and downloads

`hub.resolve_model(name)` looks in this order:

1. `$BFA_MODEL_DIR/<name>/` (`config.json` + `model.safetensors`)
2. `./models/<name>/`
3. the huggingface_hub cache (`~/.cache/huggingface`, moved with `$HF_HOME`), with no network call
4. `huggingface.co/Tabahi/<name>`, downloading those two files only (~55 MB)

The G2P model `charsiu/g2p_multilingual_byT5_small_100` (~1.2 GB) plus the `google/byt5-small`
tokenizer are fetched on the first call that phonemizes std text. `$BFA_G2P_DIR` /
`g2p_model_dir=` point at an existing copy. One G2P instance is shared per (directory, device).

A repository updated on the Hub is **not** picked up by a machine that already has it cached.
Delete it from the cache to refresh.

## Testing

```bash
python -m pytest tests/ -rs                 # the suite; -rs shows why anything was skipped
python -m pytest tests/test_std.py          # std / p4mbfa, no downloads (random-weight model)
python tests/test_buckeye.py --n 40 --seed 0                       # quick decoder experiment
python tests/test_buckeye.py --n 40 --cfg '{"boundary_bonus_weight": 0}'
```

| file | covers | needs |
|---|---|---|
| `test_core.py` | end-to-end alignment, output structure, TextGrid, embeddings | released models (downloads on first run) |
| `test_std.py` | std phonemizer, inventory check, release/checkpoint loading, full pipeline | nothing: builds a random-weight p4mbfa model |
| `test_chunking.py` | VAD, budgets, derailment detection | nothing |
| `test_cli.py` | `balign` flags, preset resolution, end-to-end CLI | released models |
| `test_eval_labels.py` | TIMIT/Buckeye label → std token tables | nothing |
| `test_buckeye.py` | not a pytest file: boundary F1 on random Buckeye segments | the Buckeye corpus (`--clip`, `--corrupt` for noisy transcripts) |

Pitfalls:

- Run from the repository root with `PYTHONPATH=.` (or an editable install,
  `pip install -e .`). A non-editable install in site-packages shadows the checkout, and
  goes stale between installs.
- `test_cli.py` runs the `balign` on your `PATH` when there is one, and **skips** (it does not fail) when a
  CLI run fails. After touching the CLI, also run
  `python -m bournemouth_aligner.cli examples/samples/audio/109867__timkahn__butterfly.wav "butterfly" --mel-path /tmp/m.png`.
- After removing a name, grep for the bare variable too: a dangling reference in a rarely-run
  path (plotting, CLI options) will not show up in the main tests.
- Scores depend on batch composition (the decoder's class prior is per batch). Compare decoder
  changes on the same files with the same `batch_size`.
- `python bournemouth_aligner/p4mbfa/model.py` (and `p3cupe/model.py`) runs the geometry self-test.

For accuracy, use the full harness in [`evaluation/`](../evaluation/README.md). The current numbers
are in [evaluation.md](evaluation.md).

## Extending BFA

### A new model for an existing language group

1. Train and export it in [bfa_models](https://github.com/tabahi/bfa_models)
   (`p4mbfa/export_hf.py export`). The release must carry the token list of the vendored
   standard_g2p (fingerprint in `standard_g2p/VENDORED.md`).
2. Test it with `BFA(ckpt_path="<release dir>", lang="...")` and score it with `evaluation/`: add it
   to `models:` in `evaluate.yaml` as a `ckpt_path` entry.
3. To make it a preset, publish it under `Tabahi/` and change the name in `presets.py`.

### A new language group (e.g. `vietnamese`)

The token list and the language membership already exist in standard_g2p. Train a model with
the group's yaml in `bfa_models/p4mbfa/`, add `"vietnamese": "mbfa-vietnamese"` to
`presets.P4MBFA_MODELS`, and add the group to [language-support.md](language-support.md).
Nothing in `core.py` changes. If the group needs a word segmenter, it is already handled in
`standard_g2p/word_segmentation.py`. Add the package as an extra in `pyproject.toml` and `setup.py`.

### Updating standard_g2p

```bash
python tools/sync_standard_g2p.py /path/to/CharsiuG2P
```

This copies the runtime modules verbatim and writes `VENDORED.md`. If the gold inventory
fingerprint changes, every released mbfa model stops loading (on purpose), and has to be retrained
or re-exported against the new tables.

### A decoder change

Make it in the training repository's `loss.py` first, then port it to
`segmental/aligner.py`. Read any new setting from `config.json` `decoder` (add it to
`DECODER_KEYS`). Measure it with `tests/test_buckeye.py` on a fixed seed, then with the full harness.
The decoder's Python loop runs once per frame, so a change that slows it is felt on every call.

### A new phoneme set

Implement a phonemizer with `phonemize_sentence(text)` and `set_backend(language)` that returns
the keys listed in `phonemizers/base.py`, plus an `Inventory` for its label space. Then register it in
`phonemizers/__init__.py:make_phonemizer`, and add it to `presets.PHONEME_SETS` / `FAMILY_OF_SET`.

## Not ported to v2

- **ONNX / C++** (`bournemouth_aligner/cpp_onnx/`): this still targets the v1 model and its
  `extract_timestamps_from_segment_simplified` pipeline. Neither exists in v2.
- **Non-forced alignment** (`enforce_all_targets=False` in v1, which could skip words missing from the
  audio): this has no counterpart in the segmental decoder. v2 always aligns every phone, and reports
  doubtful segments through `coverage_analysis["bad_alignment"]`.
