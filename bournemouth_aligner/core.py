''' 2026-02-13
Aligned Phonemes Extraction Module

This module extracts phoneme-level timestamps from audio files using a pre-trained CUPE model.
It includes functionality for processing audio segments, extracting phoneme timestamps, and pooling embeddings.
It performs forced alignment with target boosting and confidence scoring.

Inputs: SRT files with segments, audio files.
Outputs: JSON files (.vs2.json) with phoneme timestamps, phoneme groups timestamps, and confidence scores, pooled embeddings.
'''


import torch
import torchaudio
import torch.nn.functional as F
import contextlib
import math
import os
import json
import traceback

# repo modules
from .cupe2i.model2i import CUPEEmbeddingsExtractor as CUPEEmbeddingsExtractor_i
from .cupe2i.model2j import CUPEEmbeddingsExtractor as CUPEEmbeddingsExtractor_j
from .cupe2i.model3a import CUPEEmbeddingsExtractor as CUPEEmbeddingsExtractor_3a
from .cupe2i.windowing import slice_windows, stich_window_predictions, calc_spec_len_ext, frames_geometry
from .forced_alignment import AlignmentUtils
from .segmental import decode_segmental
from . import chunking as _chunking
from .chunking import (DEFAULT_CHUNK_SECONDS, DEFAULT_MAX_PHONEMES,
                       DEFAULT_PHONEMES_PER_SECOND, DEFAULT_DERAIL_FRAMES,
                       DEFAULT_DERAIL_RUN, DEFAULT_BACKOFF, DEFAULT_CONF_FLOOR,
                       DEFAULT_WEAK_RUN, DEFAULT_SNAP_TO_SILENCE_SEC,
                       MIN_KEEP_FRACTION, NO_DERAIL_TRIM_FRACTION,
                       stride4_max_phonemes)
#from .ipamappers.ph66_mapper import phoneme_mapped_index, phoneme_groups_index, phoneme_groups_mapper
from .ipamappers import ph66_phonemeizer
from .presets import get_preset
from .utils import dict_to_textgrid, weighted_pool_embeddings, _calculate_confidences, convert_to_ms, detect_device_automatically, plot_mel_alignment as _plot_mel_alignment
# Create reverse mappings for interpretability
#index_to_glabel = {v: k for k, v in phoneme_groups_index.items()}
#index_to_plabel = {v: k for k, v in phoneme_mapped_index.items()}

# Which tensor onset snapping reads, plus `'none'` for no snapping at all ("vanilla" - the raw
# Viterbi framestamps, exactly what BFA produced before onset snapping existed).
# See `boundary_snap_source` in PhonemeTimestampAligner.
SNAP_SOURCES = ('pre_transformer', 'spectral_stream', 'boundary_head', 'embeddings', 'auto',
                'none')

# What `'auto'` resolves to, BY ARCHITECTURE. `tmp/results/conclusions.md` §10 - the first
# comparison with `none` controls on BOTH corpora - measured, as gain in r_value_25ms over
# not snapping at all:
#
#                        TIMIT ue03f_e49   TIMIT ue03g_e19   Buckeye ue03f_e49
#   spectral_stream          +0.0552           +0.0565           +0.0260
#   pre_transformer          +0.0336           +0.0372           +0.0144
#   boundary_head (trained)      -              -0.0168         (control invalid)
#   embeddings               -0.0243           -0.0205           -0.0253
#
# So for model3a, `spectral_stream` is the best real source on both corpora and is above
# control on both. For model2i the ranking inverts on Buckeye (§2): on the `en` checkpoint
# `spectral_stream` scores 0.5174 against a 0.5437 control - BELOW no snapping - while
# `pre_transformer` stays above it at 0.5516. `pre_transformer` is the only source above
# control on both corpora for model2i, so the two families get different answers.
#
# A TRAINED boundary head is never selected automatically: it is the one "real" source that
# scores BELOW not snapping (-0.0168 on 6300 TIMIT files, -0.028 on the 400-clip probe in
# §8). It learned the decoder's own placement (AUC 0.864 vs its own decoder, against 0.70-0.75
# for the flux taps), which means it reproduces the decoder's lateness instead of correcting
# it. Ask for it explicitly if you want it.
AUTO_SNAP_MODEL3A = 'spectral_stream'
AUTO_SNAP_MODEL2I = 'pre_transformer'
# Back-compat alias for the pre-1.3 name, which had a single architecture-independent value.
AUTO_SNAP_FALLBACK = AUTO_SNAP_MODEL2I

# Phoneme decoders, see `decoder` in PhonemeTimestampAligner and `segmental.py`.
DECODERS = ('segmental', 'ctc')
# Windows per CUPE forward pass (one window per 80 ms of audio, so ~64 s of audio per pass).
WINDOW_BATCH = 800
# Default for `segmental_max_seconds`: longer segments fall back to the chunker, to bound
# memory and time. Measured peaks decoding one segment whole (model3a, CUDA):
#     120 s: GPU 658 MB, +460 MB RAM, 21 s        225 s: GPU 766 MB, +470 MB RAM, 45 s
# The chunker costs accuracy with this decoder (225 s Buckeye segment: 67.3 whole, 54.7
# chunked), so raise it when the hardware allows.
DEFAULT_SEGMENTAL_MAX_SECONDS = 120.0

# Default search half-width per source, in FRAMES, used when neither `boundary_snap_radius`
# nor `boundary_snap_radius_ms` is given. Frames are the unit of record because the snapper
# searches whole frames - a ms value is rounded to frames before it is used, so two different
# ms settings routinely mean the same search. (The eval CSVs show the cost of not fixing a
# unit: `cfg_snap_radius` holds `52.5` on some rows and `2`/`4`/`6` on others, i.e. ms and
# frames in one column.) `boundary_snap_radius_ms` still works and still wins over these.
#
# 4 frames is the only radius sweep on a FULL corpus (`results_TIMIT_lev.csv`, `en` model2i,
# spectral_stream): r=2 -> 0.6522, r=4 -> 0.6897, r=6 -> 0.6788. The 100-file TIMIT sweeps in
# §4 agree that the flux taps are flat over 2-4 frames.
#
# CAVEAT, and the reason this is worth re-measuring: every model3a full-corpus row on both
# corpora (§10, the rows the source defaults above rest on) was taken at r=2, not r=4, and the
# 20-file Buckeye sweeps prefer r=1 and fall with every step. There is no full-Buckeye radius
# sweep for any checkpoint. 4 stays the default because it is the only measurement on a whole
# corpus, not because the question is settled.
DEFAULT_SNAP_RADIUS_FRAMES = {
    # NOT a measurement on a trained head. §8b: this was tuned on `ue03a_e100`, whose head is
    # constant, so its "radius falloff" is a property of the fixed shift a constant head
    # produces (`1 + offset - radius` frames), not of any detector. Kept small because that
    # is the conservative direction for a source `auto` no longer selects.
    'boundary_head': 2,
}
DEFAULT_SNAP_RADIUS_FRAMES_FALLBACK = 4     # every flux tap

# Deprecated ms-valued equivalents, kept so existing imports keep working. Derived from the
# frame defaults at the 210-sample stride (13.125 ms/frame) that every released checkpoint
# uses; on a `cnn_last_stride: 1` model the frame values above are still right and these are
# not, which is the other reason frames are now the unit of record.
DEFAULT_SNAP_RADIUS_MS = {k: v * 13.125 for k, v in DEFAULT_SNAP_RADIUS_FRAMES.items()}
DEFAULT_SNAP_RADIUS_MS_FALLBACK = DEFAULT_SNAP_RADIUS_FRAMES_FALLBACK * 13.125



class PhonemeTimestampAligner:
    """
    Align phoneme-level timestamps from audio using a pre-trained CUPE model
    and Viterbi decoding for forced alignment.
    URL: https://github.com/tabahi/bournemouth-forced-aligner
    """

    def __init__(self, preset="en-us", model_name=None, cupe_ckpt_path=None, lang='en-us', mapper="ph66", duration_max=30, device="auto", silence_anchors=10, boost_targets=True, enforce_minimum=True, enforce_all_targets=True, ensure_completeness=False,ignore_noise=True, extend_soft_boundaries=True, boundary_softness=3, bad_confidence_threshold=0.6, boundary_snap_radius_ms=None, boundary_snap_radius=None, boundary_snap_offset=0, boundary_snap_source="auto", chunk_audio=True, chunk_seconds=DEFAULT_CHUNK_SECONDS, chunk_max_phonemes=DEFAULT_MAX_PHONEMES, chunk_phonemes_per_second=DEFAULT_PHONEMES_PER_SECOND, chunk_vad=True, decoder="segmental", segmental_max_seconds=DEFAULT_SEGMENTAL_MAX_SECONDS, warn_level=1):
        """
        Initialize the BFA phoneme timestamp extractor.

        Args:
            preset (str or None): Language preset for automatic model and language selection.
                If None, No model will be loaded by default, you must use load_model() with explicit parameters before using the alignment functionality.
                Supports 80+ languages including English variants, European languages,
                Indo-European families, and closely related languages. Examples: "en-us", "de", "fr", "hi", "ar", etc.
                Note: Tonal languages (Chinese, Vietnamese, Thai) and distant language families are not supported.
                See README.md for complete preset list.
            model_name (str, optional): Name of the pre-trained model to use.
                Overrides preset model selection. Available models:
                - "en_libri1000_ua01c_e4_val_GER=0.2186.ckpt" (English)
                - "multi_MLS8_uh02_e36_val_GER=0.2334.ckpt" (8 European languages)
                - "large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt" (Universal)
                - "multi_mswc38_ug20_e59_val_GER=0.5611.ckpt" (Universal)
                Models downloaded from: https://huggingface.co/Tabahi/CUPE-2i/tree/main/ckpt
            cupe_ckpt_path (str, optional): Direct path to CUPE model checkpoint.
                Highest priority - overrides both preset and model_name.
            lang (str): Language code for phonemization (espeak format).
                Examples: "en-us", "de", "fr", "hi". Only overridden by preset if using default.
                See: https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md
            mapper (str): Phoneme mapper to use (default: "ph66").
            duration_max (float): Maximum segment duration in seconds for padding (10-60 recommended).
            output_frames_key (str): --- Deprecated, as it was unused --- Frame output format. Options: "phoneme_id", "phoneme_label",
                "group_id", "group_label".
            device (str): Processing device ("cpu", "cuda", "mps", or "auto" to detect automatically).
            silence_anchors (int): Silent frames threshold for segment splitting (0 to disable).
            boost_targets (bool): Enhance target phoneme probabilities for better alignment.
            enforce_minimum (bool): Ensure minimum probability threshold for target phonemes.
            enforce_all_targets (bool): Force the (viterbi) algorithm to take the path via target phonemes regardless of their probabilities, this is what "forced" alignment means in the context of Viterbi decoding. If false, it can miss some phonemes if their probabilities are too low, but if true, it WILL have 100% coverage of the target phonemes.
            ensure_completeness (bool): This is a post-processing hack for completeness. It tries to insert missing phonemes in their most likely positions based on the surrounding aligned phonemes and the original target sequence. Can be used in conjunction with enforce_all_targets for extra safety, or can be used on its own to try to recover missed phonemes without forcing the alignment path.
            ignore_noise (bool): Skip predicted noise in alignment output.
            extend_soft_boundaries (bool): Enable the extension of the phoeneme start/end boundaries beyond the core of the phoneme. Use `boundary_softness` to control the leniency of the extension. This can help capture more of the phoneme duration, especially for softer phonemes or in cases where the model's confidence extends beyond the core frames.
            boundary_softness (int): Hyperparameter controlling leniency of boundary extension beyond the core of the phoneme. Default is 3, which corresponds to a threshold of 0.0000001. Set it to 0 or 1 if you want only the cores of the phonemes, or set it to 7 to allow more extension as long as there's any meaningful confidence in the frames between the core and the extended boundary.
            bad_confidence_threshold (float): Threshold for flagging low-confidence alignments (1 to disable), default 0.6,  0.6 would mean if 60% of phonemes have low confidence, a warning is issued, in "segments", ["coverage_analysis"]["bad_confidence"] is set to true. It's recommended to avoid bad-confidence segments for the downstream tasks.
            boundary_snap_source (str): Which tensor onset snapping reads, or "none" to disable
                snapping and emit the raw Viterbi framestamps (BFA's pre-1.2 behaviour).
                One of "auto" (default), "boundary_head", "pre_transformer", "spectral_stream",
                "embeddings", "none". "auto" branches on the checkpoint's ARCHITECTURE:
                "spectral_stream" for model3a, "pre_transformer" for model2i/model2j - the two
                families do not rank the sources the same way once Buckeye is included. It never
                selects a trained boundary head, which measures below no snapping at all. See
                AUTO_SNAP_MODEL3A at the top of this module for the numbers, and the extended
                commentary below the assignment in __init__ for the full source comparison.
            boundary_snap_radius (int or None): Search half-width in FRAMES - the unit the
                snapper actually works in, since it searches a whole number of frames either
                side of each onset. None (default) picks the measured optimum for the resolved
                source: 4 frames for the flux taps, 2 for "boundary_head". 0 disables snapping.
            boundary_snap_radius_ms (float or None): The same half-width in MILLISECONDS,
                rounded to frames at the loaded model's stride. Used only when
                `boundary_snap_radius` is None. Worth reaching for when comparing models with
                different frame strides (13.125 ms/frame at a 210-sample stride, 6.5625 at
                105), where one frame count means two different durations; for a single model,
                ask in frames. 0 disables snapping.
            boundary_snap_offset (int): Frames added after the snap argmax. 0 is correct for the
                indexing convention and was best in every sweep; only change it when probing.
            chunk_audio (bool): Align long segments by walking overlapping windows instead of
                decoding them in one shot. ON by default, and it only engages for segments past
                `chunk_seconds` of audio or `chunk_max_phonemes` targets - shorter segments take
                the single-shot path unchanged, so this cannot alter existing short-clip results.
                Set False to restore the pre-1.3 whole-segment behaviour. See `chunking.py` for
                why one-shot decoding collapses rather than degrades on long clips, and
                `set_chunking()` to retune on a live aligner.
            chunk_seconds (float): Audio window length in seconds (default 10). Windows overlap:
                each one restarts at the end of the previous window's last TRUSTED phoneme, not
                at a fixed offset, so the overlap is however much of the previous window was
                discarded.
            chunk_max_phonemes (int): Cap on the phoneme budget handed to one window (default
                200). The budget is deliberately larger than the window can hold - the surplus
                crams into the final frames at one frame apiece, and that cram is what marks
                where the window's alignment stopped being trustworthy.
            chunk_phonemes_per_second (float): Phonemes per second of SPEECH used to size each
                window's budget (default 15, a safe upper bound for English). Measured against
                speech seconds, not wall-clock seconds, which is what `chunk_vad` is for.
            chunk_vad (bool): Use a plain energy VAD to (a) measure speech seconds per window so
                a pause-heavy window is not over-budgeted, and (b) move a cut into a nearby pause
                when there is one. This is NOT `silence_anchors`: it never splits the target
                sequence and never touches the posteriors. Disabling it makes every window budget
                as if it were entirely speech.
            decoder (str): Which Viterbi places the phonemes. "segmental" (default) assigns every
                frame to a phone or a pause, with a 2-frame minimum phone duration, and places
                each onset using the posteriors AND the boundary tap's acoustic change in the
                same pass (see `segmental.py`). "ctc" is the original blank/spike CTC lattice
                followed by greedy onset snapping (BFA <= 1.1.5 behaviour). Measured on Buckeye,
                onset F1 at 25 ms: 59.0 (ctc) -> 68.2 (segmental) over 200 segments.
                `enforce_all_targets` keeps its meaning in both: True aligns every target,
                False lets the segmental decoder skip words the audio does not contain and
                absorb speech the text does not contain. The segmental decoder needs a boundary
                tap, so it falls back to "ctc" when `boundary_snap_source='none'`, and for
                sub-state models. `boost_targets`, `enforce_minimum` and `silence_anchors` are
                CTC-lattice settings and do not apply to it.
            segmental_max_seconds (float): With the segmental decoder, segments up to this long
                are decoded whole and never chunked; longer ones use the chunker (when
                `chunk_audio=True`) to bound memory and time. Default 120. Whole-segment decoding
                is more accurate (a 225 s Buckeye segment: 67.3 onset F1 whole vs 54.7 chunked)
                and its memory is modest (225 s: 766 MB GPU peak, ~470 MB extra RAM), so raise it
                if the hardware allows.
            warn_level (int): Verbosity. 0 = silent, 1 = important warnings only (default),
                2 = warnings plus informational output (checkpoint architecture, resolved frame
                geometry, snap configuration, low-confidence diagnostics). Critical errors are
                raised as exceptions regardless of warn_level.
        Parameter Priority (highest to lowest):
            1. Explicit cupe_ckpt_path
            2. Explicit model_name
            3. Preset values (only if no explicit model specified)
            4. Default values

        Available Models:
            - en_libri1000_ua01c_e4_val_GER=0.2186.ckpt: Best for English (1000hrs LibriSpeech)
            - en_libri1000_uj01d_e62_val_GER=0.2438.ckpt: For accented English speech
            - multi_MLS8_uh02_e36_val_GER=0.2334.ckpt: 8 European languages
              (English, German, French, Dutch, Italian, Spanish, Portuguese, Polish)
            - large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt: Large universal model for
              Indo-European and closely related languages (80+ languages supported)
            - multi_mswc38_ug20_e59_val_GER=0.5611.ckpt: Small Universal model for
              Indo-European and closely related languages (80+ languages supported)

        Preset Categories:
            - English: en-us, en, en-gb, etc. → English model
            - European (MLS8): de, fr, es, it, pt, pl, nl, da, sv, etc. → MLS8 model
            - Indo-European: hi, bn, ru, fa, el, etc. → Universal model
            - Related families: ar, tr, id, ms, etc. → Universal model
            - Constructed: eo, ia, jbo, etc. → Universal model

        Unsupported Language Types:
            - Tonal languages: Chinese (cmn, yue, hak), Vietnamese (vi), Thai (th)
            - Distant families: Japanese (ja), Korean (ko), Bantu languages, etc.
            - For unsupported languages, consider using explicit model_name parameter

        Examples:
            # Using presets (recommended)
            aligner = PhonemeTimestampAligner(preset="de")  # German with MLS8 model
            aligner = PhonemeTimestampAligner(preset="hi")  # Hindi with universal model

            # Explicit model selection
            aligner = PhonemeTimestampAligner(
                lang="ja",
                model_name="large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt"
            )

            # Direct checkpoint path (highest priority)
            aligner = PhonemeTimestampAligner(
                cupe_ckpt_path="/path/to/model.ckpt",
                lang="zh"
            )
        """

        # 0 = silent, 1 = important warnings (more critical for downstream tasks), 2 = all
        # warnings plus informational output (resolved architecture and frame geometry, snap
        # configuration, low-confidence phonemes, bad confidence patterns). Critical errors are
        # raised as exceptions regardless of warn_level. Set first, so that everything below -
        # including load_model and the geometry resolution - can report through _info/_warn.
        self.warn_level = int(warn_level)
        if device == "auto":
            device = detect_device_automatically()
        self.device = device
        self.extractor = None # will be loaded in load_model()
        self.lang = lang
        self.resampler_sample_rate = 16000
        self.padding_ph_label = -100
        self.ph_seq_min = 1
        #self.ph_seq_max = ph_seq_max
        
        self.seg_duration_min = 0.05  # seconds
        self.seg_duration_min_samples = int(self.seg_duration_min * self.resampler_sample_rate)
        self.seg_duration_max = duration_max  # seconds
        self.wav_len_max = int(self.seg_duration_max * self.resampler_sample_rate)

        self.selected_mapper = mapper
        if self.selected_mapper != "ph66":
            raise ValueError("Currently only 'ph66' mapper is supported.")

        if (self.selected_mapper == "ph66"):
            self.phonemizer = ph66_phonemeizer.Phonemizer(language=self.lang, remove_noise_phonemes=True)
        elif (self.selected_mapper == "ph270"):
            # placeholder for upcoming ph270 mapper implementation which will cover tonal languages and distant families. For now, raise an error to indicate it's not implemented yet.
            raise NotImplementedError("ph270 mapper is not implemented yet.")

        self.phonemes_key = self.phonemizer.phonemes_key
        self.phoneme_groups_key = self.phonemizer.phoneme_groups_key

        self.phoneme_id_to_label = self.phonemizer.index_to_plabel
        self.phoneme_label_to_id = {label: idx for idx, label in self.phoneme_id_to_label.items()}
        self.group_id_to_label = self.phonemizer.index_to_glabel
        self.group_label_to_id = {label: idx for idx, label in self.group_id_to_label.items()}
        self.phoneme_id_to_group_id = self.phonemizer.phoneme_id_to_group_id

        
        # Load the model FIRST, then resolve the geometry from it. `_setup_config` derives
        # `frame_stride_wav` by walking the feature extractor's Conv1d strides, but it used to
        # run before `load_model`, so `self.extractor` was always None and it always fell back
        # to the hard-coded 210. That was right only by coincidence: a model built with
        # `cnn_last_stride: 1` has a 105-sample frame stride, and scoring it at 210 would put
        # every timestamp out by a factor of two - silently, since nothing raises.
        if preset is not None or model_name is not None or cupe_ckpt_path is not None:
            self._setup_config(defer_geometry=True)
            # `load_model` ends by calling `_resolve_geometry` itself, so a later re-load stays
            # consistent too.
            self.load_model(preset=preset, model_name=model_name, cupe_ckpt_path=cupe_ckpt_path)
        else:
            self._setup_config()


        decoder = str(decoder).lower()
        if decoder not in DECODERS:
            raise ValueError(f"decoder must be one of {'|'.join(DECODERS)}, got {decoder!r}")
        self.decoder = decoder
        self.segmental_max_seconds = float(segmental_max_seconds)
        self.silence_anchors = silence_anchors
        self.boost_targets = boost_targets
        self.enforce_minimum = enforce_minimum
        self.enforce_all_targets = enforce_all_targets
        self.ensure_completeness = ensure_completeness
        self.ignore_noise = ignore_noise
        self.extend_soft_boundaries = extend_soft_boundaries
        self.boundary_softness = boundary_softness
        self.bad_confidence_threshold = bad_confidence_threshold
        # --- onset snapping ----------------------------------------------------------------
        # Search half-width, in FRAMES. The snapper searches `sig[f-r : f+r+1]` - a whole
        # number of frames - so frames are what the setting actually controls; 50 ms and 55 ms
        # both mean a 4-frame search at a 210-sample stride, and "4" says which one runs.
        #
        #   boundary_snap_radius = None     per-source default (DEFAULT_SNAP_RADIUS_FRAMES)
        #   boundary_snap_radius = 4        4 frames, whatever the model's stride
        #   boundary_snap_radius = 0        disables snapping entirely
        #   boundary_snap_radius_ms         optional override in MILLISECONDS, rounded to
        #                                   frames at the loaded model's stride; wins over
        #                                   `boundary_snap_radius` only if that one is None
        #
        # `boundary_snap_radius_ms` is still useful for one thing: comparing across models with
        # DIFFERENT strides (210 samples = 13.125 ms/frame, or 105 = 6.5625 ms on a finer-grid
        # model), where the same frame count is a different duration. For a single model, ask
        # in frames.
        #
        # MEASURED, r_value_25ms. The only FULL-corpus radius sweep is
        # `results_TIMIT_lev.csv`, `en` model2i, spectral_stream:
        #
        #     r=2      r=4      r=6
        #   0.6522   0.6897   0.6788          <- 4 frames, hence the default
        #
        # The 100-file TIMIT sweeps agree the flux taps are flat over 2-4 frames
        # (`results_TIMIT_lev_sweeps.csv`; tables in `conclusions.md` §4):
        #
        #   ue02a_e5                          r=1      r=2      r=3      r=4
        #     spectral_stream               0.7588   0.7616   0.7531   0.7348
        #     pre_transformer               0.7532   0.7601   0.7513   0.7387
        #     (control: none = 0.7472)
        #
        # NOT settled, and worth re-running: every model3a full-corpus row on both corpora -
        # the rows the SOURCE defaults rest on (§10) - was taken at r=2, not r=4, and the
        # 20-file Buckeye sweeps prefer r=1 and fall with every step. There is no full-Buckeye
        # radius sweep for any checkpoint. Sweep with `set_boundary_snap(radius_frames=...)`.
        self._boundary_snap_radius_ms_requested = (None if boundary_snap_radius_ms is None
                                                   else float(boundary_snap_radius_ms))
        self.boundary_snap_offset = int(boundary_snap_offset)
        self._boundary_snap_radius_override = (None if boundary_snap_radius is None
                                               else int(boundary_snap_radius))
        # The source is resolved BEFORE the radius, because the radius default depends on which
        # source was chosen and because `'none'` forces the radius to 0 - see
        # `_resolve_snap_radius`. The source's documentation is the block below.
        source = str(boundary_snap_source).lower()
        if source not in SNAP_SOURCES:
            raise ValueError(f"boundary_snap_source must be one of {'|'.join(SNAP_SOURCES)}, "
                             f"got {boundary_snap_source!r}")
        self.boundary_snap_source_requested = source
        self._resolve_snap_source()
        self._resolve_snap_radius()

        # ==================================================================================
        # WHICH TENSOR the onset snapper reads.  session_notes.md §1 / §3.
        # ==================================================================================
        #
        # THE FORWARD PATH, IN ORDER.  Every candidate is a tap on this one chain, so "which
        # source" is really "how far down the chain do I read":
        #
        #   waveform
        #     │
        #     ├─ feature_extractor ....... 4 strided convs, 44.7 ms receptive field,
        #     │                            emits one frame per 210 samples (13.125 ms)
        #     ├─ freq_attention .......... per-channel gate from a WINDOW-wide average pool;
        #     │                            constant across the window, so it cannot move a
        #     │                            boundary - it rescales channels, nothing else
        #     ├──────────────┬───────────────────────────────┐
        #     │              │                               │
        #     │      temporal_stream                 spectral_stream  ......... ['spectral_stream']
        #     │      k=7 then k=3 →                  two 1x1 convs → 44.7 ms,
        #     │      149.7 ms, MIXES TIME            CANNOT MIX TIME
        #     │              │                               │
        #     │              └──────────► concat ◄───────────┘
        #     │                            │
        #     ├─ fusion = conv1x1(concat) ......................... ['pre_transformer'] DEFAULT
        #     │    1x1, so it adds no smear of its own - but half its input is
        #     │    temporal_stream, so it INHERITS that 149.7 ms smear
        #     │
        #     ├─ window_processor ........ 4-layer transformer over the 10 frames of ONE
        #     │                            window (131.2 ms), applied BEFORE the stitch  ['embeddings']
        #     └─ classifier → posteriors → Viterbi → framestamps ...... what snapping CORRECTS
        #
        #   boundary_head ................ a learned per-frame detector read off whichever tap
        #                                  it was TRAINED on (the checkpoint carries which, in
        #                                  `bnd_source`). model3a checkpoints only.
        #                                  ['boundary_head']
        #
        #   'auto' ....................... DEFAULT. Branches on the checkpoint's ARCHITECTURE:
        #                                  `spectral_stream` for model3a, `pre_transformer` for
        #                                  model2i/model2j. It never picks a boundary head.
        #                                  Resolved once, in `_resolve_snap_source`, right after
        #                                  the model loads; `boundary_snap_source` then holds
        #                                  the RESOLVED value and
        #                                  `boundary_snap_source_requested` holds 'auto'.
        #                                  CHANGED in 1.3 - it used to pick the head whenever
        #                                  one was trained. See AUTO_SNAP_MODEL3A at the top of
        #                                  this module.
        #
        #   'none' ....................... VANILLA. No snapping: the Viterbi framestamps are
        #                                  emitted exactly as decoded, which is what BFA did
        #                                  before onset snapping existed. Identical in effect to
        #                                  `boundary_snap_radius_ms=0`, but says so by name, and
        #                                  it overrides an explicit `boundary_snap_radius`.
        #                                  Use it as the control arm: every "+N" in the table
        #                                  below is measured against exactly this.
        #
        # Note the order: `fusion` is computed BEFORE the transformer, so nothing the
        # transformer does can affect it except through backprop.
        #
        # ---------------------------------------------------------------------------------
        # MEASURED RANKING.  GAIN OVER NO SNAPPING, r_value_25ms, full corpora, radius 2
        # frames, boundary_softness=1. `conclusions.md` §10 - the first runs with a `none`
        # control on BOTH corpora, which is what makes these differences readable as gains
        # rather than as a head-to-head ordering.
        #
        #   source                          TIMIT f49   TIMIT g19   Buckeye f49
        #   ------------------------------  ----------  ----------  -----------
        #   (control) none                    0.6330      0.6242      0.5877
        #   spectral_stream                   +0.0552     +0.0565     +0.0260   <- model3a auto
        #   pre_transformer                   +0.0336     +0.0372     +0.0144
        #   boundary_head (constant shift)    +0.0234        -        +0.0472   artifact, see below
        #   boundary_head (TRAINED)              -       -0.0168     control invalid
        #   embeddings                        -0.0243     -0.0205     -0.0253
        #
        # And for model2i, where the ranking INVERTS on Buckeye (§2; `en`, radius 4 frames):
        #
        #   source                            TIMIT     Buckeye
        #   ------------------------------  ---------  ---------
        #   (control) none                    0.5812     0.5437
        #   spectral_stream                   0.6873     0.5174   <- BELOW control on Buckeye
        #   pre_transformer                   0.6556     0.5516   <- model2i auto
        #   embeddings                        0.5853     0.5128
        #
        # (MFA, for scale: 0.8035 TIMIT / 0.8164 Buckeye. Every BFA setting is still far short
        # of it. Snapping moves onsets; it does not fix the decoder.)
        #
        # What the defaults act on:
        #
        #  1. model3a gets `spectral_stream` - best real source on both corpora, above control
        #     on both. model2i gets `pre_transformer`, because `spectral_stream` is the best
        #     on TIMIT for model2i too but falls BELOW the control on Buckeye (0.5174 vs
        #     0.5437): better on read speech, worse on conversational. Hence the branch.
        #  2. `embeddings` (BFA <= 1.1.5's source) is below control on BOTH corpora for both
        #     families. It is never a default.
        #  3. A TRAINED head is never selected automatically - it is the only real source that
        #     loses to its own control.
        #
        # WARNING on 'boundary_head'. Two separate problems, and 'auto' avoids both by never
        # selecting it.
        #
        # UNTRAINED heads. `bnd_head_enabled: True` means the module exists, NOT that it
        # learned anything - that is `bnd_head_lambda`, and most released ue03* checkpoints
        # have `bnd_head_lambda: 0.0` ("boundary objective OFF", door 16). Their `out.weight`
        # is zero, so the head emits a CONSTANT (-2.2) for every frame. The argmax of a flat
        # signal is its first index, so snapping against it degenerates into a fixed shift of
        # `1 + offset - radius` frames plus the monotonic-spacing constraint. That shift can
        # still score well because it absorbs the decoder's placement bias - on Buckeye it is
        # the single best setting in the CSVs (+0.0472, ahead of spectral_stream's +0.0260),
        # which says the decoder runs about one frame late and Buckeye punishes that more than
        # TIMIT does. It is a shift artifact, not a detector. A plain global shift is not
        # reachable through this API today (`offset` only applies after a search, `radius=0`
        # disables snapping), and is worth adding as a first-class option.
        #
        # TRAINED heads. `ue03g_e19` (lambda=1.0) has one, and it does detect boundaries - frame
        # AUC 0.886 against ground truth, next to spectral_stream's 0.893 on the same
        # checkpoint. But snapping to it scores 0.0168 BELOW not snapping at all on 6300 TIMIT
        # files. Its targets come from the model's OWN Viterbi output, so what it learned best
        # is the decoder's placement (AUC 0.864 against its own decoder, vs 0.70-0.75 for the
        # flux taps) - it reproduces the decoder's lateness instead of correcting it. Signed
        # error median +6, the same as `none`, where the flux taps pull it to +3/+4.
        #
        # `model3a.CUPEEmbeddingsExtractor` records `boundary_head_trained` either way.
        # Requesting 'boundary_head' explicitly works and warns, with the relevant caveat.
        #
        # `offset` stays 0: a plain global shift of the decoder's onsets is worth +0.74 f1 held
        # out at -1 frame and 0.00 once snapping is on, because snapping already absorbs the
        # placement bias - which is what the indexing convention predicts a priori.
        #
        # This CHANGES existing alignments relative to BFA <= 1.1.5. That is deliberate:
        # `boundary_snap_source='none'` turns snapping off entirely (as does
        # `boundary_snap_radius_ms=0`), and `'embeddings'` restores the pre-1.2 source.
        # Snapping never changes the onset COUNT - `detection_ratio` moves by <0.003 across
        # every source; it moves onsets, it does not invent them.
        self._warned_pre_window = False
        self.break_at_low_confidence = False # under construction, not fully implemented yet

        # --- adaptive chunking (chunking.py) ------------------------------------------
        # Long segments are walked window by window rather than decoded in one shot. The
        # tunables below are the ones worth reaching for; the run lengths, backoff and
        # confidence floor that decide WHERE a window is cut live in `chunking.py` and are
        # settable through `set_chunking()`.
        self.chunk_audio = bool(chunk_audio)
        self.chunk_seconds = float(chunk_seconds)
        self.chunk_max_phonemes = int(chunk_max_phonemes)
        self.chunk_phonemes_per_second = float(chunk_phonemes_per_second)
        self.chunk_vad = bool(chunk_vad)
        self.chunk_derail_frames = DEFAULT_DERAIL_FRAMES
        self.chunk_derail_run = DEFAULT_DERAIL_RUN
        self.chunk_backoff = DEFAULT_BACKOFF
        self.chunk_conf_floor = DEFAULT_CONF_FLOOR
        self.chunk_weak_run = DEFAULT_WEAK_RUN
        self.chunk_snap_to_silence = DEFAULT_SNAP_TO_SILENCE_SEC
        self._warn_if_budget_overruns_lattice()


        self._setup_decoders()

        # Initialize audio processing
        self.default_resampler = torchaudio.transforms.Resample(
            orig_freq=self.resampler_sample_rate,
            new_freq=self.resampler_sample_rate,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )
        self.reset_counters()

    def _info(self, msg):
        """Informational output - resolved geometry, architecture, snap config. `warn_level>=2`."""
        if getattr(self, 'warn_level', 1) >= 2:
            print(msg)

    def _warn(self, msg):
        """A condition the caller should know about but that does not stop the run. `warn_level>=1`."""
        if getattr(self, 'warn_level', 1) >= 1:
            print(f"WARN: {msg}")

    def reset_counters(self):
        """Reset internal counters for tracking alignment statistics. Purely for debugging and analysis purposes."""
        self.total_segments_processed = 0
        self.total_segments_bad = 0
        self.total_segments_failed = 0
        self.total_phonemes_aligned = 0
        self.total_phonemes_target = 0
        self.total_phonemes_aligned_easily = 0
        self.total_phonemes_missed = 0
        self.total_phonemes_extra = 0
        self.perfect_matches = 0

    def load_model(self, preset="en-us", model_name=None, cupe_ckpt_path=None):  
        
        # Parameter priority handling:
        # 1st priority: Explicit cupe_ckpt_path (highest)
        # 2nd priority: Explicit model_name
        # 3rd priority: Preset values (only if no explicit model specified)
        # 4th priority: Default values

        # Only apply preset logic if no explicit checkpoint path or model name provided
        if cupe_ckpt_path is None and model_name is None:
            # Language preset mappings based on available models
            model_name, lang = get_preset(preset=preset, lang=self.lang)
            if self.lang != lang:
                self.phonemizer.set_backend(language=lang)
            self.lang = lang  # Update language based on preset
        if cupe_ckpt_path is not None:
            pass  # use explicit path as-is
        elif model_name is not None:
            cupe_ckpt_path = self.download_model(model_name=model_name)
        else:
            raise ValueError("Either cupe_ckpt_path or model_name must be provided.")
        if cupe_ckpt_path is None or not os.path.exists(cupe_ckpt_path):
            raise ValueError("CUPE model checkpoint not found.", cupe_ckpt_path)
        torch.random.manual_seed(42)
        # Which extractor to use is a property of the checkpoint, not of its filename. The old
        # selector was an allow-list of four filenames, which silently mis-dispatched anything
        # not on it - including every checkpoint trained since. It is gone; nothing below reads
        # the filename.
        #
        # Which architecture the checkpoint IS, decided from its own weights:
        #   `boundary_head.*`  -> model3a (contextless base + learned onset detector)
        #   `classifier_ref.*` -> a PRE-§5.1 model3a; the extractor raises with instructions,
        #                         because decoding it with the base heads would silently report
        #                         a different model from the one that produced its numbers
        #   `refiner.*` only   -> model2j (refiner over LOGITS; door 4, kept for old ckpts)
        #   neither            -> model2i (contextless base)
        arch = None
        # Read the checkpoint once, for the architecture probe AND to keep its training
        # hyperparameters so `_warn_if_stride_mismatch` can compare geometries.
        self._ckpt_hyperparams = {}
        try:
            probe = torch.load(cupe_ckpt_path, map_location='cpu', weights_only=False)
            hp = probe.get('hyper_parameters', {}) if isinstance(probe, dict) else {}
            if hasattr(hp, '__dict__') and not isinstance(hp, dict):
                hp = vars(hp)                       # argparse.Namespace
            self._ckpt_hyperparams = dict(hp) if hp else {}
            if arch is None:
                state = probe.get('state_dict', probe)
                if any('boundary_head.' in k or 'classifier_ref.' in k for k in state):
                    arch = '3a'
                elif any(k.startswith('refiner.') or '.refiner.' in k for k in state):
                    arch = 'j'
                else:
                    arch = 'i'
                del state
            del probe, hp
        except Exception as _e:
            arch = arch or 'i'
            self._warn(f"could not read checkpoint metadata ({type(_e).__name__}); "
                       f"assuming model2{arch}.")

        # Kept because `boundary_snap_source='auto'` branches on the architecture, not on
        # whether a boundary head exists - see `_resolve_snap_source`.
        self._ckpt_arch = arch
        self._info(f"Checkpoint architecture: model{'2' if arch in ('i', 'j') else ''}{arch}")
        if arch == '3a':
            self.extractor = CUPEEmbeddingsExtractor_3a(cupe_ckpt_path, device=self.device,
                                                        verbose=self.warn_level)
        elif arch == 'j':
            self.extractor = CUPEEmbeddingsExtractor_j(cupe_ckpt_path, device=self.device)
        else:
            self.extractor = CUPEEmbeddingsExtractor_i(cupe_ckpt_path, device=self.device)
        self.extractor.eval()
        
        assert self.window_size_wav is not None, "Window size in samples must be calculated before loading the model. call _setup_config() before load_model()."
        assert self.extractor.model.frames_per_window is not None, "Extractor model must have frames_per_window defined."
        self.frames_per_window = self.extractor.model.update_frames_per_window(self.window_size_wav)

        # Everything derived FROM the checkpoint has to be re-derived now that there is a new
        # one: the frame stride and sub-state topology (`_resolve_geometry`), and - because
        # `'auto'` asks the checkpoint whether it has a trained boundary head - the snap source
        # and its per-source default radius.
        self._resolve_geometry()
        self._warn_if_stride_mismatch()
        if hasattr(self, 'boundary_snap_source_requested'):
            # Only on a re-load: during __init__ these are configured just after this returns.
            self._resolve_snap_source()
            self._resolve_snap_radius()
            self._setup_decoders()

    def _setup_config(self, window_size_ms=120, stride_ms=78.75, defer_geometry=False):

        """Setup configuration parameters.

        `stride_ms` MUST be a whole number of output frames, i.e. a multiple of the feature
        extractor's total conv stride (210 samples = 13.125 ms here). 78.75 ms = 6 frames.

        It used to default to 80 ms = 6.095 frames while `stich_window_predictions` advanced
        `frames_per_window // 2` = 5. The two windows being averaged therefore described
        moments 14.4 ms apart, smearing every stitched frame - which is why `abs_err_mean`
        held at 27.7 ms with `signed_err_median` only +2 ms (a symmetric smear inflates the
        mean without shifting the median), and why a 30% PER improvement changed alignment
        by nothing. `frames_geometry` now raises rather than letting this recur silently.

        This must match the value the model was TRAINED with - a mismatch between training
        and inference geometry is worse than the original bug.
        """
        self.window_size_ms = window_size_ms
        self.stride_ms = stride_ms
        self.sample_rate = 16000 # hardcoded for now, as CUPE is trained on 16kHz audio, and the resampler will convert to this sample rate if needed. In the future, we could consider making this configurable if we want to support models trained on different sample rates, but for now it's simpler to keep it fixed.

        # Calculate window parameters
        self.window_size_wav = int(window_size_ms * self.sample_rate / 1000)
        self.stride_size_wav = int(stride_ms * self.sample_rate / 1000)

        # Phoneme class configuration. MUST precede `_resolve_geometry`, which reads
        # `phoneme_classes` to work out how many sub-states the model's class width implies.
        self.phoneme_classes = len(self.phoneme_id_to_label)-1  # exclude padding 'noise' = 66
        self.phoneme_groups = len(self.group_id_to_label)-1 # exclude padding 'noise' = 16
        self.blank_class = self.phoneme_label_to_id['noise']
        self.blank_group = self.group_label_to_id['noise']
        self.silence_class = self.phoneme_label_to_id['SIL']
        self.silence_group = self.group_label_to_id['SIL']
        self.n_substates = 1                # refined by _resolve_geometry once a model is loaded
        self.model_blank_class = self.blank_class
        # print all these:
        #print(f"Phoneme classes: {self.phoneme_classes}, groups: {self.phoneme_groups}, blank_class: {self.blank_class}, blank_group: {self.blank_group}, silence_class: {self.silence_class}, silence_group: {self.silence_group}")

        if not defer_geometry:
            self._resolve_geometry()

    def _has_trained_boundary_head(self):
        """
        Whether the loaded checkpoint carries a boundary head that was actually TRAINED.

        Three distinct states, and only the third is usable:
          - no extractor, or a model2i/model2j extractor  -> no head at all
          - a model3a extractor whose head is still zero  -> head present, emits a constant
          - a model3a extractor with a trained head       -> usable

        The middle case is the one worth naming: `bnd_head_enabled: True` only says the module
        exists. `model3a.CUPEEmbeddingsExtractor.__init__` inspects `out.weight` and records the
        answer as `boundary_head_trained`; an older extractor without that attribute is treated
        as untrained, which is the safe direction (it routes 'auto' to a flux tap).
        """
        if getattr(self, 'extractor', None) is None:
            return False
        if getattr(self.extractor, 'detect_boundaries', None) is None:
            return False
        return bool(getattr(self.extractor, 'boundary_head_trained', False))

    def _resolve_snap_source(self):
        """
        Turn `boundary_snap_source_requested` into the concrete source to read, in
        `self.boundary_snap_source`.

        Only `'auto'` needs resolving, and it is the default, so this is the path most callers
        take. It branches on the checkpoint's ARCHITECTURE - `AUTO_SNAP_MODEL3A` for model3a,
        `AUTO_SNAP_MODEL2I` for model2i/model2j - because the two families do not rank the
        sources the same way once Buckeye is taken into account. The table and the measurements
        are on `AUTO_SNAP_MODEL3A` above.

        This CHANGED in 1.3. It used to branch on whether the checkpoint had a *trained*
        boundary head, and pick the head when it did. `conclusions.md` §10 measured that head
        against a `none` control for the first time and it came out 0.0168 BELOW not snapping
        at all on 6300 TIMIT files - it is the only real source that loses to its own control -
        so `auto` no longer selects it. An explicit `boundary_snap_source='boundary_head'`
        still works.

        An EXPLICIT request is never silently rewritten - being handed a different tensor than
        you asked for is precisely the failure that makes a snap result uninterpretable - but
        the two ways it can be unsatisfiable are reported here rather than at the first batch.
        """
        requested = getattr(self, 'boundary_snap_source_requested', 'auto')
        has_head = self._has_trained_boundary_head()

        if requested == 'auto':
            arch = getattr(self, '_ckpt_arch', None)
            self.boundary_snap_source = AUTO_SNAP_MODEL3A if arch == '3a' else AUTO_SNAP_MODEL2I
            if getattr(self, 'extractor', None) is not None:
                self._info(f"boundary snap: 'auto' resolved to '{self.boundary_snap_source}' "
                           f"(model{'3a' if arch == '3a' else '2' + str(arch or 'i')})")
        else:
            self.boundary_snap_source = requested

        if self.boundary_snap_source == 'boundary_head' and getattr(self, 'extractor', None) is not None:
            if getattr(self.extractor, 'detect_boundaries', None) is None:
                raise ValueError(
                    "boundary_snap_source='boundary_head' but this checkpoint has no boundary "
                    "head (it is not a model3a checkpoint). Use 'auto', 'pre_transformer' or "
                    "'spectral_stream'.")
            if not has_head:
                self._warn(
                    "boundary_snap_source='boundary_head' but this checkpoint's head was never "
                    "trained (bnd_head_lambda=0.0, out.weight is zero), so it emits a constant "
                    "for every frame. Snapping against a constant is a fixed shift of "
                    f"{1 + self.boundary_snap_offset} - radius frames, not a detection. Use "
                    f"'auto' to get '{AUTO_SNAP_MODEL3A}' instead. (Note that on Buckeye that "
                    "shift is the single best-scoring setting in the CSVs, +0.047 r_value over "
                    "no snapping - the decoder runs late and the shift corrects it. It is still "
                    "a shift, not a detector.)")
            else:
                self._warn(
                    "boundary_snap_source='boundary_head' with a TRAINED head. Measured at "
                    "0.0168 r_value BELOW not snapping at all on 6300 TIMIT files "
                    "(conclusions.md §10): the head learned to reproduce the decoder's own "
                    "onset placement rather than correct it. 'auto' selects "
                    f"'{AUTO_SNAP_MODEL3A}' instead, and 'none' disables snapping.")
        return self.boundary_snap_source

    def _resolve_snap_radius(self):
        """
        Settle the search half-width, in FRAMES, in `self.boundary_snap_radius`.
        Called from the constructor, and again by `set_boundary_snap` when retuned.

        Frames are the unit of record. The snapper searches `sig[f-r : f+r+1]`, a whole number
        of frames, so a millisecond radius is only ever a way of naming one - 50 ms and 55 ms
        are the same 4-frame search at a 210-sample stride, and saying "4" says which search
        will actually run. Milliseconds are still accepted and still win when given; they are
        converted here and the resulting duration is reported back in
        `self.boundary_snap_radius_ms` for anyone logging it.

        Precedence, highest first:
          1. `boundary_snap_source='none'`  -> 0, overriding everything ('none' means none)
          2. `boundary_snap_radius`         -> used as-is, in frames
          3. `boundary_snap_radius_ms`      -> rounded to frames at this model's stride
          4. per-source default             -> `DEFAULT_SNAP_RADIUS_FRAMES`

        `_resolve_snap_source` must have run first, because (4) is per-source.

        A radius of 0 is how "no snapping" is expressed everywhere downstream - every call site
        already guards on `boundary_snap_radius > 0`, and `snap_onsets_to_embeddings` returns
        its input unchanged - so 'none' resolves to 0 here rather than being special-cased in
        five places.
        """
        frame_ms = 1000.0 * getattr(self, 'frame_stride_wav', 210) / self.sample_rate

        if self.boundary_snap_source == 'none':
            radius = 0
        elif self._boundary_snap_radius_override is not None:
            radius = max(0, int(self._boundary_snap_radius_override))
        elif self._boundary_snap_radius_ms_requested is not None:
            ms = float(self._boundary_snap_radius_ms_requested)
            # round, but never collapse a positive request to a 0-frame (i.e. disabled) search
            radius = 0 if ms <= 0 else max(1, int(round(ms / frame_ms)))
        else:
            radius = DEFAULT_SNAP_RADIUS_FRAMES.get(self.boundary_snap_source,
                                                    DEFAULT_SNAP_RADIUS_FRAMES_FALLBACK)

        self.boundary_snap_radius = int(radius)
        # Derived, for reporting only - the duration the resolved frame count actually spans,
        # not whatever ms value may have been asked for.
        self.boundary_snap_radius_ms = self.boundary_snap_radius * frame_ms
        self.boundary_snap_frame_ms = frame_ms
        return self.boundary_snap_radius

    def set_boundary_snap(self, radius_frames=None, radius_ms=None, offset=None, source=None,
                          verbose=None):
        """
        Retune onset snapping on a live aligner, for sweeping without reconstructing it.

        Sweep in FRAMES - `set_boundary_snap(radius_frames=n)`. That is the unit the search
        actually uses, so a frame sweep visits every distinct setting exactly once, where a ms
        sweep silently repeats itself (50 ms and 55 ms are the same 4-frame search).

        Args:
            radius_frames: search half-width in frames. `0` disables snapping; `'clear'`
                restores the per-source default from `DEFAULT_SNAP_RADIUS_FRAMES`.
            radius_ms: the same half-width in ms, rounded to frames at the loaded model's
                stride. Used only while `radius_frames` is unset, so pass `radius_frames='clear'`
                first if one was given. `'default'` clears it. Useful for comparing models with
                different frame strides; otherwise prefer frames.
            offset: frames added after the argmax (see `snap_onsets_to_embeddings`).
            source: one of `SNAP_SOURCES` - which tensor to snap against, `'auto'` to re-resolve
                against the loaded checkpoint, or `'none'` for vanilla (no snapping, the
                decoder's own framestamps). `'none'` is the control arm every gain in the table
                on `boundary_snap_source` is measured against. While the source is `'none'` the
                radius stays 0 whatever `radius_ms` or `radius_frames` say - the reported line
                prints `[DISABLED]` to show it.
            verbose: print the resolved configuration. `None` (default) follows `warn_level>=2`.

        Note that changing only `source` also changes the radius when no explicit radius was
        ever given, because the default radius is per-source.

        Returns:
            The resolved radius in frames.
        """
        # Frames first: a plain `set_boundary_snap(radius_frames=n)` must not be silently
        # outranked by a ms value set earlier, so clearing one clears the other.
        if radius_frames is not None:
            self._boundary_snap_radius_override = (None if radius_frames == 'clear'
                                                   else int(radius_frames))
            if radius_frames != 'clear':
                self._boundary_snap_radius_ms_requested = None
        if radius_ms is not None:
            self._boundary_snap_radius_ms_requested = (None if radius_ms == 'default'
                                                       else float(radius_ms))
            if radius_frames is None:
                self._boundary_snap_radius_override = None
        if offset is not None:
            self.boundary_snap_offset = int(offset)
        if source is not None:
            source = str(source).lower()
            if source not in SNAP_SOURCES:
                raise ValueError(f"source must be one of {'|'.join(SNAP_SOURCES)}, got {source!r}")
            self.boundary_snap_source_requested = source
            self._resolve_snap_source()
        r = self._resolve_snap_radius()
        line = (f"boundary snap: source {self.boundary_snap_source}, radius {r} frames "
                f"({r * self.boundary_snap_frame_ms:.3f} ms), offset {self.boundary_snap_offset}"
                f"{'  [DISABLED]' if r <= 0 else ''}")
        if verbose or (verbose is None and self.warn_level >= 2):
            print(line)
        return r

    def _warn_if_stride_mismatch(self):
        """
        Warn when the checkpoint was TRAINED at a different window stride than this aligner
        decodes at.

        `session_notes.md` §17.7: the training stitch advanced a hard-coded
        `frames_per_window // 2` = 5 frames while `window_stride_ms: 80` is 6.095 true frames,
        so training implied a fictitious 16 ms/frame grid where inference uses the real
        13.125 ms. It does not invalidate the weights - CUPE is a per-window classifier and the
        stitch is post-hoc placement - but it is worth surfacing rather than leaving silent,
        because it means the model was trained against slightly smeared supervision.
        """
        hp = getattr(self, '_ckpt_hyperparams', None) or {}
        trained = hp.get('window_stride_ms', None)
        if trained is None:
            return
        if abs(float(trained) - float(self.stride_ms)) > 1e-6:
            frame_ms = 1000.0 * self.frame_stride_wav / self.sample_rate
            self._info(
                f"checkpoint was TRAINED at window_stride_ms={trained} but this aligner "
                f"decodes at {self.stride_ms}. "
                f"({float(trained) * self.sample_rate / 1000.0 / self.frame_stride_wav:.4f} vs "
                f"{float(self.stride_ms) * self.sample_rate / 1000.0 / self.frame_stride_wav:.4f} "
                f"output frames per window step, at {frame_ms:g} ms/frame.) "
                f"Weights remain valid - CUPE is per-window and the stitch is post-hoc - the "
                f"training stitch simply advanced a whole 5 frames for a 6.095-frame step.")

    def snap_onsets_to_embeddings(self, batch_framestamps, embeddings, spectral_lens,
                                  radius=None, offset=0):
        '''
        Move each phoneme ONSET to the strongest embedding-change frame nearby.
        `session_notes.md` §4 / §15.1 / §16 - inference only, no retraining.

        ## Why

        The Viterbi reads `log_softmax(logits_class)`, the 67-d posteriors. Measured twice, on
        two different checkpoints, with the same probe:

            emb_cos  (512-d embeddings, pre-classifier)  peak-pick f1@25  53.33
            post_jsd (67-d posteriors, what Viterbi reads)                49.13
            random floor                                                 43.84

        A well-trained CUPE is near one-hot INSIDE a phone, so posterior change collapses to a
        near-binary "the argmax flipped" indicator with no graded sub-frame placement. The
        embeddings keep that gradation. The decoder is therefore placing boundaries with a
        strictly worse signal than the model has already computed.

        This keeps the decoder's phone identities and sequence - which the lattice and the text
        phonemization guarantee - and only moves WHERE each onset sits.

        ## Measured (full TIMIT, 6300 files, r_value_25ms; `tmp/results/conclusions.md`)

            checkpoint    none     pre_transformer   spectral_stream   boundary_head
            en (model2i)  0.5812        0.6556            0.6873            n/a
            ue03d_e04       -           0.6693            0.6895           0.6654

        ## Constraints

        Onsets stay strictly increasing and inside their neighbours' original positions, so
        snapping can never reorder phones or collapse two onto one frame. Each phone's end is
        then re-derived from its new start: carried to the next onset where the two were
        contiguous before snapping, and otherwise shifted by the same amount as the start so the
        duration survives a gap. Every emitted segment spans at least one frame - see the
        comment at that clamp for the zero-duration bug this replaced.

        ## Two kinds of source tensor

        `embeddings` may be either:

          - **[B, T, D]**, a feature map -> the change signal is `1 - cos(x[t], x[t+1])`, i.e.
            spectral flux when D is the CNN feature dim. This is the historical behaviour.
          - **[B, T]** (or [B, T, 1]), a per-frame boundary SCORE, e.g. `BoundaryHead`'s logit
            -> the signal is `score[t+1]`, so that `argmax + 1` lands on the scored frame
            itself. The head is trained on the FRAME index (`logit[b]` for an onset at `b`)
            while `s` lives on the PAIR index, and this shift is what reconciles them. It is
            asserted in `p2cupe/boundary_loss.py`'s `__main__`, and it fails silently if wrong.

        Everything after the signal - the argmax, the `+1`, the monotonicity constraints, the
        end-carrying - is shared, so the two sources differ ONLY in what is being maximised.

        Args:
            batch_framestamps: list per clip of (phoneme_id, start_frame, end_frame, target_idx)
            embeddings: stitched [B, T, D] feature map, or [B, T] per-frame boundary scores
            spectral_lens: valid frame count per clip
            radius: search half-width in FRAMES. `None` uses `self.boundary_snap_radius`;
                `0` disables. 4 frames = 52.5 ms at the 210-sample stride. The optimum is flat
                over 3-6 frames, so this is not a sharp tuning parameter.
            offset: frames added after the argmax. `s[t]` compares frames t and t+1, so the
                onset is `t+1`; `offset=0` was best on both slices and is what the indexing
                convention predicts a priori.
        Returns:
            New list of framestamps with snapped start (and adjusted end) frames.
        '''
        if radius is None:
            radius = int(getattr(self, 'boundary_snap_radius', 0))
        if radius <= 0 or embeddings is None:
            return batch_framestamps

        # A [B, T, 1] map carries no cosine information, so treat it as a per-frame score too.
        per_frame_score = (embeddings.dim() == 2) or (embeddings.shape[-1] == 1)

        out_batch = []
        for i, framestamps in enumerate(batch_framestamps):
            if not framestamps:
                out_batch.append(framestamps)
                continue
            T = int(spectral_lens[i]) if spectral_lens is not None else embeddings.shape[1]
            T = max(2, min(T, embeddings.shape[1]))
            if per_frame_score:
                # `score[t+1]` so that the shared `argmax + 1` below lands on the scored frame.
                score = embeddings[i, :T].float().reshape(T)
                sig = score[1:].cpu().numpy()
            else:
                emb = embeddings[i, :T].float()
                sig = (1.0 - F.cosine_similarity(emb[:-1], emb[1:], dim=-1)).cpu().numpy()
            if sig.size == 0:
                out_batch.append(framestamps)
                continue

            stamps = list(framestamps)
            new_starts = []
            lo_bound = 0
            for j, (_ph, start_frame, _end, _idx) in enumerate(stamps):
                f = int(start_frame)
                lo = max(lo_bound, f - radius, 0)
                hi = min(f + radius, len(sig) - 1)
                if hi < lo:
                    # No searchable window (the phone sits past the end of the signal, or a
                    # previous snap pushed `lo_bound` beyond it). Keep the decoder's frame, but
                    # clamp into range so the end arithmetic below cannot run off the tensor -
                    # `spectral_lens` can exceed the stitched length by a frame.
                    f = max(0, min(f, T - 1))
                    new_starts.append(f)
                    lo_bound = f + 1
                    continue
                best = int(sig[lo:hi + 1].argmax()) + lo
                cand = best + 1 + offset
                nxt = int(stamps[j + 1][1]) if j + 1 < len(stamps) else T
                cand = int(min(max(cand, lo_bound), max(lo_bound, min(nxt - 1, T - 1))))
                new_starts.append(cand)
                lo_bound = cand + 1

            snapped = []
            for j, (ph, start_frame, end_frame, idx) in enumerate(stamps):
                new_start = new_starts[j]
                # Where the NEXT phone now begins; `T` for the last one. `new_starts` is
                # strictly increasing, so this is always >= new_start + 1 and there is always
                # room for a one-frame segment.
                nxt_start = new_starts[j + 1] if j + 1 < len(stamps) else T

                if j + 1 < len(stamps) and int(end_frame) >= int(stamps[j + 1][1]) - 1:
                    # Contiguous with its neighbour before snapping: keep it contiguous, so the
                    # segmentation stays gap-free rather than only its onsets being right.
                    new_end = nxt_start
                else:
                    # Not contiguous - there was a genuine gap (a blank run) after this phone.
                    # Shift the end by the same amount as the start so the DURATION survives.
                    #
                    # This is the fix for the collapse that made snapping unusable on model2i
                    # checkpoints: the old code left `new_end` at the ORIGINAL end and then
                    # clamped with `max(new_end, new_start)`, so any onset that snapped past its
                    # own end produced a zero-length segment. `_calculate_confidences` scores
                    # such a segment as `probs[start, ph]` at a frame the phone is no longer on,
                    # which is how "b 65.6 65.6 c=0.000" came out of butterfly.wav - a real
                    # phoneme reported with zero duration and zero confidence.
                    new_end = int(end_frame) + (new_start - int(start_frame))

                new_end = min(new_end, nxt_start, T)
                new_end = max(new_end, new_start + 1)   # never emit an empty segment
                snapped.append((ph, new_start, new_end, idx))
            out_batch.append(snapped)
        return out_batch

    def _expand_substates(self, seqs, seq_lens):
        '''
        Re-label a phone sequence as its begin/middle/end triplets, for a sub-state model.

            [p, q]  ->  [p_B, p_M, p_E, q_B, q_M, q_E]      substate_id = n_sub * base + sub

        The three ids are distinct, which is what makes this a valid CTC target sequence.
        A no-op when the loaded model has one state per phone.

        Args:
            seqs: [B, S] base ph66 ids
            seq_lens: [B] sequence lengths in phones
        Returns:
            ([B, S * n_sub] expanded ids, [B] expanded lengths)
        '''
        n_sub = getattr(self, 'n_substates', 1)
        if n_sub <= 1:
            return seqs, seq_lens
        B, S = seqs.shape
        offsets = torch.arange(n_sub, device=seqs.device, dtype=seqs.dtype)
        expanded = ((seqs * n_sub).unsqueeze(-1) + offsets).reshape(B, S * n_sub)
        return expanded, seq_lens * n_sub

    def _collapse_substates(self, batch_framestamps):
        '''
        Map sub-state framestamps back to phones.

        Framestamps are `(phoneme_id, start_frame, end_frame, target_seq_idx)`. The three
        sub-states of one phone are consecutive entries sharing
        `target_seq_idx // n_sub`, so they are merged into a single entry spanning
        `min(start)` to `max(end)`: the phone's onset is the first frame of its *begin*
        sub-state, which is exactly the quantity the eval scores.

        `phoneme_id // n_sub` recovers the base ph66 id and is correct for blank as well,
        blank being the last id in both spaces.
        '''
        n_sub = getattr(self, 'n_substates', 1)
        if n_sub <= 1:
            return batch_framestamps

        collapsed_batch = []
        for framestamps in batch_framestamps:
            merged = []
            for phoneme_id, start_frame, end_frame, target_idx in framestamps:
                base_id = int(phoneme_id) // n_sub
                base_idx = int(target_idx) // n_sub if int(target_idx) >= 0 else int(target_idx)
                if merged and merged[-1][3] == base_idx and base_idx >= 0 and merged[-1][0] == base_id:
                    prev = merged[-1]
                    merged[-1] = (prev[0], min(prev[1], int(start_frame)),
                                  max(prev[2], int(end_frame)), prev[3])
                else:
                    merged.append((base_id, int(start_frame), int(end_frame), base_idx))
            collapsed_batch.append(merged)
        return collapsed_batch

    def _resolve_geometry(self):
        """
        Derive the frame geometry from the loaded model, and the sub-state topology from its
        class width.

        Samples per output frame = product of the feature extractor's conv strides. NOT
        `window_size_wav / frames_per_window` (1920/10 = 192): that is the average density
        including padding-compressed edge frames, while the interior frames are genuinely
        `stride` apart (verified by gradient centroid).

        Must be called AFTER `load_model`, or `self.extractor` is None and the stride silently
        falls back to 210 - correct for a `cnn_last_stride: 2` model, and a factor-of-two error
        for a `cnn_last_stride: 1` one.
        """
        import torch.nn as _nn
        model = getattr(self.extractor, 'model', self.extractor)
        src = getattr(model, 'feature_extractor', None)
        self.frame_stride_wav = 210
        if src is not None:
            st = 1
            for m in src.modules():
                if isinstance(m, _nn.Conv1d):
                    st *= int(m.stride[0])
            self.frame_stride_wav = st
        elif self.extractor is not None:
            # An extractor with no feature extractor is a real anomaly. With no extractor at all
            # (`preset=None`, model loaded later) the 210 fallback is just a placeholder that
            # `load_model` will overwrite, so there is nothing to report.
            self._warn("no feature_extractor found; assuming a 210-sample frame stride. "
                       "If this model was trained with cnn_last_stride: 1, every timestamp will "
                       "be out by 2x.")

        # Sub-states: the model's class width tells us how many states each phone was split
        # into. 67 = one state per phone (the original space), 199 = begin/middle/end.
        # See modules/p2cupe/substates.py - substate_id = n_sub * base_id + sub, so the
        # collapse is a plain floor-divide and is correct for blank too.
        classes_dim = getattr(model, 'classes_dim', self.phoneme_classes + 1)
        base_dim = self.phoneme_classes + 1
        self.n_substates = max(1, int(round(classes_dim / base_dim)))
        if self.n_substates * base_dim != classes_dim:
            raise ValueError(
                f"model emits {classes_dim} classes, which is not a whole multiple of the "
                f"{base_dim}-class ph66 space; cannot infer the sub-state topology")
        self.model_classes_dim = classes_dim
        # The class-space decoder runs in the MODEL's label space, so its blank is the last id
        # there (198 for a 3-state model), not the base ph66 blank (66). Getting this wrong
        # would point the Viterbi's blank state at a real phoneme's middle sub-state.
        self.model_blank_class = classes_dim - 1

        frames_per_window = int(round(self.window_size_wav / self.frame_stride_wav)) + 1
        self.stride_frames, _, _ = frames_geometry(
            self.window_size_ms, self.stride_ms, self.sample_rate,
            frames_per_window, self.frame_stride_wav, strict=True)
        self._info(f"BFA geometry: window {self.window_size_ms}ms, stride {self.stride_ms}ms = "
                   f"{self.stride_frames} frames, frame stride {self.frame_stride_wav} samples "
                   f"({1000.0*self.frame_stride_wav/self.sample_rate:g} ms/frame), "
                   f"{self.n_substates} sub-state(s)/phone, {classes_dim} classes")

    def _setup_decoders(self):
        """Setup Viterbi decoders for phoneme classes and groups."""
        
        # Alignment utilities.
        # The group decoder runs in the flat 17-way group space, unaffected by sub-states.
        # The class decoder runs in the MODEL's class space, so its blank is `model_blank_class`
        # (198 for a 3-state model, 66 otherwise) - the base ph66 blank would land on a real
        # phoneme's middle sub-state. Likewise silence is SIL's *begin* sub-state, which is id
        # 0 under `substate_id = n_sub * base + sub`, so `silence_class` needs no adjustment.
        self.alignment_utils_g = AlignmentUtils(blank_id=self.blank_group, silence_id=self.silence_group, silence_anchors=self.silence_anchors, ignore_noise=self.ignore_noise, truly_forced=self.enforce_all_targets)
        self.alignment_utils_p = AlignmentUtils(blank_id=self.model_blank_class, silence_id=self.silence_class, silence_anchors=self.silence_anchors, ignore_noise=self.ignore_noise, truly_forced=self.enforce_all_targets, n_substates=getattr(self, 'n_substates', 1))


    def download_model(self, model_name=None, local_dir=None):
        """Download the specified model from hugging face using huggingface_hub.


        -   `model_name (str)`: Name of the model checkpoint to download. Must be one of the available models in the Tabahi/CUPE-2i repository. Defaults to the English model3a checkpoint (`presets.MODEL3A_EN`); `presets.py` is the single place every checkpoint filename is written down.
        -   `local_dir (str or Path, optional)`: If provided, the downloaded file will be placed under this directory.

        """
        from huggingface_hub import hf_hub_download
        from .presets import MODEL3A_EN

        if model_name is None:
            model_name = MODEL3A_EN
        repo_id = "Tabahi/CUPE-2i"
        filename = f"ckpt/{model_name}"

        try:
            # Just return the cached path directly
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_files_only=False
            )
            self._info(f"Model available at: {model_path}")
            return model_path

        except Exception as e:
            raise Exception(f"Error downloading model: {e}")

    @torch.no_grad()
    def chop_wav(self, wav, start_frame, end_frame):
        """chop down audio segment."""
        num_frames = (end_frame - start_frame) if (end_frame != -1) else -1

        if (num_frames < self.seg_duration_min_samples):
            print(f"ERROR: Segment too short: {num_frames} frames, minimum required is {self.seg_duration_min_samples} frames.")
            return None, None, -1
         
        wav = wav[:, start_frame:end_frame]
            
        assert (wav.shape[1] <= num_frames) or (num_frames == -1)
        if wav.shape[1] < self.seg_duration_min_samples:
            print(f"Wav shape is too small: {wav.shape}, start_frame: {start_frame}, end_frame: {end_frame}")
            return None, None, -2

        # Process wav
        wav = wav.mean(dim=0)
        wav = self._rms_normalize(wav)
        wav_len = wav.shape[0]
        
        # Pad or truncate
        if wav_len > self.wav_len_max:
            wav = wav[:self.wav_len_max]
            wav_len = wav.shape[0]
        else:
            wav = torch.nn.functional.pad(
                wav, 
                (0, self.wav_len_max - wav.shape[0]), 
                'constant', 
                0
            )
        
        return wav, wav_len, 0
    
    @staticmethod
    def _rms_normalize(audio): # it's important -- CUPE is trained with RMS normalized audio
        """RMS normalize audio tensor."""
        rms = torch.sqrt(torch.mean(audio ** 2))
        if rms > 0:
            audio = audio / rms
        return audio
    



    def load_audio(self, audio, sr=None):
        """Load and preprocess audio file."""
        
        if isinstance(audio, str):
            audio_path = audio
            wav, sr = torchaudio.load(audio_path,  frame_offset=0,  normalize=True)

        elif isinstance(audio, torch.Tensor):
            if sr is None:
                raise ValueError("Sample rate 'sr' must be provided when passing audio as a tensor.")
            wav = audio
        else:
            raise ValueError("Invalid audio input type.")

        
        if sr != self.resampler_sample_rate:
            # load full
            #wav, sr = torchaudio.load(audio_path, normalize=True)
            assert wav.shape[1] > 0, "Audio data is empty"
            # Resample to target sample rate
            
            custom_resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.resampler_sample_rate,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                beta=14.769656459379492,
            )
            wav = custom_resampler(wav)
        else:
            # resample anyway for consistency
            wav = self.default_resampler(wav)
        return wav


    
    def _predict_windows(self, windows_flat, **kwargs):
        '''
        `extractor.predict` over `windows_flat` in groups of `WINDOW_BATCH` windows.

        CUPE classifies each window on its own, so splitting the batch changes nothing in the
        output - but it caps GPU memory, which otherwise grows with segment length because every
        window of the segment goes through one forward pass (~9 MB per second of audio: 2.0 GB
        for a 225 s segment).
        '''
        if windows_flat.shape[0] <= WINDOW_BATCH:
            return self.extractor.predict(windows_flat, **kwargs)
        parts = [self.extractor.predict(windows_flat[i:i + WINDOW_BATCH], **kwargs)
                 for i in range(0, windows_flat.shape[0], WINDOW_BATCH)]
        return tuple(None if parts[0][k] is None else torch.cat([p[k] for p in parts], dim=0)
                     for k in range(len(parts[0])))

    def _cupe_prediction_batch(self, audio_batch, wav_lens, extract_embeddings=False):
        """
        Process a batch of audio through the CUPE model to get logits.

        Args:
            audio_batch: Audio tensor [batch_size, samples]
            wav_lens: List of lengths of audio in samples per batch item
            extract_embeddings: Whether to extract embeddings

        Returns:
            Tuple of (logits_class, logits_group, embeddings, spectral_lens)
        """
        if audio_batch.dim() == 1:
            audio_batch = audio_batch.unsqueeze(0)
        
        
        # Window the audio
        windowed_audio = slice_windows(
            audio_batch.to(self.device), 
            self.sample_rate, 
            self.window_size_ms, 
            self.stride_ms
        )
        
        batch_size, num_windows, window_size = windowed_audio.shape
        windows_flat = windowed_audio.reshape(-1, window_size)

        # model2j checkpoints may carry a second-stage ContextualRefiner (see model2j.py);
        # model2i and model3a extractors have no such attribute. Either way it needs the
        # stitched embeddings to run, regardless of whether the caller itself wants them back.
        # (model3a's post-stitch refiner was DELETED in session_notes.md §5.1 - door 14. What a
        # model3a checkpoint adds now is a boundary head, handled below as a snap source.)
        refiner = getattr(self.extractor, 'refiner', None)
        # Onset snapping reads the stitched embeddings too when its source is 'embeddings',
        # so they must be computed whenever that is the case - not only when the caller wants
        # them returned. `boundary_snap_source` is the RESOLVED source here: 'auto' was turned
        # into a concrete tap by `_resolve_snap_source` when the model loaded, so it can never
        # reach this far.
        # The segmental decoder reads the same tap as the snapper, whether or not snapping is on.
        snapping_on = ((getattr(self, 'boundary_snap_radius', 0) or 0) > 0
                       or self._segmental_needs_tap())
        need_embeddings = (extract_embeddings or (refiner is not None)
                           or (snapping_on and self.boundary_snap_source == 'embeddings'))

        # Whether the onset snapper wants a PRE-TRANSFORMER tap. Asked for from the same forward,
        # so it costs nothing beyond the extra stitch. Only 'embeddings' reads post-transformer,
        # so every other source needs one.
        #
        # Order along the forward path, with each tap's TIMIT r_value_25ms on ue03d_e04 - the
        # full diagram and the caveats are on `boundary_snap_source` in __init__:
        #
        #   feature_extractor → temporal ‖ spectral → fusion → window_processor → posteriors
        #                                   ↑           ↑            ↑                ↑
        #                        'spectral_stream'  'pre_transformer'  'embeddings'  'none'
        #                            0.6895          0.6693           0.6146      control
        #
        # 'none' is vanilla - snapping off, the decoder's own framestamps. It resolves to
        # radius 0 in `_resolve_snap_radius`, so every guard below already handles it.
        #
        want_pre_window = snapping_on and self.boundary_snap_source in (
            'pre_transformer', 'spectral_stream', 'boundary_head')

        # WHICH tap. For 'boundary_head' it is not a choice: the head was trained on one tensor
        # and the checkpoint carries which (`bnd_source`), so feeding it the other would score
        # low without raising.
        if self.boundary_snap_source == 'spectral_stream':
            want_tap = 'spectral_stream'
        elif self.boundary_snap_source == 'boundary_head':
            want_tap = str(getattr(self.extractor, 'bnd_source', 'fusion'))
        else:
            want_tap = 'fusion'

        # Get predictions
        pre_window = None
        if need_embeddings or want_pre_window:
            if want_pre_window:
                try:
                    logits_class, logits_group, embeddings, pre_window = self._predict_windows(
                        windows_flat, return_pre_window=True, groups_only=False, tap=want_tap)
                except TypeError:
                    # An extractor that predates `return_pre_window` / `tap`. Say so rather than
                    # silently snapping against a different tensor than the caller asked for -
                    # a snap result read off the wrong tap is not interpretable.
                    if self.boundary_snap_source in ('boundary_head', 'spectral_stream'):
                        raise ValueError(
                            f"boundary_snap_source={self.boundary_snap_source!r} needs an "
                            f"extractor that can supply the requested pre-transformer tap "
                            f"({want_tap})"
                            + (" and a checkpoint with a trained BoundaryHead (model3a)"
                               if self.boundary_snap_source == 'boundary_head' else "")
                            + ". This one cannot; use 'auto', 'pre_transformer' or 'embeddings'.")
                    if self.boundary_snap_source == 'pre_transformer' and not self._warned_pre_window:
                        self._warn("this extractor cannot supply pre-transformer features; "
                                   "falling back to boundary_snap_source='embeddings', which "
                                   "measures barely above no snapping at all (TIMIT r_value "
                                   "0.5853 vs 0.5812 for 'none').")
                        self._warned_pre_window = True
                    logits_class, logits_group, embeddings = self._predict_windows(
                        windows_flat, return_embeddings=True, groups_only=False)
            else:
                logits_class, logits_group, embeddings = self._predict_windows(
                    windows_flat,
                    return_embeddings=True,
                    groups_only=False
                )
        else:
            logits_class, logits_group = self._predict_windows(
                windows_flat,
                return_embeddings=False,
                groups_only=False
            )
            embeddings = None

        frames_per_window = logits_group.shape[1]
        
        # Reshape outputs
        logits_class = logits_class.reshape(batch_size, num_windows, frames_per_window, -1)
        logits_group = logits_group.reshape(batch_size, num_windows, frames_per_window, -1)
        
        # Get original audio length
        original_audio_length = audio_batch.size(-1)
        
        # Stitch window predictions
        logits_class = stich_window_predictions(
            logits_class, 
            original_audio_length=original_audio_length,
            cnn_output_size=frames_per_window, 
            sample_rate=self.sample_rate, 
            window_size_ms=self.window_size_ms, 
            stride_ms=self.stride_ms,
            stride_frames=self.stride_frames
        )
        
        logits_group = stich_window_predictions(
            logits_group, 
            original_audio_length=original_audio_length,
            cnn_output_size=frames_per_window, 
            sample_rate=self.sample_rate, 
            window_size_ms=self.window_size_ms, 
            stride_ms=self.stride_ms,
            stride_frames=self.stride_frames
        )
        
        if need_embeddings and embeddings is not None:
            embeddings = embeddings.reshape(batch_size, num_windows, frames_per_window, -1)
            embeddings = stich_window_predictions(
                embeddings,
                original_audio_length=original_audio_length,
                cnn_output_size=frames_per_window,
                sample_rate=self.sample_rate,
                window_size_ms=self.window_size_ms,
                stride_ms=self.stride_ms,
                stride_frames=self.stride_frames
            )

        # The pre-transformer map is stitched on the SAME geometry, so an onset index means the
        # same moment in both tensors and the snapper's arithmetic is unchanged.
        if pre_window is not None:
            pre_window = stich_window_predictions(
                pre_window.reshape(batch_size, num_windows, frames_per_window, -1),
                original_audio_length=original_audio_length,
                cnn_output_size=frames_per_window,
                sample_rate=self.sample_rate,
                window_size_ms=self.window_size_ms,
                stride_ms=self.stride_ms,
                stride_frames=self.stride_frames
            )

        # Calculate spectral length
        spectral_lens_t = calc_spec_len_ext(
            torch.tensor(wav_lens),
            self.window_size_ms,
            self.stride_ms,
            self.sample_rate,
            torch.tensor(frames_per_window, dtype=torch.long),
            stride_frames=self.stride_frames
        )
        spectral_lens = spectral_lens_t.tolist()

        # Stage 2a: model3a - the LEARNED onset detector, read off the stitched feature map.
        # Mirrors `p2cupe/trainer.py::forward`, which is the definition of correct: the head was
        # trained on the stitched tensor, so it is applied to the stitched tensor.
        #
        # It touches nothing else. The posteriors, the decoder, the phone identities and the
        # sequence are exactly model2i's - this only offers `snap_onsets_to_embeddings` a
        # different signal to maximise, selected by `boundary_snap_source`.
        boundary_logit = None
        if self.boundary_snap_source == 'boundary_head' and pre_window is not None:
            detect = getattr(self.extractor, 'detect_boundaries', None)
            if detect is None:
                # `_resolve_snap_source` already rejects this at load time; kept as a guard for
                # a caller that reassigned `boundary_snap_source` directly.
                raise ValueError(
                    "boundary_snap_source='boundary_head' but this checkpoint has no "
                    "BoundaryHead (it is not a model3a). Use 'auto' or 'pre_transformer'.")
            boundary_logit = detect(pre_window)

        # Stage 2: contextual refinement over the whole clip (model2j only). Mirrors
        # maestro6.ComposedModel.forward - zero-initialised heads + residual on the base
        # logits mean a checkpoint with an untrained/no-op refiner reproduces the base
        # model exactly, so this is safe to always apply when a refiner is present.
        if refiner is not None:
            with torch.no_grad():
                logits_class, logits_group = refiner(
                    embeddings,
                    lengths=spectral_lens_t.to(embeddings.device),
                    base_logits_class=logits_class,
                    base_logits_group=logits_group,
                )
                self._info(f"Refiner applied: logits_class shape {logits_class.shape}, "
                           f"logits_group shape {logits_group.shape}")
            if not extract_embeddings and not (getattr(self, 'boundary_snap_radius', 0) or 0) > 0:
                embeddings = None

        # The snap source, resolved here so the caller does not re-derive the policy:
        #   'boundary_head'                      -> the head's per-frame logit  [B, T]
        #   'pre_transformer' / 'spectral_stream'-> the stitched CNN map        [B, T, D]
        #   'embeddings'                         -> the 512-d embeddings        [B, T, D]
        snap_source = boundary_logit if boundary_logit is not None else (
            pre_window if pre_window is not None else embeddings)

        return logits_class, logits_group, embeddings, spectral_lens, snap_source
    
    def ensure_target_coverage(self, phoneme_sequences, aligned_frames, seq_lens=None, _silence_class=0, debug=False):
        '''

        Argument:
            phoneme_sequences: List of target phoneme sequences for each batch item (list of lists or list of tensors), shape: [batch_size, target_seq_len] (may include padding)
            aligned_frames: List of lists of aligned phoneme tuples for each batch item (list of lists of tuples, where each tuple is (phoneme_id, start_frame, end_frame, target_seq_idx)), shape: [batch_size, num_aligned_frames]
            seq_lens: Actual (unpadded) length of each phoneme sequence. If None, full length of each sequence is used.

        Returns:
            aligned_frames: Updated list of aligned frames with ensured target coverage (same format as input)

        '''


        for batch_idx in range(len(aligned_frames)):
            
            self.total_phonemes_aligned += len(aligned_frames[batch_idx])
            # CTC output length = max end_frame across all aligned phonemes (computed before any modifications)
            ctc_len = max(f[2] for f in aligned_frames[batch_idx]) if aligned_frames[batch_idx] else 0
            full_seq = phoneme_sequences[batch_idx].tolist() if hasattr(phoneme_sequences[batch_idx], 'tolist') else list(phoneme_sequences[batch_idx])
            # Strip padding using actual sequence length
            actual_len = int(seq_lens[batch_idx]) if seq_lens is not None else len(full_seq)
            target_phoneme_ids = full_seq[:actual_len]
            self.total_phonemes_target += len(target_phoneme_ids)
            targets_found = [0] * len(target_phoneme_ids)

            invalid_target_indices = set()
            for api in range(len(aligned_frames[batch_idx])):
                target_idx = int(aligned_frames[batch_idx][api][3])
                if target_idx < len(target_phoneme_ids) and target_idx != -1:
                    targets_found[target_idx] += 1
                else:
                    invalid_target_indices.add(target_idx)


            repeated_targets = {idx for idx, count in enumerate(targets_found) if count > 1}
            if repeated_targets:
                repeated_labels = [self.phonemizer.index_to_plabel.get(target_phoneme_ids[idx], f'UNK_{target_phoneme_ids[idx]}') for idx in sorted(repeated_targets)]
                if self.warn_level>=2 or debug: print(f"WARNING: Repeated target phonemes found at positions {sorted(repeated_targets)}: {repeated_labels}")
                self.total_phonemes_extra += sum(targets_found[idx] - 1 for idx in repeated_targets)

            missing_targets = [idx for idx, count in enumerate(targets_found) if count == 0]
            if missing_targets:
                missing_labels = [self.phonemizer.index_to_plabel.get(target_phoneme_ids[idx], f'UNK_{target_phoneme_ids[idx]}') for idx in missing_targets]
                if self.warn_level>=2 or debug: print(f"WARNING: {len(missing_targets)} target phonemes not found in alignment at positions {missing_targets}: {missing_labels}")
                self.total_phonemes_missed += len(missing_targets)  

            # Remove aligned phonemes with invalid target indices
            if invalid_target_indices:
                if self.warn_level>=2 or debug: print(f"WARNING: Found {len(invalid_target_indices)} aligned phonemes with invalid target indices in batch {batch_idx}.")
                aligned_frames[batch_idx] = [frame for frame in aligned_frames[batch_idx] if int(frame[3]) not in invalid_target_indices]
                if self.warn_level>=2 or debug: print(f"WARNING: Removed aligned phonemes with invalid target indices {invalid_target_indices} from batch {batch_idx}.")

            # Deduplicate repeated target phonemes: merge consecutive, else keep longest
            if repeated_targets and self.ensure_completeness:
                repeated_frames = {}
                new_aligned_frames = []
                for frame in aligned_frames[batch_idx]:
                    target_idx = int(frame[3])
                    if target_idx in repeated_targets:
                        repeated_frames.setdefault(target_idx, []).append(frame)
                    else:
                        new_aligned_frames.append(frame)

                for target_idx in sorted(repeated_frames.keys()):
                    frames = sorted(repeated_frames[target_idx], key=lambda x: x[1])
                    # Merge consecutive/overlapping frames
                    merged = [(frames[0][0], frames[0][1], frames[0][2], frames[0][3])]
                    for f in frames[1:]:
                        prev = merged[-1]
                        if f[1] <= prev[2]:  # consecutive or overlapping
                            merged[-1] = (prev[0], prev[1], max(prev[2], f[2]), prev[3])
                        else:
                            merged.append((f[0], f[1], f[2], f[3]))
                    # Keep the one with the most frames; tie-break: earliest
                    best = max(merged, key=lambda x: x[2] - x[1])
                    new_aligned_frames.append(best)

                aligned_frames[batch_idx] = new_aligned_frames

            # Insert missing target phonemes with estimated frame positions
            skipped_sil_indices = set()
            if missing_targets and self.ensure_completeness:
                # Build target_idx -> aligned frame lookup
                target_to_frame = {}
                for frame in aligned_frames[batch_idx]:
                    target_to_frame[int(frame[3])] = frame

                # Group consecutive missing targets so we can distribute frames evenly within each gap
                missing_groups = [[missing_targets[0]]]
                for i in range(1, len(missing_targets)):
                    if missing_targets[i] == missing_targets[i - 1] + 1:
                        missing_groups[-1].append(missing_targets[i])
                    else:
                        missing_groups.append([missing_targets[i]])

                for group in missing_groups:
                    n_missing = len(group)

                    # Find nearest anchor before this group
                    prev_frame = None
                    for tidx in range(group[0] - 1, -1, -1):
                        if tidx in target_to_frame:
                            prev_frame = target_to_frame[tidx]
                            break

                    # Find nearest anchor after this group
                    next_frame = None
                    for tidx in range(group[-1] + 1, len(target_phoneme_ids)):
                        if tidx in target_to_frame:
                            next_frame = target_to_frame[tidx]
                            break

                    # Determine the frame range available for this group
                    if prev_frame and next_frame:
                        gap_start = prev_frame[2]   # end_frame of previous
                        gap_end = next_frame[1]      # start_frame of next
                    elif prev_frame:
                        # --- Trailing missing targets (no anchor after) ---

                        # Skip trailing SIL phonemes — silence at the end doesn't need insertion
                        non_sil_group = [tidx for tidx in group if target_phoneme_ids[tidx] != _silence_class]
                        sil_in_group = [tidx for tidx in group if target_phoneme_ids[tidx] == _silence_class]
                        if sil_in_group:
                            if debug: print(f"  Skipping {len(sil_in_group)} trailing SIL phoneme(s)")
                            skipped_sil_indices.update(sil_in_group)
                        if not non_sil_group:
                            continue

                        n_needed = len(non_sil_group)

                        # Try to make room by contracting SIL-aligned phonemes backward,
                        # then shifting all subsequent frames back to free frames at the end
                        sorted_frames = sorted(aligned_frames[batch_idx], key=lambda x: x[1])
                        frames_freed = 0

                        # Pass 1: contract SIL phonemes only
                        for scan_idx in range(len(sorted_frames) - 1, -1, -1):
                            if frames_freed >= n_needed:
                                break
                            f = sorted_frames[scan_idx]
                            span = f[2] - f[1]
                            if f[0] == _silence_class and span > 1:
                                can_give = min(span - 1, n_needed - frames_freed)
                                sorted_frames[scan_idx] = (*f[:2], f[2] - can_give, *f[3:])
                                for j in range(scan_idx + 1, len(sorted_frames)):
                                    sj = sorted_frames[j]
                                    sorted_frames[j] = (sj[0], sj[1] - can_give, sj[2] - can_give, *sj[3:])
                                frames_freed += can_give

                        # Pass 2: if still not enough, contract any phoneme (last first)
                        if frames_freed < n_needed:
                            for steal_idx in range(len(sorted_frames) - 1, -1, -1):
                                if frames_freed >= n_needed:
                                    break
                                f = sorted_frames[steal_idx]
                                span = f[2] - f[1]
                                if span > 1:
                                    can_steal = min(span - 1, n_needed - frames_freed)
                                    sorted_frames[steal_idx] = (*f[:2], f[2] - can_steal, *f[3:])
                                    for j in range(steal_idx + 1, len(sorted_frames)):
                                        sj = sorted_frames[j]
                                        sorted_frames[j] = (sj[0], sj[1] - can_steal, sj[2] - can_steal, *sj[3:])
                                    frames_freed += can_steal

                        aligned_frames[batch_idx] = list(sorted_frames)
                        target_to_frame = {int(f[3]): f for f in aligned_frames[batch_idx]}
                        last_end = max(f[2] for f in aligned_frames[batch_idx])

                        for i, tidx in enumerate(non_sil_group):
                            est_start = min(last_end + i, ctc_len - 1)
                            est_end = min(est_start + 1, ctc_len)
                            new_frame = (target_phoneme_ids[tidx], est_start, est_end, tidx, True)
                            self.total_phonemes_aligned += 1
                            aligned_frames[batch_idx].append(new_frame)
                            target_to_frame[tidx] = new_frame
                        continue

                    elif next_frame:
                        gap_end = next_frame[1]
                        gap_start = max(0, gap_end - n_missing)
                    else:
                        gap_start = 0
                        gap_end = n_missing

                    # Distribute frames evenly across the group, clamped to valid CTC range
                    gap_size = max(gap_end - gap_start, n_missing)
                    frames_per_phoneme = gap_size / n_missing

                    for i, tidx in enumerate(group):
                        est_start = min(int(gap_start + i * frames_per_phoneme), ctc_len - 1)
                        est_end = min(max(int(gap_start + (i + 1) * frames_per_phoneme), est_start + 1), ctc_len)
                        new_frame = (target_phoneme_ids[tidx], est_start, est_end, tidx, True)
                        self.total_phonemes_aligned += 1
                        aligned_frames[batch_idx].append(new_frame)
                        target_to_frame[tidx] = new_frame

            # Sort by start_frame to maintain order
            aligned_frames[batch_idx].sort(key=lambda x: x[1])

            # add False flag to non-estimated frames for consistency
            for i in range(len(aligned_frames[batch_idx])):
                if len(aligned_frames[batch_idx][i]) == 4:
                    aligned_frames[batch_idx][i] = (*aligned_frames[batch_idx][i], False)
                    self.total_phonemes_aligned_easily += 1

            if self.ensure_completeness:
                expected_count = len(target_phoneme_ids) - len(skipped_sil_indices)
                targets_found2 = [0] * len(target_phoneme_ids)
                for api in range(len(aligned_frames[batch_idx])):
                    target_idx = int(aligned_frames[batch_idx][api][3])
                    if target_idx < len(target_phoneme_ids) and target_idx != -1:
                        targets_found2[target_idx] += 1
                actual_covered = sum(targets_found2)
                if actual_covered != expected_count or len(aligned_frames[batch_idx]) != expected_count:
                    raise Exception(f"Post-processing error: target coverage mismatch for segment {batch_idx}. Expected {expected_count}, got {actual_covered} covered, {len(aligned_frames[batch_idx])} aligned. Skipped SIL: {skipped_sil_indices}")

        return aligned_frames
    

    def extend_soft_boundaries_func(self, log_probs, framestamps, boundary_softness=3, debug=False):
        '''
        Extend start and end boundaries of aligned phonemes based on confidence scores from log probabilities.
        Three passes:
          1. Extend start boundaries with strict thresholds
          2. Extend end boundaries with lenient thresholds
          3. Extend start boundaries again with even more lenient thresholds (to fill remaining gaps)

        Args:
            log_probs: Tensor of shape [B, T, C] containing log probabilities.
            framestamps: List of lists of tuples (phoneme_id, start_frame, end_frame, target_seq_idx, is_estimated).
            boundary_softness: Hyperparameter controlling leniency of boundary extension. Higher means more lenient (allows extension with lower confidence). Default is 3, which corresponds to a threshold of 0.0000001.

        Returns:
            Updated framestamps with extended boundaries (same structure as input).
        '''
        max_extension_factor = 10.0
        boundary_softness_initial = max(2, 7 - 4)  # initial pass is stricter, then we relax it in subsequent passes
        _thresh_1 = 10.0 ** (-1*boundary_softness_initial)  # e.g. 0.00001 for boundary_softness_initial of 5
        _thresh_2 = 10.0 ** (-1*boundary_softness)  # e.g. 0.001 for boundary_softness of 3
        updated_batch = []
        for b in range(len(framestamps)):
            probs = torch.exp(log_probs[b])  # [T, C]
            num_frames = probs.shape[0]
            tuples_list = list(framestamps[b])

            # Precompute mean probs for each phoneme using original boundaries
            mean_probs = []
            for phoneme_id, start_frame, end_frame, _, _ in tuples_list:
                if start_frame < num_frames and phoneme_id < probs.shape[1] and start_frame < end_frame:
                    mean_probs.append(probs[start_frame:end_frame, phoneme_id].mean().item())
                else:
                    mean_probs.append(0.001)

            # --- Pass 1: Extend start boundaries (strict) ---
            for i in range(len(tuples_list)):
                phoneme_id, start_frame, end_frame, target_seq_idx, is_estimated = tuples_list[i]
                if start_frame >= num_frames or phoneme_id >= probs.shape[1]:
                    continue
                original_duration = end_frame - start_frame

                min_start = max(0, int(start_frame - original_duration * max_extension_factor))
                if i > 0:
                    min_start = max(min_start, min(tuples_list[i - 1][2]+10, start_frame))  # prev end_frame + 10

                start_threshold = min(mean_probs[i] * _thresh_1, _thresh_1) # strict threshold for initial extension, based on mean confidence of the phoneme's original segment
                new_start = start_frame
                for f in range(start_frame - 1, min_start - 1, -1):
                    if probs[f, phoneme_id].item() >= start_threshold:
                        new_start = f
                    else:
                        break

                tuples_list[i] = (phoneme_id, new_start, end_frame, target_seq_idx, is_estimated)

            # --- Pass 2: Extend end boundaries (lenient) ---
            for i in range(len(tuples_list)):
                phoneme_id, start_frame, end_frame, target_seq_idx, is_estimated = tuples_list[i]
                if start_frame >= num_frames or phoneme_id >= probs.shape[1]:
                    continue
                original_duration = end_frame - start_frame

                max_end = min(num_frames, int(end_frame + original_duration * max_extension_factor))
                if i + 1 < len(tuples_list):
                    max_end = min(max_end, min(end_frame, tuples_list[i + 1][1]-10))  # next start_frame

                end_threshold = min(mean_probs[i] * _thresh_1, _thresh_1) # strict threshold for initial extension, based on mean confidence of the phoneme's original segment
                new_end = end_frame
                for f in range(end_frame, max_end):
                    if probs[f, phoneme_id].item() >= end_threshold:
                        new_end = f + 1
                    else:
                        break
                tuples_list[i] = (phoneme_id, start_frame, new_end, target_seq_idx, is_estimated)

            # --- Pass 3: Extend start boundaries again (lenient, fills remaining gaps) ---
            for i in range(len(tuples_list)):
                phoneme_id, start_frame, end_frame, target_seq_idx, is_estimated = tuples_list[i]
                if start_frame >= num_frames or phoneme_id >= probs.shape[1]:
                    continue

                min_start = 0
                if i > 0:
                    min_start = tuples_list[i - 1][2]  # prev end_frame (already extended)

                # Only extend if there's a gap to fill
                if start_frame <= min_start:
                    continue

                new_start = start_frame
                for f in range(start_frame - 1, min_start - 1, -1):
                    if probs[f, phoneme_id].item() >= _thresh_2: # _thresh_2 of 7 means threshold of 0.0000001, which is very lenient and allows extension as long as there's any meaningful confidence
                        new_start = f
                    else:
                        break

                tuples_list[i] = (phoneme_id, new_start, end_frame, target_seq_idx, is_estimated)


            # --- Pass 4: Extend end boundaries  (lenient, fills remaining gaps) ---
            for i in range(len(tuples_list)):
                phoneme_id, start_frame, end_frame, target_seq_idx, is_estimated = tuples_list[i]
                if start_frame >= num_frames or phoneme_id >= probs.shape[1]:
                    continue
                original_duration = end_frame - start_frame

                max_end = min(num_frames, int(end_frame + original_duration * max_extension_factor))
                if i + 1 < len(tuples_list):
                    max_end = min(max_end, tuples_list[i + 1][1])  # next start_frame

                #end_threshold = min(mean_probs[i] * 0.00001, 0.000001)
                new_end = end_frame
                for f in range(end_frame, max_end):
                    if probs[f, phoneme_id].item() >= _thresh_2:
                        new_end = f + 1
                    else:
                        break

                if debug and (i < 5 or i >= len(tuples_list) - 5):
                    label = self.phonemizer.index_to_plabel.get(phoneme_id, f'UNK_{phoneme_id}')
                    print(f"Extended {label} (idx {target_seq_idx}): [{framestamps[b][i][1]}, {framestamps[b][i][2]}) -> [{start_frame}, {new_end}), mean={mean_probs[i]:.4f}")
                    if i == 4: print("...")

                tuples_list[i] = (phoneme_id, start_frame, new_end, target_seq_idx, is_estimated)

            updated_batch.append(tuples_list)

        return updated_batch

    def _segmental_needs_tap(self):
        '''Whether the segmental decoder will run, and so needs the boundary tap computed.'''
        return (getattr(self, 'decoder', 'ctc') == 'segmental'
                and getattr(self, 'n_substates', 1) == 1
                and getattr(self, 'boundary_snap_source', 'none') != 'none')

    def _decode_segmental_batch(self, log_probs, ph_seqs, ph_seq_lens, spectral_lens,
                                change_source, word_sequences=None):
        '''
        Run `segmental.decode_segmental` per batch item. Returns framestamps in the same
        `(phoneme_id, start_frame, end_frame, target_seq_idx)` form as the CTC decoder.

        `change_source` is the snap tap: a [B, T, D] feature map, whose adjacent-frame cosine
        distance is the spectral flux, or a [B, T] per-frame boundary score.
        '''
        per_frame = (change_source.dim() == 2) or (change_source.shape[-1] == 1)
        out = []
        for b in range(log_probs.shape[0]):
            T = int(spectral_lens[b])
            n = int(ph_seq_lens[b])
            if n == 0:
                out.append([])
                continue
            T_src = min(T, change_source.shape[1])
            if per_frame:
                change = change_source[b, :T_src].float().reshape(-1).cpu().numpy()
            else:
                emb = change_source[b, :T_src].float()
                change = (1.0 - F.cosine_similarity(emb[:-1], emb[1:], dim=-1)).cpu().numpy()
            words = None
            if word_sequences is not None and word_sequences[b] is not None \
                    and len(word_sequences[b]) >= n:
                words = list(word_sequences[b])[:n]
            out.append(decode_segmental(
                log_probs[b, :T], ph_seqs[b, :n].tolist(), self.model_blank_class,
                self.silence_class, change=change, word_idx=words,
                enforce=self.enforce_all_targets, change_per_frame=per_frame))
        return out

    def extract_timestamps_from_segment_batch(self, wavs, wav_lens, phoneme_sequences, start_offset_times=0,
                                      group_sequences=None, extract_embeddings=True, do_groups=True,
                                      debug=False, word_sequences=None):
        """
        Extract phoneme and group timestamps from multiple audio segments (batch processing).

        Args:
            wavs: Batch of audio tensors for the segments.
            wav_lens: List of lengths of the audio segments in samples.
            phoneme_sequences: List of phoneme sequences (each as list or tensor of phoneme indices).
            start_offset_times: Start time offsets in seconds for each segment (list or single value).
            group_sequences: Optional list of phoneme group sequences.
            extract_embeddings: Whether to extract pooled embeddings.
            do_groups: Whether to extract phoneme group timestamps.
            debug: Enable debug output.
            word_sequences: Optional word index per phoneme, per segment (the phonemizer's
                `word_num`). Only the segmental decoder uses it: pauses are allowed between
                words, and `enforce_all_targets=False` skips whole words. Without it every
                phoneme is treated as its own word.

        Returns:
            timestamp_dicts: List of dictionaries with 'phoneme_timestamps' and 'group_timestamps'.
                    `timestamp_dicts = [
                    {
                        'phoneme_timestamps': frame_phonemes[b],
                        'group_timestamps': frame_groups[b],
                    }
                    for b in range(len(frame_phonemes))
                ]`
            pooled_embeddings_phonemes_list: List of pooled phoneme embeddings or list of None.
            pooled_embeddings_groups_list: List of pooled group embeddings or list of None.
        """
    

        # Compute actual sequence lengths BEFORE padding to tensor
        if isinstance(phoneme_sequences, torch.Tensor):
            # Already a tensor — count non-padding elements per row
            ph_seq_lens = [(seq != self.blank_class).sum().item() for seq in phoneme_sequences]
        else:
            ph_seq_lens = [len(seq) for seq in phoneme_sequences]

        # Convert sequences to tensors
        if not isinstance(phoneme_sequences, torch.Tensor):
            #padding phoneme sequences to have same length
            max_len = max(len(seq) for seq in phoneme_sequences)
            phoneme_sequences = torch.tensor([seq + [self.blank_class] * (max_len - len(seq)) for seq in phoneme_sequences], dtype=torch.long)

        if group_sequences is not None and not isinstance(group_sequences, torch.Tensor):
            #padding
            max_len = max(len(seq) for seq in group_sequences)
            group_sequences = torch.tensor([seq + [self.blank_group] * (max_len - len(seq)) for seq in group_sequences], dtype=torch.long)
            #
        # Generate group sequence if not provided
        
        if group_sequences is None:
            mapped = []
            # Support both list-of-lists and 2D tensor inputs
            if isinstance(phoneme_sequences, torch.Tensor) and phoneme_sequences.dim() == 2:
                seq_iter = [row.tolist() for row in phoneme_sequences]
            else:
                seq_iter = list(phoneme_sequences)

            for seq in seq_iter:
                grp = self._map_phonemes_to_groups(seq)
                if not isinstance(grp, torch.Tensor):
                    grp = torch.tensor(grp, dtype=torch.long)
                mapped.append(grp)

            # Pad sequences to same length and stack
            if len(mapped) == 0:
                group_sequences = torch.empty(0, dtype=torch.long)
            else:
                max_len = max(m.size(0) for m in mapped)
                padded = [
                    m if m.size(0) == max_len else torch.nn.functional.pad(m, (0, max_len - m.size(0)), value=self.blank_group)
                    for m in mapped
                ]
                group_sequences = torch.stack(padded, dim=0)

        assert self.extractor is not None, "CUPE extractor model is not loaded. Use load_model() to load the model before calling this method."

        logits_class, logits_group, embeddings, spectral_lens, snap_source = self._cupe_prediction_batch(wavs, wav_lens, extract_embeddings)


        # Prepare sequences for alignment
        ph_seqs = phoneme_sequences.to(self.device)
        grp_seqs = group_sequences.to(self.device)
        ph_seq_lens = torch.tensor(ph_seq_lens, dtype=torch.long).to(self.device)
        spectral_lens = torch.tensor(spectral_lens, dtype=torch.long).to(self.device)
        
        # Get log probabilities
        log_probs_g = F.log_softmax(logits_group, dim=2)
        log_probs_p = F.log_softmax(logits_class, dim=2)

        
        # With a sub-state model the class head emits p_B/p_M/p_E, so the phone lattice is
        # built over the EXPANDED sequence and the framestamps are collapsed straight back.
        # Doing it at this boundary rather than inside the Viterbi keeps the decoder itself
        # (and door 7's settled blank-stride behaviour) completely untouched: the triplet
        # simply occupies the slots one phone used to, and the three sub-state labels are
        # distinct, so CTC's skip transition already permits a direct B->M->E walk.
        use_segmental = self._segmental_needs_tap() and snap_source is not None
        if use_segmental:
            # Onsets are placed with the boundary tap INSIDE the Viterbi, so there is nothing
            # left for the greedy snapper below to do.
            frame_phonemes = self._decode_segmental_batch(
                log_probs_p, ph_seqs, ph_seq_lens, spectral_lens, snap_source, word_sequences)
        else:
            decode_ph_seqs, decode_ph_lens = self._expand_substates(ph_seqs, ph_seq_lens)

            frame_phonemes = self.alignment_utils_p.decode_alignments(
                        log_probs_p,
                        true_seqs=decode_ph_seqs,
                        pred_lens=spectral_lens,
                        true_seqs_lens=decode_ph_lens,
                        forced_alignment=True,
                        boost_targets=self.boost_targets,
                        enforce_minimum=self.enforce_minimum,
                        debug=debug
                    )
            frame_phonemes = self._collapse_substates(frame_phonemes)

        # §4 route 1: re-place the onsets using the embedding change signal, which carries
        # boundary information the posteriors have lost. Identities and order are untouched.
        # Applied BEFORE ensure_target_coverage/extend_soft_boundaries so those still see a
        # well-formed, monotonic segmentation. `boundary_snap_radius=0` disables it.
        if (not use_segmental) and (getattr(self, 'boundary_snap_radius', 0) or 0) > 0:
            # `snap_source` is already resolved by `_cupe_prediction_batch` from
            # `boundary_snap_source`: the head's per-frame logit, the pre-transformer map, or
            # the 512-d embeddings. `snap_onsets_to_embeddings` reads the per-frame and the
            # feature-map forms differently but places onsets identically.
            frame_phonemes = self.snap_onsets_to_embeddings(
                frame_phonemes, snap_source, spectral_lens,
                radius=self.boundary_snap_radius, offset=self.boundary_snap_offset)

        if use_segmental:
            # Groups are a many-to-one relabelling of the phonemes, so they take the phoneme
            # segmentation directly: consistent with it by construction, and one Viterbi
            # instead of two.
            grp_list = grp_seqs.tolist()
            frame_groups = [[(int(grp_list[b][ti]), sf, ef, ti) for (_ph, sf, ef, ti) in fp]
                            for b, fp in enumerate(frame_phonemes)]
        else:
            # Forced alignment with target boosting
            frame_groups = self.alignment_utils_g.decode_alignments(
                log_probs_g,
                true_seqs=grp_seqs,
                pred_lens=spectral_lens,
                true_seqs_lens=ph_seq_lens,
                forced_alignment=True,
                boost_targets=self.boost_targets,
                enforce_minimum=self.enforce_minimum,
            )


        frame_phonemes = self.ensure_target_coverage(phoneme_sequences, aligned_frames=frame_phonemes, seq_lens=ph_seq_lens, _silence_class=self.silence_class, debug=debug)
        frame_groups = self.ensure_target_coverage(group_sequences, aligned_frames=frame_groups, seq_lens=ph_seq_lens, _silence_class=self.silence_group, debug=debug)

        if self.extend_soft_boundaries and self.boundary_softness > 0:
            if debug: print("Extending end boundaries of aligned phonemes/groups based on confidence scores...")
            frame_phonemes = self.extend_soft_boundaries_func(log_probs_p, frame_phonemes, boundary_softness=self.boundary_softness, debug=debug)
            frame_groups = self.extend_soft_boundaries_func(log_probs_g, frame_groups, boundary_softness=self.boundary_softness, debug=False)
            
        # Calculate confidence scores
        
        for b in range(len(frame_phonemes)): # loop over batch items
            frame_phonemes[b] = _calculate_confidences(log_probs_p[b], frame_phonemes[b])
            frame_groups[b] = _calculate_confidences(log_probs_g[b], frame_groups[b])

            frame_phonemes[b] = convert_to_ms(
                frame_phonemes[b],
                spectral_lens[b],
                start_offset_times[b] if isinstance(start_offset_times, (list, tuple)) else start_offset_times,
                wav_lens[b],
                self.resampler_sample_rate,
                frame_stride_samples=self.frame_stride_wav
            )
            frame_groups[b] = convert_to_ms(
                frame_groups[b],
                spectral_lens[b],
                start_offset_times[b] if isinstance(start_offset_times, (list, tuple)) else start_offset_times,
                wav_lens[b],
                self.resampler_sample_rate,
                frame_stride_samples=self.frame_stride_wav
            )

            # Resort phonemes and groups by timestamp after adding missing phonemes
            frame_phonemes[b] = sorted(frame_phonemes[b], key=lambda x: x[6])  # sort by start timestamp (ms)
            frame_groups[b] = sorted(frame_groups[b], key=lambda x: x[6])  # sort by start timestamp (ms)
            
        timestamp_dicts = [
            {
                'phoneme_timestamps': frame_phonemes[b],
                'group_timestamps': frame_groups[b],
            }
            for b in range(len(frame_phonemes))
        ]

        # Extract pooled embeddings if requested
        pooled_embeddings_phonemes_list = []
        pooled_embeddings_groups_list = []

        if extract_embeddings and embeddings is not None:
            for b in range(len(frame_phonemes)):
                pooled_emb_ph = weighted_pool_embeddings(
                    embeddings[b][:spectral_lens[b]],
                    log_probs_p[b][:spectral_lens[b]],
                    frame_phonemes[b]
                )
                pooled_embeddings_phonemes_list.append(pooled_emb_ph)

                if do_groups:
                    pooled_emb_gr = weighted_pool_embeddings(
                        embeddings[b][:spectral_lens[b]],
                        log_probs_g[b][:spectral_lens[b]],
                        frame_groups[b]
                    )
                    pooled_embeddings_groups_list.append(pooled_emb_gr)

            if do_groups:
                return timestamp_dicts, pooled_embeddings_phonemes_list, pooled_embeddings_groups_list
            else:
                return timestamp_dicts, pooled_embeddings_phonemes_list, [None] * len(frame_phonemes)

        return timestamp_dicts, [None] * len(frame_phonemes), [None] * len(frame_phonemes)

    
    # ======================================================================================
    # Adaptive chunking for long segments.  See `chunking.py` for the rationale and the
    # measurement that motivated it.
    # ======================================================================================

    def set_chunking(self, enabled=None, seconds=None, max_phonemes=None,
                     phonemes_per_second=None, vad=None, derail_frames=None,
                     derail_run=None, backoff=None, conf_floor=None, weak_run=None,
                     snap_to_silence=None, verbose=None):
        """
        Retune the long-segment chunker on a live aligner, for sweeping without
        reconstructing it. Every argument left as None keeps its current value.

        Args:
            enabled: master switch (`chunk_audio`). False restores whole-segment decoding.
            seconds: audio window length.
            max_phonemes: cap on one window's phoneme budget.
            phonemes_per_second: budget rate, measured over SPEECH seconds.
            vad: use the energy VAD for speech-seconds and cut snapping.
            derail_frames: a phoneme this short (frames) is the decoder out of audio.
            derail_run: consecutive short phonemes that constitute a cram.
            backoff: minimum phonemes dropped before a detected cut.
            conf_floor: confidence below which a phoneme counts as unseen.
            weak_run: consecutive below-floor phonemes that derail a window on their own.
            snap_to_silence: seconds a cut may be moved forward to land in a pause; 0 off.
            verbose: print the resolved configuration. None follows `warn_level>=2`.

        Returns:
            The resolved configuration as a dict.
        """
        if enabled is not None:
            self.chunk_audio = bool(enabled)
        if seconds is not None:
            self.chunk_seconds = float(seconds)
        if max_phonemes is not None:
            self.chunk_max_phonemes = int(max_phonemes)
        if phonemes_per_second is not None:
            self.chunk_phonemes_per_second = float(phonemes_per_second)
        if vad is not None:
            self.chunk_vad = bool(vad)
        if derail_frames is not None:
            self.chunk_derail_frames = int(derail_frames)
        if derail_run is not None:
            self.chunk_derail_run = int(derail_run)
        if backoff is not None:
            self.chunk_backoff = int(backoff)
        if conf_floor is not None:
            self.chunk_conf_floor = float(conf_floor)
        if weak_run is not None:
            self.chunk_weak_run = int(weak_run)
        if snap_to_silence is not None:
            self.chunk_snap_to_silence = float(snap_to_silence)
        # Re-checked here, not only in __init__, because all three of `seconds`,
        # `max_phonemes` and `phonemes_per_second` feed the same constraint: whichever of
        # the rate and the cap binds first has to stay under what the window's frames can
        # hold at CTC stride 4 (152 phonemes for an 8s window, 190 for 10s).
        self._warn_if_budget_overruns_lattice()

        cfg = {
            'enabled': self.chunk_audio, 'seconds': self.chunk_seconds,
            'max_phonemes': self.chunk_max_phonemes,
            'phonemes_per_second': self.chunk_phonemes_per_second,
            'vad': self.chunk_vad, 'derail_frames': self.chunk_derail_frames,
            'derail_run': self.chunk_derail_run, 'backoff': self.chunk_backoff,
            'conf_floor': self.chunk_conf_floor, 'weak_run': self.chunk_weak_run,
            'snap_to_silence': self.chunk_snap_to_silence,
        }
        if verbose or (verbose is None and self.warn_level >= 2):
            print("chunking: " + ", ".join(f"{k}={v}" for k, v in cfg.items()))
        return cfg

    def _warn_if_budget_overruns_lattice(self):
        """
        Warn when a window's phoneme budget cannot fit the CTC lattice at stride 4.

        `chunk_seconds` and the budget are not independent. The forced decoder builds a
        `stride * S + 1` lattice and walks `stride` down from 4 until it fits the window's
        frames; below 4 there is less blank room between phones and placement degrades.

        Both knobs are checked because either can cause it: raising the rate, or shortening
        the window without lowering the cap (an 8s window holds 152 phonemes at stride 4
        where a 10s window holds 190). Nothing is silently adjusted - the caller may have a
        reason - but it is said out loud rather than surfacing later as a bad alignment.

        This is a guard against a condition no tested configuration has actually hit; the
        measured cost of a high rate comes from somewhere else (see
        `chunking.DEFAULT_PHONEMES_PER_SECOND`). Keep the warning quiet rather than loud.
        """
        if not getattr(self, 'chunk_audio', False):
            return
        frame_ms = 1000.0 * getattr(self, 'frame_stride_wav', 210) / self.sample_rate
        safe = stride4_max_phonemes(self.chunk_seconds, 1000.0 / frame_ms)
        rate_ask = int(math.ceil(self.chunk_phonemes_per_second * self.chunk_seconds))
        budget = min(self.chunk_max_phonemes, rate_ask)
        if budget > safe:
            driver = ("chunk_max_phonemes" if self.chunk_max_phonemes <= rate_ask
                      else f"chunk_phonemes_per_second={self.chunk_phonemes_per_second:g}")
            self._warn(
                f"a full {self.chunk_seconds:g}s window can budget {budget} phonemes "
                f"({driver}), but only {safe} fit the CTC lattice at stride 4. The decoder "
                f"will compress the lattice and windows may derail internally. Use "
                f"chunk_max_phonemes<={safe}, or chunking.stride4_max_phonemes"
                f"({self.chunk_seconds:g}) to derive it.")

    def detect_speech_segments(self, audio_wav, **kwargs):
        """
        Energy VAD over a waveform. Returns `[(start_sec, end_sec), ...]` for the speech
        spans, in order.

        Exposed because it is useful on its own (marking `[SIL]` regions, trimming, sanity
        checks), but its job inside BFA is narrow: size each chunk's phoneme budget from
        SPEECH seconds rather than wall-clock seconds, and let a chunk cut land in a pause.
        It is independent of `silence_anchors`, which is a different mechanism operating on
        the posteriors and the target sequence - see `chunking.py`.

        Args:
            audio_wav: waveform tensor, 1-D or (C, T).
            **kwargs: forwarded to `chunking.detect_speech_segments` (frame_ms, hop_ms,
                noise_margin_db, dynamic_range_db, min_speech_ms, min_silence_ms, pad_ms).
        """
        return _chunking.detect_speech_segments(audio_wav, self.resampler_sample_rate, **kwargs)

    @contextlib.contextmanager
    def _chunk_window_scope(self):
        """
        Run one chunk window without letting it touch the segment-level bookkeeping.

        A window is an internal step, not a unit of work the caller asked for, and most of
        one is thrown away. Left alone it would corrupt both halves of the reporting:

          - `ensure_target_coverage` counts every window's targets, so a phoneme aligned in
            the discarded tail of window 1 and again in window 2 is counted twice. On
            30s.wav that turned 356 targets into a reported 920.
          - it also warns about targets it could not place, which for the over-provisioned
            tail is the EXPECTED outcome - those phonemes are the next window's job.

        So the counters are snapshotted and restored, and `warn_level` is held at 1 for the
        duration (silencing the level-2 coverage chatter but not real warnings).
        `align_segment_chunked` then attributes the segment's totals once, from the
        assembled result.

        SILENCE ANCHORING IS ALSO DISABLED inside a window, and that one is not just
        bookkeeping. `silence_anchors` splits the target sequence at SIL tokens from the
        transcript's punctuation and decodes each span separately - a mitigation for
        exactly the long-segment collapse the chunker now handles directly, applied to a
        window that is already short enough not to need it. Measured on 30s.wav, chunked:

            silence_anchors    0      3      10
            mean confidence  0.403  0.330  0.403     (0.330 also trips `bad_alignment`)

        0 and 10 are identical because at 10 no span qualifies inside a 10 s window, so it
        falls through to the plain Viterbi. 3 does fire, and it only costs accuracy: the
        window gets cut at punctuation that may not be where the speaker paused, on top of
        the cut the chunker is about to make on evidence. Holding it at 0 makes chunked
        results independent of a setting that no longer applies to them. The parameter
        still does its original job when `chunk_audio=False`.
        """
        names = ('total_segments_processed', 'total_segments_bad', 'total_segments_failed',
                 'total_phonemes_aligned', 'total_phonemes_target',
                 'total_phonemes_aligned_easily', 'total_phonemes_missed',
                 'total_phonemes_extra', 'perfect_matches')
        saved = {n: getattr(self, n) for n in names}
        saved_warn = self.warn_level
        self.warn_level = min(self.warn_level, 1)
        # `decode_alignments` reads `anchor_pauses=self.silence_anchors > 0` off these, so
        # zeroing them is what turns the segmented path off for the window.
        utils = [u for u in (getattr(self, 'alignment_utils_p', None),
                             getattr(self, 'alignment_utils_g', None)) if u is not None]
        saved_anchors = [u.silence_anchors for u in utils]
        for u in utils:
            u.silence_anchors = 0
        try:
            yield
        finally:
            for u, v in zip(utils, saved_anchors):
                u.silence_anchors = v
            self.warn_level = saved_warn
            for n, v in saved.items():
                setattr(self, n, v)

    def _needs_chunking(self, duration_sec, n_phonemes):
        """
        Whether a segment goes down the chunked path.

        Both conditions matter and neither implies the other: 40 s of sparse speech holds
        few phonemes but still overruns the decoder's usable window, while 30 s of rapid
        speech overruns the phoneme budget first.

        The test is a plain `>`, so `chunk_seconds` means what it says: anything longer is
        chunked. It used to carry 10% slack on each side, which made the real threshold 11 s
        for a nominal 10 - so a 10.2 s segment took the single-shot path and collapsed, while
        the setting that would have caught it (`chunk_seconds`) appeared to be satisfied.
        A segment of exactly `chunk_seconds` still takes the cheaper single-shot path.
        """
        if not self.chunk_audio:
            return False
        if self._segmental_needs_tap():
            # The segmental decoder does not collapse on long segments (recall is flat across
            # position on 30 s segments, and 30s.wav keeps its confidence to the last word), so
            # windowing only adds cut points that can land mid-phone: on 40 Buckeye segments,
            # onset f1@25 69.2 whole vs 68.3 chunked, and 30s.wav mean confidence 0.728 vs
            # 0.661, and on a single 225 s Buckeye segment 67.3 vs 54.7. It is kept only as a
            # memory/time guard above `segmental_max_seconds`.
            return duration_sec > self.segmental_max_seconds
        return (duration_sec > self.chunk_seconds
                or n_phonemes > self.chunk_max_phonemes)

    @torch.no_grad()
    def align_segment_whole(self, clip_wav, start_sec, end_sec, phoneme_sequence,
                            group_sequence=None, extract_embeddings=False, do_groups=False,
                            debug=False, word_sequence=None):
        '''
        Align one segment in a single decode on its own, UNPADDED audio.

        This is the path for segments longer than `duration_max` that are not chunked. The
        batched path pads every segment to `duration_max` and would otherwise truncate a
        longer one silently, dropping its tail. Same inputs and return value as
        `align_segment_chunked`.
        '''
        sr = self.resampler_sample_rate
        wav = clip_wav.mean(dim=0) if clip_wav.dim() > 1 else clip_wav
        a = max(0, int(start_sec * sr))
        b = int(end_sec * sr) if (end_sec is not None and end_sec > 0) else wav.shape[0]
        wav = self._rms_normalize(wav[a:min(b, wav.shape[0])])
        res, pe, ge = self.extract_timestamps_from_segment_batch(
            wav.unsqueeze(0), [wav.shape[0]], [list(phoneme_sequence)],
            start_offset_times=[start_sec],
            group_sequences=[list(group_sequence)] if (do_groups and group_sequence is not None) else None,
            extract_embeddings=extract_embeddings, do_groups=do_groups, debug=debug,
            word_sequences=[list(word_sequence)] if word_sequence is not None else None)
        return res[0], pe[0], ge[0]

    @torch.no_grad()
    def align_segment_chunked(self, clip_wav, start_sec, end_sec, phoneme_sequence,
                              group_sequence=None, extract_embeddings=False,
                              do_groups=False, debug=False, word_sequence=None):
        """
        Align one long segment by walking overlapping windows, and return exactly what one
        element of `extract_timestamps_from_segment_batch`'s output looks like - so the
        caller cannot tell which path a segment took.

        The loop is the one described in `chunking.py`: over-provision each window's
        phoneme budget from its speech seconds, align it, find where the surplus starts
        cramming into one frame apiece (corroborated by the confidence collapse), keep
        everything before that, and restart at the end of the last kept phoneme with the
        phonemes that were not kept.

        Args:
            clip_wav: the PARENT clip's waveform, 1-D or (C, T). Sliced here, not by
                `chop_wav` - windows must not be padded out to `duration_max`, which would
                cost several times the compute per window and change the window count.
            start_sec, end_sec: segment bounds within the clip, in seconds.
            phoneme_sequence: target phoneme ids for the whole segment.
            group_sequence: matching group ids, or None to derive them per window.
            extract_embeddings, do_groups, debug: as `extract_timestamps_from_segment_batch`.
            word_sequence: word index per phoneme (phonemizer `word_num`), or None.

        Returns:
            `(result_dict, pooled_phoneme_embeddings, pooled_group_embeddings)` where
            `result_dict` has 'phoneme_timestamps' and 'group_timestamps' as 8-tuples
            `(id, start_frame, end_frame, target_idx, is_estimated, confidence,
            start_ms, end_ms)`. Frames and target indices are SEGMENT-global, not
            window-local, and the ms values carry `start_sec` like the single-shot path.
        """
        sr = self.resampler_sample_rate
        n_total = len(phoneme_sequence)

        wav = clip_wav
        if wav.dim() > 1:
            wav = wav.mean(dim=0)
        seg_a = max(0, int(start_sec * sr))
        seg_b = int(end_sec * sr) if (end_sec is not None and end_sec > 0) else wav.shape[0]
        seg_b = min(seg_b, wav.shape[0])
        wav = wav[seg_a:seg_b]
        # Normalise ONCE over the whole segment rather than per window. CUPE wants
        # RMS-normalised input, but normalising each window separately would hand the same
        # moment of audio a different gain depending on which window it fell in, so the
        # overlap would not agree with itself.
        wav = self._rms_normalize(wav)
        seg_dur = wav.shape[0] / sr

        speech = self.detect_speech_segments(wav) if self.chunk_vad else []
        if debug or self.warn_level >= 2:
            sp = _chunking.speech_seconds(speech, 0.0, seg_dur)
            print(f"Chunked alignment: {seg_dur:.2f}s segment, {n_total} phonemes, "
                  f"VAD speech {sp:.2f}s in {len(speech)} span(s)")

        # Each entry is `(framestamp, pooled_embedding_row_or_None)` - see the comment at
        # the extend below for why the two travel together rather than in parallel lists.
        ph_rows, grp_rows = [], []
        ph_cursor = 0
        t_cursor = 0.0
        # One window always consumes at least one phoneme, so `n_total` iterations is a
        # hard ceiling; the +8 covers the audio-only advances over pure-silence windows.
        for _ in range(n_total + 8):
            if ph_cursor >= n_total or (seg_dur - t_cursor) < self.seg_duration_min:
                break
            remaining = n_total - ph_cursor

            chunk_end = min(t_cursor + self.chunk_seconds, seg_dur)
            # Extend to the end only when what remains could not be aligned AT ALL.
            #
            # This used to collapse whenever the remainder was under a quarter of a window,
            # which measured the wrong quantity and created a dead zone: the next window does
            # not start at `chunk_end`, it starts at the CUT, which is typically 20-25%
            # earlier. So a 12 s segment saw "2 s left, less than 2.5 s, collapse" and became
            # one 12 s window - i.e. the single-shot decode the chunker exists to avoid -
            # when the real next window would have been [~7.5 s, 12 s], a perfectly alignable
            # 4.5 s. Everything between `chunk_seconds` and 1.25x `chunk_seconds` was
            # silently un-chunked.
            #
            # Because the restart is at the cut, a small remainder here never becomes a small
            # window; it is absorbed into a window that starts well before it.
            if (seg_dur - chunk_end) < self.seg_duration_min:
                chunk_end = seg_dur
            audio_exhausted = chunk_end >= seg_dur - 1e-6

            sp_sec = _chunking.speech_seconds(speech, t_cursor, chunk_end)
            if sp_sec < 0.15 and not audio_exhausted:
                # A window with no speech in it. Skip the audio rather than forcing
                # phonemes into it - this is the one place the VAD decides something on its
                # own, and getting it wrong only costs a window of overlap.
                nxt = next((a for a, _ in speech if a > t_cursor), chunk_end)
                skipped_from = t_cursor
                t_cursor = max(min(nxt, chunk_end), t_cursor + 0.25)
                if debug:
                    print(f"  window {skipped_from:6.2f}-{chunk_end:6.2f}s is silent, "
                          f"skipping to {t_cursor:.2f}s")
                continue

            budget = int(math.ceil(self.chunk_phonemes_per_second * sp_sec))
            budget = max(1, min(budget, self.chunk_max_phonemes))
            n = min(remaining, budget)
            if audio_exhausted:
                # Last window: there is no more audio to walk into, so everything left has
                # to go in here whether it fits or not. A cram at this point is the text
                # genuinely outrunning the audio, not a budget artifact.
                n = remaining
            final = (n == remaining) and audio_exhausted

            res, pe, ge, used_n = self._align_chunk(
                wav, t_cursor, chunk_end, start_sec,
                list(phoneme_sequence[ph_cursor:ph_cursor + n]),
                list(group_sequence[ph_cursor:ph_cursor + n]) if (do_groups and group_sequence is not None) else None,
                extract_embeddings=extract_embeddings, do_groups=do_groups, debug=debug,
                words=list(word_sequence[ph_cursor:ph_cursor + n]) if word_sequence is not None else None)
            if res is None:
                # Window could not be aligned at all even after shrinking the budget.
                # Advance the audio and try again rather than aborting the segment.
                t_cursor = min(seg_dur, t_cursor + max(0.5, self.chunk_seconds * 0.25))
                continue
            n = used_n
            final = final and (n == remaining)

            stamps = res['phoneme_timestamps']
            if not stamps:
                # The window decoded but placed nothing (every target fell outside the
                # valid frame range). Advance past it rather than dividing by it below.
                if debug:
                    print(f"  window {t_cursor:6.2f}-{chunk_end:6.2f}s produced no stamps, skipping")
                t_cursor = min(seg_dur, t_cursor + max(0.5, self.chunk_seconds * 0.25))
                continue
            keep, reason = self._chunk_keep_count(stamps, n, final)

            frame_offset = int(round(t_cursor * sr / self.frame_stride_wav))
            keep_mask = [0 <= int(t[3]) < keep for t in stamps]
            kept = [t for t, m in zip(stamps, keep_mask) if m]
            if not kept:
                # Nothing survived the cut. Take the window's first phoneme on trust so the
                # walk cannot stall, and let the coverage/confidence warnings flag it.
                kept = stamps[:1]
                keep = max(1, min(keep, 1))
                keep_mask = [i == 0 for i in range(len(stamps))]

            last_end_ms = max(float(t[7]) for t in kept)
            cut_rel = (last_end_ms / 1000.0) - start_sec

            if (not final) and self.chunk_snap_to_silence > 0:
                # Restarting on a pause gives the next window's first phoneme a clean onset
                # to latch onto. Forward only - moving the cut back would make the next
                # window re-align audio already assigned to a kept phoneme.
                snapped = _chunking.snap_cut_to_silence(
                    cut_rel, speech, seg_dur, self.chunk_snap_to_silence)
                if snapped > cut_rel:
                    cut_rel = snapped

            # Stamps and their pooled embedding rows are carried together, because the
            # final sort below can reorder them: two windows' kept stamps are each in time
            # order, but a snapped or soft-extended boundary at a window edge can still put
            # one window's last stamp after the next window's first. Appending to two
            # parallel lists and sorting only one would silently pair each embedding with a
            # different phoneme.
            keep_ids = [i for i, m in enumerate(keep_mask) if m]
            ph_rows.extend(zip(
                self._reindex_stamps(kept, ph_cursor, frame_offset),
                (pe[keep_ids] if (extract_embeddings and pe is not None) else [None] * len(kept))))

            if do_groups:
                g_all = res.get('group_timestamps') or []
                g_ids = [i for i, t in enumerate(g_all) if 0 <= int(t[3]) < keep]
                g_kept = [g_all[i] for i in g_ids]
                grp_rows.extend(zip(
                    self._reindex_stamps(g_kept, ph_cursor, frame_offset),
                    (ge[g_ids] if (extract_embeddings and ge is not None
                                   and len(ge) == len(g_all)) else [None] * len(g_kept))))

            if debug:
                why = f"derailed at {reason}" if reason else "kept whole window"
                print(f"  window {t_cursor:6.2f}-{chunk_end:6.2f}s  speech {sp_sec:5.2f}s  "
                      f"budget {n:3d}  kept {keep:3d} ({why})  next cut {cut_rel:.2f}s")

            ph_cursor += keep
            # Phoneme progress is guaranteed (keep >= 1), so the loop always terminates;
            # this only keeps the audio cursor from stalling and re-decoding the same
            # window over and over when a cut lands on or before where it started.
            t_cursor = max(cut_rel, t_cursor + 1e-3) if cut_rel > t_cursor else \
                min(seg_dur, t_cursor + max(0.25, self.chunk_seconds * 0.25))
            if final:
                break

        if ph_cursor < n_total and (debug or self.warn_level >= 1):
            self._warn(f"chunked alignment ran out of audio with {n_total - ph_cursor} of "
                       f"{n_total} phonemes unaligned ({seg_dur:.2f}s segment). The text is "
                       f"longer than the audio, or the audio is mostly silence.")

        ph_rows.sort(key=lambda r: (r[0][6], r[0][3]))
        grp_rows.sort(key=lambda r: (r[0][6], r[0][3]))
        ph_stamps = [r[0] for r in ph_rows]
        grp_stamps = [r[0] for r in grp_rows]

        # Attribute the segment's coverage stats ONCE, from the assembled result. The
        # per-window numbers were rolled back by `_chunk_window_scope` because they count
        # each over-provisioned tail as "missed" and each re-aligned phoneme twice.
        covered = {int(t[3]) for t in ph_stamps if int(t[3]) >= 0}
        self.total_phonemes_target += n_total
        self.total_phonemes_aligned += len(ph_stamps)
        self.total_phonemes_aligned_easily += sum(1 for t in ph_stamps if not t[4])
        self.total_phonemes_missed += max(0, n_total - len(covered))
        self.total_phonemes_extra += max(0, len(ph_stamps) - len(covered))

        result = {'phoneme_timestamps': ph_stamps, 'group_timestamps': grp_stamps}
        p_emb = self._stack_rows(ph_rows)
        g_emb = self._stack_rows(grp_rows)
        return result, p_emb, g_emb

    @staticmethod
    def _stack_rows(rows):
        """`[(stamp, emb_row), ...]` -> `[N, D]` tensor, or None if no window produced any."""
        embs = [r[1] for r in rows if r[1] is not None]
        if not embs or len(embs) != len(rows):
            return None
        return torch.stack(embs, dim=0)

    @staticmethod
    def _reindex_stamps(stamps, ph_offset, frame_offset):
        """
        Move one window's framestamps into segment-global coordinates.

        Only the frames and the target index move; the ms values already carry the
        window's `start_offset_times`, which `_align_chunk` sets to the window's absolute
        position, so they are global on arrival.
        """
        out = []
        for (ph, sf, ef, ti, est, conf, sms, ems) in stamps:
            out.append((ph, int(sf) + frame_offset, int(ef) + frame_offset,
                        int(ti) + ph_offset if int(ti) >= 0 else int(ti),
                        est, conf, sms, ems))
        return out

    def _align_chunk(self, wav, t0, t1, seg_start_sec, phonemes, groups,
                     extract_embeddings=False, do_groups=False, debug=False, words=None):
        """
        Align one window. Returns `(result, pooled_p, pooled_g, n_phonemes_used)`, or
        `(None, None, None, 0)` if the window could not be aligned at all.

        The budget is over-provisioned by design, so it can overshoot far enough that the
        CTC lattice no longer fits the window's frames and `decode_with_forced_alignment`
        raises. That is a budget that was too greedy, not a broken window, so it is halved
        and retried rather than propagated.
        """
        sr = self.resampler_sample_rate
        a, b = int(t0 * sr), int(t1 * sr)
        chunk = wav[a:b]
        if chunk.shape[0] < self.seg_duration_min_samples or not phonemes:
            return None, None, None, 0

        n = len(phonemes)
        for _ in range(4):
            try:
                with self._chunk_window_scope():
                    res, pe, ge = self.extract_timestamps_from_segment_batch(
                        chunk.unsqueeze(0), [chunk.shape[0]], [list(phonemes[:n])],
                        start_offset_times=[seg_start_sec + t0],
                        group_sequences=[list(groups[:n])] if (do_groups and groups) else None,
                        extract_embeddings=extract_embeddings, do_groups=do_groups,
                        debug=False,
                        word_sequences=[list(words[:n])] if words else None)
                return res[0], pe[0], ge[0], n
            except ValueError as ex:
                n = n // 2
                if n < 1:
                    if debug or self.warn_level >= 1:
                        self._warn(f"could not align window {t0:.2f}-{t1:.2f}s: {ex}")
                    return None, None, None, 0
                if debug:
                    print(f"  window {t0:.2f}-{t1:.2f}s over-budgeted ({ex}); "
                          f"retrying with {n} phonemes")
        return None, None, None, 0

    def _chunk_keep_count(self, stamps, n_targets, final):
        """
        How many of a window's `n_targets` phonemes to trust.

        `stamps` are the window's aligned 8-tuples, in start-time order, with window-local
        target indices. They are re-keyed by target index here (first occurrence wins;
        `ensure_target_coverage` can insert estimated duplicates) so a missing target shows
        up as `None` and counts as derailed.

        Returns `(keep_count, reason)` where reason is 'duration', 'confidence', 'trim' or
        None.
        """
        durations = [None] * n_targets
        confidences = [None] * n_targets
        for (_ph, sf, ef, ti, _est, conf, _sms, _ems) in stamps:
            ti = int(ti)
            if 0 <= ti < n_targets and durations[ti] is None:
                durations[ti] = int(ef) - int(sf)
                confidences[ti] = float(conf)

        min_keep = max(1, int(MIN_KEEP_FRACTION * n_targets))
        cut, reason = _chunking.find_derailment(
            durations, confidences, min_keep=min_keep,
            derail_frames=self.chunk_derail_frames, derail_run=self.chunk_derail_run,
            conf_floor=self.chunk_conf_floor, weak_run=self.chunk_weak_run,
            backoff=self.chunk_backoff)

        if final:
            # The LAST window keeps everything, derailed or not, and this has to be checked
            # BEFORE the cut. A cut defers phonemes to the next window; on the final window
            # there is no next window, so the same cut DELETES them. That is how the last
            # three phonemes of 30s.wav ("find", at 31.4-31.9 s) went missing: the window was
            # 33 phonemes, `find_derailment` called a confidence cut at 30, and 31-33 had
            # nowhere to go. A badly-placed stamp is strictly better than no stamp - it is
            # flagged by its own confidence, and by `coverage_analysis`, where a dropped
            # phoneme is silent.
            #
            # `reason` is still reported so the debug line can say the window derailed even
            # though nothing was trimmed; a cram here means the text genuinely outruns the
            # audio rather than the budget being too greedy.
            return n_targets, (f'{reason} (final, kept anyway)' if reason else None)
        if cut is not None:
            return max(min_keep, min(cut, n_targets)), reason
        # No cram and audio still to come means the budget was too SMALL for this window:
        # its last phonemes were stretched to fill audio that belongs to the ones that
        # never got in. Drop a short tail so the next window re-decodes them.
        trim = max(2, int(NO_DERAIL_TRIM_FRACTION * n_targets))
        return max(min_keep, n_targets - trim), 'trim'

    def extract_timestamps_from_segment_simplified(self, wavs, wav_lens, phoneme_sequences, start_offset_times=0.0, debug=False):
        '''
        Simplified version of extract_timestamps_from_segment_batch.
        No group timestamps, no embeddings, no target boosting, no silence anchoring.
        Straight CUPE -> log_softmax -> Viterbi -> timestamps.
        To be ported to C++ for real-time applications.
        '''

        # Compute sequence lengths before padding
        if isinstance(phoneme_sequences, torch.Tensor):
            ph_seq_lens = [(seq != self.blank_class).sum().item() for seq in phoneme_sequences]
        else:
            ph_seq_lens = [len(seq) for seq in phoneme_sequences]

        # Pad phoneme sequences to tensor
        if not isinstance(phoneme_sequences, torch.Tensor):
            max_len = max(len(seq) for seq in phoneme_sequences)
            phoneme_sequences = torch.tensor(
                [seq + [self.blank_class] * (max_len - len(seq)) for seq in phoneme_sequences],
                dtype=torch.long,
            )

        # CUPE forward pass (no embeddings)
        logits_class, _, _, spectral_lens, _ = self._cupe_prediction_batch(wavs, wav_lens, extract_embeddings=False)

        # Log softmax
        log_probs_p = F.log_softmax(logits_class, dim=2)

        # Simple Viterbi alignment
        ph_seqs = phoneme_sequences.to(self.device)
        ph_seq_lens_t = torch.tensor(ph_seq_lens, dtype=torch.long, device=self.device)
        spectral_lens_t = torch.tensor(spectral_lens, dtype=torch.long, device=self.device)

        frame_phonemes = self.alignment_utils_p.decode_alignments_simple(
            log_probs_p,
            true_seqs=ph_seqs,
            pred_lens=spectral_lens_t,
            true_seqs_lens=ph_seq_lens_t,
        )

        # Convert frames to millisecond timestamps
        for b in range(len(frame_phonemes)):
            offset = start_offset_times[b] if isinstance(start_offset_times, (list, tuple)) else start_offset_times
            frame_phonemes[b] = convert_to_ms(
                frame_phonemes[b], spectral_lens[b], offset, wav_lens[b], self.resampler_sample_rate,
            )

        timestamp_dicts = [{'phoneme_timestamps': frame_phonemes[b]} for b in range(len(frame_phonemes))]

        return timestamp_dicts



    def _map_phonemes_to_groups(self, phoneme_sequence):
        """Map phoneme indices to phoneme group indices."""
        group_sequence = []
        for ph_idx in phoneme_sequence:
            # Handle both tensor and int inputs
            if isinstance(ph_idx, torch.Tensor):
                ph_idx_val = ph_idx.item()
            else:
                ph_idx_val = int(ph_idx)
            group_idx = self.phoneme_id_to_group_id.get(ph_idx_val, self.blank_group)
            group_sequence.append(group_idx)
        return torch.tensor(group_sequence, dtype=torch.long)
    
    
    def _align_words(self, phoneme_ts, word_num, words_list):
        '''
        phoneme_ts: List of phoneme timestamps [
                {
                    "phoneme_id": int(ph_idx),
                    "phoneme_label": index_to_plabel.get(ph_idx, f"UNK_{ph_idx}"),
                    "start_ms": float(start_ms),
                    "end_ms": float(end_ms),
                    "confidence": float(conf)
                }
        word_num: List[int] e.g., [0,0,0,1,1,1,1,1] where each element corresponds to the word index for the phoneme
        words_list: ["Hello", "world"]
        '''
        #print(len(phoneme_ts), len(word_num), len(words_list))
        #assert len(phoneme_ts) == len(word_num), "Phoneme timestamps and word numbers must match"
        
        if not phoneme_ts or not word_num:
            return []
        
        words_ts = []
        current_word_idx = word_num[0]
        current_word_start = phoneme_ts[0]["start_ms"]
        current_word_phonemes = []

        for i in range(min(len(word_num), len(phoneme_ts))):
            # If we've moved to a new word or reached the end
            if word_num[i] != current_word_idx or i == len(word_num) - 1:
                # If we're at the end and still on the same word, include the current phoneme
                if i == len(word_num) - 1 and word_num[i] == current_word_idx:
                    current_word_phonemes.append(phoneme_ts[i])
                
                # Calculate word-level metrics
                word_end = current_word_phonemes[-1]["end_ms"]
                
                # Calculate average confidence for the word
                total_confidence = sum(ph["confidence"] for ph in current_word_phonemes)
                avg_confidence = total_confidence / len(current_word_phonemes)
                
                # Create word timestamp entry
                word_ts = {
                    "word": words_list[current_word_idx] if current_word_idx < len(words_list) else f"UNK_WORD_{current_word_idx}",
                    "start_ms": current_word_start,
                    "end_ms": word_end,
                    "confidence": avg_confidence,
                    "ph66": [ph["phoneme_id"] for ph in current_word_phonemes],
                    "ipa": [ph["ipa_label"] for ph in current_word_phonemes]
                }
                words_ts.append(word_ts)
                
                # Start tracking the new word (unless we're at the end)
                if i < len(word_num) - 1:
                    current_word_idx = word_num[i]
                    current_word_start = phoneme_ts[i]["start_ms"]
                    current_word_phonemes = [phoneme_ts[i]]
            else:
                # Continue with the same word
                current_word_phonemes.append(phoneme_ts[i])
        
        return words_ts

    def phonemize_sentence(self, text):
        """Phonemize a sentence.
        Args:
            text (str): The input text to phonemize.
        Return
            - segment_out dict with keys:
                - text: original sentence
                - eipa: list of original espeak phonemes in IPA format (one-to-one with mapped phonemes)
                - mipa: list of mapped phonemes in IPA format (after ph66 reduction)
                - ph66: list of phoneme class indices (mapped to phoneme_mapped_index)
                - pg16: list of phoneme group indices (mapped to phoneme_groups_mapper)
                - words: list of words corresponding to the phonemes
                - word_num: list of word indices corresponding to the phonemes
        """
        
        return self.phonemizer.phonemize_sentence(text)


    def post_process_segment(self, segment, ts, phoneme_sequence, phoneme_timestamps, group_timestamps=None, debug=False):


        coverage_analysis = self.analyze_alignment_coverage(
            phoneme_sequence,
            phoneme_timestamps,
            self.phonemizer.index_to_plabel
        )

        if debug:
            print(f"Alignment Coverage Analysis:")
            print(f"  Target phonemes: {coverage_analysis['target_count']}")
            print(f"  Aligned phonemes: {coverage_analysis['aligned_count']}")
            print(f"  Coverage ratio: {coverage_analysis['coverage_ratio']:.2%}")
            if coverage_analysis['missing_phonemes']:
                print(f"  Missing: {coverage_analysis['missing_phonemes']}")
            if coverage_analysis['extra_phonemes']:
                print(f"  Extra: {coverage_analysis['extra_phonemes']}")


        ts_out_segment = segment.copy()
        ts_out_segment["coverage_analysis"] = coverage_analysis
        ts_out_segment["ipa"] = ts.get("eipa", "")
        ts_out_segment["word_num"] = ts.get("word_num", "")
        ts_out_segment["words"] = ts.get("words", "")

        ts_out_segment["phoneme_ts"] = [
            {
                "phoneme_id": int(ph_idx),
                "phoneme_label": self.phonemizer.index_to_plabel.get(ph_idx, f"UNK_{ph_idx}"),
                "ipa_label": ts_out_segment["ipa"][target_seq_idx] if 0 <= target_seq_idx < len(ts_out_segment["ipa"]) else "overflow",
                "start_ms": float(start_ms),
                "end_ms": float(end_ms),
                "confidence": float(avg_confidence),
                "is_estimated": bool(is_estimated),
                "target_seq_idx": int(target_seq_idx),
                "index": idx
            }
            for idx, (ph_idx, start_frame, end_frame, target_seq_idx, is_estimated, avg_confidence, start_ms, end_ms) in enumerate(phoneme_timestamps)
        ]

        if debug:
            print(f"\nSegment: '{segment['text']}'")
            for tsi in range(len(ts_out_segment["phoneme_ts"])):
                if tsi < 20 or tsi >= len(ts_out_segment["phoneme_ts"]) - 20:  # Print first and last 20 phonemes for brevity
                    ts_item = ts_out_segment["phoneme_ts"][tsi]
                    print(f"{tsi}  Phoneme ID: {ts_item['phoneme_id']}, ph66: {ts_item['phoneme_label']}, IPA: {ts_item['ipa_label']}, Start: {int(ts_item['start_ms'])} ms, End: {int(ts_item['end_ms'])} ms, Confidence: {ts_item['confidence']:.3f}")
                elif tsi == 20:
                    print("... (omitting middle phonemes for brevity) ...")

        if group_timestamps is not None:
            ts_out_segment["group_ts"] = [
                {
                    "group_id": int(grp_idx),
                    "group_label": self.phonemizer.index_to_glabel.get(grp_idx, f"UNK_{grp_idx}"),
                    "start_ms": float(start_ms),
                    "end_ms": float(end_ms),
                    "confidence": float(avg_confidence),
                    "is_estimated": bool(is_estimated),
                    "target_seq_idx": int(target_seq_idx),
                    "index": idx
                }
                for idx, (grp_idx, start_frame, end_frame, target_seq_idx, is_estimated, avg_confidence, start_ms, end_ms) in enumerate(group_timestamps)
            ]

        ts_out_segment["words_ts"] = self._align_words(
            ts_out_segment["phoneme_ts"],
            ts.get("word_num", []),
            ts.get("words", [])
        )
        return ts_out_segment
    
    def process_segments(self, srt_data, audio_wavs, extract_embeddings=False, do_groups=False, batch_size=16, debug=False):
        """
        Process multiple audio clips in a batch, each with multiple time-bounded segments.

        Args:
            srt_data: List of dicts (one per clip in the batch), each with "segments" key
                containing a list of {"start", "end", "text"} dicts. Or single dict with "segments".
                Example: [{"segments": [{"start": 0.0, "end": 2.5, "text": "Hello world"}, ...]}, ...]
            audio_wavs: List of audio waveform tensors (one per clip). Each clip's waveform
                is shared across all its segments. Can also be a single (C, T) or batched (B, C, T) tensor.
            extract_embeddings: Whether to extract pooled embeddings.
            do_groups: Whether to extract phoneme group timestamps.
            batch_size: Number of segments to process in a single batch for model inference. Set to 1 if out of memory issues arise.
            debug: Enable debug output.

        Returns:
            List of dicts (one per batch item), each with "segments" key containing processed segments.
            If extract_embeddings=True, returns tuple:
                (batch_results, batch_phoneme_embeddings, batch_group_embeddings)
                where embeddings are nested lists [batch_item][segment] = embedding_tensor.
        """

        # --- Input normalization ---
        if isinstance(audio_wavs, torch.Tensor):
            if audio_wavs.dim() == 3:  # (B, C, T)
                audio_wavs = [audio_wavs[i] for i in range(audio_wavs.size(0))]
            elif audio_wavs.dim() == 2:  # (C, T)
                audio_wavs = [audio_wavs]
            else:
                raise ValueError(f"Expected audio_wavs of 2D (C,T) or 3D (B,C,T), got {audio_wavs.dim()}D")

        if isinstance(srt_data, dict):
            srt_data = [srt_data]

        if len(srt_data) != len(audio_wavs):
            raise ValueError(f"Batch size mismatch: {len(srt_data)} srt items vs {len(audio_wavs)} audio waveforms.")

        # --- Validate structure ---
        for bi, batch_item in enumerate(srt_data):
            if "segments" not in batch_item:
                raise ValueError(f"Batch item {bi} missing 'segments' key. Keys: {list(batch_item.keys())}")
            for si, seg in enumerate(batch_item["segments"]):
                if not all(k in seg for k in ("start", "end", "text")):
                    raise ValueError(f"Batch {bi}, segment {si} missing required keys (start/end/text). Has: {list(seg.keys())}")

        num_batch = len(srt_data)

        # --- Flatten sub-segments across all clips, tracking batch index ---
        flat_items = []  # (batch_idx, segment_dict, clip_audio_wav)
        for bi, (batch_item, clip_wav) in enumerate(zip(srt_data, audio_wavs)):
            for seg in batch_item["segments"]:
                flat_items.append((bi, seg, clip_wav))

        if not flat_items:
            empty = [{"segments": []} for _ in range(num_batch)]
            if extract_embeddings:
                return empty, [[] for _ in range(num_batch)], [[] for _ in range(num_batch)]
            return empty

        # --- Phonemize all sub-segments ---
        ts_outs = [self.phonemize_sentence(seg["text"]) for _, seg, _ in flat_items]
        phoneme_sequences = [ts[self.phonemes_key] for ts in ts_outs]
        group_sequences = [ts[self.phoneme_groups_key] for ts in ts_outs] if do_groups else [None] * len(flat_items)
        word_sequences = [ts.get("word_num") for ts in ts_outs]

        for (_, seg, _), ph_seq, grp_seq in zip(flat_items, phoneme_sequences, group_sequences):
            seg[self.phonemes_key] = ph_seq
            seg[self.phoneme_groups_key] = grp_seq

        # --- Filter sub-segments with insufficient phoneme sequences ---
        valid_indices = []
        for i, ((bi, seg, _), ph_seq) in enumerate(zip(flat_items, phoneme_sequences)):
            if not ph_seq or len(ph_seq) < self.ph_seq_min:
                if debug or self.warn_level >= 1:
                    print(f"Skipping clip {bi}, segment '{seg.get('text', '')[:30]}': insufficient phoneme sequence ({len(ph_seq) if ph_seq else 0})")
                continue
            valid_indices.append(i)

        # Initialize per-batch-item result containers
        batch_results = [{"segments": []} for _ in range(num_batch)]
        batch_p_embds = [[] for _ in range(num_batch)]
        batch_g_embds = [[] for _ in range(num_batch)]

        if not valid_indices:
            if extract_embeddings:
                return batch_results, batch_p_embds, batch_g_embds
            return batch_results

        # Build filtered lists
        flat_items_f = [flat_items[i] for i in valid_indices]
        ph_seqs_f = [phoneme_sequences[i] for i in valid_indices]
        grp_seqs_f = [group_sequences[i] for i in valid_indices]
        word_seqs_f = [word_sequences[i] for i in valid_indices]
        ts_outs_f = [ts_outs[i] for i in valid_indices]

        # --- Chop audio from each sub-segment's parent clip ---
        wavs_and_lens = [
            self.chop_wav(
                clip_wav,
                int(seg["start"] * self.resampler_sample_rate),
                int(seg["end"] * self.resampler_sample_rate)
            )
            for _, seg, clip_wav in flat_items_f
        ]
        wavs, wav_lens, error_codes = zip(*wavs_and_lens)

        # analyze errors and filter out segments with audio issues
        valid_indices = [i for i, code in enumerate(error_codes) if code == 0]
        if len(valid_indices) < len(flat_items_f):
        
            available_wav_duration = clip_wav.size(1) / self.resampler_sample_rate
            
            for i, code in enumerate(error_codes):
                if code != 0:
                    bi, seg, _ = flat_items_f[i]
                    if debug or self.warn_level >= 1:
                        print(f"Skipping clip {bi}, segment start: {seg['start']}, end: {seg['end']} due to chopping error ({code})",)
            # print totals:
            if debug or self.warn_level >= 1:
                print(f"Chopping errors detected in {len(flat_items_f) - len(valid_indices)} segments. Clip total audio duration: {available_wav_duration:.2f} seconds.")
                print(f"Total segments: {len(flat_items_f)}, valid: {len(valid_indices)}, skipped: {len(flat_items_f) - len(valid_indices)}")
                    
            # Filter lists to only include valid segments
            flat_items_f = [flat_items_f[i] for i in valid_indices]
            ph_seqs_f = [ph_seqs_f[i] for i in valid_indices]
            grp_seqs_f = [grp_seqs_f[i] for i in valid_indices]
            word_seqs_f = [word_seqs_f[i] for i in valid_indices]
            ts_outs_f = [ts_outs_f[i] for i in valid_indices]
            wavs = [wavs[i] for i in valid_indices]
            wav_lens = [wav_lens[i] for i in valid_indices]

        # raise error if all wavs are invalid
        if not wavs:
            raise ValueError("All segments have audio chopping errors. Cannot proceed with timestamp extraction.")
        
        wavs = torch.stack(wavs, dim=0)
        start_times = [seg["start"] for _, seg, _ in flat_items_f]

        results = [None] * len(flat_items_f)
        pooled_emb_p_list = [None] * len(flat_items_f)
        pooled_emb_g_list = [None] * len(flat_items_f)

        # --- Route long segments to the adaptive chunker -----------------------------------
        # A long segment is not just a bigger short one. Past roughly `chunk_seconds` of
        # audio or `chunk_max_phonemes` targets, the single-shot Viterbi stops degrading and
        # starts collapsing - on 30s.wav it packed 353 phonemes into the first 18.3s of a 32s
        # clip and emitted the last ~200 at one frame each. Those segments are walked window
        # by window instead (chunking.py); everything shorter takes the batched single-shot
        # path below, unchanged, so short-clip results are bit-identical to before.
        long_ids, whole_ids, short_ids = [], [], []
        for i, ((_bi, seg, _cw), ph_seq) in enumerate(zip(flat_items_f, ph_seqs_f)):
            seg_dur = float(seg["end"]) - float(seg["start"])
            if self._needs_chunking(seg_dur, len(ph_seq)):
                long_ids.append(i)
            elif seg_dur > self.seg_duration_max:
                # Longer than the batch padding: decode on its own rather than be truncated.
                whole_ids.append(i)
            else:
                short_ids.append(i)

        for i in whole_ids:
            bi, seg, clip_wav = flat_items_f[i]
            try:
                res, pe, ge = self.align_segment_whole(
                    clip_wav, float(seg["start"]), float(seg["end"]), ph_seqs_f[i],
                    group_sequence=grp_seqs_f[i] if do_groups else None,
                    extract_embeddings=extract_embeddings, do_groups=do_groups, debug=debug,
                    word_sequence=word_seqs_f[i])
            except ValueError as ex:
                self._warn(f"alignment failed for clip {bi}, segment "
                           f"{seg['start']:.2f}-{seg['end']:.2f}s: {ex}")
                res, pe, ge = {'phoneme_timestamps': [], 'group_timestamps': []}, None, None
            results[i], pooled_emb_p_list[i], pooled_emb_g_list[i] = res, pe, ge

        for i in long_ids:
            bi, seg, clip_wav = flat_items_f[i]
            if debug or self.warn_level >= 2:
                print(f"Clip {bi}: segment {seg['start']:.2f}-{seg['end']:.2f}s with "
                      f"{len(ph_seqs_f[i])} phonemes -> chunked alignment "
                      f"({self.chunk_seconds:g}s windows)")
            try:
                res, pe, ge = self.align_segment_chunked(
                    clip_wav, float(seg["start"]), float(seg["end"]), ph_seqs_f[i],
                    group_sequence=grp_seqs_f[i] if do_groups else None,
                    extract_embeddings=extract_embeddings, do_groups=do_groups, debug=debug,
                    word_sequence=word_seqs_f[i])
            except ValueError as ex:
                self._warn(f"chunked alignment failed for clip {bi}, segment "
                           f"{seg['start']:.2f}-{seg['end']:.2f}s: {ex}")
                res, pe, ge = {'phoneme_timestamps': [], 'group_timestamps': []}, None, None
            results[i], pooled_emb_p_list[i], pooled_emb_g_list[i] = res, pe, ge

        # --- Model inference for the rest (single batched call across all segments) ---
        if short_ids:
            eff_batch = batch_size if batch_size < len(flat_items) else len(short_ids)
            if debug and eff_batch < len(short_ids):
                print(f"Processing {len(short_ids)} segments in batches of {eff_batch}...")
            for i in range(0, len(short_ids), eff_batch):
                ids = short_ids[i:i + eff_batch]
                try:
                    r_slice, p_slice, g_slice = self.extract_timestamps_from_segment_batch(
                        torch.stack([wavs[j] for j in ids], dim=0),
                        [wav_lens[j] for j in ids],
                        [ph_seqs_f[j] for j in ids],
                        start_offset_times=[start_times[j] for j in ids],
                        group_sequences=[grp_seqs_f[j] for j in ids] if do_groups else None,
                        extract_embeddings=extract_embeddings,
                        do_groups=do_groups,
                        debug=debug,
                        word_sequences=[word_seqs_f[j] for j in ids],
                    )
                except ValueError as ex:
                    if debug or self.warn_level >= 1:
                        print(f"ValueError processing segments {ids}: {ex}. This may be due to audio duration too short for the phoneme sequence. Check the segment durations and phoneme counts. Expected at least 2 frames per phone.")
                        # print audio duration and phoneme sequences for each segment in the batch slice
                        for j in ids:
                            bi, seg, _ = flat_items_f[j]
                            ph_seq = ph_seqs_f[j]
                            grp_seq = grp_seqs_f[j] if do_groups else None
                            seg_duration = wav_lens[j] / self.resampler_sample_rate
                            print(f"  Clip {bi}, segment '{seg['text'][:30]}...', duration: {seg_duration:.2f}s, phoneme count: {len(ph_seq)}, group count: {len(grp_seq) if grp_seq is not None else 'N/A'}")

                    # One placeholder PER SEGMENT IN THIS SLICE. The old code emitted
                    # `batch_size` of them regardless, so a short final slice shifted every
                    # later segment's result onto the wrong segment.
                    r_slice = [{'phoneme_timestamps': [], 'group_timestamps': []} for _ in ids]
                    p_slice = [None] * len(ids)
                    g_slice = [None] * len(ids)
                except Exception as e:
                    print(f"Error processing segments {ids}: {e}")
                    raise e

                for k, j in enumerate(ids):
                    results[j] = r_slice[k]
                    pooled_emb_p_list[j] = p_slice[k]
                    pooled_emb_g_list[j] = g_slice[k]

        # --- Post-process and regroup results by batch item ---
        for idx, ((bi, seg, _), result, ts) in enumerate(zip(flat_items_f, results, ts_outs_f)):
            ph_seq = seg[self.phonemes_key]
            processed = self.post_process_segment(
                seg, ts, ph_seq,
                result["phoneme_timestamps"],
                result["group_timestamps"] if do_groups else None,
                debug=debug
            )
            batch_results[bi]["segments"].append(processed)

            if extract_embeddings:
                pooled_emb_p = pooled_emb_p_list[idx]
                if pooled_emb_p is not None:
                    batch_p_embds[bi].append(pooled_emb_p.detach().cpu())
                if do_groups:
                    pooled_emb_g = pooled_emb_g_list[idx]
                    if pooled_emb_g is not None:
                        batch_g_embds[bi].append(pooled_emb_g.detach().cpu())

        # --- Confidence analysis ---
        total_phonemes = 0
        total_confidence = 0.0

        for bi, batch_item in enumerate(batch_results):
            for si, seg_out in enumerate(batch_item["segments"]):
                self.total_segments_processed += 1
                if not seg_out.get("phoneme_ts"):
                    self.total_segments_failed += 1
                    continue

                phoneme_ts = seg_out["phoneme_ts"]
                total_phonemes += len(phoneme_ts)
                total_confidence += sum(ts_item["confidence"] for ts_item in phoneme_ts)

                # Check if sequence matches perfectly
                predicted_sequence = [ts_item["phoneme_id"] for ts_item in phoneme_ts]
                if predicted_sequence == seg_out[self.phonemes_key]:
                    self.perfect_matches += 1
                elif debug:
                    for pidx, (pred_id, target_id) in enumerate(zip(predicted_sequence, seg_out[self.phonemes_key])):
                        if pred_id != target_id:
                            print(f"  Mismatch at position {pidx}: predicted {pred_id} ({self.phonemizer.index_to_plabel.get(pred_id, f'UNK_{pred_id}')}) vs target {target_id} ({self.phonemizer.index_to_plabel.get(target_id, f'UNK_{target_id}')})")
                            break

                if len(phoneme_ts) > 60:
                    confidences = [ts_item["confidence"] for ts_item in phoneme_ts]
                    avg_confidence = sum(confidences) / len(confidences)
                    low_confidence_count = sum(1 for c in confidences if c < 0.5)
                    low_confidence_ratio = low_confidence_count / len(confidences)

                    # `si` counts SURVIVING segments, so it does not index the caller's input
                    # once anything has been skipped (empty phonemization, a chop error).
                    # The time range does identify the segment unambiguously, so both
                    # warnings below carry it - chasing "segment 44" back to an SRT that has
                    # a different segment 44 is a waste of everyone's afternoon.
                    seg_where = (f"Clip {bi}, segment {si+1} "
                                 f"({float(seg_out.get('start', 0)):.2f}-"
                                 f"{float(seg_out.get('end', 0)):.2f}s)")

                    if low_confidence_ratio > self.bad_confidence_threshold:
                        if debug or self.warn_level >= 2:
                            print(f"  WARNING: {seg_where} has too many low-confidence phonemes: {low_confidence_ratio:.2%} ({low_confidence_count}/{len(confidences)}) avg={avg_confidence:.3f}")
                        batch_results[bi]["segments"][si]["coverage_analysis"]["bad_alignment"] = True
                        self.total_segments_bad += 1
                    # The head-vs-tail confidence split. This is the signature of an
                    # alignment that starts on the right audio and then runs away from it:
                    # confidence collapses as the decoder loses the thread, well before the
                    # durations degenerate. `chunking.find_derailment` now uses the same
                    # signal WITHIN a window to decide where to cut, but this check stays -
                    # it is the last line of defence, reporting a segment that came out bad
                    # even after chunking (or with chunking turned off).
                    first_20_confidences = sum(confidences[10:30]) / 20
                    last_20_confidences = sum(confidences[-30:-10]) / 20

                    if first_20_confidences > 0.1 and last_20_confidences < 0.1:
                        seg_dur = float(seg_out.get("end", 0)) - float(seg_out.get("start", 0))
                        n_ph = len(seg_out[self.phonemes_key])
                        was_chunked = self._needs_chunking(seg_dur, n_ph)
                        pps = (n_ph / seg_dur) if seg_dur > 0 else 0.0
                        if not was_chunked and self.silence_anchors == 0 and not self.chunk_audio:
                            raise Exception(f"Bad confidence pattern in {seg_where}: first 20 avg {first_20_confidences:.3f} vs last 20 avg {last_20_confidences:.3f}. Both mitigations are off - enable `chunk_audio=True` (recommended for segments over ~{self.chunk_seconds:g}s) or set `silence_anchors=3`.")

                        # Three distinct situations, and the advice differs for each. The
                        # middle one used to be reported as "enable chunk_audio=True" even
                        # when it already was - the segment simply sat under the threshold.
                        if was_chunked:
                            hint = (f"the chunker's cut points are landing late; try a smaller "
                                    f"`chunk_seconds` (now {self.chunk_seconds:g}s) or a lower "
                                    f"`chunk_phonemes_per_second` (now "
                                    f"{self.chunk_phonemes_per_second:g}) via `set_chunking()`")
                        elif self.chunk_audio:
                            hint = (f"this {seg_dur:.1f}s / {n_ph}-phoneme segment is UNDER the "
                                    f"chunking threshold (chunk_seconds={self.chunk_seconds:g}s, "
                                    f"chunk_max_phonemes={self.chunk_max_phonemes}), so it took "
                                    f"the single-shot path. Chunking is already on - lower the "
                                    f"threshold, e.g. `set_chunking(seconds={max(2.0, seg_dur * 0.6):.0f})`")
                        else:
                            hint = (f"this {seg_dur:.1f}s segment took the single-shot path and "
                                    f"`chunk_audio` is off - enable it")
                        dense = (f" Note {n_ph} phonemes in {seg_dur:.1f}s is {pps:.1f}/s, above "
                                 f"the {self.chunk_phonemes_per_second:g}/s this aligner assumes "
                                 f"speech runs at; if the audio is not unusually fast, check the "
                                 f"transcript for text that is not spoken in this segment."
                                 if pps > self.chunk_phonemes_per_second else "")
                        if debug or self.warn_level >= 1:
                            print(f"  WARNING: Bad confidence pattern in {seg_where}: first 20 avg {first_20_confidences:.3f} vs last 20 avg {last_20_confidences:.3f}. Likely {hint}.{dense}")
                        batch_results[bi]["segments"][si]["coverage_analysis"]["bad_alignment"] = True
                        self.total_segments_bad += 1

        if debug:
            overall_avg_confidence = total_confidence / total_phonemes if total_phonemes > 0 else 0.0
            total_segments = sum(len(item["segments"]) for item in batch_results)
            print(f"\n{'='*60}")
            print(f"PROCESSING SUMMARY")
            print(f"{'='*60}")
            print(f"Clip items: {num_batch}, Segments processed: {total_segments}")
            print(f"Overall average confidence: {overall_avg_confidence:.3f}")
            print(f"Totals, Segments processed: {self.total_segments_processed}", f"\tSegments bad: {self.total_segments_bad}", f"\tPhonemes aligned: {self.total_phonemes_aligned}",
                   f"\tTarget phonemes: {self.total_phonemes_target}", f"\tPhonemes missed: {self.total_phonemes_missed}",
                   f"\tPhonemes extra: {self.total_phonemes_extra}", f"\tPhonemes aligned easily: {self.total_phonemes_aligned_easily}") # self.total_phonemes_aligned_easily counts phonemes that were not enforced in post-processing
            print(f"{'='*60}")

        if extract_embeddings:
            return batch_results, batch_p_embds, batch_g_embds
        return batch_results


    def process_srt_file(self, srt_path, audio_path, ts_out_path=None, extract_embeddings=False, vspt_path=None, do_groups=False, batch_size=16, debug=False):
        """
        Read sentences from SRT file and process them.
        Process entire srt file and generate vs2 output with timestamps.
        
        Args:
            srt_path: Path to input SRT file (JSON format), example: {"segments": [{"start": 0.0, "end": 9.23, "text": "text"}], }
            audio_path: Path to audio file  
            ts_out_path: Path to output vs2 file
            extract_embeddings: Whether to extract embeddings
            vspt_path: Path to save embeddings (.pt file)
            do_groups: Whether to extract phoneme group timestamps
            batch_size: Batch size for processing segments (default: 16). Set to 1 if out of memory issues arise.
            debug: Whether to print debug information
        """
        # Read srt data
        srt_data = None
        try:
            if os.path.exists(srt_path):
                with open(srt_path, "r") as file:
                    srt_data = json.load(file)
                if "segments" not in srt_data:
                    raise ValueError("SRT data does not contain 'segments' key")
                if len(srt_data["segments"]) == 0:
                    print("SRT file is empty or has no segments:", srt_path)
                    return None
                if "text" not in srt_data["segments"][0] or "start" not in srt_data["segments"][0] or "end" not in srt_data["segments"][0]:
                    print("SRT file segments do not contain required phoneme keys:", srt_path)
                    return None
                if debug:
                    print(f"Loaded SRT file with {len(srt_data['segments'])} segments from {srt_path}")
            else:
                print("SRT file does not exist:", srt_path)
                return None
                
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"Error reading SRT file {srt_path}: {e}") from e

        audio_wav = self.load_audio(audio_path)



        srt_batch = [{"segments": srt_data["segments"]}]
        result = self.process_segments(srt_batch, [audio_wav], extract_embeddings=extract_embeddings, do_groups=do_groups, batch_size=batch_size, debug=debug)

        if extract_embeddings:
            batch_results, batch_p_embds, batch_g_embds = result
            vs_out_data = batch_results[0]
        else:
            vs_out_data = result[0]

        if ts_out_path is not None:
            if os.path.dirname(ts_out_path):
                os.makedirs(os.path.dirname(ts_out_path), exist_ok=True)
            with open(ts_out_path, 'w') as f:
                json.dump(vs_out_data, f, indent=4 if debug else None, ensure_ascii=False)
            if extract_embeddings and vspt_path is not None and batch_p_embds:
                torch.save((batch_p_embds[0], batch_g_embds[0]), vspt_path)
                if debug:
                    print(f"Embeddings saved to: {vspt_path}")
            if debug:
                print(f"Results saved to: {ts_out_path}")

        return vs_out_data
    
    def process_sentence(self, text, audio_wav, extract_embeddings=False, do_groups=False, debug=False):
        """
        Process a single text sentence along with it's audio waveform, returning an output dict with timestamps.

        Args:
            text: Text (str).
            audio_wav: Audio waveform tensor (torch.Tensor).
            ts_out_path: Path to output vs2 file (optional).
            extract_embeddings: Whether to extract embeddings (bool, optional).
            do_groups: Whether to extract group timestamps (bool, optional).
            debug: Whether to print debug information (bool, optional).


        Returns:
            If extract_embeddings is False:
                vs2 output dict for the sentence.
            If extract_embeddings is True:
                Tuple of (timestamps_output_dict, phoneme_embeddings, group_embeddings)
                where timestamps_output_dict is the output dict,
                phoneme_embeddings and group_embeddings are embedding tensors.
        """

        duration = audio_wav.shape[1] / self.sample_rate
        srt_data = [ {"segments": [{"start": 0.0, "end": duration, "text": text.strip()}]} ]  # create whisper style SRT data

        result = self.process_segments(srt_data, [audio_wav], extract_embeddings=extract_embeddings, do_groups=do_groups, debug=debug)

        # Unwrap single batch item
        if extract_embeddings:
            timestamps_output_dict, batch_p_embds, batch_g_embds = result
            return timestamps_output_dict[0], batch_p_embds[0], batch_g_embds[0]
        return result[0]

    def process_sentences_batch(self, texts, audio_wavs, extract_embeddings=False, do_groups=False, debug=False):
        """
        Process a batch of sentences and audio waveforms, generating vs2 output with timestamps.

        Args:
            texts: List of text strings.
            audio_wavs: List of audio waveform tensors (torch.Tensor).
            extract_embeddings: Whether to extract embeddings (bool, optional).
            do_groups: Whether to extract group timestamps (bool, optional).
            debug: Whether to print debug information (bool, optional).

        Returns:
            If extract_embeddings is False:
                List of timestamps_output_dicts, one per sentence.
            If extract_embeddings is True:
                Tuple of (timestamps_list, phoneme_embeddings, group_embeddings)
                where timestamps_list is a list of timestamps_output_dicts,
                phoneme_embeddings and group_embeddings are lists of embedding tensors.
        """
        assert len(texts) == len(audio_wavs), f"Number of texts ({len(texts)}) must match number of audio waveforms ({len(audio_wavs)})"
        srt_data = []
        for text, audio_wav in zip(texts, audio_wavs):
            duration = audio_wav.shape[1] / self.sample_rate
            srt_data.append({"segments": [{"start": 0.0, "end": duration, "text": text.strip()}]})

        result = self.process_segments(srt_data, audio_wavs, extract_embeddings=extract_embeddings, do_groups=do_groups, debug=debug)

        if extract_embeddings:
            timestamps_output_dicts, batch_p_embds, batch_g_embds = result
            return timestamps_output_dicts, batch_p_embds, batch_g_embds
        return result



    def convert_to_textgrid(self, timestamps_dict, output_file=None, include_confidence=False):
        """
        Convert VS2 data to TextGrid format.
        """

        textgrid_content = dict_to_textgrid(timestamps_dict, output_file=None, include_confidence=include_confidence)

        if output_file and textgrid_content is not None:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(textgrid_content)

        return textgrid_content

    def plot_mel_alignment(self, timestamps_dict, audio_wav, save_path,
                           segment_idx=0, vocoder_config=None):
        """
        Generate and save a mel spectrogram with phoneme boundary markers.

        Combines mel extraction, frame-wise assortment, and plotting in one call.

        Args:
            timestamps_dict (dict): Output from process_sentence / process_srt_file,
                containing a ``"segments"`` key.
            audio_wav (Tensor): Waveform tensor of shape (1, T) at the aligner's
                internal sample rate (``self.resampler_sample_rate``).
            save_path (str): Destination path for the image (.png) or tensor (.pt).
                When the extension is ``.pt`` the raw mel tensor is saved instead
                of a plot.
            segment_idx (int): Which segment to plot (default 0).
            vocoder_config (dict | None): Vocoder config forwarded to
                ``extract_mel_spectrum``.  Defaults to the BigVGAN 22 kHz preset.

        Returns:
            str: The path the file was saved to.
        """
        import os, torch

        if vocoder_config is None:
            vocoder_config = {
                'num_mels': 80, 'num_freq': 1025, 'n_fft': 1024,
                'hop_size': 256, 'win_size': 1024, 'sampling_rate': 22050,
                'fmin': 0, 'fmax': 8000,
                'model': 'nvidia/bigvgan_v2_22khz_80band_fmax8k_256x'
            }

        segments = timestamps_dict.get('segments', [])
        if not segments:
            raise ValueError("timestamps_dict has no segments")

        segment = segments[min(segment_idx, len(segments) - 1)]
        seg_start = segment.get('start', 0.0)
        seg_end   = segment.get('end', audio_wav.shape[1] / self.resampler_sample_rate)
        phoneme_ts = segment.get('phoneme_ts', [])

        # Crop audio to this segment
        s = int(seg_start * self.resampler_sample_rate)
        e = int(seg_end   * self.resampler_sample_rate)
        segment_wav = audio_wav[:, s:e]

        mel = self.extract_mel_spectrum(segment_wav, self.resampler_sample_rate, vocoder_config)

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        if save_path.lower().endswith('.pt'):
            torch.save(mel, save_path)
            return save_path

        # Build a title from the segment text if available
        text_snippet = segment.get('text', '')
        title = f'"{text_snippet}"' if text_snippet else None

        _plot_mel_alignment(
            mel=mel,
            phoneme_ts=phoneme_ts,
            save_path=save_path,
            segment_start_ms=seg_start * 1000.0,
            title=title,
        )
        return save_path

    # Additional helper function for analysis
    def analyze_alignment_coverage(self, target_sequence, aligned_timestamps, index_to_label):
        """
        Analyze how well the alignment covers the target sequence.
        
        Args:
            target_sequence: Original target phoneme sequence
            aligned_timestamps: List of aligned phoneme timestamps
            index_to_label: Mapping from indices to phoneme labels
            
        Returns:
            dict: Analysis results
        """
        target_set = set(target_sequence.tolist() if hasattr(target_sequence, 'tolist') else target_sequence)
        aligned_set = set([ts[0] for ts in aligned_timestamps])
        
        missing = target_set - aligned_set
        extra = aligned_set - target_set
        
        coverage = len(target_set - missing) / len(target_set) if target_set else 1.0
        
        analysis = {
            'target_count': len(target_set),
            'aligned_count': len(aligned_set),
            'missing_count': len(missing),
            'extra_count': len(extra),
            'coverage_ratio': coverage,
            'missing_phonemes': [index_to_label.get(p, f'UNK_{p}') for p in missing],
            'extra_phonemes': [index_to_label.get(p, f'UNK_{p}') for p in extra],
            'bad_alignment': coverage < 0.8  # arbitrary threshold for flagging bad alignments
        }
        
        return analysis


    def extract_mel_spectrum(self, wav, wav_sample_rate, vocoder_config={'num_mels': 80, 'num_freq': 1025, 'n_fft': 1024, 'hop_size': 256, 'win_size': 1024, 'sampling_rate': 22050, 'fmin': 0, 'fmax': 8000, 'model': 'nvidia/bigvgan_v2_22khz_80band_fmax8k_256x'}):
        '''
        Args:
            wav: Input waveform tensor of shape (1, T)
            wav_sample_rate: Sample rate of the input waveform
            vocoder_config: Configuration dictionary for HiFiGAN/BigVGAN vocoder. Use the same config to generate mel-spectrum as bigvan vocoder model, so that the mel-spectrum can be converted back to audio easily.

        Returns:
            mel: Mel spectrogram tensor of shape (num_mels, T)

        Ensure compatability with HiFiGAN/BigVGAN vocoder https://github.com/NVIDIA/BigVGAN
        '''
        from .helpers.mel_spec import mel_spectrogram

        assert (wav.dim() == 2) and (wav.shape[0] == 1), f"Expected input shape (1, T), got {wav.shape}"

        if wav_sample_rate != vocoder_config['sampling_rate']:
            print ("extract_mel_spectrum: Resampling waveform from {} to {} for vocoder_config compatibility".format(wav_sample_rate, vocoder_config['sampling_rate']))
            # use librosa resampling:
            import librosa
            wav = librosa.resample(wav.numpy(), orig_sr=wav_sample_rate, target_sr=vocoder_config['sampling_rate'])
            wav = torch.from_numpy(wav)

        mel = mel_spectrogram(
            wav,
            n_fft=vocoder_config['n_fft'],
            num_mels=vocoder_config['num_mels'],
            sampling_rate=vocoder_config['sampling_rate'],
            hop_size=vocoder_config['hop_size'],
            win_size=vocoder_config['win_size'],
            fmin=vocoder_config['fmin'],
            fmax=vocoder_config['fmax']
        )

        if mel.dim() == 3 and mel.shape[0] == 1 and mel.shape[1] == vocoder_config['num_mels']:
            mel = mel.squeeze(0)
        else:
            raise ValueError(f"Unexpected mel shape: {mel.shape}, expected [1, mel_dim, mel_len] or [mel_dim, mel_len]")
        
        mel = mel.transpose(0, 1) # (T, mel_dim)

        return mel
        


    def ceil(self, float_value): 
        return int(float_value) + (float_value % 1 > 0)

    def compress_frames(self, frames_list):
        '''
        Given frames list as [0,0,0,0,1,1,1,1,3,4,5,4,5,2,2,2], return compressed list as [(0,4),(1,4),(3,1),(4,1),(5,1),(4,1),(5,1),(2,3)] where each tuple represents (frame_value, frame_count)
        '''
        if not frames_list:
            return []

        compressed = []
        current_value = frames_list[0]
        current_count = 1

        for i in range(1, len(frames_list)):
            if frames_list[i] == current_value:
                current_count += 1
            else:
                compressed.append((current_value, current_count))
                current_value = frames_list[i]
                current_count = 1

        compressed.append((current_value, current_count))
        return compressed

    
    def decompress_frames(self, compressed_frames):
        """Decompress phoneme frames from compressed format"""
        decompressed = []
        for phn_id, count in compressed_frames:
            decompressed.extend([phn_id] * count)
        return decompressed

    def framewise_assortment(self, aligned_ts, total_frames, frames_per_second, gap_contraction=5, select_key="phoneme_id", offset_ms=0):
        """
        Perform frame-wise assortment of aligned timestamps.
        Args:
            aligned_ts: Dictionary containing segment-level timestamp information. It can be either "phoneme_ts" or "word_ts", "group_ts".
            total_frames: Total number of frames in the mel spectrogram.
            frames_per_second: Frame rate of the mel spectrogram.
            gap_contraction: extra gaps (in frames) to fill during the assortment process. Compensate for silence overwhelment on either side of unvoiced segments.
            select_key: Key to select timestamps from the aligned_ts dictionary. Default is "phoneme_id". It can be set to "ipa_label" to use IPA labels instead of phoneme IDs, or "group_id" to use group IDs.
            offset_ms: Offset in milliseconds to adjust the start and end times of each timestamp. Pass `timestamps["segments"][segment_id]["start"]*1000` to align with the original audio.
        """
        '''
        
        where <aligned_ts> is expected to be in the following format:
        "phoneme_ts": [{"phoneme_id": 4, "phoneme_label": "ɪ", "start_ms": 33.03478240966797, "end_ms": 49.55217361450195, "confidence": 0.9967153072357178}, {"phoneme_id": 53, "phoneme_label": "n", "start_ms": 49.55217361450195, "end_ms": 82.58695983886719, "confidence": 0.7659286260604858}, {"phoneme_id": 29, "phoneme_label": "b", "start_ms": 181.69129943847656, "end_ms": 198.2086944580078, "confidence": 0.978035569190979}, ...], 
        "group_ts": [{"group_id": 1, "group_label": "front_vowels", "start_ms": 33.03478240966797, "end_ms": 49.55217361450195, "confidence": 0.9991531372070312}, {"group_id": 12, "group_label": "nasals", "start_ms": 49.55217361450195, "end_ms": 82.58695983886719, "confidence": 0.7814053893089294},  ... ], 
        "words_ts": [{"word": "in", "start_ms": 33.03478240966797, "end_ms": 82.58695983886719, "confidence": 0.8813219666481018, "ph66": [4, 53], "ipa": ["ɪ", "n"]}, {"word": "being", "start_ms": 181.69129943847656, "end_ms": 396.4173889160156, "confidence": 0.6953825578093529, "ph66": [29, 2, 4, 55], "ipa": ["b", "i:", "ɪ", "ŋ"]}, {"word": "comparatively", "start_ms": 429.4521789550781, "end_ms": 1271.839111328125, "confidence": 0.6755372906724612, "ph66": [32, 8, 52, 28, 20, 59, 8, 30, 4, 44, 56, 1], "ipa": ["k", "ə", "m", "p", "æ", "ɹ", "ə", "t", "ɪ", "v", "l", "i"]}, {"word": "modern", "start_ms": 1354.4261474609375, "end_ms": 1783.8782958984375, "confidence": 0.4999569676816463, "ph66": [52, 19, 31, 9, 53], "ipa": ["m", "a:", "d", "ɚ", "n"]}`
        
        and `select_ts` and `select_key` are used to specify which timestamps to align.
        '''
        
        gap_fill_frame_value = -1  # sentinel: must not collide with any valid id (SIL=0)

        ms_per_frame = 1000.0 / frames_per_second

        # first sort by start_ms
        aligned_ts.sort(key=lambda x: x["start_ms"])

        # fill with gap frames
        framewise_label = [gap_fill_frame_value] * total_frames

        # Stage 1: Fill frames using exact timestamp boundaries (no tolerance)
        for i, ts_item in enumerate(aligned_ts):
            # Calculate exact frame boundaries without any gap tolerance
            framewise_start_index = max(int((ts_item["start_ms"]-offset_ms) / ms_per_frame), 0)
            framewise_end_index = min(self.ceil((ts_item["end_ms"]-offset_ms) / ms_per_frame), total_frames)

            if framewise_end_index-framewise_start_index > total_frames:
                if self.warn_level >= 2:
                    print(f"Warning: Timestamp start frame {framewise_start_index} exceeds total frames {total_frames}, skipping.")
                continue
            elif framewise_end_index > total_frames:
                if self.warn_level >= 2:
                    print(f"Warning: Timestamp ending frame {framewise_end_index} exceeds total frames {total_frames}, adjusting end index.")
                framewise_end_index = total_frames

            # Simple assignment - fill the exact range
            if select_key not in ts_item:
                raise ValueError(f"select_key '{select_key}' not found in timestamp item", ts_item)
            for frame_i in range(max(framewise_start_index-1, 0), min(framewise_end_index+1, total_frames)):
                if framewise_label[frame_i] == gap_fill_frame_value:
                    framewise_label[frame_i] = ts_item[select_key]

        # Stage 2: Contract silent gaps

        i = 0
        gaps = []
        while i < total_frames:
            if framewise_label[i] == gap_fill_frame_value:
                gap_start = i
                while i < total_frames and framewise_label[i] == gap_fill_frame_value:
                    i += 1
                gap_end = i
                gaps.append((gap_start, gap_end))
            else:
                i += 1


        # Contract gaps from both sides using the frame values from either side:
        for gap_start, gap_end in gaps:
            if gap_start > 0 and gap_end <= total_frames:
                left_value = framewise_label[gap_start - 1]
                if gap_end < total_frames:
                    right_value = framewise_label[gap_end]
                else:
                    # For end-of-sequence gaps, use the left phoneme to fill
                    right_value = gap_fill_frame_value
                if left_value != gap_fill_frame_value and (right_value != gap_fill_frame_value or gap_end == total_frames):
                    gap_size = gap_end - gap_start

                    # Only fill gaps up to gap_contraction * 2 in size

                    if gap_size <= gap_contraction:
                        # For small gaps, fill entire gap with left value
                        for j in range(gap_start, gap_end):
                            framewise_label[j] = left_value
                    else:

                        if gap_size <= gap_contraction * 2:
                            # For medium gaps, fill equally from both sides towards the center
                            midpoint = gap_start + gap_size // 2
                            for j in range(gap_start, gap_end):
                                if j < midpoint:
                                    framewise_label[j] = left_value
                                else:
                                    framewise_label[j] = right_value
                        else:
                            # contract gaps from eitherside +- gap_contraction
                            # first left side
                            for j in range(gap_start, min(gap_start + gap_contraction, gap_end)):
                                framewise_label[j] = left_value
                            # then right side
                            for j in range(gap_end - 1, max(gap_end - gap_contraction - 1, gap_start - 1), -1):
                                framewise_label[j] = right_value


        return framewise_label





if __name__ == "__main__":
    torch.random.manual_seed(42)
    
