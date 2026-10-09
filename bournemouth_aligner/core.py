'''
Bournemouth Forced Aligner - phoneme-level timestamps from audio and text.

Text is phonemized, audio goes through a CNN phoneme model (120 ms context-free windows, one
posterior per 5 ms frame), and the two are matched by the segmental Viterbi the model was
trained against. Inputs are a sentence or an SRT-style segment list; outputs are per-phoneme,
per-group and per-word timestamps with confidences, optionally as a Praat TextGrid or with
pooled embeddings.

## Two phoneme sets, one aligner

    phoneme_set="std"   (default)  standard_g2p phonemes, decoded by a p4mbfa checkpoint
    phoneme_set="ph66"             espeak-ng + ph66,      decoded by a p3cupe checkpoint

A phoneme set is a phonemizer and the label space its ids live in (`phonemizers/`); a model
family is a checkpoint format whose head emits exactly that label space (`p3cupe/`,
`p4mbfa/`). The pairing is fixed, so naming either one settles the other, and a checkpoint
path settles both. Everything below this line - windowing, decoding, chunking, the output
format - is the same code for the two.

The one structural difference: p3cupe has a single label space for every language, while a
p4mbfa checkpoint belongs to one LANGUAGE GROUP (latin, cyrillic, cjk, ...). The language picks
the group and the group picks the checkpoint (`presets.py`).

## What places a boundary

The model's posteriors ARE the boundary detector. Each phone is trained to peak at 1.0 across its
own segment and fall to exactly 0.5 at each of its two boundaries, where the neighbour rises
through 0.5, so a boundary is the CROSSING of two posteriors - a continuous quantity, read off
by interpolation rather than snapped to a frame. There is no blank class, no CTC lattice and no
post-hoc onset snapping; `segmental/aligner.py` places every onset inside one global search.

Every timestamp comes from `time_ms(frame f) = f * 5.0`, exactly: the window stride advances a
whole number of output frames, so the stitched grid is the model's own conv grid with no
correction term. See `segmental/windowing.py`.
'''

import contextlib
import json
import math
import os
import traceback

import torch
import torchaudio
import torch.nn.functional as F

from . import chunking as _chunking
from .chunking import (DEFAULT_CHUNK_SECONDS, DEFAULT_MAX_PHONEMES,
                       DEFAULT_PHONEMES_PER_SECOND, DEFAULT_DERAIL_FRAMES,
                       DEFAULT_DERAIL_RUN, DEFAULT_BACKOFF, DEFAULT_CONF_FLOOR,
                       DEFAULT_WEAK_RUN, DEFAULT_SNAP_TO_SILENCE_SEC,
                       MIN_KEEP_FRACTION, NO_DERAIL_TRIM_FRACTION,
                       TRUST_WINDOW_FRACTION)
from . import hub
from . import presets as _presets
from .extractors import load_extractor
from .phonemizers import make_phonemizer
from .segmental import align_batch
from .utils import (dict_to_textgrid, weighted_pool_embeddings, _calculate_confidences,
                    convert_to_ms, detect_device_automatically,
                    plot_mel_alignment as _plot_mel_alignment)

# Segments longer than this are handed to the chunker instead of being decoded in one pass.
# The decoder itself does not degrade on long audio - it has no band constraint and nothing to
# collapse - so this is purely a time and memory bound: the DP is a python loop over frames at
# 200 frames per second of audio, which measures ~1.8 s for a 32 s clip of 356 phonemes.
DEFAULT_SEGMENT_MAX_SECONDS = 120.0


class PhonemeTimestampAligner:
    """
    Align phoneme-level timestamps from audio and text.
    URL: https://github.com/tabahi/bournemouth-forced-aligner
    """

    def __init__(self, preset="en-us", model_name=None, cupe_ckpt_path=None, lang='en-us',
                 mapper=None, duration_max=30, device="auto", boundary_bonus_weight=None,
                 sub_frame_onsets=True, min_phone_frames=None, soft_phone_frames=None,
                 bad_confidence_threshold=0.6, low_confidence_level=0.25, chunk_audio=True,
                 chunk_seconds=DEFAULT_CHUNK_SECONDS, chunk_max_phonemes=DEFAULT_MAX_PHONEMES,
                 chunk_phonemes_per_second=DEFAULT_PHONEMES_PER_SECOND, chunk_vad=True,
                 segment_max_seconds=DEFAULT_SEGMENT_MAX_SECONDS, warn_level=1,
                 phoneme_set=None, ckpt_path=None, g2p_model_dir=None, g2p_device=None,
                 g2p_cache_size=None, punctuation_sil=False):
        """
        Initialize the BFA phoneme timestamp extractor.

        Args:
            preset (str or None): Language preset, e.g. "en-us", "de", "ru", "zh". Selects the
                language and, through it, the checkpoint. `None` loads no model, and
                `load_model()` must be called before aligning anything.
            model_name (str, optional): A published model, overriding the preset's choice:
                "mbfa-latin", "mbfa-latin-benchmark", "mbfa-cyrillic", "mbfa-cjk",
                "mbfa-japanese" (std) or "CUPE-3-en" (ph66), or any Hugging Face repository
                id in the same layout. Downloaded on first use (`hub.resolve_model`).
            cupe_ckpt_path (str, optional): Older name of `ckpt_path`.
            lang (str): Language code for phonemization. Only overridden by `preset` while it
                is still the default. For the std phoneme set any BCP 47 code standard_g2p
                covers, in any case ("en", "en-GB", "pt-br", "zh-Hant", "ja"); for ph66 an
                espeak-ng code, see
                https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md
            mapper (str, optional): Deprecated name of `phoneme_set`.
            phoneme_set (str, optional): "std" (default) or "ph66". "std" phonemizes with
                standard_g2p and decodes with the p4mbfa checkpoint of the language's group;
                "ph66" phonemizes with espeak-ng and decodes with a p3cupe checkpoint. Left
                unset with a `ckpt_path` or `model_name`, it is whatever that checkpoint is.
                A language the std set cannot serve raises rather than falling back, because
                the two sets' ids and output keys differ.
            ckpt_path (str, optional): Direct path to a model of either family on disk: a
                release directory (config.json + model.safetensors) or a training checkpoint
                file. Highest priority - overrides both `preset` and `model_name` - and
                decides the phoneme set by itself. A p4mbfa model also fixes the language
                group, so `lang` must be a language of that group.
            g2p_model_dir (str, optional): Directory already holding the std set's G2P model.
                Default: $BFA_G2P_DIR, else charsiu/g2p_multilingual_byT5_small_100 is
                downloaded from the Hugging Face Hub on first use (~1.2 GB, once).
            g2p_device (str, optional): Device for the G2P model. Default: `device`.
            g2p_cache_size (int, optional): Word -> IPA entries the G2P remembers. Default
                500k (~55 MB); 0 for no limit.
            punctuation_sil (bool): std only. Write a silence where the text has pause
                punctuation, as ph66 always does. Off by default because the p4mbfa models
                were trained without one; the aligner may skip any silence at no cost, so
                turning it on only offers the path somewhere to put a pause.
            duration_max (float): Padding width, in seconds, for the batched short-segment
                path. Segments longer than this are decoded on their own, unpadded, so this
                trades memory against how many segments share one forward pass.
            device (str): "cpu", "cuda", "mps", or "auto" to detect.
            boundary_bonus_weight (float or None): Weight, in nats per standard deviation, on
                the log-mel acoustic-change bonus added to every phone-onset transition inside
                the Viterbi. `None` (default) uses the value the checkpoint was TRAINED with,
                which is 30 for the shipped one; 0 turns the bonus off.

                Worth knowing before changing it, because the evidence points both ways.
                Against: scored per phone ONSET on TIMIT the bonus costs about 7 f1@25,
                because it drags stop onsets onto the burst and pulls vowel onsets early
                (`model3p/memory.md`, decoder sweep on the English pa01d checkpoint: 68.0 at
                weight 30 against 74.6 at weight 0). For: measured here on Buckeye with the
                unpublished MLS checkpoint (pa07a), 40 segments x 3 seeds, it is the better
                setting on every one -

                    f1@25   seed 0  60.2 -> 64.6   seed 1  59.8 -> 61.5   seed 2  60.8 -> 63.2
                    f1@10   seed 0  36.2 -> 43.8   seed 1  34.5 -> 40.1   seed 2  39.1 -> 45.1

                and the median absolute onset error falls from 12-14 ms to 9-11 ms. The
                difference is most likely the checkpoint: this one was trained on MLS without
                English and its English posteriors are weak, so the aligner leans on acoustic
                change for placement more than a strong English model would. Re-measure it
                when an English p3cupe checkpoint exists.
            sub_frame_onsets (bool): Place each onset at the interpolated crossing of the two
                neighbouring posteriors instead of on the 5 ms frame grid. This is the readout
                the loss was shaped for; `False` gives plain frame indices.
            min_phone_frames (int or None): Shortest phone the aligner may produce, in 5 ms
                frames. `None` (default) takes the checkpoint's own training value, which is
                what its targets were built with. A clip whose text will not fit at this floor
                gets a shorter one automatically rather than losing its alignment.
            soft_phone_frames (int or None): Frames above `min_phone_frames` that a phone may
                give up, at a cost each - the soft half of the duration prior. `None` takes
                the checkpoint's value.
            bad_confidence_threshold (float): Share of low-confidence phonemes above which a
                segment is flagged (`coverage_analysis["bad_alignment"]`). 1 disables it.
            low_confidence_level (float): Confidence below which a phoneme counts towards that
                share. **Not comparable with BFA <= 1.2's 0.5**, which was set against CTC
                posteriors that sit near 1.0 inside a phone. p3cupe's loss deliberately does
                not saturate, so its confidences are lower throughout: measured on the MLS
                checkpoint (pa07a), the mean peak posterior of a correctly placed phone is 0.33 on
                English read speech. This default is calibrated against that one checkpoint and
                should be re-checked when an English-trained p3cupe checkpoint exists.
            chunk_audio (bool): Align segments longer than `segment_max_seconds` by walking
                overlapping windows instead of decoding them in one pass. See `chunking.py`
                and `set_chunking()`. `False` always decodes whole, whatever the length.
            chunk_seconds (float): Audio window length for that walk, in seconds.
            chunk_max_phonemes (int): Cap on one window's phoneme budget.
            chunk_phonemes_per_second (float): Phonemes per second of SPEECH used to size each
                window's budget. Measured against speech seconds, which is what `chunk_vad` is
                for.
            chunk_vad (bool): Use an energy VAD to measure speech seconds per window and to
                move a cut into a nearby pause.
            segment_max_seconds (float): Segments up to this long are decoded whole. Whole
                decoding is more accurate than chunking - chunk cuts can land mid-phone - so
                raise it if the hardware allows; the cost is a python-loop DP that runs at
                roughly 1.8 s per 32 s of audio.
            warn_level (int): 0 = silent, 1 = warnings only (default), 2 = warnings plus
                informational output (resolved geometry, decoder settings, coverage details).
                Errors are raised as exceptions regardless.

        Parameter priority (highest to lowest): `ckpt_path`, `model_name`, `preset`.

        Examples:
            aligner = PhonemeTimestampAligner(preset="de")
            aligner = PhonemeTimestampAligner(preset="de", phoneme_set="ph66")
            aligner = PhonemeTimestampAligner(model_name="mbfa-latin-benchmark")
            aligner = PhonemeTimestampAligner(ckpt_path="path/to/mbfa-cjk", lang="zh")
        """
        # Set first, so load_model and the geometry resolution can report through _info/_warn.
        self.warn_level = int(warn_level)
        if device == "auto":
            device = detect_device_automatically()
        self.device = device
        self.extractor = None                   # set by load_model()
        self.lang = lang
        # True until an explicit language is given: only then may a preset choose it.
        self._lang_is_default = (not lang) or lang == _presets.DEFAULT_LANG
        self.resampler_sample_rate = 16000
        self.sample_rate = 16000                # both model families are trained at 16 kHz
        self.ph_seq_min = 1

        self.seg_duration_min = 0.05            # seconds
        self.seg_duration_min_samples = int(self.seg_duration_min * self.resampler_sample_rate)
        self.seg_duration_max = duration_max
        self.wav_len_max = int(self.seg_duration_max * self.resampler_sample_rate)

        # --- phoneme set ------------------------------------------------------------------
        # What the caller ASKED for, or None. The set actually in use (`self.phoneme_set`) is
        # settled by `load_model`, because a checkpoint path decides it on its own.
        self._requested_set = (None if phoneme_set is None and mapper is None
                               else _presets.resolve_phoneme_set(phoneme_set, mapper))
        self._g2p_options = dict(device=g2p_device or device, model_dir=g2p_model_dir,
                                 cache_size=g2p_cache_size, punctuation_sil=punctuation_sil)
        self.phoneme_set = None
        self.phonemizer = None
        self.inventory = None

        # --- decoder settings -------------------------------------------------------------
        # `min_phone_frames` / `soft_phone_frames` default to the checkpoint's own training
        # values: the posteriors were shaped by targets built with that topology, so decoding
        # with a different one decodes a model that was never trained.
        self.sub_frame_onsets = bool(sub_frame_onsets)
        self._bonus_weight_override = boundary_bonus_weight
        self._min_phone_frames_override = min_phone_frames
        self._soft_phone_frames_override = soft_phone_frames
        self.boundary_bonus_weight = float(boundary_bonus_weight or 0.0)
        self.bad_confidence_threshold = bad_confidence_threshold
        self.low_confidence_level = float(low_confidence_level)
        self.segment_max_seconds = float(segment_max_seconds)

        # --- chunking (chunking.py) -------------------------------------------------------
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

        ckpt_path = ckpt_path or cupe_ckpt_path
        if preset is not None or model_name is not None or ckpt_path is not None:
            self.load_model(preset=preset, model_name=model_name, ckpt_path=ckpt_path)
        else:
            # No model yet: phonemization still works, in the set that was asked for.
            phoneme_set = self._requested_set or _presets.DEFAULT_PHONEME_SET
            self._set_phonemizer(phoneme_set, self._resolve_lang(phoneme_set, self.lang))

        self.default_resampler = torchaudio.transforms.Resample(
            orig_freq=self.resampler_sample_rate,
            new_freq=self.resampler_sample_rate,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )
        self.reset_counters()

    # ======================================================================================
    # Setup
    # ======================================================================================

    def _info(self, msg):
        """Informational output - resolved geometry, decoder settings. `warn_level >= 2`."""
        if getattr(self, 'warn_level', 1) >= 2:
            print(msg)

    def _warn(self, msg):
        """A condition the caller should know about that does not stop the run. `>= 1`."""
        if getattr(self, 'warn_level', 1) >= 1:
            print(f"WARN: {msg}")

    def _resolve_lang(self, phoneme_set, lang):
        """
        A language code as the phoneme set's own phonemizer spells it: canonical BCP 47 for
        std ("en-us" -> "en", "cmn" -> "zh"), untouched for ph66, whose codes are espeak-ng's.
        """
        if phoneme_set != "std":
            return lang
        code, note = _presets.normalize_std_lang(lang)
        if note:
            self._warn(note)
        return code

    def _set_phonemizer(self, phoneme_set, lang):
        """Build (or retarget) the phonemizer and adopt its label space."""
        if self.phonemizer is None or phoneme_set != self.phoneme_set:
            self.phonemizer = make_phonemizer(phoneme_set, lang, verbose=self.warn_level,
                                              **self._g2p_options)
        elif lang != self.lang:
            self.phonemizer.set_backend(language=lang)
        self.phoneme_set = phoneme_set
        self.selected_mapper = phoneme_set          # the name older code reads
        self.lang = lang
        self._setup_labels()

    def _setup_labels(self):
        """Label inventory and the special class ids, from the phonemizer's inventory."""
        inv = self.inventory = self.phonemizer.inventory
        self.phonemes_key = inv.phonemes_key
        self.phoneme_groups_key = inv.groups_key

        self.phoneme_id_to_label = inv.id_to_label
        self.phoneme_label_to_id = inv.label_to_id
        self.group_id_to_label = inv.group_id_to_label
        self.group_label_to_id = inv.group_label_to_id
        self.phoneme_id_to_group_id = inv.phoneme_id_to_group_id

        # Silence is the one class the decoder treats specially (it pads an optional one onto
        # both ends and may skip any it meets), so it has to be the id the model was trained
        # with: 0 in ph66, 1 in every std language group.
        self.silence_class = inv.sil_id
        self.silence_group = inv.sil_group
        # Filler for ragged batches, never a target: ph66's unused 'noise' slot, std's
        # '<blank>'. NOT 'noise' in std, where that is a real label the G2P can produce.
        self.pad_class = inv.pad_id
        self.pad_group = inv.pad_group

    def reset_counters(self):
        """Reset the alignment statistics counters. For debugging and reporting only."""
        self.total_segments_processed = 0
        self.total_segments_bad = 0
        self.total_segments_failed = 0
        self.total_phonemes_aligned = 0
        self.total_phonemes_target = 0
        self.total_phonemes_missed = 0
        self.perfect_matches = 0

    def load_model(self, preset="en-us", model_name=None, ckpt_path=None, cupe_ckpt_path=None):
        """
        Load a model and adopt its phoneme set, geometry and decoder settings.

        Priority: `ckpt_path` (or its older name `cupe_ckpt_path`), then `model_name`, then
        `preset`. Whichever names the model, the preset still supplies the LANGUAGE as long as
        none was given explicitly.

        `model_name` is a published model ('mbfa-latin', 'CUPE-3-en', or a full Hub repository
        id), downloaded on first use; `ckpt_path` is a release directory or a training
        checkpoint file on disk.

        The model has the last word on the phoneme set: a p3cupe one is ph66 and a p4mbfa one
        is std, whatever the default is. Asking for one set by name and handing over a model
        of the other raises.
        """
        ckpt_path = ckpt_path or cupe_ckpt_path
        want_set = self._requested_set
        lang = self.lang

        if ckpt_path is None and model_name is None:
            phoneme_set = want_set or _presets.DEFAULT_PHONEME_SET
            if phoneme_set == "std":
                asked = preset if (preset and self._lang_is_default) else lang
                model_name, lang, _group, note = _presets.get_std_preset(asked)
                if note:
                    self._warn(note)
            else:
                model_name, lang = _presets.get_preset(
                    preset=preset, lang=_presets.DEFAULT_LANG if self._lang_is_default else lang)
        elif preset and self._lang_is_default:
            lang = preset

        if ckpt_path is None:
            # On disk first ($BFA_MODEL_DIR, ./models), then the model's Hub repository.
            ckpt_path = hub.resolve_model(model_name)
        elif not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"ckpt_path {ckpt_path!r} does not exist. It takes a release directory "
                f"(config.json + model.safetensors) or a training checkpoint file; a "
                f"published model is named with model_name=, e.g. 'mbfa-latin'.")
        shown = model_name if model_name and not os.path.exists(model_name) \
            else os.path.basename(os.path.normpath(ckpt_path))

        extractor = load_extractor(ckpt_path, device=self.device,
                                   sample_rate=self.resampler_sample_rate,
                                   verbose=self.warn_level)
        phoneme_set = _presets.SET_OF_FAMILY[extractor.family]
        if want_set is not None and want_set != phoneme_set:
            raise ValueError(
                f"phoneme_set='{want_set}' was asked for, but {shown} is a "
                f"{extractor.family} model, which decodes the '{phoneme_set}' set. Drop "
                f"`phoneme_set` to use it as it is, or pass a "
                f"{_presets.FAMILY_OF_SET[want_set]} model.")

        self._set_phonemizer(phoneme_set, self._resolve_lang(phoneme_set, lang))
        # Before the extractor is adopted: a checkpoint whose head does not emit this
        # inventory would align without error and return nonsense.
        extractor.check_inventory(self.inventory)
        self.extractor = extractor

        # Geometry comes from the checkpoint, never from a constant here.
        self.sample_rate = self.extractor.sample_rate
        self.window_size_ms = self.extractor.window_size_ms
        self.stride_ms = self.extractor.stride_ms
        self.frame_stride_wav = self.extractor.frame_stride_samples
        self.ms_per_frame = self.extractor.ms_per_frame
        self.frames_per_window = self.extractor.frames_per_window
        self._resolve_decoder()

        self._info(f"BFA: {self.extractor.family} model {shown}, "
                   f"phoneme set {self.inventory.name}, language {self.lang}")
        self._info(f"BFA geometry: window {self.window_size_ms:g} ms, stride "
                   f"{self.stride_ms:g} ms = {self.extractor.stride_frames} frames, frame "
                   f"stride {self.frame_stride_wav} samples ({self.ms_per_frame:g} ms/frame), "
                   f"{self.extractor.classes_dim} classes")
        return self.extractor

    def _resolve_decoder(self):
        """
        Settle the aligner's topology: the checkpoint's training values, with any explicit
        constructor override applied on top.
        """
        hp = dict(self.extractor.decoder_hp)
        if self._min_phone_frames_override is not None:
            hp['min_phone_frames'] = int(self._min_phone_frames_override)
        if self._soft_phone_frames_override is not None:
            hp['soft_phone_frames'] = int(self._soft_phone_frames_override)
        if self._bonus_weight_override is not None:
            hp['boundary_bonus_weight'] = float(self._bonus_weight_override)
        self.decoder_hp = hp
        self.min_phone_frames = int(hp['min_phone_frames'])
        self.soft_phone_frames = int(hp['soft_phone_frames'])
        self.boundary_bonus_weight = float(hp['boundary_bonus_weight'])
        self._info(
            f"decoder: min {self.min_phone_frames} + {self.soft_phone_frames} soft frames "
            f"({self.min_phone_frames * self.ms_per_frame:g} ms minimum phone), "
            f"sil_skippable={hp['sil_skippable']}, class_prior_norm={hp['class_prior_norm']}, "
            f"onset bonus {self.boundary_bonus_weight:g} "
            f"({self.extractor.flux_source} flux), "
            f"sub-frame onsets {'on' if self.sub_frame_onsets else 'off'}")

    def download_model(self, model_name=None, local_dir=None):
        """
        Resolve a model name to a path on disk, downloading it if needed.

        Looks on disk first (`hub.model_search_dirs`: $BFA_MODEL_DIR, then `./models`), then
        fetches `config.json` + `model.safetensors` from the model's Hugging Face repository.

        Args:
            model_name (str): a published model ('mbfa-latin', 'CUPE-3-en', ...) or a full
                repository id. Defaults to the current phoneme set's model for the current
                language.
            local_dir (str, optional): download into this directory instead of the
                huggingface_hub cache.
        """
        phoneme_set = self.phoneme_set or self._requested_set or _presets.DEFAULT_PHONEME_SET
        if model_name is None:
            model_name = (_presets.get_std_preset(self.lang)[0] if phoneme_set == "std"
                          else _presets.DEFAULT_MODEL)
        path = hub.resolve_model(model_name, local_dir=local_dir)
        self._info(f"Model available at: {path}")
        return path

    # ======================================================================================
    # Audio
    # ======================================================================================

    @torch.no_grad()
    def chop_wav(self, wav, start_frame, end_frame):
        """Cut one segment out of a clip, RMS-normalise it, and pad it to `duration_max`."""
        num_frames = (end_frame - start_frame) if (end_frame != -1) else -1

        if num_frames < self.seg_duration_min_samples:
            self._warn(f"segment too short: {num_frames} samples, minimum is "
                       f"{self.seg_duration_min_samples}")
            return None, None, -1

        wav = wav[:, start_frame:end_frame]
        assert (wav.shape[1] <= num_frames) or (num_frames == -1)
        if wav.shape[1] < self.seg_duration_min_samples:
            self._warn(f"segment audio too small: {tuple(wav.shape)} for "
                       f"[{start_frame}, {end_frame})")
            return None, None, -2

        wav = self._rms_normalize(wav.mean(dim=0))
        wav_len = wav.shape[0]
        if wav_len > self.wav_len_max:
            wav = wav[:self.wav_len_max]
            wav_len = wav.shape[0]
        else:
            wav = torch.nn.functional.pad(wav, (0, self.wav_len_max - wav_len), 'constant', 0)
        return wav, wav_len, 0

    @staticmethod
    def _rms_normalize(audio):
        """RMS normalise. The model is trained on RMS-normalised audio; this is not optional."""
        rms = torch.sqrt(torch.mean(audio ** 2))
        if rms > 0:
            audio = audio / rms
        return audio

    def load_audio(self, audio, sr=None):
        """Load a file (or take a tensor) and resample it to the model's 16 kHz."""
        if isinstance(audio, str):
            wav, sr = torchaudio.load(audio, frame_offset=0, normalize=True)
        elif isinstance(audio, torch.Tensor):
            if sr is None:
                raise ValueError("Sample rate 'sr' must be provided when passing a tensor.")
            wav = audio
        else:
            raise ValueError("Invalid audio input type.")

        if sr != self.resampler_sample_rate:
            assert wav.shape[1] > 0, "Audio data is empty"
            custom_resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.resampler_sample_rate,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                beta=14.769656459379492,
            )
            return custom_resampler(wav)
        return self.default_resampler(wav)

    # ======================================================================================
    # Model + decoder
    # ======================================================================================

    def _model_prediction_batch(self, audio_batch, wav_lens, extract_embeddings=False):
        """
        Run a batch of waveforms through the loaded model.

        Returns:
            (logits_class, logits_group, embeddings|None, spectral_lens, flux|None)
        """
        if audio_batch.dim() == 1:
            audio_batch = audio_batch.unsqueeze(0)

        out = self.extractor.encode_clip(
            audio_batch, return_embeddings=extract_embeddings,
            return_flux=bool(self.boundary_bonus_weight))

        spectral_lens = self.extractor.spectral_lengths(wav_lens)
        spectral_lens = spectral_lens.clamp(max=out['ph'].shape[1])
        return out['ph'], out['group'], out['emb'], spectral_lens, out['flux']

    def _framestamps_from_alignment(self, aligned, n_targets):
        """
        One clip's `segmental.align_batch` result -> `(phoneme_id, start_frame, end_frame,
        target_idx)` tuples, one per target the path actually visited.

        Frames are FLOATS when `sub_frame_onsets` is on: the onset is the interpolated crossing
        of two posteriors, which is the quantity the model was trained to place, and rounding
        it to the 5 ms grid here would throw that away before `convert_to_ms` ever sees it.

        A phone ends where the next VISITED phone starts, so the segmentation is gap-free by
        construction - there is no blank run between phones to extend into. The optional SIL
        pads carry a `target_idx` of -1 and are dropped: they are not in the caller's sequence,
        but they still bound the neighbours that are.
        """
        if not aligned:
            return []
        onsets, used = aligned['onsets'], aligned['used']
        phones, target_idx = aligned['phones'], aligned['target_idx']
        n_frames = float(aligned['n_frames'])

        visited = [i for i in range(len(phones)) if used[i]]
        stamps = []
        for k, i in enumerate(visited):
            start = min(max(float(onsets[i]), 0.0), n_frames)
            end = float(onsets[visited[k + 1]]) if k + 1 < len(visited) else n_frames
            end = min(max(end, start + 1.0), n_frames)
            ti = int(target_idx[i])
            if 0 <= ti < n_targets:
                stamps.append((int(phones[i]), start, end, ti))
        return stamps

    def _decode_batch(self, log_probs, ph_seqs, ph_seq_lens, spectral_lens, flux=None):
        """
        Force-align a batch and return framestamps per clip.

        Raises:
            ValueError: when a clip's sequence cannot be fitted into its frames even at one
                frame per phone. The chunker catches this and retries with a smaller budget.
        """
        hp = self.decoder_hp
        aligned = align_batch(
            log_probs, ph_seqs, ph_seq_lens, spectral_lens,
            min_frames=self.min_phone_frames, soft_frames=self.soft_phone_frames,
            soft_cost=hp['soft_frame_cost'], sil_skippable=hp['sil_skippable'],
            optional_sil=hp['optional_sil'], class_prior_norm=hp['class_prior_norm'],
            sil_class=self.silence_class, flux=flux,
            bonus_weight=self.boundary_bonus_weight,
            contrast_frames=hp['boundary_contrast_frames'],
            sub_frame=self.sub_frame_onsets)

        out = []
        for b, entry in enumerate(aligned):
            n_targets = int(ph_seq_lens[b])
            if not entry:
                raise ValueError(
                    f"Audio too short to align: {n_targets} phonemes cannot be fitted into "
                    f"{int(spectral_lens[b])} frames "
                    f"({int(spectral_lens[b]) * self.ms_per_frame / 1000.0:.2f}s).")
            out.append(self._framestamps_from_alignment(entry, n_targets))
        return out

    def _account_targets(self, framestamps, n_targets, debug=False):
        """
        Update the coverage counters and report anything the aligner could not place.

        Forced alignment places every target that is not a skippable SIL, so a miss here is
        almost always a SIL the phonemizer wrote for a comma the speaker did not pause at -
        which is the aligner doing its job, not failing at it. Nothing is inserted to paper
        over it: the caller can see exactly which targets were placed from `target_seq_idx`.
        """
        self.total_phonemes_target += n_targets
        self.total_phonemes_aligned += len(framestamps)

        placed = {int(t[3]) for t in framestamps}
        missing = [i for i in range(n_targets) if i not in placed]
        self.total_phonemes_missed += len(missing)
        if missing and (debug or self.warn_level >= 2):
            labels = [self.phoneme_id_to_label.get(int(t[0]), '?')
                      for t in framestamps]
            print(f"  {len(missing)} of {n_targets} targets not placed "
                  f"(positions {missing[:12]}{'...' if len(missing) > 12 else ''}); "
                  f"{len(labels)} placed")
        return framestamps

    def extract_timestamps_from_segment_batch(self, wavs, wav_lens, phoneme_sequences,
                                              start_offset_times=0, group_sequences=None,
                                              extract_embeddings=True, do_groups=True,
                                              debug=False):
        """
        Extract phoneme and group timestamps for a batch of audio segments.

        Args:
            wavs: audio tensor [B, samples].
            wav_lens: valid sample count per segment.
            phoneme_sequences: list of phoneme id sequences in the loaded model's own label
                space - ph66 ids, or a language group's local tokens (or a [B, S] tensor
                padded with `self.pad_class`).
            start_offset_times: segment start time(s) in seconds, added to every timestamp.
            group_sequences: matching phoneme-group sequences, or None to derive them.
            extract_embeddings: also return confidence-pooled embeddings per phoneme.
            do_groups: also emit group timestamps.
            debug: print per-segment detail.

        Returns:
            (timestamp_dicts, pooled_phoneme_embeddings, pooled_group_embeddings) where each
            timestamp dict has 'phoneme_timestamps' and 'group_timestamps' as 8-tuples
            `(id, start_frame, end_frame, target_idx, is_estimated, confidence, start_ms,
            end_ms)`.
        """
        assert self.extractor is not None, \
            "No model loaded. Use load_model() before aligning."

        if isinstance(phoneme_sequences, torch.Tensor):
            ph_seq_lens = [(seq != self.pad_class).sum().item() for seq in phoneme_sequences]
        else:
            ph_seq_lens = [len(seq) for seq in phoneme_sequences]
            max_len = max(ph_seq_lens)
            phoneme_sequences = torch.tensor(
                [list(seq) + [self.pad_class] * (max_len - len(seq))
                 for seq in phoneme_sequences], dtype=torch.long)

        if group_sequences is None:
            seq_iter = ([row.tolist() for row in phoneme_sequences]
                        if phoneme_sequences.dim() == 2 else list(phoneme_sequences))
            mapped = [self._map_phonemes_to_groups(seq) for seq in seq_iter]
            max_len = max((m.size(0) for m in mapped), default=0)
            group_sequences = torch.stack([
                F.pad(m, (0, max_len - m.size(0)), value=self.pad_group) for m in mapped
            ], dim=0) if mapped else torch.empty(0, dtype=torch.long)
        elif not isinstance(group_sequences, torch.Tensor):
            max_len = max(len(seq) for seq in group_sequences)
            group_sequences = torch.tensor(
                [list(seq) + [self.pad_group] * (max_len - len(seq))
                 for seq in group_sequences], dtype=torch.long)

        logits_class, logits_group, embeddings, spectral_lens, flux = \
            self._model_prediction_batch(wavs, wav_lens, extract_embeddings)

        # The DP is a python loop over frames on tiny tensors, so it is launch-bound on a GPU:
        # measured 1.8 s on CPU against 5.0 s on CUDA for a 32 s clip, bit-identical. The
        # posteriors move once, here, and everything downstream reads them on the CPU too.
        log_probs_p = F.log_softmax(logits_class.float(), dim=2).cpu()
        log_probs_g = F.log_softmax(logits_group.float(), dim=2).cpu()
        ph_seqs = phoneme_sequences.cpu()
        grp_seqs = group_sequences.cpu()
        ph_seq_lens_t = torch.tensor(ph_seq_lens, dtype=torch.long)
        spectral_lens = spectral_lens.cpu()

        frame_phonemes = self._decode_batch(log_probs_p, ph_seqs, ph_seq_lens_t, spectral_lens,
                                            flux=None if flux is None else flux.cpu())

        # Groups are a many-to-one relabelling of the phonemes, so they take the phoneme
        # segmentation directly: consistent with it by construction, and one DP instead of two.
        grp_list = grp_seqs.tolist()
        frame_groups = [[(int(grp_list[b][ti]), sf, ef, ti) for (_ph, sf, ef, ti) in fp]
                        for b, fp in enumerate(frame_phonemes)]

        timestamp_dicts = []
        for b in range(len(frame_phonemes)):
            self._account_targets(frame_phonemes[b], int(ph_seq_lens[b]), debug=debug)
            offset = (start_offset_times[b] if isinstance(start_offset_times, (list, tuple))
                      else start_offset_times)

            rows = {}
            for key, stamps, probs in (('phoneme_timestamps', frame_phonemes[b], log_probs_p[b]),
                                       ('group_timestamps', frame_groups[b], log_probs_g[b])):
                # `is_estimated` is always False now - forced alignment places every target it
                # reports - but the field stays in the output schema.
                stamps = _calculate_confidences(probs, [(*t, False) for t in stamps])
                stamps = convert_to_ms(stamps, spectral_lens[b], offset, wav_lens[b],
                                       self.resampler_sample_rate,
                                       frame_stride_samples=self.frame_stride_wav)
                rows[key] = sorted(stamps, key=lambda x: x[6])
            timestamp_dicts.append(rows)

        pooled_p, pooled_g = [None] * len(frame_phonemes), [None] * len(frame_phonemes)
        if extract_embeddings and embeddings is not None:
            embeddings = embeddings.cpu()
            for b in range(len(frame_phonemes)):
                n = int(spectral_lens[b])
                pooled_p[b] = weighted_pool_embeddings(
                    embeddings[b][:n], log_probs_p[b][:n],
                    timestamp_dicts[b]['phoneme_timestamps'])
                if do_groups:
                    pooled_g[b] = weighted_pool_embeddings(
                        embeddings[b][:n], log_probs_g[b][:n],
                        timestamp_dicts[b]['group_timestamps'])
        return timestamp_dicts, pooled_p, pooled_g

    # ======================================================================================
    # Long segments.  See `chunking.py`.
    # ======================================================================================

    def set_chunking(self, enabled=None, seconds=None, max_phonemes=None,
                     phonemes_per_second=None, vad=None, derail_frames=None,
                     derail_run=None, backoff=None, conf_floor=None, weak_run=None,
                     snap_to_silence=None, segment_max_seconds=None, verbose=None):
        """
        Retune the long-segment chunker on a live aligner. Every argument left as None keeps
        its current value.

        Args:
            enabled: master switch (`chunk_audio`).
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
            segment_max_seconds: length above which a segment is chunked at all.
            verbose: print the resolved configuration. None follows `warn_level >= 2`.

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
        if segment_max_seconds is not None:
            self.segment_max_seconds = float(segment_max_seconds)

        cfg = {
            'enabled': self.chunk_audio, 'seconds': self.chunk_seconds,
            'max_phonemes': self.chunk_max_phonemes,
            'phonemes_per_second': self.chunk_phonemes_per_second,
            'vad': self.chunk_vad, 'derail_frames': self.chunk_derail_frames,
            'derail_run': self.chunk_derail_run, 'backoff': self.chunk_backoff,
            'conf_floor': self.chunk_conf_floor, 'weak_run': self.chunk_weak_run,
            'snap_to_silence': self.chunk_snap_to_silence,
            'segment_max_seconds': self.segment_max_seconds,
        }
        if verbose or (verbose is None and self.warn_level >= 2):
            print("chunking: " + ", ".join(f"{k}={v}" for k, v in cfg.items()))
        return cfg

    def detect_speech_segments(self, audio_wav, **kwargs):
        """
        Energy VAD over a waveform. Returns `[(start_sec, end_sec), ...]` for the speech spans.

        Exposed because it is useful on its own (marking silent regions, trimming, sanity
        checks); inside BFA its only jobs are to size a chunk window's phoneme budget from
        SPEECH seconds and to let a cut land in a pause.

        Args:
            audio_wav: waveform tensor, 1-D or (C, T).
            **kwargs: forwarded to `chunking.detect_speech_segments`.
        """
        return _chunking.detect_speech_segments(audio_wav, self.resampler_sample_rate, **kwargs)

    @contextlib.contextmanager
    def _chunk_window_scope(self):
        """
        Run one chunk window without letting it touch the segment-level bookkeeping.

        A window is an internal step, not a unit of work the caller asked for, and most of one
        is thrown away. Left alone it would double-count every phoneme aligned in a discarded
        tail and re-aligned in the next window, and report the over-provisioned surplus as
        missing when being unplaceable there is the expected outcome.
        `align_segment_chunked` attributes the segment's totals once, from the assembled result.
        """
        names = ('total_segments_processed', 'total_segments_bad', 'total_segments_failed',
                 'total_phonemes_aligned', 'total_phonemes_target',
                 'total_phonemes_missed', 'perfect_matches')
        saved = {n: getattr(self, n) for n in names}
        saved_warn = self.warn_level
        self.warn_level = min(self.warn_level, 1)
        try:
            yield
        finally:
            self.warn_level = saved_warn
            for n, v in saved.items():
                setattr(self, n, v)

    def _needs_chunking(self, duration_sec, n_phonemes):
        """
        Whether a segment goes down the chunked path.

        Length only. The segmental decoder does not collapse on long audio - it has no band
        constraint and no lattice to compress, and every phone is placed in one global search -
        so windowing only adds cut points that can land mid-phone. It is a time and memory
        guard, nothing more.
        """
        if not self.chunk_audio:
            return False
        return duration_sec > self.segment_max_seconds

    @torch.no_grad()
    def align_segment_whole(self, clip_wav, start_sec, end_sec, phoneme_sequence,
                            group_sequence=None, extract_embeddings=False, do_groups=False,
                            debug=False):
        '''
        Align one segment in a single decode on its own, UNPADDED audio.

        This is the path for segments longer than `duration_max` that are not chunked. The
        batched path pads every segment to `duration_max` and would otherwise truncate a longer
        one silently, dropping its tail.
        '''
        sr = self.resampler_sample_rate
        wav = clip_wav.mean(dim=0) if clip_wav.dim() > 1 else clip_wav
        a = max(0, int(start_sec * sr))
        b = int(end_sec * sr) if (end_sec is not None and end_sec > 0) else wav.shape[0]
        wav = self._rms_normalize(wav[a:min(b, wav.shape[0])])
        res, pe, ge = self.extract_timestamps_from_segment_batch(
            wav.unsqueeze(0), [wav.shape[0]], [list(phoneme_sequence)],
            start_offset_times=[start_sec],
            group_sequences=[list(group_sequence)] if (do_groups and group_sequence is not None)
            else None,
            extract_embeddings=extract_embeddings, do_groups=do_groups, debug=debug)
        return res[0], pe[0], ge[0]

    @torch.no_grad()
    def align_segment_chunked(self, clip_wav, start_sec, end_sec, phoneme_sequence,
                              group_sequence=None, extract_embeddings=False,
                              do_groups=False, debug=False):
        """
        Align one very long segment by walking overlapping windows, returning exactly what one
        element of `extract_timestamps_from_segment_batch`'s output looks like - so the caller
        cannot tell which path a segment took.

        Each window gets a phoneme budget sized from its SPEECH seconds, is aligned, and is
        trusted up to the point where its alignment stops looking healthy (`chunking.py`); the
        next window restarts at the end of the last trusted phoneme with the phonemes that
        were not kept.

        Args:
            clip_wav: the PARENT clip's waveform, 1-D or (C, T). Sliced here, not by
                `chop_wav` - windows must not be padded out to `duration_max`.
            start_sec, end_sec: segment bounds within the clip, in seconds.
            phoneme_sequence: target phoneme ids for the whole segment.
            group_sequence: matching group ids, or None to derive them per window.
            extract_embeddings, do_groups, debug: as `extract_timestamps_from_segment_batch`.

        Returns:
            `(result_dict, pooled_phoneme_embeddings, pooled_group_embeddings)`. Frames and
            target indices are SEGMENT-global, and the ms values carry `start_sec` like the
            single-shot path.
        """
        sr = self.resampler_sample_rate
        n_total = len(phoneme_sequence)

        wav = clip_wav
        if wav.dim() > 1:
            wav = wav.mean(dim=0)
        seg_a = max(0, int(start_sec * sr))
        seg_b = int(end_sec * sr) if (end_sec is not None and end_sec > 0) else wav.shape[0]
        seg_b = min(seg_b, wav.shape[0])
        # Normalise ONCE over the whole segment. Normalising each window separately would hand
        # the same moment of audio a different gain depending on which window it fell in, so
        # the overlap would not agree with itself.
        wav = self._rms_normalize(wav[seg_a:seg_b])
        seg_dur = wav.shape[0] / sr

        speech = self.detect_speech_segments(wav) if self.chunk_vad else []
        if debug or self.warn_level >= 2:
            sp = _chunking.speech_seconds(speech, 0.0, seg_dur)
            print(f"Chunked alignment: {seg_dur:.2f}s segment, {n_total} phonemes, "
                  f"VAD speech {sp:.2f}s in {len(speech)} span(s)")

        # Each entry is `(framestamp, pooled_embedding_row_or_None)` - the two travel together
        # because the final sort can reorder them, and sorting only one list would silently
        # pair each embedding with a different phoneme.
        ph_rows, grp_rows = [], []
        ph_cursor = 0
        t_cursor = 0.0
        # One window always consumes at least one phoneme, so `n_total` iterations is a hard
        # ceiling; the +8 covers the audio-only advances over pure-silence windows.
        for _ in range(n_total + 8):
            if ph_cursor >= n_total or (seg_dur - t_cursor) < self.seg_duration_min:
                break
            remaining = n_total - ph_cursor

            chunk_end = min(t_cursor + self.chunk_seconds, seg_dur)
            if (seg_dur - chunk_end) < self.seg_duration_min:
                chunk_end = seg_dur
            audio_exhausted = chunk_end >= seg_dur - 1e-6

            sp_sec = _chunking.speech_seconds(speech, t_cursor, chunk_end)
            if sp_sec < 0.15 and not audio_exhausted:
                # A window with no speech in it. Skip the audio rather than forcing phonemes
                # into it - this is the one place the VAD decides something on its own, and
                # getting it wrong only costs a window of overlap.
                nxt = next((a for a, _ in speech if a > t_cursor), chunk_end)
                skipped_from = t_cursor
                t_cursor = max(min(nxt, chunk_end), t_cursor + 0.25)
                if debug:
                    print(f"  window {skipped_from:6.2f}-{chunk_end:6.2f}s is silent, "
                          f"skipping to {t_cursor:.2f}s")
                continue

            budget = int(math.ceil(self.chunk_phonemes_per_second * sp_sec))
            budget = max(1, min(budget, self.chunk_max_phonemes))
            # Never hand a window more phonemes than its frames can hold at the duration
            # floor: past that the alignment is infeasible, not merely crowded.
            fits = max(1, int((chunk_end - t_cursor) * 1000.0
                              / (self.ms_per_frame * self.min_phone_frames)))
            n = min(remaining, budget, fits)
            if audio_exhausted:
                # Last window: there is no more audio to walk into, so everything left has to
                # go in here whether it fits or not.
                n = remaining
            final = (n == remaining) and audio_exhausted

            res, pe, ge, used_n = self._align_chunk(
                wav, t_cursor, chunk_end, start_sec,
                list(phoneme_sequence[ph_cursor:ph_cursor + n]),
                list(group_sequence[ph_cursor:ph_cursor + n])
                if (do_groups and group_sequence is not None) else None,
                extract_embeddings=extract_embeddings, do_groups=do_groups, debug=debug)
            if res is None:
                # Window could not be aligned at all even after shrinking the budget. Advance
                # the audio and try again rather than aborting the segment.
                t_cursor = min(seg_dur, t_cursor + max(0.5, self.chunk_seconds * 0.25))
                continue
            n = used_n
            final = final and (n == remaining)

            stamps = res['phoneme_timestamps']
            if not stamps:
                if debug:
                    print(f"  window {t_cursor:6.2f}-{chunk_end:6.2f}s produced no stamps, "
                          f"skipping")
                t_cursor = min(seg_dur, t_cursor + max(0.5, self.chunk_seconds * 0.25))
                continue
            keep, reason = self._chunk_keep_count(
                stamps, n, final,
                window_ms=((start_sec + t_cursor) * 1000.0, (start_sec + chunk_end) * 1000.0))

            frame_offset = t_cursor * sr / self.frame_stride_wav
            keep_mask = [0 <= int(t[3]) < keep for t in stamps]
            kept = [t for t, m in zip(stamps, keep_mask) if m]
            if not kept:
                # Nothing survived the cut. Take the window's first phoneme on trust so the
                # walk cannot stall, and let the confidence warnings flag it.
                kept = stamps[:1]
                keep = max(1, min(keep, 1))
                keep_mask = [i == 0 for i in range(len(stamps))]

            last_end_ms = max(float(t[7]) for t in kept)
            cut_rel = (last_end_ms / 1000.0) - start_sec

            if (not final) and self.chunk_snap_to_silence > 0:
                # Restarting on a pause gives the next window's first phoneme a clean onset to
                # latch onto. Forward only - moving the cut back would make the next window
                # re-align audio already assigned to a kept phoneme.
                snapped = _chunking.snap_cut_to_silence(
                    cut_rel, speech, seg_dur, self.chunk_snap_to_silence)
                if snapped > cut_rel:
                    cut_rel = snapped

            keep_ids = [i for i, m in enumerate(keep_mask) if m]
            ph_rows.extend(zip(
                self._reindex_stamps(kept, ph_cursor, frame_offset),
                (pe[keep_ids] if (extract_embeddings and pe is not None)
                 else [None] * len(kept))))

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
            # Phoneme progress is guaranteed (keep >= 1), so the loop always terminates; this
            # only keeps the audio cursor from stalling and re-decoding the same window when a
            # cut lands on or before where it started.
            t_cursor = max(cut_rel, t_cursor + 1e-3) if cut_rel > t_cursor else \
                min(seg_dur, t_cursor + max(0.25, self.chunk_seconds * 0.25))
            if final:
                break

        if ph_cursor < n_total:
            self._warn(f"chunked alignment ran out of audio with {n_total - ph_cursor} of "
                       f"{n_total} phonemes unaligned ({seg_dur:.2f}s segment). The text is "
                       f"longer than the audio, or the audio is mostly silence.")

        ph_rows.sort(key=lambda r: (r[0][6], r[0][3]))
        grp_rows.sort(key=lambda r: (r[0][6], r[0][3]))
        ph_stamps = [r[0] for r in ph_rows]
        grp_stamps = [r[0] for r in grp_rows]

        # Attribute the segment's coverage stats ONCE, from the assembled result; the
        # per-window numbers were rolled back by `_chunk_window_scope`.
        covered = {int(t[3]) for t in ph_stamps if int(t[3]) >= 0}
        self.total_phonemes_target += n_total
        self.total_phonemes_aligned += len(ph_stamps)
        self.total_phonemes_missed += max(0, n_total - len(covered))

        result = {'phoneme_timestamps': ph_stamps, 'group_timestamps': grp_stamps}
        return result, self._stack_rows(ph_rows), self._stack_rows(grp_rows)

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

        Only the frames and the target index move; the ms values already carry the window's
        `start_offset_times`, which `_align_chunk` sets to the window's absolute position, so
        they are global on arrival.
        """
        out = []
        for (ph, sf, ef, ti, est, conf, sms, ems) in stamps:
            out.append((ph, float(sf) + frame_offset, float(ef) + frame_offset,
                        int(ti) + ph_offset if int(ti) >= 0 else int(ti),
                        est, conf, sms, ems))
        return out

    def _align_chunk(self, wav, t0, t1, seg_start_sec, phonemes, groups,
                     extract_embeddings=False, do_groups=False, debug=False):
        """
        Align one window. Returns `(result, pooled_p, pooled_g, n_phonemes_used)`, or
        `(None, None, None, 0)` if the window could not be aligned at all.

        The budget can still overshoot what the audio holds, which makes the alignment
        infeasible rather than merely crowded. That is a budget that was too greedy, not a
        broken window, so it is halved and retried.
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
                        debug=False)
                return res[0], pe[0], ge[0], n
            except ValueError as ex:
                n = n // 2
                if n < 1:
                    self._warn(f"could not align window {t0:.2f}-{t1:.2f}s: {ex}")
                    return None, None, None, 0
                if debug:
                    print(f"  window {t0:.2f}-{t1:.2f}s over-budgeted ({ex}); "
                          f"retrying with {n} phonemes")
        return None, None, None, 0

    def _chunk_keep_count(self, stamps, n_targets, final, window_ms=None):
        """
        How many of a window's `n_targets` phonemes to trust.

        `stamps` are the window's aligned 8-tuples, in start-time order, with window-local
        target indices. They are re-keyed by target index here so a target the aligner could
        not place shows up as `None` and counts as derailed.

        Args:
            window_ms: the window's own `(start_ms, end_ms)` audio span. Used for the time
                trim - see the return of 'trim' below.

        Returns `(keep_count, reason)` where reason is 'duration', 'confidence', 'trim' or
        None.
        """
        durations = [None] * n_targets
        confidences = [None] * n_targets
        ends_ms = [None] * n_targets
        for (_ph, sf, ef, ti, _est, conf, _sms, ems) in stamps:
            ti = int(ti)
            if 0 <= ti < n_targets and durations[ti] is None:
                durations[ti] = float(ef) - float(sf)
                confidences[ti] = float(conf)
                ends_ms[ti] = float(ems)

        min_keep = max(1, int(MIN_KEEP_FRACTION * n_targets))
        cut, reason = _chunking.find_derailment(
            durations, confidences, min_keep=min_keep,
            derail_frames=self.chunk_derail_frames, derail_run=self.chunk_derail_run,
            conf_floor=self.chunk_conf_floor, weak_run=self.chunk_weak_run,
            backoff=self.chunk_backoff)

        if final:
            # The LAST window keeps everything, derailed or not, and this has to be checked
            # BEFORE the cut. A cut defers phonemes to the next window; on the final window
            # there is no next window, so the same cut DELETES them. A badly placed stamp is
            # strictly better than no stamp - it is flagged by its own confidence, where a
            # dropped phoneme is silent.
            return n_targets, (f'{reason} (final, kept anyway)' if reason else None)
        # Never trust a window past `TRUST_WINDOW_FRACTION` of its own AUDIO, whatever the
        # derailment detector says. Everything after that is handed back to the next window,
        # which will see it with real audio on both sides instead of at its edge.
        #
        # By time, not by count, and as a CAP rather than a fallback. This decoder cannot cram
        # - it enforces a minimum phone duration and reports infeasibility instead - so an
        # over-budgeted window places its surplus phonemes early rather than announcing
        # itself, and both the count trim and the confidence rule then keep too many of them.
        # The walk runs ahead of the audio and the error compounds window after window.
        time_keep = None
        if window_ms is not None:
            a, b = window_ms
            cut_ms = a + TRUST_WINDOW_FRACTION * (b - a)
            time_keep = sum(1 for e in ends_ms if e is not None and e <= cut_ms)

        if cut is not None:
            keep = cut if time_keep is None else min(cut, max(time_keep, 1))
            return max(min_keep, min(keep, n_targets)), reason
        if time_keep:
            return max(min_keep, min(time_keep, n_targets)), 'trim'
        # No timing to go on: drop a short tail by count so the next window re-decodes it.
        trim = max(2, int(NO_DERAIL_TRIM_FRACTION * n_targets))
        return max(min_keep, n_targets - trim), 'trim'

    # ======================================================================================
    # Text, segments and output
    # ======================================================================================

    def _map_phonemes_to_groups(self, phoneme_sequence):
        """Map phoneme ids to their phoneme-group ids, in the loaded inventory."""
        group_sequence = []
        for ph_idx in phoneme_sequence:
            ph_idx_val = ph_idx.item() if isinstance(ph_idx, torch.Tensor) else int(ph_idx)
            group_sequence.append(self.phoneme_id_to_group_id.get(ph_idx_val, self.pad_group))
        return torch.tensor(group_sequence, dtype=torch.long)

    def _align_words(self, phoneme_ts, word_num, words_list):
        '''
        Group phoneme timestamps into words.

        Phonemes are matched to words by their `target_seq_idx`, NOT by their position in
        `phoneme_ts`. The two differ whenever the aligner declined a target - a SIL the
        phonemizer wrote for a comma the speaker did not pause at is the normal case - and
        pairing them positionally would shift every later word onto the wrong phonemes.

        Args:
            phoneme_ts: the segment's phoneme_ts dicts, in time order.
            word_num: word index per TARGET phoneme, from the phonemizer.
            words_list: the segment's words.
        '''
        if not phoneme_ts or not word_num:
            return []

        by_word = {}
        for ph in phoneme_ts:
            ti = int(ph.get("target_seq_idx", -1))
            if not (0 <= ti < len(word_num)):
                continue
            by_word.setdefault(int(word_num[ti]), []).append(ph)

        words_ts = []
        for word_idx in sorted(by_word):
            phones = sorted(by_word[word_idx], key=lambda p: p["start_ms"])
            words_ts.append({
                "word": (words_list[word_idx] if word_idx < len(words_list)
                         else f"UNK_WORD_{word_idx}"),
                "start_ms": phones[0]["start_ms"],
                "end_ms": phones[-1]["end_ms"],
                "confidence": sum(p["confidence"] for p in phones) / len(phones),
                self.phonemes_key: [p["phoneme_id"] for p in phones],
                "ipa": [p["ipa_label"] for p in phones],
            })
        return words_ts

    def phonemize_sentence(self, text):
        """Phonemize a sentence.

        Args:
            text (str): The input text to phonemize.
        Returns:
            dict with keys:
                - text: original sentence
                - eipa: the G2P's phonemes in IPA (one-to-one with the mapped phonemes)
                - mipa: the inventory label each one was mapped to
                - `self.phonemes_key` ("local_ph" for std, "ph66" for ph66): phoneme ids
                - `self.phoneme_groups_key` ("gold_phg" / "pg16"): phoneme group ids
                - words: the words
                - word_num: word index per phoneme
            and, for the std set, `tone`, `stress` and `length` per phoneme.
        """
        return self.phonemizer.phonemize_sentence(text)

    def _phonemize_segment(self, seg):
        """
        One input segment -> its phonemized dict.

        Normally that is the segment's text through the phonemizer. A segment that already
        carries `gold_ph` - a standard_g2p `.gs.json` segment - is used as it is in the std
        set, so a corpus phonemized once is not phonemized again (and the G2P model is never
        loaded for it). `gold_ph` is language-group independent, which is why it is the one
        pre-phonemized form accepted here; a caller holding ids in the model's own label
        space can hand them to `extract_timestamps_from_segment_batch` directly.
        """
        if self.phoneme_set == "std" and seg.get("gold_ph"):
            return self.phonemizer.phonemize_gold(seg)
        return self.phonemize_sentence(seg["text"])

    def post_process_segment(self, segment, ts, phoneme_sequence, phoneme_timestamps,
                             group_timestamps=None, debug=False):
        """Turn one segment's framestamps into the output dicts, with coverage and words."""
        coverage_analysis = self.analyze_alignment_coverage(
            phoneme_sequence, phoneme_timestamps, self.phoneme_id_to_label)

        if debug:
            print("Alignment Coverage Analysis:")
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
                "phoneme_label": self.phoneme_id_to_label.get(ph_idx, f"UNK_{ph_idx}"),
                "ipa_label": (ts_out_segment["ipa"][target_seq_idx]
                              if 0 <= target_seq_idx < len(ts_out_segment["ipa"])
                              else "overflow"),
                "start_ms": float(start_ms),
                "end_ms": float(end_ms),
                "confidence": float(avg_confidence),
                "is_estimated": bool(is_estimated),
                "target_seq_idx": int(target_seq_idx),
                "index": idx
            }
            for idx, (ph_idx, start_frame, end_frame, target_seq_idx, is_estimated,
                      avg_confidence, start_ms, end_ms) in enumerate(phoneme_timestamps)
        ]
        # The std set's suprasegmental layers, one value per target phoneme. They come from
        # the G2P, not from the audio: what the dictionary says, placed on the aligned phone.
        for layer in ("tone", "stress", "length"):
            values = ts.get(layer)
            if values is None:
                continue
            ts_out_segment[layer] = values
            for item in ts_out_segment["phoneme_ts"]:
                ti = item["target_seq_idx"]
                item[layer] = int(values[ti]) if 0 <= ti < len(values) else 0

        if debug:
            print(f"\nSegment: '{segment['text']}'")
            n = len(ts_out_segment["phoneme_ts"])
            for tsi, item in enumerate(ts_out_segment["phoneme_ts"]):
                if tsi < 20 or tsi >= n - 20:
                    print(f"{tsi}  Phoneme ID: {item['phoneme_id']}, "
                          f"label: {item['phoneme_label']}, IPA: {item['ipa_label']}, "
                          f"Start: {int(item['start_ms'])} ms, End: {int(item['end_ms'])} ms, "
                          f"Confidence: {item['confidence']:.3f}")
                elif tsi == 20:
                    print("... (omitting middle phonemes for brevity) ...")

        if group_timestamps is not None:
            ts_out_segment["group_ts"] = [
                {
                    "group_id": int(grp_idx),
                    "group_label": self.group_id_to_label.get(grp_idx, f"UNK_{grp_idx}"),
                    "start_ms": float(start_ms),
                    "end_ms": float(end_ms),
                    "confidence": float(avg_confidence),
                    "is_estimated": bool(is_estimated),
                    "target_seq_idx": int(target_seq_idx),
                    "index": idx
                }
                for idx, (grp_idx, start_frame, end_frame, target_seq_idx, is_estimated,
                          avg_confidence, start_ms, end_ms) in enumerate(group_timestamps)
            ]

        ts_out_segment["words_ts"] = self._align_words(
            ts_out_segment["phoneme_ts"], ts.get("word_num", []), ts.get("words", []))
        return ts_out_segment

    def _result_header(self):
        """
        An empty result: the segment list, plus what the ids in it mean. A `phoneme_id` is
        only meaningful next to its inventory - 5 is one phoneme in ph66 and another in every
        std language group - so every result says which one it was written in.
        """
        return {"phoneme_set": self.inventory.name, "lang": self.lang, "segments": []}

    def process_segments(self, srt_data, audio_wavs, extract_embeddings=False, do_groups=False,
                         batch_size=16, debug=False):
        """
        Process multiple audio clips, each with multiple time-bounded segments.

        Args:
            srt_data: list of dicts (one per clip), each with a "segments" key holding
                `{"start", "end", "text"}` dicts. A single such dict is also accepted. In the
                std set a segment that also carries `gold_ph` (a standard_g2p `.gs.json`
                segment) is aligned from those phonemes and its text is not re-phonemized.
            audio_wavs: list of waveform tensors, one per clip. A single (C, T) or batched
                (B, C, T) tensor is also accepted.
            extract_embeddings: also return confidence-pooled embeddings.
            do_groups: also emit phoneme group timestamps.
            batch_size: segments per forward pass. Lower it if memory is tight.
            debug: print per-segment detail.

        Returns:
            List of dicts (one per clip) with a "segments" key. With
            `extract_embeddings=True`, `(batch_results, phoneme_embeddings, group_embeddings)`.
        """
        if isinstance(audio_wavs, torch.Tensor):
            if audio_wavs.dim() == 3:
                audio_wavs = [audio_wavs[i] for i in range(audio_wavs.size(0))]
            elif audio_wavs.dim() == 2:
                audio_wavs = [audio_wavs]
            else:
                raise ValueError(f"Expected audio_wavs of 2D (C,T) or 3D (B,C,T), got "
                                 f"{audio_wavs.dim()}D")

        if isinstance(srt_data, dict):
            srt_data = [srt_data]
        if len(srt_data) != len(audio_wavs):
            raise ValueError(f"Batch size mismatch: {len(srt_data)} srt items vs "
                             f"{len(audio_wavs)} audio waveforms.")

        for bi, batch_item in enumerate(srt_data):
            if "segments" not in batch_item:
                raise ValueError(f"Batch item {bi} missing 'segments' key. "
                                 f"Keys: {list(batch_item.keys())}")
            for si, seg in enumerate(batch_item["segments"]):
                if not all(k in seg for k in ("start", "end", "text")):
                    raise ValueError(f"Batch {bi}, segment {si} missing required keys "
                                     f"(start/end/text). Has: {list(seg.keys())}")

        num_batch = len(srt_data)

        # --- flatten sub-segments across all clips, tracking the clip they came from -------
        flat_items = []                       # (batch_idx, segment_dict, clip_audio_wav)
        for bi, (batch_item, clip_wav) in enumerate(zip(srt_data, audio_wavs)):
            for seg in batch_item["segments"]:
                flat_items.append((bi, seg, clip_wav))

        if not flat_items:
            empty = [self._result_header() for _ in range(num_batch)]
            if extract_embeddings:
                return empty, [[] for _ in range(num_batch)], [[] for _ in range(num_batch)]
            return empty

        ts_outs = [self._phonemize_segment(seg) for _, seg, _ in flat_items]
        phoneme_sequences = [ts[self.phonemes_key] for ts in ts_outs]
        group_sequences = ([ts[self.phoneme_groups_key] for ts in ts_outs] if do_groups
                           else [None] * len(flat_items))

        for (_, seg, _), ph_seq, grp_seq in zip(flat_items, phoneme_sequences, group_sequences):
            seg[self.phonemes_key] = ph_seq
            seg[self.phoneme_groups_key] = grp_seq

        valid_indices = []
        for i, ((bi, seg, _), ph_seq) in enumerate(zip(flat_items, phoneme_sequences)):
            if not ph_seq or len(ph_seq) < self.ph_seq_min:
                self._warn(f"skipping clip {bi}, segment '{seg.get('text', '')[:30]}': "
                           f"insufficient phoneme sequence ({len(ph_seq) if ph_seq else 0})")
                continue
            valid_indices.append(i)

        batch_results = [self._result_header() for _ in range(num_batch)]
        batch_p_embds = [[] for _ in range(num_batch)]
        batch_g_embds = [[] for _ in range(num_batch)]

        if not valid_indices:
            if extract_embeddings:
                return batch_results, batch_p_embds, batch_g_embds
            return batch_results

        flat_items_f = [flat_items[i] for i in valid_indices]
        ph_seqs_f = [phoneme_sequences[i] for i in valid_indices]
        grp_seqs_f = [group_sequences[i] for i in valid_indices]
        ts_outs_f = [ts_outs[i] for i in valid_indices]

        wavs_and_lens = [
            self.chop_wav(clip_wav,
                          int(seg["start"] * self.resampler_sample_rate),
                          int(seg["end"] * self.resampler_sample_rate))
            for _, seg, clip_wav in flat_items_f
        ]
        wavs, wav_lens, error_codes = zip(*wavs_and_lens)

        valid_indices = [i for i, code in enumerate(error_codes) if code == 0]
        if len(valid_indices) < len(flat_items_f):
            for i, code in enumerate(error_codes):
                if code != 0:
                    bi, seg, _ = flat_items_f[i]
                    self._warn(f"skipping clip {bi}, segment {seg['start']}-{seg['end']}s "
                               f"due to a chopping error ({code})")
            flat_items_f = [flat_items_f[i] for i in valid_indices]
            ph_seqs_f = [ph_seqs_f[i] for i in valid_indices]
            grp_seqs_f = [grp_seqs_f[i] for i in valid_indices]
            ts_outs_f = [ts_outs_f[i] for i in valid_indices]
            wavs = [wavs[i] for i in valid_indices]
            wav_lens = [wav_lens[i] for i in valid_indices]

        if not wavs:
            raise ValueError("All segments have audio chopping errors. "
                             "Cannot proceed with timestamp extraction.")

        start_times = [seg["start"] for _, seg, _ in flat_items_f]
        results = [None] * len(flat_items_f)
        pooled_emb_p_list = [None] * len(flat_items_f)
        pooled_emb_g_list = [None] * len(flat_items_f)

        # --- route each segment to the path its length calls for --------------------------
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
                    extract_embeddings=extract_embeddings, do_groups=do_groups, debug=debug)
            except ValueError as ex:
                self._warn(f"alignment failed for clip {bi}, segment "
                           f"{seg['start']:.2f}-{seg['end']:.2f}s: {ex}")
                res, pe, ge = {'phoneme_timestamps': [], 'group_timestamps': []}, None, None
            results[i], pooled_emb_p_list[i], pooled_emb_g_list[i] = res, pe, ge

        for i in long_ids:
            bi, seg, clip_wav = flat_items_f[i]
            self._info(f"Clip {bi}: segment {seg['start']:.2f}-{seg['end']:.2f}s with "
                       f"{len(ph_seqs_f[i])} phonemes -> chunked alignment "
                       f"({self.chunk_seconds:g}s windows)")
            try:
                res, pe, ge = self.align_segment_chunked(
                    clip_wav, float(seg["start"]), float(seg["end"]), ph_seqs_f[i],
                    group_sequence=grp_seqs_f[i] if do_groups else None,
                    extract_embeddings=extract_embeddings, do_groups=do_groups, debug=debug)
            except ValueError as ex:
                self._warn(f"chunked alignment failed for clip {bi}, segment "
                           f"{seg['start']:.2f}-{seg['end']:.2f}s: {ex}")
                res, pe, ge = {'phoneme_timestamps': [], 'group_timestamps': []}, None, None
            results[i], pooled_emb_p_list[i], pooled_emb_g_list[i] = res, pe, ge

        # --- the rest go through the batched single-shot path -----------------------------
        for start in range(0, len(short_ids), max(1, batch_size)):
            ids = short_ids[start:start + max(1, batch_size)]
            r_slice, p_slice, g_slice = self._align_slice(
                ids, wavs, wav_lens, ph_seqs_f, grp_seqs_f, start_times, flat_items_f,
                extract_embeddings, do_groups, debug)
            for k, j in enumerate(ids):
                results[j] = r_slice[k]
                pooled_emb_p_list[j] = p_slice[k]
                pooled_emb_g_list[j] = g_slice[k]

        for idx, ((bi, seg, _), result, ts) in enumerate(zip(flat_items_f, results, ts_outs_f)):
            processed = self.post_process_segment(
                seg, ts, seg[self.phonemes_key], result["phoneme_timestamps"],
                result["group_timestamps"] if do_groups else None, debug=debug)
            batch_results[bi]["segments"].append(processed)

            if extract_embeddings:
                if pooled_emb_p_list[idx] is not None:
                    batch_p_embds[bi].append(pooled_emb_p_list[idx].detach().cpu())
                if do_groups and pooled_emb_g_list[idx] is not None:
                    batch_g_embds[bi].append(pooled_emb_g_list[idx].detach().cpu())

        self._report_confidence(batch_results, debug=debug)

        if debug:
            total_segments = sum(len(item["segments"]) for item in batch_results)
            print(f"\n{'=' * 60}\nPROCESSING SUMMARY\n{'=' * 60}")
            print(f"Clip items: {num_batch}, Segments processed: {total_segments}")
            print(f"Totals, Segments processed: {self.total_segments_processed}",
                  f"\tSegments bad: {self.total_segments_bad}",
                  f"\tPhonemes aligned: {self.total_phonemes_aligned}",
                  f"\tTarget phonemes: {self.total_phonemes_target}",
                  f"\tPhonemes missed: {self.total_phonemes_missed}")
            print(f"{'=' * 60}")

        if extract_embeddings:
            return batch_results, batch_p_embds, batch_g_embds
        return batch_results

    def _align_slice(self, ids, wavs, wav_lens, ph_seqs_f, grp_seqs_f, start_times,
                     flat_items_f, extract_embeddings, do_groups, debug):
        """
        One batched forward+decode over `ids`, falling back to one segment at a time if any
        of them cannot be aligned.

        The fallback matters because the decode raises for the WHOLE batch as soon as one
        segment's text outruns its audio; retrying individually costs one extra pass over a
        handful of segments and keeps the rest of the slice.
        """
        def run(subset):
            return self.extract_timestamps_from_segment_batch(
                torch.stack([wavs[j] for j in subset], dim=0),
                [wav_lens[j] for j in subset],
                [ph_seqs_f[j] for j in subset],
                start_offset_times=[start_times[j] for j in subset],
                group_sequences=[grp_seqs_f[j] for j in subset] if do_groups else None,
                extract_embeddings=extract_embeddings, do_groups=do_groups, debug=debug)

        try:
            return run(ids)
        except ValueError as ex:
            if len(ids) == 1:
                bi, seg, _ = flat_items_f[ids[0]]
                self._warn(f"could not align clip {bi}, segment '{seg['text'][:40]}' "
                           f"({wav_lens[ids[0]] / self.resampler_sample_rate:.2f}s, "
                           f"{len(ph_seqs_f[ids[0]])} phonemes): {ex}")
                return ([{'phoneme_timestamps': [], 'group_timestamps': []}], [None], [None])
            self._info(f"  a segment in this batch of {len(ids)} could not be aligned "
                       f"({ex}); retrying them one at a time")

        r_out, p_out, g_out = [], [], []
        for j in ids:
            r, p, g = self._align_slice([j], wavs, wav_lens, ph_seqs_f, grp_seqs_f,
                                        start_times, flat_items_f, extract_embeddings,
                                        do_groups, debug)
            r_out += r
            p_out += p
            g_out += g
        return r_out, p_out, g_out

    def _report_confidence(self, batch_results, debug=False):
        """
        Flag segments whose alignment does not look trustworthy.

        Two patterns, both written into `coverage_analysis["bad_alignment"]`:
          - too many low-confidence phonemes overall (`bad_confidence_threshold`);
          - a healthy head and a collapsed tail, which is the signature of an alignment that
            starts on the right audio and then runs away from it.
        """
        for bi, batch_item in enumerate(batch_results):
            for si, seg_out in enumerate(batch_item["segments"]):
                self.total_segments_processed += 1
                phoneme_ts = seg_out.get("phoneme_ts")
                if not phoneme_ts:
                    self.total_segments_failed += 1
                    continue

                predicted_sequence = [t["phoneme_id"] for t in phoneme_ts]
                if predicted_sequence == seg_out.get(self.phonemes_key):
                    self.perfect_matches += 1

                if len(phoneme_ts) <= 60:
                    continue

                confidences = [t["confidence"] for t in phoneme_ts]
                avg_confidence = sum(confidences) / len(confidences)
                low = sum(1 for c in confidences if c < self.low_confidence_level)
                low_ratio = low / len(confidences)

                # `si` counts SURVIVING segments, so it does not index the caller's input once
                # anything has been skipped. The time range does identify the segment, so both
                # warnings carry it.
                seg_where = (f"Clip {bi}, segment {si + 1} "
                             f"({float(seg_out.get('start', 0)):.2f}-"
                             f"{float(seg_out.get('end', 0)):.2f}s)")

                flagged = False
                if low_ratio > self.bad_confidence_threshold:
                    self._info(f"  WARNING: {seg_where} has too many phonemes below "
                               f"confidence {self.low_confidence_level:g}: {low_ratio:.2%} "
                               f"({low}/{len(confidences)}) avg={avg_confidence:.3f}")
                    flagged = True

                head = sum(confidences[10:30]) / 20
                tail = sum(confidences[-30:-10]) / 20
                if head > 0.1 and tail < 0.1:
                    seg_dur = float(seg_out.get("end", 0)) - float(seg_out.get("start", 0))
                    n_ph = len(seg_out.get(self.phonemes_key) or phoneme_ts)
                    pps = (n_ph / seg_dur) if seg_dur > 0 else 0.0
                    dense = (f" Note {n_ph} phonemes in {seg_dur:.1f}s is {pps:.1f}/s, above "
                             f"the {self.chunk_phonemes_per_second:g}/s this aligner assumes "
                             f"speech runs at; if the audio is not unusually fast, check the "
                             f"transcript for text that is not spoken in this segment."
                             if pps > self.chunk_phonemes_per_second else "")
                    self._warn(f"Bad confidence pattern in {seg_where}: first 20 avg "
                               f"{head:.3f} vs last 20 avg {tail:.3f}. The alignment starts on "
                               f"the right audio and then loses it.{dense}")
                    flagged = True

                if flagged:
                    seg_out["coverage_analysis"]["bad_alignment"] = True
                    self.total_segments_bad += 1

    def process_srt_file(self, srt_path, audio_path, ts_out_path=None, extract_embeddings=False,
                         vspt_path=None, do_groups=False, batch_size=16, debug=False):
        """
        Align every segment of an SRT-style JSON transcript against an audio file.

        Args:
            srt_path: input JSON, e.g. `{"segments": [{"start": 0.0, "end": 9.2, "text": ""}]}`
            audio_path: audio file to align against
            ts_out_path: where to write the output JSON
            extract_embeddings: also pool per-phoneme embeddings
            vspt_path: where to save those embeddings (.pt)
            do_groups: also emit group timestamps
            batch_size: segments per forward pass
            debug: print detailed progress
        """
        try:
            if not os.path.exists(srt_path):
                self._warn(f"SRT file does not exist: {srt_path}")
                return None
            with open(srt_path, "r") as file:
                srt_data = json.load(file)
            if "segments" not in srt_data:
                raise ValueError("SRT data does not contain 'segments' key")
            if len(srt_data["segments"]) == 0:
                self._warn(f"SRT file is empty or has no segments: {srt_path}")
                return None
            first = srt_data["segments"][0]
            if not all(k in first for k in ("text", "start", "end")):
                self._warn(f"SRT segments do not contain the required keys: {srt_path}")
                return None
            self._info(f"Loaded SRT file with {len(srt_data['segments'])} segments "
                       f"from {srt_path}")
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"Error reading SRT file {srt_path}: {e}") from e

        audio_wav = self.load_audio(audio_path)
        result = self.process_segments([{"segments": srt_data["segments"]}], [audio_wav],
                                       extract_embeddings=extract_embeddings,
                                       do_groups=do_groups, batch_size=batch_size, debug=debug)

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
                self._info(f"Embeddings saved to: {vspt_path}")
            self._info(f"Results saved to: {ts_out_path}")

        return vs_out_data

    def process_sentence(self, text, audio_wav, extract_embeddings=False, do_groups=False,
                         debug=False):
        """
        Align one sentence against one waveform.

        Args:
            text: the sentence.
            audio_wav: waveform tensor at the aligner's sample rate.
            extract_embeddings: also pool per-phoneme embeddings.
            do_groups: also emit group timestamps.
            debug: print detailed progress.

        Returns:
            The output dict, or `(output_dict, phoneme_embeddings, group_embeddings)` when
            `extract_embeddings` is set.
        """
        duration = audio_wav.shape[1] / self.sample_rate
        srt_data = [{"segments": [{"start": 0.0, "end": duration, "text": text.strip()}]}]
        result = self.process_segments(srt_data, [audio_wav],
                                       extract_embeddings=extract_embeddings,
                                       do_groups=do_groups, debug=debug)
        if extract_embeddings:
            timestamps_output_dict, batch_p_embds, batch_g_embds = result
            return timestamps_output_dict[0], batch_p_embds[0], batch_g_embds[0]
        return result[0]

    def process_sentences_batch(self, texts, audio_wavs, extract_embeddings=False,
                                do_groups=False, debug=False):
        """
        Align a batch of sentences against their waveforms.

        Returns a list of output dicts, or `(list, phoneme_embeddings, group_embeddings)` when
        `extract_embeddings` is set.
        """
        assert len(texts) == len(audio_wavs), \
            f"Number of texts ({len(texts)}) must match number of waveforms ({len(audio_wavs)})"
        srt_data = []
        for text, audio_wav in zip(texts, audio_wavs):
            duration = audio_wav.shape[1] / self.sample_rate
            srt_data.append({"segments": [{"start": 0.0, "end": duration,
                                           "text": text.strip()}]})
        return self.process_segments(srt_data, audio_wavs,
                                     extract_embeddings=extract_embeddings,
                                     do_groups=do_groups, debug=debug)

    # ======================================================================================
    # Export helpers
    # ======================================================================================

    def convert_to_textgrid(self, timestamps_dict, output_file=None, include_confidence=False):
        """Convert an output dict to Praat TextGrid format."""
        textgrid_content = dict_to_textgrid(timestamps_dict, output_file=None,
                                            include_confidence=include_confidence)
        if output_file and textgrid_content is not None:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(textgrid_content)
        return textgrid_content

    def plot_mel_alignment(self, timestamps_dict, audio_wav, save_path,
                           segment_idx=0, vocoder_config=None):
        """
        Generate and save a mel spectrogram with phoneme boundary markers.

        Args:
            timestamps_dict (dict): output from `process_sentence` / `process_srt_file`.
            audio_wav (Tensor): waveform of shape (1, T) at `resampler_sample_rate`.
            save_path (str): destination. `.pt` saves the raw mel tensor instead of a plot.
            segment_idx (int): which segment to plot.
            vocoder_config (dict | None): forwarded to `extract_mel_spectrum`.

        Returns:
            str: the path the file was saved to.
        """
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
        seg_end = segment.get('end', audio_wav.shape[1] / self.resampler_sample_rate)
        phoneme_ts = segment.get('phoneme_ts', [])

        s = int(seg_start * self.resampler_sample_rate)
        e = int(seg_end * self.resampler_sample_rate)
        mel = self.extract_mel_spectrum(audio_wav[:, s:e], self.resampler_sample_rate,
                                        vocoder_config)

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        if save_path.lower().endswith('.pt'):
            torch.save(mel, save_path)
            return save_path

        text_snippet = segment.get('text', '')
        _plot_mel_alignment(mel=mel, phoneme_ts=phoneme_ts, save_path=save_path,
                            segment_start_ms=seg_start * 1000.0,
                            title=f'"{text_snippet}"' if text_snippet else None)
        return save_path

    def analyze_alignment_coverage(self, target_sequence, aligned_timestamps, index_to_label):
        """
        Analyse how well the alignment covers the target sequence.

        Args:
            target_sequence: the target phoneme sequence
            aligned_timestamps: the aligned framestamps
            index_to_label: mapping from indices to phoneme labels

        Returns:
            dict: coverage counts, ratios and the labels that went missing.
        """
        target_set = set(target_sequence.tolist() if hasattr(target_sequence, 'tolist')
                         else target_sequence)
        aligned_set = set([ts[0] for ts in aligned_timestamps])

        missing = target_set - aligned_set
        extra = aligned_set - target_set
        coverage = len(target_set - missing) / len(target_set) if target_set else 1.0

        return {
            'target_count': len(target_set),
            'aligned_count': len(aligned_set),
            'missing_count': len(missing),
            'extra_count': len(extra),
            'coverage_ratio': coverage,
            'missing_phonemes': [index_to_label.get(p, f'UNK_{p}') for p in missing],
            'extra_phonemes': [index_to_label.get(p, f'UNK_{p}') for p in extra],
            'bad_alignment': coverage < 0.8,
        }

    def extract_mel_spectrum(self, wav, wav_sample_rate,
                             vocoder_config={'num_mels': 80, 'num_freq': 1025, 'n_fft': 1024,
                                             'hop_size': 256, 'win_size': 1024,
                                             'sampling_rate': 22050, 'fmin': 0, 'fmax': 8000,
                                             'model': 'nvidia/bigvgan_v2_22khz_80band_fmax8k_256x'}):
        '''
        Mel spectrogram compatible with the HiFiGAN/BigVGAN vocoders, so it can be inverted
        back to audio. See https://github.com/NVIDIA/BigVGAN

        Args:
            wav: input waveform of shape (1, T)
            wav_sample_rate: its sample rate
            vocoder_config: the vocoder's mel configuration

        Returns:
            mel: [T, num_mels]
        '''
        from .helpers.mel_spec import mel_spectrogram

        assert (wav.dim() == 2) and (wav.shape[0] == 1), \
            f"Expected input shape (1, T), got {tuple(wav.shape)}"

        if wav_sample_rate != vocoder_config['sampling_rate']:
            self._info(f"extract_mel_spectrum: resampling {wav_sample_rate} -> "
                       f"{vocoder_config['sampling_rate']} for vocoder compatibility")
            import librosa
            wav = torch.from_numpy(librosa.resample(
                wav.numpy(), orig_sr=wav_sample_rate,
                target_sr=vocoder_config['sampling_rate']))

        mel = mel_spectrogram(wav, n_fft=vocoder_config['n_fft'],
                              num_mels=vocoder_config['num_mels'],
                              sampling_rate=vocoder_config['sampling_rate'],
                              hop_size=vocoder_config['hop_size'],
                              win_size=vocoder_config['win_size'],
                              fmin=vocoder_config['fmin'], fmax=vocoder_config['fmax'])

        if mel.dim() == 3 and mel.shape[0] == 1 and mel.shape[1] == vocoder_config['num_mels']:
            mel = mel.squeeze(0)
        else:
            raise ValueError(f"Unexpected mel shape: {mel.shape}, expected "
                             f"[1, mel_dim, mel_len]")
        return mel.transpose(0, 1)          # (T, mel_dim)

    def ceil(self, float_value):
        return int(float_value) + (float_value % 1 > 0)

    def compress_frames(self, frames_list):
        '''
        Run-length encode a frame-wise label list:
        `[0,0,0,0,1,1,1,1,3]` -> `[(0,4),(1,4),(3,1)]`, each tuple `(value, count)`.
        '''
        if not frames_list:
            return []
        compressed = []
        current_value = frames_list[0]
        current_count = 1
        for value in frames_list[1:]:
            if value == current_value:
                current_count += 1
            else:
                compressed.append((current_value, current_count))
                current_value, current_count = value, 1
        compressed.append((current_value, current_count))
        return compressed

    def decompress_frames(self, compressed_frames):
        """Inverse of `compress_frames`."""
        decompressed = []
        for phn_id, count in compressed_frames:
            decompressed.extend([phn_id] * count)
        return decompressed

    def framewise_assortment(self, aligned_ts, total_frames, frames_per_second,
                             gap_contraction=5, select_key="phoneme_id", offset_ms=0):
        """
        Spread timestamped labels over a frame grid of the caller's choosing - e.g. a mel
        spectrogram's hop grid, which is not the model's 5 ms grid.

        Args:
            aligned_ts: a segment's `phoneme_ts`, `group_ts` or `words_ts` list.
            total_frames: frames in the target grid.
            frames_per_second: that grid's frame rate.
            gap_contraction: how many frames of an unlabelled gap to fill from each side.
            select_key: which field to write into each frame, e.g. "phoneme_id", "ipa_label",
                "group_id".
            offset_ms: subtracted from every timestamp. Pass the segment's own start when the
                grid begins at the segment rather than at the clip.
        """
        gap_fill_frame_value = -1       # sentinel: must not collide with any valid id (SIL=0)
        ms_per_frame = 1000.0 / frames_per_second

        aligned_ts.sort(key=lambda x: x["start_ms"])
        framewise_label = [gap_fill_frame_value] * total_frames

        # Stage 1: fill frames from the exact timestamp boundaries.
        for ts_item in aligned_ts:
            start_index = max(int((ts_item["start_ms"] - offset_ms) / ms_per_frame), 0)
            end_index = min(self.ceil((ts_item["end_ms"] - offset_ms) / ms_per_frame),
                            total_frames)

            if end_index - start_index > total_frames:
                self._info(f"Warning: timestamp start frame {start_index} exceeds total "
                           f"frames {total_frames}, skipping.")
                continue
            elif end_index > total_frames:
                self._info(f"Warning: timestamp end frame {end_index} exceeds total frames "
                           f"{total_frames}, clamping.")
                end_index = total_frames

            if select_key not in ts_item:
                raise ValueError(f"select_key '{select_key}' not found in timestamp item",
                                 ts_item)
            for frame_i in range(max(start_index - 1, 0), min(end_index + 1, total_frames)):
                if framewise_label[frame_i] == gap_fill_frame_value:
                    framewise_label[frame_i] = ts_item[select_key]

        # Stage 2: contract the gaps that are left.
        gaps = []
        i = 0
        while i < total_frames:
            if framewise_label[i] == gap_fill_frame_value:
                gap_start = i
                while i < total_frames and framewise_label[i] == gap_fill_frame_value:
                    i += 1
                gaps.append((gap_start, i))
            else:
                i += 1

        for gap_start, gap_end in gaps:
            if not (gap_start > 0 and gap_end <= total_frames):
                continue
            left_value = framewise_label[gap_start - 1]
            right_value = (framewise_label[gap_end] if gap_end < total_frames
                           else gap_fill_frame_value)
            if left_value == gap_fill_frame_value or not (
                    right_value != gap_fill_frame_value or gap_end == total_frames):
                continue

            gap_size = gap_end - gap_start
            if gap_size <= gap_contraction:
                # Small gap: fill it entirely from the left.
                for j in range(gap_start, gap_end):
                    framewise_label[j] = left_value
            elif gap_size <= gap_contraction * 2:
                # Medium gap: meet in the middle.
                midpoint = gap_start + gap_size // 2
                for j in range(gap_start, gap_end):
                    framewise_label[j] = left_value if j < midpoint else right_value
            else:
                # Large gap: contract `gap_contraction` frames from each side and leave the
                # rest unlabelled.
                for j in range(gap_start, min(gap_start + gap_contraction, gap_end)):
                    framewise_label[j] = left_value
                for j in range(gap_end - 1, max(gap_end - gap_contraction - 1, gap_start - 1),
                               -1):
                    framewise_label[j] = right_value

        return framewise_label
