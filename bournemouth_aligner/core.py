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
import os
import json
import time
import traceback

# repo modules
from .cupe2i.model2i import CUPEEmbeddingsExtractor
from .cupe2i.windowing import slice_windows, stich_window_predictions, calc_spec_len_ext
from .forced_alignment import AlignmentUtils 
#from .ipamappers.ph66_mapper import phoneme_mapped_index, phoneme_groups_index, phoneme_groups_mapper
from .ipamappers import ph66_phonemeizer
from .presets import get_preset
from .utils import dict_to_textgrid, weighted_pool_embeddings, _calculate_confidences, convert_to_ms, detect_device_automatically
# Create reverse mappings for interpretability
#index_to_glabel = {v: k for k, v in phoneme_groups_index.items()}
#index_to_plabel = {v: k for k, v in phoneme_mapped_index.items()}



class PhonemeTimestampAligner:
    """
    Align phoneme-level timestamps from audio using a pre-trained CUPE model
    and Viterbi decoding for forced alignment.
    URL: https://github.com/tabahi/bournemouth-forced-aligner
    """

    def __init__(self, preset="en-us", model_name=None, cupe_ckpt_path=None, lang='en-us', mapper="ph66", duration_max=10, device="auto", silence_anchors=0, boost_targets=True, enforce_minimum=True, enforce_all_targets=True, ignore_noise=True, extend_soft_boundaries=True, boundary_softness=7, bad_confidence_threshold=0.6):
        """
        Initialize the BFA phoneme timestamp extractor.

        Args:
            preset (str): Language preset for automatic model and language selection.
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
            enforce_all_targets (bool): Force all target phonemes to appear in alignment.
            ignore_noise (bool): Skip predicted noise in alignment output.
            extend_soft_boundaries (bool): Enable the extension of the phoeneme start/end boundaries beyond the core of the phoneme. Use `boundary_softness` to control the leniency of the extension. This can help capture more of the phoneme duration, especially for softer phonemes or in cases where the model's confidence extends beyond the core frames.
            boundary_softness (int): Hyperparameter controlling leniency of boundary extension beyond the core of the phoneme. Default is 7, which corresponds to a threshold of 0.0000001. Set it to 2 or 3 if you want only the cores of the phonemes, or set it to 7 to allow more extension as long as there's any meaningful confidence in the frames between the core and the extended boundary.
            bad_confidence_threshold (float): Threshold for flagging low-confidence alignments (1 to disable), default 0.6,  0.6 would mean if 60% of phonemes have low confidence, a warning is issued, in "segments", ["coverage_analysis"]["bad_confidence"] is set to true. It's recommended to avoid bad-confidence segments for the downstream tasks.

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
        if device == "auto":
            device = detect_device_automatically()
        self.device = device

        # Parameter priority handling:
        # 1st priority: Explicit cupe_ckpt_path (highest)
        # 2nd priority: Explicit model_name
        # 3rd priority: Preset values (only if no explicit model specified)
        # 4th priority: Default values

        # Only apply preset logic if no explicit checkpoint path or model name provided
        if cupe_ckpt_path is None and model_name is None:
            # Language preset mappings based on available models
            model_name, lang = get_preset(preset=preset, lang=lang)
        if cupe_ckpt_path is not None:
            pass  # use explicit path as-is
        elif model_name is not None:
            cupe_ckpt_path = self.download_model(model_name=model_name)
        else:
            raise ValueError("Either cupe_ckpt_path or model_name must be provided.")
        if cupe_ckpt_path is None or not os.path.exists(cupe_ckpt_path):
            raise ValueError("CUPE model checkpoint not found.", cupe_ckpt_path)
        self.extractor = CUPEEmbeddingsExtractor(cupe_ckpt_path, device=self.device)
        
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
            self.phonemizer = ph66_phonemeizer.Phonemizer(language=lang,remove_noise_phonemes=True)

        self.phonemes_key = self.phonemizer.phonemes_key
        self.phoneme_groups_key = self.phonemizer.phoneme_groups_key

        self.phoneme_id_to_label = self.phonemizer.index_to_plabel
        self.phoneme_label_to_id = {label: idx for idx, label in self.phoneme_id_to_label.items()}
        self.group_id_to_label = self.phonemizer.index_to_glabel
        self.group_label_to_id = {label: idx for idx, label in self.group_id_to_label.items()}
        self.phoneme_id_to_group_id = self.phonemizer.phoneme_id_to_group_id


        self.silence_anchors = silence_anchors
        self.boost_targets = boost_targets
        self.enforce_minimum = enforce_minimum
        self.enforce_all_targets = enforce_all_targets
        self.ignore_noise = ignore_noise
        self.extend_soft_boundaries = extend_soft_boundaries
        self.boundary_softness = boundary_softness
        self.bad_confidence_threshold = bad_confidence_threshold
        self.break_at_low_confidence = False # under construction, not fully implemented yet
        self._setup_config()
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

    def reset_counters(self):
        """Reset internal counters for tracking alignment statistics. Purely for debugging and analysis purposes."""
        self.total_segments_processed = 0
        self.total_segments_bad = 0
        self.total_phonemes_aligned = 0
        self.total_phonemes_target = 0
        self.total_phonemes_aligned_easily = 0
        self.total_phonemes_missed = 0
        self.total_phonemes_extra = 0
        self.perfect_matches = 0

    def _setup_config(self, window_size_ms=120, stride_ms=80):
        """Setup configuration parameters."""
        self.window_size_ms = window_size_ms
        self.stride_ms = stride_ms
        self.sample_rate = 16000
        
        # Calculate window parameters
        self.window_size_wav = int(window_size_ms * self.sample_rate / 1000)
        self.stride_size_wav = int(stride_ms * self.sample_rate / 1000)
        self.frames_per_window = self.extractor.model.update_frames_per_window(self.window_size_wav)
        
        # Phoneme class configuration
        self.phoneme_classes = len(self.phoneme_id_to_label)-1  # exclude padding 'noise' = 66
        self.phoneme_groups = len(self.group_id_to_label)-1 # exclude padding 'noise' = 16  
        self.blank_class = self.phoneme_label_to_id['noise']
        self.blank_group = self.group_label_to_id['noise']
        self.silence_class = self.phoneme_label_to_id['SIL']
        self.silence_group = self.group_label_to_id['SIL']
        # print all these:
        #print(f"Phoneme classes: {self.phoneme_classes}, groups: {self.phoneme_groups}, blank_class: {self.blank_class}, blank_group: {self.blank_group}, silence_class: {self.silence_class}, silence_group: {self.silence_group}")

    def _setup_decoders(self):
        """Setup Viterbi decoders for phoneme classes and groups."""
        
        # Alignment utilities
        self.alignment_utils_g = AlignmentUtils(blank_id=self.blank_group, silence_id=self.silence_group, silence_anchors=self.silence_anchors, ignore_noise=self.ignore_noise)
        self.alignment_utils_p = AlignmentUtils(blank_id=self.blank_class, silence_id=self.silence_class, silence_anchors=self.silence_anchors, ignore_noise=self.ignore_noise)


    def download_model(self, model_name="en_libri1000_ua01c_e4_val_GER=0.2186.ckpt", model_dir="./models"):
        """Download the specified model from hugging face using huggingface_hub."""
        from huggingface_hub import hf_hub_download
        
        repo_id = "Tabahi/CUPE-2i"
        filename = f"ckpt/{model_name}"
        
        try:
            # Just return the cached path directly
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_files_only=False
            )
            print(f"Model available at: {model_path}")
            return model_path
            
        except Exception as e:
            raise Exception(f"Error downloading model: {e}")

    @torch.no_grad()
    def chop_wav(self, wav, start_frame, end_frame):
        """chop down audio segment."""
        num_frames = (end_frame - start_frame) if (end_frame != -1) else -1

        if (num_frames < self.seg_duration_min_samples):
            raise ValueError(f"Segment too short: {num_frames} frames, minimum required is {self.seg_duration_min_samples} frames.")
         
        wav = wav[:, start_frame:end_frame]
            
        assert (wav.shape[1] <= num_frames) or (num_frames == -1)
        if wav.shape[1] < self.seg_duration_min_samples:
            raise Exception("Wav shape is too small:", wav.shape, start_frame, end_frame)
        
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
        
        return wav, wav_len
    
    @staticmethod
    def _rms_normalize(audio): # it's important -- CUPE is trained with RMS normalized audio
        """RMS normalize audio tensor."""
        rms = torch.sqrt(torch.mean(audio ** 2))
        if rms > 0:
            audio = audio / rms
        return audio
    



    def load_audio(self, audio, backend = "ffmpeg", sr=None):
        """Load and preprocess audio file."""
        
        if isinstance(audio, str):
            audio_path = audio
            wav, sr = torchaudio.load(audio_path,  frame_offset=0,  normalize=True, backend=backend)

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


    def _cupe_prediction(self, audio_batch, wav_len, extract_embeddings=False):
        """
        Process audio through the CUPE model to get logits.
        
        Args:
            audio_batch: Audio tensor [1, samples]
            wav_len: Length of audio in samples
            extract_embeddings: Whether to extract embeddings
            
        Returns:
            Tuple of (logits_class, logits_group, embeddings, spectral_len)
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
        
        # Get predictions
        if extract_embeddings:
            logits_class, logits_group, embeddings = self.extractor.predict(
                windows_flat, 
                return_embeddings=True, 
                groups_only=False
            )
        else:
            logits_class, logits_group = self.extractor.predict(
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
            stride_ms=self.stride_ms
        )
        
        logits_group = stich_window_predictions(
            logits_group, 
            original_audio_length=original_audio_length,
            cnn_output_size=frames_per_window, 
            sample_rate=self.sample_rate, 
            window_size_ms=self.window_size_ms, 
            stride_ms=self.stride_ms
        )
        
        if extract_embeddings and embeddings is not None:
            embeddings = embeddings.reshape(batch_size, num_windows, frames_per_window, -1)
            embeddings = stich_window_predictions(
                embeddings, 
                original_audio_length=original_audio_length,
                cnn_output_size=frames_per_window, 
                sample_rate=self.sample_rate, 
                window_size_ms=self.window_size_ms, 
                stride_ms=self.stride_ms
            )
        
        # Calculate spectral length
        spectral_len = calc_spec_len_ext(
            torch.tensor([wav_len]), 
            self.window_size_ms, 
            self.stride_ms, 
            self.sample_rate, 
            torch.tensor(frames_per_window, dtype=torch.long)
        )[0].item()
        
        return logits_class, logits_group, embeddings, spectral_len
    
    
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
        
        # Get predictions
        if extract_embeddings:
            logits_class, logits_group, embeddings = self.extractor.predict(
                windows_flat, 
                return_embeddings=True, 
                groups_only=False
            )
        else:
            logits_class, logits_group = self.extractor.predict(
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
            stride_ms=self.stride_ms
        )
        
        logits_group = stich_window_predictions(
            logits_group, 
            original_audio_length=original_audio_length,
            cnn_output_size=frames_per_window, 
            sample_rate=self.sample_rate, 
            window_size_ms=self.window_size_ms, 
            stride_ms=self.stride_ms
        )
        
        if extract_embeddings and embeddings is not None:
            embeddings = embeddings.reshape(batch_size, num_windows, frames_per_window, -1)
            embeddings = stich_window_predictions(
                embeddings, 
                original_audio_length=original_audio_length,
                cnn_output_size=frames_per_window, 
                sample_rate=self.sample_rate, 
                window_size_ms=self.window_size_ms, 
                stride_ms=self.stride_ms
            )
        
        # Calculate spectral length
        spectral_lens = calc_spec_len_ext(
            torch.tensor(wav_lens), 
            self.window_size_ms, 
            self.stride_ms, 
            self.sample_rate, 
            torch.tensor(frames_per_window, dtype=torch.long)
        ).tolist()

        return logits_class, logits_group, embeddings, spectral_lens
    
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
                if debug: print(f"WARNING: Repeated target phonemes found at positions {sorted(repeated_targets)}: {repeated_labels}")
                self.total_phonemes_extra += sum(targets_found[idx] - 1 for idx in repeated_targets)

            missing_targets = [idx for idx, count in enumerate(targets_found) if count == 0]
            if missing_targets:
                missing_labels = [self.phonemizer.index_to_plabel.get(target_phoneme_ids[idx], f'UNK_{target_phoneme_ids[idx]}') for idx in missing_targets]
                if debug: print(f"WARNING: {len(missing_targets)} target phonemes not found in alignment at positions {missing_targets}: {missing_labels}")
                self.total_phonemes_missed += len(missing_targets)  

            # Remove aligned phonemes with invalid target indices
            if invalid_target_indices:
                if debug: print(f"WARNING: Found {len(invalid_target_indices)} aligned phonemes with invalid target indices in batch {batch_idx}.")
                aligned_frames[batch_idx] = [frame for frame in aligned_frames[batch_idx] if int(frame[3]) not in invalid_target_indices]
                if debug: print(f"Removed aligned phonemes with invalid target indices {invalid_target_indices} from batch {batch_idx}.")

            # Deduplicate repeated target phonemes: merge consecutive, else keep longest
            if repeated_targets and self.enforce_all_targets:
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
            if missing_targets and self.enforce_all_targets:
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

            if self.enforce_all_targets:
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
    

    def extend_soft_boundaries_func(self, log_probs, framestamps, boundary_softness=7, debug=False):
        '''
        Extend start and end boundaries of aligned phonemes based on confidence scores from log probabilities.
        Three passes:
          1. Extend start boundaries with strict thresholds
          2. Extend end boundaries with lenient thresholds
          3. Extend start boundaries again with even more lenient thresholds (to fill remaining gaps)

        Args:
            log_probs: Tensor of shape [B, T, C] containing log probabilities.
            framestamps: List of lists of tuples (phoneme_id, start_frame, end_frame, target_seq_idx, is_estimated).
            boundary_softness: Hyperparameter controlling leniency of boundary extension. Higher means more lenient (allows extension with lower confidence). Default is 7, which corresponds to a threshold of 0.0000001.

        Returns:
            Updated framestamps with extended boundaries (same structure as input).
        '''
        max_extension_factor = 10.0
        boundary_softness_initial = max(2, 7 - 4)  # initial pass is stricter, then we relax it in subsequent passes
        _thresh_1 = 10.0 ** (-1*boundary_softness_initial)  # e.g. 0.00001 for boundary_softness_initial of 5
        _thresh_2 = 10.0 ** (-1*boundary_softness)  # e.g. 0.0000001 for boundary_softness of 7
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

    def extract_timestamps_from_segment_batch(self, wavs, wav_lens, phoneme_sequences, start_offset_times=0,
                                      group_sequences=None, extract_embeddings=True, do_groups=True,
                                      debug=True, ):
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


        logits_class, logits_group, embeddings, spectral_lens = self._cupe_prediction_batch(wavs, wav_lens, extract_embeddings)


        # Prepare sequences for alignment
        ph_seqs = phoneme_sequences.to(self.device)
        grp_seqs = group_sequences.to(self.device)
        ph_seq_lens = torch.tensor(ph_seq_lens, dtype=torch.long).to(self.device)
        spectral_lens = torch.tensor(spectral_lens, dtype=torch.long).to(self.device)
        
        # Get log probabilities
        log_probs_g = F.log_softmax(logits_group, dim=2)
        log_probs_p = F.log_softmax(logits_class, dim=2)


        frame_phonemes = self.alignment_utils_p.decode_alignments(
                    log_probs_p,
                    true_seqs=ph_seqs,
                    pred_lens=spectral_lens,
                    true_seqs_lens=ph_seq_lens,
                    forced_alignment=True,
                    boost_targets=self.boost_targets,
                    enforce_minimum=self.enforce_minimum,
                    debug=debug
                )

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

        if self.extend_soft_boundaries:
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
                self.resampler_sample_rate
            )
            frame_groups[b] = convert_to_ms(
                frame_groups[b],
                spectral_lens[b],
                start_offset_times[b] if isinstance(start_offset_times, (list, tuple)) else start_offset_times,
                wav_lens[b],
                self.resampler_sample_rate
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
                - ipa: list of phonemes in IPA format
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
    
    def process_segments(self, srt_data, audio_wavs, extract_embeddings=False, do_groups=False, debug=False):
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

        for (_, seg, _), ph_seq, grp_seq in zip(flat_items, phoneme_sequences, group_sequences):
            seg[self.phonemes_key] = ph_seq
            seg[self.phoneme_groups_key] = grp_seq

        # --- Filter sub-segments with insufficient phoneme sequences ---
        valid_indices = []
        for i, ((bi, seg, _), ph_seq) in enumerate(zip(flat_items, phoneme_sequences)):
            if not ph_seq or len(ph_seq) < self.ph_seq_min:
                if debug:
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
        wavs, wav_lens = zip(*wavs_and_lens)
        wavs = torch.stack(wavs, dim=0)
        start_times = [seg["start"] for _, seg, _ in flat_items_f]

        # --- Model inference (single batched call across all segments) ---
        results, pooled_emb_p_list, pooled_emb_g_list = self.extract_timestamps_from_segment_batch(
            wavs, wav_lens, ph_seqs_f,
            start_offset_times=start_times,
            group_sequences=grp_seqs_f if do_groups else None,
            extract_embeddings=extract_embeddings,
            do_groups=do_groups,
            debug=debug
        )

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

                    if low_confidence_ratio > self.bad_confidence_threshold:
                        print(f"  WARNING: Clip {bi}, segment {si+1} has too many low-confidence phonemes: {low_confidence_ratio:.2%} ({low_confidence_count}/{len(confidences)}) avg={avg_confidence:.3f}")
                        batch_results[bi]["segments"][si]["coverage_analysis"]["bad_alignment"] = True
                        self.total_segments_bad += 1
                    first_20_confidences = sum(confidences[10:30]) / 20
                    last_20_confidences = sum(confidences[-30:-10]) / 20

                    if first_20_confidences > 0.1 and last_20_confidences < 0.1:
                        if self.silence_anchors == 0:
                            raise Exception(f"Bad confidence pattern in clip {bi}, segment {si+1}: first 20 avg {first_20_confidences:.3f} vs last 20 avg {last_20_confidences:.3f}. Consider setting `silence_anchors=3`.")
                        else:
                            print(f"  WARNING: Bad confidence pattern in clip {bi}, segment {si+1}: first 20 avg {first_20_confidences:.3f} vs last 20 avg {last_20_confidences:.3f}. Consider adjusting `silence_anchors`.")
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


    def process_srt_file(self, srt_path, audio_path, ts_out_path=None, extract_embeddings=False, vspt_path=None, do_groups=False, debug=True):
        """
        Read sentences from SRT file and process them.
        Process entire srt file and generate vs2 output with timestamps.
        
        Args:
            srt_path: Path to input SRT file (JSON format), example: {"segments": [{"start": 0.0, "end": 9.23, "text": "text"}], }
            audio_path: Path to audio file  
            ts_out_path: Path to output vs2 file
            extract_embeddings: Whether to extract embeddings
            vspt_path: Path to save embeddings (.pt file)
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
        result = self.process_segments(srt_batch, [audio_wav], extract_embeddings=extract_embeddings, do_groups=do_groups, debug=debug)

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
            print ("Resampling waveform from {} to {} for vocoder compatibility".format(wav_sample_rate, vocoder_config['sampling_rate']))
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
                print(f"Warning: Timestamp start frame {framewise_start_index} exceeds total frames {total_frames}, skipping.")
                continue
            elif framewise_end_index > total_frames:
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
    
