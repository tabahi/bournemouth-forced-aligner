''' 2025-08-07
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
from .utils import dict_to_textgrid
# Create reverse mappings for interpretability
#index_to_glabel = {v: k for k, v in phoneme_groups_index.items()}
#index_to_plabel = {v: k for k, v in phoneme_mapped_index.items()}



class PhonemeTimestampAligner:
    """
    Align phoneme-level timestamps from audio using a pre-trained CUPE model
    and Viterbi decoding for forced alignment.
    URL: https://github.com/tabahi/bournemouth-forced-aligner
    """

    def __init__(self, model_name = "en_libri1000_uj01d_e199_val_GER=0.2307.ckpt", cupe_ckpt_path=None, lang='en-us', mapper="ph66", duration_max=10, output_frames_key="phoneme_idx", device="cpu", silence_anchors=10, boost_targets=True, enforce_minimum=True, enforce_all_targets=True, ignore_noise=True):
        """
        Initialize the phoneme timestamp extractor.
        
        Args:
            model_name: Name of the pre-trained model to use. Use the ckpt base filenames from:  https://huggingface.co/Tabahi/CUPE-2i/tree/main/ckpt
            cupe_ckpt_path: Path to the CUPE model checkpoint. Download from: https://huggingface.co/Tabahi/CUPE-2i/tree/main/ckpt
            lang: Language for phonemization (use espeak lang codes, see https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md)
            mapper: Phoneme mapper to use (default: "ph66")
            duration_max: Maximum duration in seconds (Padding purposes when using batch processing). You can set it to 10 to 60 seconds.
            ms_per_frame: Milliseconds per frame in the framewise assortment of phoneme labels. Set to `-1` to disable framewise assortment. Set to 1 to 80ms if your next task requires a specific frame rate. This does not effect the model or the alignment accuracy. See `framewise_assortment()`.
            output_frames_key: Set which of the ouputs to use to assort frames (using ms_per_frame). Options: "phoneme_idx"(default), "phoneme_label", "group_idx", "group_label"
            device: Device to run inference on
            silence_anchors: Number of silent frames to anchor pauses (i.e., split segments when at least `silence_anchors` frames are silent). Set `0` to disable. Default is `10`. Set a lower value to increase sensitivity to silences. Best set `enforce_all_targets=True` when using this.
            boost_targets: Boost the probabilities of target phonemes to ensure they can be aligned.
            enforce_minimum: Ensure target phonemes meet a minimum probability threshold in the predicted frames.
            enforce_all_targets: Whether to enforce all target phonemes to be present. Band-aid postprocessing patch: It will insert phonemes missed by viterbi decoding at their expected positions based on targets.
            ignore_noise: Whether to ignore the predicted "noise" in the alignment. If set to True, noise will be skipped over. If False, long noisy/silent segments will be included as "noise" timestamps.
        """
        self.device = device

        if cupe_ckpt_path is not None:
            cupe_ckpt_path = cupe_ckpt_path
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
        self.output_frames_key = output_frames_key
        
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
        self.alignment_utils_g = AlignmentUtils(blank_id=self.blank_group, silence_id=self.silence_class, silence_anchors=self.silence_anchors, ignore_noise=self.ignore_noise)
        self.alignment_utils_p = AlignmentUtils(blank_id=self.blank_class, silence_id=self.silence_group, silence_anchors=self.silence_anchors, ignore_noise=self.ignore_noise)


    def download_model(self, model_name="en_libri1000_uj01d_e199_val_GER=0.2307.ckpt", model_dir="./models"):
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
    



    def load_audio(self, audio_path, backend = "ffmpeg"):
        """Load and preprocess audio file."""
        
        wav, sr = torchaudio.load(audio_path,  frame_offset=0,  normalize=True, backend=backend)
        
        if sr != self.resampler_sample_rate:
            print(f"Resampling {audio_path} from {sr}Hz to {self.resampler_sample_rate}Hz")
            # load full
            wav, sr = torchaudio.load(audio_path, normalize=True)
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
    
    def extract_timestamps_from_segment(self, wav, wav_len, phoneme_sequence, start_offset_time=0,
                                      group_sequence=None, extract_embeddings=True, do_groups=True, 
                                      debug=True, ):
        """
        Extract phoneme and group timestamps from a single audio segment.

        Args:
            wav: Audio tensor for the segment.
            wav_len: Length of the audio segment in samples.
            phoneme_sequence: List or tensor of phoneme indices.
            start_offset_time: Start time offset in seconds for the segment.
            group_sequence: Optional list or tensor of phoneme group indices.
            extract_embeddings: Whether to extract pooled embeddings.
            do_groups: Whether to extract phoneme group timestamps.
            debug: Enable debug output.

        Returns:
            timestamp_dict: Dictionary with 'phoneme_timestamps' and 'group_timestamps'.
            pooled_embeddings_phonemes: Pooled phoneme embeddings or None.
            pooled_embeddings_groups: Pooled group embeddings or None.
        """
    

        # Convert sequences to tensors
        if not isinstance(phoneme_sequence, torch.Tensor):
            phoneme_sequence = torch.tensor(phoneme_sequence, dtype=torch.long)
        
        if group_sequence is not None and not isinstance(group_sequence, torch.Tensor):
            group_sequence = torch.tensor(group_sequence, dtype=torch.long)
        
        # Generate group sequence if not provided
        if group_sequence is None:
            group_sequence = self._map_phonemes_to_groups(phoneme_sequence)
        
        # Process audio through the model
        logits_class, logits_group, embeddings, spectral_len = self._cupe_prediction(wav, wav_len, extract_embeddings)
        
        # Prepare sequences for alignment
        ph_seqs = phoneme_sequence.unsqueeze(0).to(self.device)
        grp_seqs = group_sequence.unsqueeze(0).to(self.device)
        ph_seq_lens = torch.tensor([len(phoneme_sequence)], dtype=torch.long).to(self.device)
        spectral_lens = torch.tensor([spectral_len], dtype=torch.long).to(self.device)
        
        # Get log probabilities
        log_probs_g = F.log_softmax(logits_group, dim=2)
        log_probs_p = F.log_softmax(logits_class, dim=2)
        
        if debug:
            print(f"Target phonemes: {len(phoneme_sequence)}, Expected: {[self.phonemizer.index_to_plabel.get(p.item(), f'UNK_{p}') for p in phoneme_sequence]}")
            print(f"Spectral length: {spectral_len}")
        
        t0 = time.time()
        
        # Forced alignment with target boosting
        frame_groups = self.alignment_utils_g.decode_alignments(
            log_probs_g, 
            true_seqs=grp_seqs, 
            pred_lens=spectral_lens, 
            true_seqs_lens=ph_seq_lens, 
            forced_alignment=True,
            boost_targets=self.boost_targets,
            enforce_minimum=self.enforce_minimum,
            enforce_all_targets=self.enforce_all_targets
        )
        
        frame_phonemes = self.alignment_utils_p.decode_alignments(
            log_probs_p, 
            true_seqs=ph_seqs, 
            pred_lens=spectral_lens, 
            true_seqs_lens=ph_seq_lens, 
            forced_alignment=True,
            boost_targets=self.boost_targets,
            enforce_minimum=self.enforce_minimum,
            enforce_all_targets=self.enforce_all_targets
        )
        
        
        if debug:
            t1 = time.time()
            print(f"Forced alignment took {(t1-t0)*1000:.3f} ms")
            print(f"Aligned phonemes: {len(frame_phonemes[0])}")
            print(f"Target phonemes: {len(phoneme_sequence)}")
            
            # Check which phonemes were successfully aligned
            aligned_phoneme_ids = [p[0] for p in frame_phonemes[0]]
            target_phoneme_ids = phoneme_sequence.tolist()
            
            missing_phonemes = set(target_phoneme_ids) - set(aligned_phoneme_ids)
            if missing_phonemes:
                print(f"WARNING: Still missing {len(missing_phonemes)} phonemes: {[self.phonemizer.index_to_plabel.get(p, f'UNK_{p}') for p in missing_phonemes]}")
            else:
                print("SUCCESS: All target phonemes were aligned!")
        

        
        # Calculate confidence scores
        frame_phonemes[0] = self._calculate_confidences(log_probs_p[0], frame_phonemes[0])
        frame_groups[0] = self._calculate_confidences(log_probs_g[0], frame_groups[0])

        frame_phonemes[0] = self.convert_to_ms(frame_phonemes[0], spectral_lens[0], start_offset_time, wav_len, self.resampler_sample_rate)
        frame_groups[0] = self.convert_to_ms(frame_groups[0], spectral_lens[0], start_offset_time, wav_len, self.resampler_sample_rate)


        
        if debug:
            print("Predicted phonemes", len(frame_phonemes[0]))
            print("Predicted groups", len(frame_groups[0]))
            print("start_offset_time", start_offset_time)
            for i, (ph, grp) in enumerate(zip(frame_phonemes[0], frame_groups[0])):
                print(f"{i+1:2d}: {self.phonemizer.index_to_plabel[ph[0]]:>3s}, {self.phonemizer.index_to_glabel[grp[0]]:>3s}  -> ({grp[4]:.3f} - {grp[5]:.3f}), Confidence: {grp[3]:.3f}")
        
        
        timestamp_dict = {
            'phoneme_timestamps': frame_phonemes[0],
            'group_timestamps': frame_groups[0],
        }
        
        pooled_embeddings_phonemes = None
        if extract_embeddings and embeddings is not None:
            pooled_embeddings_phonemes = self.weighted_pool_embeddings(embeddings[0][:spectral_len], log_probs_p[0][:spectral_len], frame_phonemes[0])
            
            if do_groups:
                pooled_embeddings_groups = self.weighted_pool_embeddings(embeddings[0][:spectral_len], log_probs_g[0][:spectral_len], frame_groups[0])
                return timestamp_dict, pooled_embeddings_phonemes, pooled_embeddings_groups
            else: 
                return timestamp_dict, pooled_embeddings_phonemes

        return timestamp_dict, None, None

    
    
    def weighted_pool_embeddings(self, embeddings, log_probs, framestamps):
        """
        Average Weighted by confidence embeddings over frame ranges for each phoneme timestamp.
        
        Args:
            embeddings: Tensor of shape [T, D] where T is number of frames and D is embedding dimension
            log_probs:  log_probs: Log probabilities [T, C], can pass either phoneme or group log_probs
            timestamps: List of tuples (phoneme_idx, start_frame, end_frame, start_ms, end_ms)
        
        Returns:
            pooled_embeddings: Tensor of shape [N, D] where N is length of timestamps
        """
        if len(framestamps) == 0:
            return torch.empty(0, embeddings.shape[1], device=embeddings.device)
        
        assert embeddings.dim() == 2, "Embeddings should be of shape [T, D] remove the batch dim"
        assert log_probs.shape[0] == embeddings.shape[0], "Log probabilities and embeddings must have the same number of frames"

        probs = torch.exp(log_probs.to(embeddings.device))
        pooled_embeddings = []
        
        for phoneme_idx, start_frame, end_frame, avg_confidence, start_ms, end_ms in framestamps:
            # Clamp frame indices to valid range
            start_frame = max(0, int(start_frame))
            end_frame = min(embeddings.shape[0], int(end_frame))
            
            if start_frame < end_frame:
                # Get segment embeddings and confidence weights
                segment_embeddings = embeddings[start_frame:end_frame]  # Shape: [num_frames, D]
                confidence_weights = probs[start_frame:end_frame, phoneme_idx]  # Shape: [num_frames]
                
                # Compute weighted average
                # Expand weights to match embedding dimensions: [num_frames, 1] 
                weights_expanded = confidence_weights.unsqueeze(1)  # Shape: [num_frames, 1]
                
                # Weighted sum: multiply each embedding by its confidence weight
                weighted_embeddings = segment_embeddings * weights_expanded  # Shape: [num_frames, D]
                
                # Sum along frame dimension and normalize by total weight
                sum_weights = confidence_weights.sum()  # Scalar
                if sum_weights > 0:
                    pooled_embedding = weighted_embeddings.sum(dim=0) / sum_weights  # Shape: [D]
                else:
                    # Fallback to uniform average if all weights are zero
                    pooled_embedding = segment_embeddings.mean(dim=0)  # Shape: [D]
                    
                pooled_embeddings.append(pooled_embedding)
            else:
                # Handle edge case where start_frame >= end_frame
                # Use a zero embedding or the closest frame
                if start_frame < embeddings.shape[0]:
                    pooled_embeddings.append(embeddings[start_frame])
                else:
                    # If completely out of bounds, use zero embedding
                    pooled_embeddings.append(torch.zeros(embeddings.shape[1], device=embeddings.device))
        
        # Stack all pooled embeddings
        pooled_embeddings = torch.stack(pooled_embeddings, dim=0)  # Shape: [N, D]
        
        return pooled_embeddings

    def _calculate_confidences(self, log_probs, framestamps):
        """
        Calculate confidence scores for each timestamped phoneme/group.
        
        Args:
            log_probs: Log probabilities [T, C]
            framestamps: List of (phoneme_idx, start_frame, end_frame, ) tuples
            
        Returns:
            List of confidence scores
        """
        probs = torch.exp(log_probs)
        updated_tuples = []
        
        for phoneme_idx, start_frame, end_frame in framestamps:
            # Clamp to valid range
            start_frame = max(0, int(start_frame))
            end_frame = min(log_probs.shape[0], int(end_frame))
            avg_confidence = probs[start_frame, phoneme_idx]
            
            if start_frame < end_frame and phoneme_idx < log_probs.shape[1]:

                half_confidence = avg_confidence/2
                #if (half_confidence < 0.01): half_confidence = avg_confidence*2
                last_good_frame = start_frame
                total_good_frames = 1

                # since there can be blanks after the first one, we only take probablities if at least prob > 0.5 compared to the first frame to avoid for-sure blanks
                for f in range(start_frame+1, end_frame):
                    frame_prob = probs[f, phoneme_idx]
                    if (frame_prob > half_confidence) or (frame_prob > 0.1):
                        avg_confidence += frame_prob
                        last_good_frame = f
                        total_good_frames += 1
                if total_good_frames > 1:
                    avg_confidence /= total_good_frames
                    end_frame = min(log_probs.shape[0], int(last_good_frame + 1 )) # end_frame is exclusive, so we add 1

                    max_confidence = probs[start_frame:end_frame, phoneme_idx].max()
                    if avg_confidence < max_confidence/2:
                        #print(avg_confidence, max_confidence)
                        avg_confidence = max_confidence
                
            updated_tuples.append((phoneme_idx, start_frame, end_frame, avg_confidence.item()))
        
        return updated_tuples

    def convert_to_ms(self, framestamps, spectral_length, start_offset_time, wav_len, sample_rate):
        '''
        Args:
            framestamps: List of tuples (phoneme_idx, start_frame, end_frame, avg_confidence)
            spectral_length: Number of spectral frames (int)
            start_time: Start time of the segment in seconds, used to offset the timestamps
            wav_len: Length of the audio segment in samples, used to estimate the duration per spectral-frame
            sample_rate: Sample rate of the audio, used to convert frames to milliseconds
        Returns:7
            updated_tuples: List of tuples (phoneme_idx, start_frame, end_frame, avg_confidence, start_ms, end_ms)
        '''
        duration_in_seconds = wav_len / sample_rate
        duration_per_frame = duration_in_seconds / spectral_length if spectral_length > 0 else 0

        updated_tuples = []
        for tup in framestamps:
            if len(tup) == 4:
                phoneme_idx, start_frame, end_frame, avg_confidence = tup
            else:
                # fallback for tuples with different length
                phoneme_idx, start_frame, end_frame = tup[:3]
                avg_confidence = tup[3] if len(tup) > 3 else 0.0

            # Calculate start and end times in seconds
            start_sec = start_offset_time + (start_frame * duration_per_frame)
            end_sec = start_offset_time + (end_frame * duration_per_frame)
            # Convert to milliseconds
            start_ms = start_sec * 1000
            end_ms = end_sec * 1000

            updated_tuples.append((phoneme_idx, start_frame, end_frame, avg_confidence, start_ms, end_ms))

        return updated_tuples
        



    def _map_phonemes_to_groups(self, phoneme_sequence):
        """Map phoneme indices to phoneme group indices."""
        group_sequence = []
        for ph_idx in phoneme_sequence:
            group_idx = self.phoneme_id_to_group_id.get(ph_idx.item(), self.blank_group)
            group_sequence.append(group_idx)
        return torch.tensor(group_sequence, dtype=torch.long)
    
    def _align_words(self, phoneme_ts, word_num, words_list):
        '''
        phoneme_ts: List of phoneme timestamps [
                {
                    "phoneme_idx": int(ph_idx),
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
                    "ph66": [ph["phoneme_idx"] for ph in current_word_phonemes],
                    "ipa": [ph["phoneme_label"] for ph in current_word_phonemes]
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

    def process_segments(self, srt_data, audio_wav, ts_out_path=None, extract_embeddings=False, vspt_path=None, do_groups=True,  debug=False):

        # Process each segment
        vs2_segments = [] # timestamps for each phoneme in the segment
        vspt_g_embds = []
        vspt_p_embd  = []

    
        for i, segment in enumerate(srt_data):
            start_time = segment["start"] # segment start time
            end_time = segment["end"]

            ts_out = self.phonemize_sentence(segment["text"])
            


            phoneme_sequence = ts_out[self.phonemes_key]
            group_sequence = ts_out[self.phoneme_groups_key] if do_groups else None

            segment[self.phonemes_key] = phoneme_sequence
            segment[self.phoneme_groups_key] = group_sequence

            if not phoneme_sequence or len(phoneme_sequence) < self.ph_seq_min:
                print(f"Skipping segment {i+1} due to insufficient phoneme sequence length: {len(phoneme_sequence)}")
                continue

            phoneme_labels = [self.phonemizer.index_to_plabel.get(ph, f'UNK_{ph}') for ph in phoneme_sequence]
            if debug: 
                print(f"Expected phonemes: {phoneme_labels}")
            
            # Extract timestamps for this segment
            wav, wav_len = self.chop_wav(audio_wav, int(start_time * self.resampler_sample_rate), int(end_time * self.resampler_sample_rate))

            result, pooled_embeddings_p, pooled_embeddings_g = self.extract_timestamps_from_segment(wav, wav_len, phoneme_sequence, start_offset_time=start_time, group_sequence=group_sequence, extract_embeddings=extract_embeddings, do_groups=do_groups, debug=debug)

            coverage_analysis = self.analyze_alignment_coverage(
                phoneme_sequence, 
                result["phoneme_timestamps"], 
                self.phonemizer.index_to_plabel
            )
        
            if debug:
                # Analyze alignment coverage
                
                print(f"Alignment Coverage Analysis:")
                print(f"  Target phonemes: {coverage_analysis['target_count']}")
                print(f"  Aligned phonemes: {coverage_analysis['aligned_count']}")
                print(f"  Coverage ratio: {coverage_analysis['coverage_ratio']:.2%}")
                if coverage_analysis['missing_phonemes']:
                    print(f"  Missing: {coverage_analysis['missing_phonemes']}")
                if coverage_analysis['extra_phonemes']:
                    print(f"  Extra: {coverage_analysis['extra_phonemes']}")
            
            if extract_embeddings:
                pooled_embeddings_p = pooled_embeddings_p.detach()
                if pooled_embeddings_p.device != torch.device("cpu"): pooled_embeddings_p = pooled_embeddings_p.cpu()
                vspt_p_embd.append(pooled_embeddings_p)
                assert len(vspt_p_embd[-1]) == len(result["phoneme_timestamps"]), "Embeddings length does not match phoneme timestamps length"
                if (do_groups):
                    pooled_embeddings_g = pooled_embeddings_g.detach()
                    if pooled_embeddings_g.device != torch.device("cpu"): pooled_embeddings_g = pooled_embeddings_g.cpu()
                    vspt_g_embds.append(pooled_embeddings_g)
                    assert len(vspt_g_embds[-1]) == len(result["group_timestamps"]), "Embeddings length does not match phoneme groups timestamps length"
                


            
            # Create vs2 segment (copy original segment data and add timestamps)
            vs2_segment = segment.copy()
            vs2_segment["coverage_analysis"] = coverage_analysis
            vs2_segment["ipa"] = ts_out.get("ipa", "")
            vs2_segment["word_num"] = ts_out.get("word_num", "")
            vs2_segment["words"] = ts_out.get("words", "")
            
            
            # Add phoneme timestamps (convert 5-tuple back to simple format for JSON)
            vs2_segment["phoneme_ts"] = [
                {
                    "phoneme_idx": int(ph_idx),
                    "phoneme_label": self.phonemizer.index_to_plabel.get(ph_idx, f"UNK_{ph_idx}"),
                    "start_ms": float(start_ms),
                    "end_ms": float(end_ms),
                    "confidence": float(conf)
                }
                for (ph_idx, start_frame, end_frame, conf, start_ms, end_ms) in result["phoneme_timestamps"]
            ]
            
            # Add group timestamps
            vs2_segment["group_ts"] = [
                {
                    "group_idx": int(grp_idx),
                    "group_label": self.phonemizer.index_to_glabel.get(grp_idx, f"UNK_{grp_idx}"),
                    "start_ms": float(start_ms),
                    "end_ms": float(end_ms),
                    "confidence": float(conf)
                }
                for (grp_idx, start_frame, end_frame, conf, start_ms, end_ms) in  result["group_timestamps"]
            ]

            vs2_segment["words_ts"] = self._align_words(vs2_segment["phoneme_ts"], ts_out.get("word_num", []), ts_out.get("words", []))
            
            vs2_segments.append(vs2_segment)
        
        if debug: 
            print(f"\n{'='*60}")
            print(f"PROCESSING SUMMARY")
            print(f"{'='*60}")
            print(f"Total segments processed: {len(vs2_segments)}")
        
        # Calculate overall statistics
        total_phonemes = 0
        total_confidence = 0.0
        perfect_matches = 0
        
        for segment in vs2_segments:
            if segment["phoneme_ts"]:
                total_phonemes += len(segment["phoneme_ts"])
                total_confidence += sum(ts["confidence"] for ts in segment["phoneme_ts"])
                
                # Check if sequence matches perfectly
                predicted_sequence = [ts["phoneme_idx"] for ts in segment["phoneme_ts"]]
                if predicted_sequence == segment[self.phonemes_key]:
                    perfect_matches += 1
        
        if debug:
            overall_avg_confidence = total_confidence / total_phonemes if total_phonemes > 0 else 0.0
            print(f"Perfect sequence matches: {perfect_matches}/{len(vs2_segments)+1e-8:.0f} ({perfect_matches/(len(vs2_segments)+1e-8)*100:.1f}%)")
            print(f"Total phonemes aligned: {total_phonemes}")
            print(f"Overall average confidence: {overall_avg_confidence:.3f}")
            print(f"{'='*60}")
            
        # Create vs2 output
        vs2_data = {"segments": vs2_segments}
        
        if ts_out_path is not None:
            # Save vs2 file
            if (os.path.dirname(ts_out_path)):
                os.makedirs(os.path.dirname(ts_out_path), exist_ok=True)
            with open(ts_out_path, 'w') as f:
                json.dump(vs2_data, f, indent=4 if (debug) else None, ensure_ascii=False)
            
            if extract_embeddings and vspt_path is not None and vspt_p_embd:
                torch.save((vspt_p_embd, vspt_g_embds), vspt_path)
                if debug: 
                    print(f"Embeddings saved to: {vspt_path}")

            if debug:  
                print(f"Results saved to: {ts_out_path}")
        return vs2_data
    
    def process_srt_file(self, srt_path, audio_path, ts_out_path=None, extract_embeddings=False, vspt_path=None, do_groups=True, debug=True):
        """
        New version, read setences from SRT file and process them.
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

        return self.process_segments(srt_data["segments"], audio_wav, ts_out_path, extract_embeddings, vspt_path, do_groups, debug)
    
    def process_sentence(self, text, audio_wav, ts_out_path=None, extract_embeddings=False, vspt_path=None, do_groups=True, debug=False):
        """
        Process a single transcription and audio waveform, generating vs2 output with timestamps.

        Args:
            text: Text (str).
            audio_wav: Audio waveform tensor (torch.Tensor).
            ts_out_path: Path to output vs2 file (optional).
            extract_embeddings: Whether to extract embeddings (bool, optional).
            vspt_path: Path to save embeddings (.pt file, optional).
            do_groups: Whether to extract group timestamps (bool, optional).
            debug: Whether to print debug information (bool, optional).
        """

        duration = audio_wav.shape[1] / self.sample_rate
        srt_data = {"segments": [{"start": 0.0, "end": duration, "text": text.strip()}]}  # create whisper style SRT data

        return self.process_segments(srt_data["segments"], audio_wav, ts_out_path=ts_out_path, extract_embeddings=extract_embeddings, vspt_path=vspt_path, do_groups=do_groups, debug=debug)

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
            'extra_phonemes': [index_to_label.get(p, f'UNK_{p}') for p in extra]
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

    def framewise_assortment(self, aligned_ts, total_frames, frames_per_second, gap_contraction=5, select_key="phoneme_idx", offset_ms=0):
        """
        Perform frame-wise assortment of aligned timestamps.
        Args:
            aligned_ts: Dictionary containing segment-level timestamp information. It can be either "phoneme_ts" or "word_ts", "group_ts".
            total_frames: Total number of frames in the mel spectrogram.
            frames_per_second: Frame rate of the mel spectrogram.
            gap_contraction: extra gaps (in frames) to fill during the assortment process. Compensate for silence overwhelment on either side of unvoiced segments.
            select_key: Key to select timestamps from the aligned_ts dictionary. Default is "phoneme_idx".
            offset_ms: Offset in milliseconds to adjust the start and end times of each timestamp. Pass `timestamps["segments"][segment_id]["start"]*1000` to align with the original audio.
        """
        '''
        
        where <aligned_ts> is expected to be in the following format:
        "phoneme_ts": [{"phoneme_idx": 4, "phoneme_label": "ɪ", "start_ms": 33.03478240966797, "end_ms": 49.55217361450195, "confidence": 0.9967153072357178}, {"phoneme_idx": 53, "phoneme_label": "n", "start_ms": 49.55217361450195, "end_ms": 82.58695983886719, "confidence": 0.7659286260604858}, {"phoneme_idx": 29, "phoneme_label": "b", "start_ms": 181.69129943847656, "end_ms": 198.2086944580078, "confidence": 0.978035569190979}, {"phoneme_idx": 2, "phoneme_label": "i:", "start_ms": 198.2086944580078, "end_ms": 214.72608947753906, "confidence": 0.8755438327789307}, {"phoneme_idx": 4, "phoneme_label": "ɪ", "start_ms": 280.795654296875, "end_ms": 313.8304443359375, "confidence": 0.31954720616340637}, {"phoneme_idx": 55, "phoneme_label": "ŋ", "start_ms": 363.3825988769531, "end_ms": 396.4173889160156, "confidence": 0.6084036231040955}, {"phoneme_idx": 32, "phoneme_label": "k", "start_ms": 429.4521789550781, "end_ms": 462.4869384765625, "confidence": 0.5099813342094421}, {"phoneme_idx": 8, "phoneme_label": "ə", "start_ms": 512.0391235351562, "end_ms": 528.5565185546875, "confidence": 0.8283558487892151}, {"phoneme_idx": 52, "phoneme_label": "m", "start_ms": 528.5565185546875, "end_ms": 561.59130859375, "confidence": 0.9580954909324646}, {"phoneme_idx": 28, "phoneme_label": "p", "start_ms": 594.6260986328125, "end_ms": 627.660888671875, "confidence": 0.6376017928123474}, {"phoneme_idx": 20, "phoneme_label": "æ", "start_ms": 759.7999877929688, "end_ms": 776.3174438476562, "confidence": 0.8247546553611755}, {"phoneme_idx": 59, "phoneme_label": "ɹ", "start_ms": 776.3174438476562, "end_ms": 809.3521728515625, "confidence": 0.5612516403198242}, {"phoneme_idx": 8, "phoneme_label": "ə", "start_ms": 858.9043579101562, "end_ms": 875.4217529296875, "confidence": 0.2321549355983734}, {"phoneme_idx": 30, "phoneme_label": "t", "start_ms": 941.4913330078125, "end_ms": 991.0435180664062, "confidence": 0.8491405844688416}, {"phoneme_idx": 4, "phoneme_label": "ɪ", "start_ms": 1024.0782470703125, "end_ms": 1040.595703125, "confidence": 0.793982744216919}, {"phoneme_idx": 44, "phoneme_label": "v", "start_ms": 1040.595703125, "end_ms": 1123.1826171875, "confidence": 0.9437225461006165}, {"phoneme_idx": 56, "phoneme_label": "l", "start_ms": 1172.7347412109375, "end_ms": 1222.2869873046875, "confidence": 0.5690894722938538}, {"phoneme_idx": 1, "phoneme_label": "i", "start_ms": 1255.32177734375, "end_ms": 1271.839111328125, "confidence": 0.3983164429664612}, {"phoneme_idx": 52, "phoneme_label": "m", "start_ms": 1354.4261474609375, "end_ms": 1387.4608154296875, "confidence": 0.864766538143158}, {"phoneme_idx": 19, "phoneme_label": "a:", "start_ms": 1437.0130615234375, "end_ms": 1453.5303955078125, "confidence": 0.056571315973997116}, {"phoneme_idx": 31, "phoneme_label": "d", "start_ms": 1602.1868896484375, "end_ms": 1618.704345703125, "confidence": 0.48222440481185913}, {"phoneme_idx": 9, "phoneme_label": "ɚ", "start_ms": 1668.2564697265625, "end_ms": 1684.77392578125, "confidence": 0.9793221354484558}, {"phoneme_idx": 53, "phoneme_label": "n", "start_ms": 1767.3609619140625, "end_ms": 1783.8782958984375, "confidence": 0.11690044403076172}, {"phoneme_idx": 0, "phoneme_label": "SIL", "start_ms": 1833.430419921875, "end_ms": 1882.982666015625, "confidence": 0.07832614332437515}], 
        "group_ts": [{"group_idx": 1, "group_label": "front_vowels", "start_ms": 33.03478240966797, "end_ms": 49.55217361450195, "confidence": 0.9991531372070312}, {"group_idx": 12, "group_label": "nasals", "start_ms": 49.55217361450195, "end_ms": 82.58695983886719, "confidence": 0.7814053893089294}, {"group_idx": 7, "group_label": "voiced_stops", "start_ms": 165.17391967773438, "end_ms": 198.2086944580078, "confidence": 0.4964541494846344}, {"group_idx": 1, "group_label": "front_vowels", "start_ms": 198.2086944580078, "end_ms": 231.24346923828125, "confidence": 0.993299126625061}, {"group_idx": 1, "group_label": "front_vowels", "start_ms": 280.795654296875, "end_ms": 313.8304443359375, "confidence": 0.21188485622406006}, {"group_idx": 12, "group_label": "nasals", "start_ms": 363.3825988769531, "end_ms": 396.4173889160156, "confidence": 0.5997515320777893}, {"group_idx": 6, "group_label": "voiceless_stops", "start_ms": 429.4521789550781, "end_ms": 462.4869384765625, "confidence": 0.5166441202163696}, {"group_idx": 2, "group_label": "central_vowels", "start_ms": 512.0391235351562, "end_ms": 528.5565185546875, "confidence": 0.9326215386390686}, {"group_idx": 12, "group_label": "nasals", "start_ms": 528.5565185546875, "end_ms": 561.59130859375, "confidence": 0.748111367225647}, {"group_idx": 6, "group_label": "voiceless_stops", "start_ms": 561.59130859375, "end_ms": 627.660888671875, "confidence": 0.995503842830658}, {"group_idx": 4, "group_label": "low_vowels", "start_ms": 759.7999877929688, "end_ms": 776.3174438476562, "confidence": 0.8065245151519775}, {"group_idx": 14, "group_label": "rhotics", "start_ms": 776.3174438476562, "end_ms": 809.3521728515625, "confidence": 0.5473693013191223}, {"group_idx": 2, "group_label": "central_vowels", "start_ms": 858.9043579101562, "end_ms": 875.4217529296875, "confidence": 0.15379419922828674}, {"group_idx": 6, "group_label": "voiceless_stops", "start_ms": 924.973876953125, "end_ms": 991.0435180664062, "confidence": 0.9740506410598755}, {"group_idx": 1, "group_label": "front_vowels", "start_ms": 1024.0782470703125, "end_ms": 1040.595703125, "confidence": 0.7481966018676758}, {"group_idx": 9, "group_label": "voiced_fricatives", "start_ms": 1040.595703125, "end_ms": 1139.7000732421875, "confidence": 0.9575645327568054}, {"group_idx": 13, "group_label": "laterals", "start_ms": 1172.7347412109375, "end_ms": 1205.76953125, "confidence": 0.8053812384605408}, {"group_idx": 1, "group_label": "front_vowels", "start_ms": 1255.32177734375, "end_ms": 1271.839111328125, "confidence": 0.9730117917060852}, {"group_idx": 12, "group_label": "nasals", "start_ms": 1271.839111328125, "end_ms": 1387.4608154296875, "confidence": 0.540493369102478}, {"group_idx": 4, "group_label": "low_vowels", "start_ms": 1437.0130615234375, "end_ms": 1453.5303955078125, "confidence": 0.1977187544107437}, {"group_idx": 7, "group_label": "voiced_stops", "start_ms": 1602.1868896484375, "end_ms": 1618.704345703125, "confidence": 0.460404634475708}, {"group_idx": 2, "group_label": "central_vowels", "start_ms": 1618.704345703125, "end_ms": 1684.77392578125, "confidence": 0.5910724997520447}, {"group_idx": 12, "group_label": "nasals", "start_ms": 1750.843505859375, "end_ms": 1800.3956298828125, "confidence": 0.1525062620639801}, {"group_idx": 0, "group_label": "SIL", "start_ms": 1833.430419921875, "end_ms": 1882.982666015625, "confidence": 0.07381139695644379}], 
        "words_ts": [{"word": "in", "start_ms": 33.03478240966797, "end_ms": 82.58695983886719, "confidence": 0.8813219666481018, "ph66": [4, 53], "ipa": ["ɪ", "n"]}, {"word": "being", "start_ms": 181.69129943847656, "end_ms": 396.4173889160156, "confidence": 0.6953825578093529, "ph66": [29, 2, 4, 55], "ipa": ["b", "i:", "ɪ", "ŋ"]}, {"word": "comparatively", "start_ms": 429.4521789550781, "end_ms": 1271.839111328125, "confidence": 0.6755372906724612, "ph66": [32, 8, 52, 28, 20, 59, 8, 30, 4, 44, 56, 1], "ipa": ["k", "ə", "m", "p", "æ", "ɹ", "ə", "t", "ɪ", "v", "l", "i"]}, {"word": "modern", "start_ms": 1354.4261474609375, "end_ms": 1783.8782958984375, "confidence": 0.4999569676816463, "ph66": [52, 19, 31, 9, 53], "ipa": ["m", "a:", "d", "ɚ", "n"]}`
        
        and `select_ts` and `select_key` are used to specify which timestamps to align.
        '''
        
        gap_fill_frame_value = 0

        ms_per_frame = 1000.0 / frames_per_second


        # first sort by start_ms
        aligned_ts.sort(key=lambda x: x["start_ms"])

        framewise_label = []

        # fill with gap frames
        framewise_label = [gap_fill_frame_value] * total_frames


        # Stage 1: Fill frames using exact timestamp boundaries (no tolerance)
        for i, ts_item in enumerate(aligned_ts):
            # Calculate exact frame boundaries without any gap tolerance
            framewise_start_index = max(int((ts_item["start_ms"]-offset_ms) / ms_per_frame), 0)
            framewise_end_index = min(self.ceil((ts_item["end_ms"]-offset_ms) / ms_per_frame), total_frames)
            
            
            if framewise_end_index-framewise_start_index >= total_frames:
                print(f"Warning: Timestamp start frame {framewise_start_index} exceeds total frames {total_frames}, skipping.")
                continue
            elif framewise_end_index > total_frames:
                print(f"Warning: Timestamp ending frame {framewise_end_index} exceeds total frames {total_frames}, adjusting end index.")
                framewise_end_index = total_frames

            # Simple assignment - fill the exact range
            for frame_i in range(max(framewise_start_index-1, 0), min(framewise_end_index+1, total_frames)):
                if select_key not in ts_item:
                    raise ValueError(f"select_key '{select_key}' not found in timestamp item", ts_item)
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

    def framewise_assortment__old(self, segment_ts_dict, ms_per_frame=10.0, select_ts="phoneme_ts", select_key="phoneme_idx", gap_fill_frame_value=0, gap_intolerance=1, total_frames=None):
        """
        Perform frame-wise assortment of aligned timestamps.
        Args:
            segment_ts_dict: Dictionary containing segment-level timestamp information.
            ms_per_frame: Milliseconds per frame for the frame-wise assortment.
            gap_fill_frame_value: Value to fill in for gaps in the frame-wise assortment. Default `0` maps to silence for phoneme_ts and group_ts.
            gap_intolerance: extra gaps (in frames) to fill during the assortment process.
            total_frames: Expected total number of frames for the segment. If not provided, it will be calculated based on the (segment_duration_ms / ms_per_frame).

        """
        '''
        
        where <segment_ts_dict> is expected to be in the following format:
        `{"start": 0.0, "end": 1.899546485260771, 
        "phoneme_ts": [{"phoneme_idx": 4, "phoneme_label": "ɪ", "start_ms": 33.03478240966797, "end_ms": 49.55217361450195, "confidence": 0.9967153072357178}, {"phoneme_idx": 53, "phoneme_label": "n", "start_ms": 49.55217361450195, "end_ms": 82.58695983886719, "confidence": 0.7659286260604858}, {"phoneme_idx": 29, "phoneme_label": "b", "start_ms": 181.69129943847656, "end_ms": 198.2086944580078, "confidence": 0.978035569190979}, {"phoneme_idx": 2, "phoneme_label": "i:", "start_ms": 198.2086944580078, "end_ms": 214.72608947753906, "confidence": 0.8755438327789307}, {"phoneme_idx": 4, "phoneme_label": "ɪ", "start_ms": 280.795654296875, "end_ms": 313.8304443359375, "confidence": 0.31954720616340637}, {"phoneme_idx": 55, "phoneme_label": "ŋ", "start_ms": 363.3825988769531, "end_ms": 396.4173889160156, "confidence": 0.6084036231040955}, {"phoneme_idx": 32, "phoneme_label": "k", "start_ms": 429.4521789550781, "end_ms": 462.4869384765625, "confidence": 0.5099813342094421}, {"phoneme_idx": 8, "phoneme_label": "ə", "start_ms": 512.0391235351562, "end_ms": 528.5565185546875, "confidence": 0.8283558487892151}, {"phoneme_idx": 52, "phoneme_label": "m", "start_ms": 528.5565185546875, "end_ms": 561.59130859375, "confidence": 0.9580954909324646}, {"phoneme_idx": 28, "phoneme_label": "p", "start_ms": 594.6260986328125, "end_ms": 627.660888671875, "confidence": 0.6376017928123474}, {"phoneme_idx": 20, "phoneme_label": "æ", "start_ms": 759.7999877929688, "end_ms": 776.3174438476562, "confidence": 0.8247546553611755}, {"phoneme_idx": 59, "phoneme_label": "ɹ", "start_ms": 776.3174438476562, "end_ms": 809.3521728515625, "confidence": 0.5612516403198242}, {"phoneme_idx": 8, "phoneme_label": "ə", "start_ms": 858.9043579101562, "end_ms": 875.4217529296875, "confidence": 0.2321549355983734}, {"phoneme_idx": 30, "phoneme_label": "t", "start_ms": 941.4913330078125, "end_ms": 991.0435180664062, "confidence": 0.8491405844688416}, {"phoneme_idx": 4, "phoneme_label": "ɪ", "start_ms": 1024.0782470703125, "end_ms": 1040.595703125, "confidence": 0.793982744216919}, {"phoneme_idx": 44, "phoneme_label": "v", "start_ms": 1040.595703125, "end_ms": 1123.1826171875, "confidence": 0.9437225461006165}, {"phoneme_idx": 56, "phoneme_label": "l", "start_ms": 1172.7347412109375, "end_ms": 1222.2869873046875, "confidence": 0.5690894722938538}, {"phoneme_idx": 1, "phoneme_label": "i", "start_ms": 1255.32177734375, "end_ms": 1271.839111328125, "confidence": 0.3983164429664612}, {"phoneme_idx": 52, "phoneme_label": "m", "start_ms": 1354.4261474609375, "end_ms": 1387.4608154296875, "confidence": 0.864766538143158}, {"phoneme_idx": 19, "phoneme_label": "a:", "start_ms": 1437.0130615234375, "end_ms": 1453.5303955078125, "confidence": 0.056571315973997116}, {"phoneme_idx": 31, "phoneme_label": "d", "start_ms": 1602.1868896484375, "end_ms": 1618.704345703125, "confidence": 0.48222440481185913}, {"phoneme_idx": 9, "phoneme_label": "ɚ", "start_ms": 1668.2564697265625, "end_ms": 1684.77392578125, "confidence": 0.9793221354484558}, {"phoneme_idx": 53, "phoneme_label": "n", "start_ms": 1767.3609619140625, "end_ms": 1783.8782958984375, "confidence": 0.11690044403076172}, {"phoneme_idx": 0, "phoneme_label": "SIL", "start_ms": 1833.430419921875, "end_ms": 1882.982666015625, "confidence": 0.07832614332437515}], 
        "group_ts": [{"group_idx": 1, "group_label": "front_vowels", "start_ms": 33.03478240966797, "end_ms": 49.55217361450195, "confidence": 0.9991531372070312}, {"group_idx": 12, "group_label": "nasals", "start_ms": 49.55217361450195, "end_ms": 82.58695983886719, "confidence": 0.7814053893089294}, {"group_idx": 7, "group_label": "voiced_stops", "start_ms": 165.17391967773438, "end_ms": 198.2086944580078, "confidence": 0.4964541494846344}, {"group_idx": 1, "group_label": "front_vowels", "start_ms": 198.2086944580078, "end_ms": 231.24346923828125, "confidence": 0.993299126625061}, {"group_idx": 1, "group_label": "front_vowels", "start_ms": 280.795654296875, "end_ms": 313.8304443359375, "confidence": 0.21188485622406006}, {"group_idx": 12, "group_label": "nasals", "start_ms": 363.3825988769531, "end_ms": 396.4173889160156, "confidence": 0.5997515320777893}, {"group_idx": 6, "group_label": "voiceless_stops", "start_ms": 429.4521789550781, "end_ms": 462.4869384765625, "confidence": 0.5166441202163696}, {"group_idx": 2, "group_label": "central_vowels", "start_ms": 512.0391235351562, "end_ms": 528.5565185546875, "confidence": 0.9326215386390686}, {"group_idx": 12, "group_label": "nasals", "start_ms": 528.5565185546875, "end_ms": 561.59130859375, "confidence": 0.748111367225647}, {"group_idx": 6, "group_label": "voiceless_stops", "start_ms": 561.59130859375, "end_ms": 627.660888671875, "confidence": 0.995503842830658}, {"group_idx": 4, "group_label": "low_vowels", "start_ms": 759.7999877929688, "end_ms": 776.3174438476562, "confidence": 0.8065245151519775}, {"group_idx": 14, "group_label": "rhotics", "start_ms": 776.3174438476562, "end_ms": 809.3521728515625, "confidence": 0.5473693013191223}, {"group_idx": 2, "group_label": "central_vowels", "start_ms": 858.9043579101562, "end_ms": 875.4217529296875, "confidence": 0.15379419922828674}, {"group_idx": 6, "group_label": "voiceless_stops", "start_ms": 924.973876953125, "end_ms": 991.0435180664062, "confidence": 0.9740506410598755}, {"group_idx": 1, "group_label": "front_vowels", "start_ms": 1024.0782470703125, "end_ms": 1040.595703125, "confidence": 0.7481966018676758}, {"group_idx": 9, "group_label": "voiced_fricatives", "start_ms": 1040.595703125, "end_ms": 1139.7000732421875, "confidence": 0.9575645327568054}, {"group_idx": 13, "group_label": "laterals", "start_ms": 1172.7347412109375, "end_ms": 1205.76953125, "confidence": 0.8053812384605408}, {"group_idx": 1, "group_label": "front_vowels", "start_ms": 1255.32177734375, "end_ms": 1271.839111328125, "confidence": 0.9730117917060852}, {"group_idx": 12, "group_label": "nasals", "start_ms": 1271.839111328125, "end_ms": 1387.4608154296875, "confidence": 0.540493369102478}, {"group_idx": 4, "group_label": "low_vowels", "start_ms": 1437.0130615234375, "end_ms": 1453.5303955078125, "confidence": 0.1977187544107437}, {"group_idx": 7, "group_label": "voiced_stops", "start_ms": 1602.1868896484375, "end_ms": 1618.704345703125, "confidence": 0.460404634475708}, {"group_idx": 2, "group_label": "central_vowels", "start_ms": 1618.704345703125, "end_ms": 1684.77392578125, "confidence": 0.5910724997520447}, {"group_idx": 12, "group_label": "nasals", "start_ms": 1750.843505859375, "end_ms": 1800.3956298828125, "confidence": 0.1525062620639801}, {"group_idx": 0, "group_label": "SIL", "start_ms": 1833.430419921875, "end_ms": 1882.982666015625, "confidence": 0.07381139695644379}], 
        "words_ts": [{"word": "in", "start_ms": 33.03478240966797, "end_ms": 82.58695983886719, "confidence": 0.8813219666481018, "ph66": [4, 53], "ipa": ["ɪ", "n"]}, {"word": "being", "start_ms": 181.69129943847656, "end_ms": 396.4173889160156, "confidence": 0.6953825578093529, "ph66": [29, 2, 4, 55], "ipa": ["b", "i:", "ɪ", "ŋ"]}, {"word": "comparatively", "start_ms": 429.4521789550781, "end_ms": 1271.839111328125, "confidence": 0.6755372906724612, "ph66": [32, 8, 52, 28, 20, 59, 8, 30, 4, 44, 56, 1], "ipa": ["k", "ə", "m", "p", "æ", "ɹ", "ə", "t", "ɪ", "v", "l", "i"]}, {"word": "modern", "start_ms": 1354.4261474609375, "end_ms": 1783.8782958984375, "confidence": 0.4999569676816463, "ph66": [52, 19, 31, 9, 53], "ipa": ["m", "a:", "d", "ɚ", "n"]}`
        

        and `select_ts` and `select_key` are used to specify which timestamps to align.
        '''
        aligned_ts = segment_ts_dict[select_ts]
        # first sort by start_ms
        aligned_ts.sort(key=lambda x: x["start_ms"])

        
        if total_frames is None:
            total_frames = self.ceil((segment_ts_dict["end"] - segment_ts_dict["start"]) * 1000 / ms_per_frame)

        framewise_label = []

        # fill with gap frames
        if total_frames is not None:
            framewise_label = [gap_fill_frame_value] * total_frames

        
        for ts_item in aligned_ts:
            #print("Processing timestamp item:", ts_item)
            framewise_start_index = max(int(max(0.0, ts_item["start_ms"]-(ms_per_frame / 2)) / ms_per_frame)-gap_intolerance, 0)
            framewise_end_index = min(self.ceil(ts_item["end_ms"] / ms_per_frame)+gap_intolerance, total_frames)
            #print(f"Filling frames {framewise_start_index} to {framewise_end_index} for timestamp item", ts_item)
            for frame_i in range(framewise_start_index, framewise_end_index):
                if framewise_label[frame_i]==gap_fill_frame_value or (frame_i > framewise_start_index+gap_intolerance):
                    
                    if select_key not in ts_item:
                        raise ValueError(f"select_key '{select_key}' not found in timestamp item", ts_item)
                    framewise_label[frame_i] = ts_item[select_key]

        
        return framewise_label

def process_sentence(transcription, audio_path, model_name="en_libri1000_uj01d_e199_val_GER=0.2307.ckpt", lang="en-us", duration_max=10, ts_out_path=None, device="cpu"):

    extractor = PhonemeTimestampAligner(model_name=model_name, lang=lang, duration_max=duration_max, device=device)

    audio_wav = extractor.load_audio(audio_path) # can replace it with custom audio source


    return extractor.process_sentence(transcription, audio_wav, ts_out_path=ts_out_path, extract_embeddings=False, vspt_path=None, do_groups=True, debug=False)



def process_single_clip(cupe_ckpt_path, audio_path, srt_path, ts_out_path = None, vspt_path=None,  duration_max=10, lang="en-us", device="cuda:0", extract_embeddings=False, debug=True):
    """
    Process a single audio clip for phoneme timestamp extraction.

    Args:
        cupe_ckpt_path (str): Path to CUPE model checkpoint.
        audio_path (str): Path to audio file.
        srt_path (str): Path to SRT file (JSON format), e.g., {"segments": [{"start": 0.0, "end": 9.23, "text": "text"}], }
        ts_out_path (str, optional): Path to output vs2 file. Timestamps output json format.
        vspt_path (str, optional): Path to save embeddings (.pt file).
        duration_max (float): Maximum segment duration in seconds.
        lang (str): Language code for phonemization (e.g., "en-us").
        device (str): Device to run inference on (e.g., "cuda:0" or "cpu").
        extract_embeddings (bool): Whether to extract embeddings (default: False).
        debug (bool): Whether to print debug information (default: True).
    """
    extractor = PhonemeTimestampAligner(cupe_ckpt_path=cupe_ckpt_path, lang=lang, duration_max=duration_max, device=device)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    
    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"SRT file does not exist: {srt_path}")

    if (vspt_path is None) and extract_embeddings:
        vspt_path = srt_path.replace('.srt', '.vs.pt')
        if not vspt_path.endswith('.vs.pt'):
            vspt_path = vspt_path.rsplit('.', 1)[0] + '.vs.pt'
        
    return extractor.process_srt_file(srt_path, audio_path, ts_out_path, extract_embeddings, vspt_path, debug=debug)





def example_audio_timestamps():

    transcription = "butterfly"
    audio_path = "samples/audio/109867__timkahn__butterfly.wav"
    
    model_name = "en_libri1000_uj01d_e199_val_GER=0.2307.ckpt" 
    extractor = PhonemeTimestampAligner(model_name=model_name, lang='en-us', duration_max=10, device='cpu')

    audio_wav = extractor.load_audio(audio_path) # can replace it with custom audio source

    t0 = time.time()

    timestamps = extractor.process_sentence(transcription, audio_wav, ts_out_path=None, extract_embeddings=False, vspt_path=None, do_groups=True, debug=True)

    t1 = time.time()
    print("Timestamps:")
    print(json.dumps(timestamps, indent=4, ensure_ascii=False))
    print(f"Processing time: {t1 - t0:.2f} seconds")




if __name__ == "__main__":
    torch.random.manual_seed(42)
    example_audio_timestamps()
