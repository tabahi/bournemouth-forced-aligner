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
from .mapper66 import phoneme_mapped_index, phoneme_groups_index, phoneme_groups_mapper
from . import universal_phonemeizer
from .utils import dict_to_textgrid
# Create reverse mappings for interpretability
index_to_glabel = {v: k for k, v in phoneme_groups_index.items()}
index_to_plabel = {v: k for k, v in phoneme_mapped_index.items()}



class PhonemeTimestampAligner:
    """
    Align phoneme-level timestamps from audio using a pre-trained CUPE model
    and Viterbi decoding for forced alignment.
    URL: https://github.com/tabahi/bournemouth-forced-aligner
    """

    def __init__(self, model_name = "en_libri1000_uj01d_e199_val_GER=0.2307.ckpt", cupe_ckpt_path=None, lang='en-us', duration_max=10, ph_seq_max=64, device="cuda", boost_targets=True, enforce_minimum=True):
        """
        Initialize the phoneme timestamp extractor.
        
        Args:
            model_name: Name of the pre-trained model to use. Use the ckpt base filenames from:  https://huggingface.co/Tabahi/CUPE-2i/tree/main/ckpt
            cupe_ckpt_path: Path to the CUPE model checkpoint. Download from: https://huggingface.co/Tabahi/CUPE-2i/tree/main/ckpt
            lang: Language for phonemization (use espeak lang codes, see https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md)
            duration_max: Maximum duration in seconds (Padding purposes when using batch processing). You can set it to 10 to 60 seconds.
            ph_seq_max: Maximum phoneme sequence length
            device: Device to run inference on
            boost_targets: Boost the probabilities of target phonemes to ensure they can be aligned.
            enforce_minimum: Ensure target phonemes have minimum probability at each frame.
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
        self.ph_seq_max = ph_seq_max
        self.seg_duration_min = 0.05  # seconds
        self.seg_duration_min_samples = int(self.seg_duration_min * self.resampler_sample_rate)
        self.seg_duration_max = duration_max  # seconds
        self.wav_len_max = int(self.seg_duration_max * self.resampler_sample_rate)
        self.phonemizer = universal_phonemeizer.Phonemizer(language=lang,remove_noise_phonemes=True)

        self.phonemes_key = self.phonemizer.phonemes_key
        self.phoneme_groups_key = self.phonemizer.phoneme_groups_key

        self._setup_config()
        self._setup_decoders()

        self.boost_targets = boost_targets
        self.enforce_minimum = enforce_minimum

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
        self.phoneme_classes = 66
        self.phoneme_groups = 16
        self.blank_class = self.phoneme_classes
        self.blank_group = self.phoneme_groups
        
    def _setup_decoders(self):
        """Setup Viterbi decoders for phoneme classes and groups."""
        
        # Alignment utilities
        self.alignment_utils_g = AlignmentUtils(blank_id=self.blank_group)
        self.alignment_utils_p = AlignmentUtils(blank_id=self.blank_class)

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
    



    def load_audio(self, audio_path):
        """Load and preprocess audio file."""
        
        wav, sr = torchaudio.load(audio_path,  frame_offset=0,  normalize=True)
        
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
            print(f"Target phonemes: {len(phoneme_sequence)}, Expected: {[index_to_plabel.get(p.item(), f'UNK_{p}') for p in phoneme_sequence]}")
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
            enforce_minimum=self.enforce_minimum
        )
        
        frame_phonemes = self.alignment_utils_p.decode_alignments(
            log_probs_p, 
            true_seqs=ph_seqs, 
            pred_lens=spectral_lens, 
            true_seqs_lens=ph_seq_lens, 
            forced_alignment=True,
            boost_targets=self.boost_targets,
            enforce_minimum=self.enforce_minimum
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
                print(f"WARNING: Still missing {len(missing_phonemes)} phonemes: {[index_to_plabel.get(p, f'UNK_{p}') for p in missing_phonemes]}")
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
                print(f"{i+1:2d}: {index_to_plabel[ph[0]]:>3s}, {index_to_glabel[grp[0]]:>3s}  -> ({grp[4]:.3f} - {grp[5]:.3f}), Confidence: {grp[3]:.3f}")
        
        
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
            group_idx = phoneme_groups_mapper.get(ph_idx.item(), self.blank_group)
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


    def process_segments(self, srt_data, audio_wav, ts_out_path=None, extract_embeddings=False, vspt_path=None, do_groups=True, debug=False):
        
        # Process each segment
        vs2_segments = [] # timestamps for each phoneme in the segment
        vspt_g_embds = []
        vspt_p_embd  = []

    
        for i, segment in enumerate(srt_data):
            start_time = segment["start"] # segment start time
            end_time = segment["end"]

            ts_out = self.phonemizer.phonemize_sentence(segment["text"])
            


            phoneme_sequence = ts_out[self.phonemes_key]
            group_sequence = ts_out[self.phoneme_groups_key] if do_groups else None

            segment[self.phonemes_key] = phoneme_sequence
            segment[self.phoneme_groups_key] = group_sequence

            if not phoneme_sequence or len(phoneme_sequence) < self.ph_seq_min:
                print(f"Skipping segment {i+1} due to insufficient phoneme sequence length: {len(phoneme_sequence)}")
                continue

            phoneme_labels = [index_to_plabel.get(ph, f'UNK_{ph}') for ph in phoneme_sequence]
            if debug: 
                print(f"Expected phonemes: {phoneme_labels}")
            
            # Extract timestamps for this segment
            wav, wav_len = self.chop_wav(audio_wav, int(start_time * self.resampler_sample_rate), int(end_time * self.resampler_sample_rate))

            result, pooled_embeddings_p, pooled_embeddings_g = self.extract_timestamps_from_segment(wav, wav_len, phoneme_sequence, start_offset_time=start_time, group_sequence=group_sequence, extract_embeddings=extract_embeddings, do_groups=do_groups, debug=debug)

            coverage_analysis = self.analyze_alignment_coverage(
                phoneme_sequence, 
                result["phoneme_timestamps"], 
                index_to_plabel
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
                    "phoneme_label": index_to_plabel.get(ph_idx, f"UNK_{ph_idx}"),
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
                    "group_label": index_to_glabel.get(grp_idx, f"UNK_{grp_idx}"),
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
    
    def process_transcription(self, text, audio_wav, ts_out_path=None, extract_embeddings=False, vspt_path=None, do_groups=True, debug=False):
        """
        Process a single transcription and audio waveform, generating vs2 output with timestamps.

        Args:
            text: Transcription text (str).
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

        if output_file:
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



def process_transcription(transcription, audio_path, model_name="en_libri1000_uj01d_e199_val_GER=0.2307.ckpt", lang="en-us", duration_max=10, ts_out_path=None, device="cpu"):

    extractor = PhonemeTimestampAligner(model_name=model_name, lang=lang, duration_max=duration_max, device=device)

    audio_wav = extractor.load_audio(audio_path) # can replace it with custom audio source


    return extractor.process_transcription(transcription, audio_wav, ts_out_path=ts_out_path, extract_embeddings=False, vspt_path=None, do_groups=True, debug=False)



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

    timestamps = extractor.process_transcription(transcription, audio_wav, ts_out_path=None, extract_embeddings=False, vspt_path=None, do_groups=True, debug=True)

    t1 = time.time()
    print("Timestamps:")
    print(json.dumps(timestamps, indent=4, ensure_ascii=False))
    print(f"Processing time: {t1 - t0:.2f} seconds")




if __name__ == "__main__":
    torch.random.manual_seed(42)
    example_audio_timestamps()
