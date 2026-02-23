'''
Self-contained BFA pipeline for C++ porting reference.
All required code is inline — no cross-module imports.
'''
import torch
import numpy as np
import librosa
import torch.nn.functional as F
import os
import time
import math
import onnxruntime as ort

try:
    from bournemouth_aligner.ipamappers.ph66_phonemeizer import Phonemizer as _Phonemizer
except ImportError:
    import sys
    sys.path.append('.')
    sys.path.append('./bournemouth_aligner')
    from ipamappers.ph66_phonemeizer import Phonemizer as _Phonemizer

# ============================================================
# Phoneme & group index tables (from ph66_mapper.py)
# ============================================================

phoneme_mapped_index = { # keep it, it's correct
    # Special token
    'SIL': 0,
    
    # High front vowels and commonly confused similar vowels
    'i': 1,        # High front unrounded
    'i:': 2,       # Long high front unrounded
    'ɨ': 3,        # High central (grouped here due to high confusion with 'i')
    'ɪ': 4,        # Near-high front unrounded
    
    # Mid front vowels
    'e': 5,        # Mid front unrounded
    'e:': 6,       # Long mid front unrounded
    'ɛ': 7,        # Open-mid front unrounded
    
    # Central vowels
    'ə': 8,        # Schwa (mid central)
    'ɚ': 9,        # R-colored schwa
    'ʌ': 10,       # Open-mid back unrounded
    
    # Back vowels
    'u': 11,       # High back rounded
    'u:': 12,      # Long high back rounded
    'ʊ': 13,       # Near-high back rounded
    'ɯ': 14,       # High back unrounded
    'o': 15,       # Mid back rounded
    'o:': 16,      # Long mid back rounded
    'ɔ': 17,       # Open-mid back rounded
    
    # Low vowels
    'a': 18,       # Open central/front unrounded
    'a:': 19,      # Long open central/front unrounded
    'æ': 20,       # Near-open front unrounded
    
    # Front rounded vowels
    'y': 21,       # High front rounded
    'ø': 22,       # Mid front rounded
    
    # Diphthongs
    'aɪ': 23,      # Open central to high front
    'eɪ': 24,      # Mid front to high front
    'aʊ': 25,      # Open central to high back
    'oʊ': 26,      # Mid back to high back
    'ɔɪ': 27,      # Open-mid back to high front
    
    # Stops (organized by place of articulation)
    'p': 28,       # Voiceless bilabial
    'b': 29,       # Voiced bilabial
    't': 30,       # Voiceless alveolar
    'd': 31,       # Voiced alveolar
    'k': 32,       # Voiceless velar
    'g': 33,       # Voiced velar
    'q': 34,       # Voiceless uvular
    
    # Affricates and related sibilant fricatives (grouped by similarity)
    'ts': 35,      # Voiceless alveolar affricate
    's': 36,       # Voiceless alveolar fricative
    'z': 37,       # Voiced alveolar fricative
    'tʃ': 38,      # Voiceless postalveolar affricate
    'dʒ': 39,      # Voiced postalveolar affricate
    'ʃ': 40,       # Voiceless postalveolar fricative
    'ʒ': 41,       # Voiced postalveolar fricative
    'ɕ': 42,       # Voiceless alveolo-palatal fricative
    
    # Other fricatives (organized by place)
    'f': 43,       # Voiceless labiodental
    'v': 44,       # Voiced labiodental
    'θ': 45,       # Voiceless dental
    'ð': 46,       # Voiced dental
    'ç': 47,       # Voiceless palatal
    'x': 48,       # Voiceless velar
    'ɣ': 49,       # Voiced velar
    'h': 50,       # Voiceless glottal
    'ʁ': 51,       # Voiced uvular
    
    # Nasals (organized by place)
    'm': 52,       # Bilabial
    'n': 53,       # Alveolar
    'ɲ': 54,       # Palatal
    'ŋ': 55,       # Velar
    
    # Liquids and approximants
    'l': 56,       # Alveolar lateral
    'ɭ': 57,       # Retroflex lateral
    'ɾ': 58,       # Alveolar tap
    'ɹ': 59,       # Alveolar approximant
    'j': 60,       # Palatal approximant
    'w': 61,       # Labial-velar approximant
    
    # Palatalized consonants
    'tʲ': 62,      # Palatalized t
    'nʲ': 63,      # Palatalized n
    'rʲ': 64,      # Palatalized r
    'ɭʲ': 65,      # Palatalized retroflex lateral
    
    # Special token
    'noise': 66
}

phoneme_groups_mapper = {0: 0, 1: 1, 2: 1, 3: 3, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 2, 10: 2, 11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 4, 19: 4, 20: 4, 21: 1, 22: 1, 23: 5, 24: 5, 25: 5, 26: 5, 27: 5, 28: 6, 29: 7, 30: 6, 31: 7, 32: 6, 33: 7, 34: 6, 35: 10, 36: 8, 37: 9, 38: 10, 39: 11, 40: 8, 41: 9, 42: 8, 43: 8, 44: 9, 45: 8, 46: 9, 47: 8, 48: 8, 49: 9, 50: 8, 51: 9, 52: 12, 53: 12, 54: 12, 55: 12, 56: 13, 57: 13, 58: 14, 59: 14, 60: 15, 61: 15, 62: 6, 63: 12, 64: 14, 65: 13, 66: 16}

phoneme_groups_index = {'SIL': 0, 'front_vowels': 1, 'central_vowels': 2, 'back_vowels': 3, 'low_vowels': 4, 'diphthongs': 5, 'voiceless_stops': 6, 'voiced_stops': 7, 'voiceless_fricatives': 8, 'voiced_fricatives': 9, 'voiceless_affricates': 10, 'voiced_affricates': 11, 'nasals': 12, 'laterals': 13, 'rhotics': 14, 'glides': 15, 'noise': 16}

# Reverse lookup tables
index_to_plabel = {v: k for k, v in phoneme_mapped_index.items()}
index_to_glabel = {v: k for k, v in phoneme_groups_index.items()}

# ============================================================
# Phonemizer stub — phonemization is handled externally in C++.
# This stub wraps the Python espeak backend for testing only.
# ============================================================

class Phonemizer:
    """Thin wrapper around espeak-ng for testing. Not needed in C++."""

    def __init__(self, language='en-us', remove_noise_phonemes=True):
        self.remove_noise_phonemes = remove_noise_phonemes
        self.phonemes_key = "ph66"
        self.phoneme_groups_key = "pg16"
        self.index_to_plabel = index_to_plabel
        self.index_to_glabel = index_to_glabel
        self.phoneme_id_to_group_id = phoneme_groups_mapper

        # import the mapper data lazily so the rest of the file stays self-contained
        #from bournemouth_aligner.ipamappers.ph66_phonemeizer import Phonemizer as _Phonemizer
        self._inner = _Phonemizer(language=language, remove_noise_phonemes=remove_noise_phonemes)

    def phonemize_sentence(self, text):
        return self._inner.phonemize_sentence(text)

# ============================================================
# ONNX predictor
# ============================================================

class CUPEONNXPredictor: # keep it
    def __init__(self, onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']):
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        # Check if batch axis is dynamic or fixed
        input_shape = self.sess.get_inputs()[0].shape
        self.dynamic_batch = not isinstance(input_shape[0], int)  # symbolic = dynamic

    def predict(self, audio_batch, return_embeddings=True):
        """audio_batch: torch.Tensor [batch_size, wav_length]."""
        x_np = audio_batch.detach().cpu().numpy().astype(np.float32)
        input_name = self.sess.get_inputs()[0].name

        if self.dynamic_batch or x_np.shape[0] == 1:
            # Dynamic batch or single window — run in one shot
            outputs = self.sess.run(None, {input_name: x_np})
        else:
            # Fixed-batch ONNX model — process one window at a time
            all_outputs = [self.sess.run(None, {input_name: x_np[i:i+1]}) for i in range(x_np.shape[0])]
            num_outputs = len(all_outputs[0])
            outputs = [np.concatenate(np.array([o[k] for o in all_outputs]), axis=0) for k in range(num_outputs)]

        logits_class = torch.from_numpy(outputs[0])
        logits_group = torch.from_numpy(outputs[1])
        embeddings = torch.from_numpy(outputs[2]) if return_embeddings else None

        return logits_class, logits_group, embeddings


def slice_windows(audio_batch: torch.Tensor,
                        sample_rate: int = 16000,
                        window_size_ms: int = 160,
                        stride_ms: int = 80) -> torch.Tensor:
    '''
    Keep it, it's correct. Slices a batch of audio into overlapping windows for CUPE processing.
    '''


    audio_batch = audio_batch.squeeze(1)  # [batch_size, max_audio_length]
    batch_size, max_audio_length = audio_batch.shape
    
    # Calculate window parameters
    window_size = int(window_size_ms * sample_rate / 1000)
    stride = int(stride_ms * sample_rate / 1000)
    num_windows = ((max_audio_length - window_size) // stride) + 1
    
    # Create indices for all windows at once
    offsets = torch.arange(0, window_size, device=audio_batch.device)
    starts = torch.arange(0, num_windows * stride, stride, device=audio_batch.device)
    
    # Create a indices matrix [num_windows, window_size]
    indices = starts.unsqueeze(1) + offsets.unsqueeze(0)
    
    # Handle out-of-bounds indices
    valid_indices = indices < max_audio_length
    indices = torch.minimum(indices, torch.tensor(max_audio_length - 1, device=audio_batch.device))
    
    # Expand indices for batching [batch_size, num_windows, window_size]
    batch_indices = torch.arange(batch_size, device=audio_batch.device)[:, None, None]
    
    # Gather windows using expanded indices
    windows = audio_batch[batch_indices, indices]
    
    # Zero out invalid regions
    windows = windows * valid_indices.float()
    
    return windows


def stich_window_predictions(window_logits: torch.Tensor,
                             original_audio_length: int,
                             cnn_output_size: int,
                             sample_rate: int = 16000,
                             window_size_ms: int = 160,
                             stride_ms: int = 80) -> torch.Tensor:
    '''
    Keep it, it's correct. Stitches overlapping window predictions back into a single sequence of frame-level logits.
    '''


    device = window_logits.device
    batch_size, num_windows, frames_per_window, num_phonemes = window_logits.shape
    
    # Pre-compute constants
    window_size_samples = int(window_size_ms * sample_rate / 1000)
    stride_samples = int(stride_ms * sample_rate / 1000)
    num_windows_total = ((original_audio_length - window_size_samples) // stride_samples) + 1
    total_frames = ((num_windows_total * cnn_output_size) // 2)
    stride_frames = frames_per_window // 2
    
    # Pre-compute weights once and cache
    window_weights = torch.cos(torch.linspace(-math.pi/2, math.pi/2, frames_per_window, device=device))
    window_weights = window_weights.view(1, frames_per_window, 1)
    
    # Pre-allocate output tensors
    combined = torch.zeros(batch_size, total_frames, num_phonemes, device=device)
    weight_sum = torch.zeros(batch_size, total_frames, 1, device=device)
    
    # Process all windows at once when possible
    full_windows = num_windows - 1  # Leave last window for special handling
    if full_windows > 0:
        # Get all start frames at once
        #start_frames = torch.arange(0, full_windows * stride_frames, stride_frames, device=device)
        
        # Process full windows in a single operation
        full_slices = window_logits[:, :full_windows]  # [batch_size, full_windows, frames_per_window, num_phonemes]
        
        for i in range(full_windows):
            start_frame = i * stride_frames
            end_frame = start_frame + frames_per_window
            combined[:, start_frame:end_frame] += full_slices[:, i] * window_weights
            weight_sum[:, start_frame:end_frame] += window_weights
    
    # Handle last window separately due to potential size mismatch
    if num_windows > 0:
        start_frame = (num_windows - 1) * stride_frames
        end_frame = start_frame + frames_per_window
        
        if end_frame > total_frames:
            frames_to_use = total_frames - start_frame
            window_logits_slice = window_logits[:, -1, :frames_to_use]
            weights = window_weights[:, :frames_to_use]
        else:
            window_logits_slice = window_logits[:, -1]
            weights = window_weights
        
        combined[:, start_frame:start_frame + window_logits_slice.size(1)] += window_logits_slice * weights
        weight_sum[:, start_frame:start_frame + weights.size(1)] += weights
    
    # Normalize with stable division
    combined = combined / (weight_sum + 1e-8)
    return combined

def calc_spec_len_ext(wav_lens, window_size_ms, stride_ms, sample_rate, frames_per_window, disable_windowing=False, wav_len_max=1*16000):
    '''
    Keep it, it's correct. It's a direct copy
    '''
    print("wav_lens:", wav_lens)
        
    if (not disable_windowing):
        #window_size_samples = int(self.window_size_ms * self.sample_rate / 1000)
        #stride_samples = int(self.stride_ms * self.sample_rate / 1000)
        
        # move self.frames_per_window to the same device if not already:
        frames_per_window = frames_per_window.to(wav_lens.device)
        window_size_wav = int(window_size_ms * sample_rate / 1000)  # 1920
        stride_size_wav = int(stride_ms * sample_rate / 1000)    # 1280
        spectral_lens = []
        print("window_size_wav:", window_size_wav, "stride_size_wav:", stride_size_wav, "frames_per_window:", frames_per_window)

        for wav_len in wav_lens:
            # Handle case where audio is shorter than window size
            if wav_len <= window_size_wav:
                # For short clips, use a single window with scaled output frames
                # Scale proportionally to actual length relative to window size
                num_windows = wav_len.float() / window_size_wav
                total_frames = torch.ceil(frames_per_window * num_windows).long()
            else:
                # Standard calculation for normal-length audio
                # Calculate number of windows
                num_windows = ((wav_len - window_size_wav) // stride_size_wav) + 1
                # Calculate total frames after combining windows
                total_frames = ((num_windows * frames_per_window) // 2)  # divide by 2 due to window overlap
            
            print("total_frames:", total_frames)
            if (total_frames < 2):
                raise Exception("WARN: spectral_len < 2, wav_lens:", wav_len.item(), "output frames:", total_frames.item(), "num_windows:", num_windows.item(), "Expected at least", window_size_ms, "ms", "got", (1000*wav_len.item()/sample_rate), "ms")
            spectral_lens.append(total_frames)
        
        spectral_lens = torch.tensor(spectral_lens, device=wav_lens.device, dtype=torch.long)


    else:
        # Given that there are 149 frames per 3 seconds,  49 frames per 1 seconds, we can calculate the number of frames for the whole audio clip
            
        #max_seconds = self.wav_len_max / self.sample_rate
        #max_frames = int(max_seconds * 50) # 49 frames per second, 20ms per frame
        print("\n\nCalculating spectral lens with windowing disenabled\n\n")

        frames_per_window = frames_per_window.to(wav_lens.device)
        wav_len_per_frame = (wav_len_max / frames_per_window).clone().detach().to(wav_lens.device)

        spectral_lens = torch.tensor([frames_per_window]).repeat(len(wav_lens)).to(wav_lens.device)    # initialize with the max possible frames per clip
        # wav_lens is the real length of the audio clip in samples
        for wi in range(len(wav_lens)):
            #wav_len = wav_lens[wi]      # raw length of the audio clip
            #frames_per_clip = int(wav_lens[wi]/wav_len_per_frame)  # calculate the number of frames for the whole audio clip
            spectral_lens[wi] = torch.ceil(wav_lens[wi]/wav_len_per_frame)
            if (spectral_lens[wi] > frames_per_window):
                raise Exception("WARN: spectral_len > frames_per_window, wav_lens:", spectral_lens[wi], frames_per_window, wav_lens[wi])
            
    print("spectral_lens:", spectral_lens)
    return spectral_lens




class ViterbiDecoder:
    """
    Viterbi decoder that ensures all phonemes in the target sequence are aligned,
    even when they have very low probabilities.
    """
    def __init__(self, blank_id, silence_id, silence_anchors=3,  min_phoneme_prob=1e-8, ignore_noise=True):
        self.blank_id = blank_id
        self.silence_id = silence_id
        self.silence_anchors = silence_anchors  # Number of silence frames to anchor pauses
        self.min_phoneme_prob = min_phoneme_prob  # Minimum probability floor for phonemes
        self.ignore_noise = ignore_noise
        self._neg_inf = -1000.0
        
    def set_blank_id(self, blank_id):
        """Set the blank token ID"""
        self.blank_id = blank_id
    
    

    def _viterbi_decode(self, log_probs, ctc_path, ctc_len, ctc_path_true_idx=None, band_width=0, debug=False):
        """
        Standard Viterbi decoding implementation.

        Args:
            log_probs: Log probabilities tensor [T, C]
            ctc_path: CTC path tensor [ctc_len]
            ctc_len: Length of CTC path
            ctc_path_true_idx: Optional mapping from CTC states to target phoneme indices
            band_width: Sakoe-Chiba band width for diagonal constraint (0 = disabled).
                When > 0, restricts valid CTC states at each frame to a band around the
                expected diagonal position, preventing the alignment from rushing ahead
                or lagging behind. Useful for long sequences.
        """
        if debug: print("Starting Viterbi decoding...")
        num_frames = log_probs.shape[0]
        device = log_probs.device

        # Initialize DP table
        dp = torch.full((num_frames, ctc_len), self._neg_inf, device=device, dtype=torch.float32)
        backpointers = torch.zeros((num_frames, ctc_len), dtype=torch.long, device=device)

        # Precompute diagonal band limits (Sakoe-Chiba band)
        if band_width > 0 and num_frames > 1 and ctc_len > 1:
            pace = (ctc_len - 1) / (num_frames - 1)
            state_indices = torch.arange(ctc_len, device=device, dtype=torch.float32)
            use_band = True
        else:
            use_band = False

        # Initialize first frame
        dp[0, 0] = log_probs[0, self.blank_id]
        if ctc_len > 1:
            dp[0, 1] = log_probs[0, ctc_path[1]]

        # Precompute transition masks
        can_advance = torch.zeros(ctc_len, dtype=torch.bool, device=device)
        can_skip = torch.zeros(ctc_len, dtype=torch.bool, device=device)

        can_advance[1:] = True
        for s in range(2, ctc_len):
            if ctc_path[s] != ctc_path[s-2]:
                can_skip[s] = True

        # Forward pass
        for t in range(1, num_frames):
            frame_log_probs = log_probs[t, ctc_path]
            prev_dp = dp[t-1]

            # Stay transitions
            stay_scores = prev_dp + frame_log_probs

            # Advance transitions
            advance_scores = torch.full_like(stay_scores, self._neg_inf)
            advance_scores[1:] = prev_dp[:-1] + frame_log_probs[1:]

            # Skip transitions
            skip_scores = torch.full_like(stay_scores, self._neg_inf)
            skip_scores[2:] = torch.where(
                can_skip[2:],
                prev_dp[:-2] + frame_log_probs[2:],
                self._neg_inf
            )

            # Find best transitions
            all_scores = torch.stack([stay_scores, advance_scores, skip_scores], dim=1)
            all_prev_states = torch.stack([
                torch.arange(ctc_len, device=device),
                torch.arange(ctc_len, device=device) - 1,
                torch.arange(ctc_len, device=device) - 2
            ], dim=1)

            # Mask invalid transitions
            transition_mask = torch.stack([
                torch.ones(ctc_len, dtype=torch.bool, device=device),  # Can always stay
                can_advance,
                can_skip
            ], dim=1)

            all_scores = torch.where(transition_mask, all_scores, self._neg_inf)

            # Update DP table
            best_transitions = torch.argmax(all_scores, dim=1)
            dp[t] = all_scores[torch.arange(ctc_len), best_transitions]
            backpointers[t] = all_prev_states[torch.arange(ctc_len), best_transitions]

            # Apply diagonal band constraint: mask out states outside the band
            if use_band:
                center = t * pace
                outside_band = (state_indices < center - band_width) | (state_indices > center + band_width)
                dp[t, outside_band] = self._neg_inf
        
    
        # Find best final state
        final_scores = dp[num_frames-1]
        valid_mask = final_scores > self._neg_inf
        if valid_mask.any():
            valid_indices = torch.where(valid_mask)[0]
            final_state = valid_indices[torch.argmax(final_scores[valid_indices])]
        else:
            final_state = torch.argmax(final_scores)
        
        # Backtrack
        path_states = torch.zeros(num_frames, dtype=torch.long, device=device)
        path_states[num_frames-1] = final_state
        
        for t in range(num_frames-2, -1, -1):
            path_states[t] = backpointers[t+1, path_states[t+1]]
        
        
        frame_phonemes = ctc_path[path_states]
        

        if ctc_path_true_idx is not None:
            # Map back to true phoneme indices if provided
            frame_phonemes_idx = ctc_path_true_idx[path_states]

            return frame_phonemes, frame_phonemes_idx
        else: return frame_phonemes, None
    
    
    def _boost_target_phonemes(self, log_probs, target_phonemes, boost_factor=5.0):
        """
        Boost the probabilities of target phonemes to ensure they can be aligned.

        Args:
            log_probs: Original log probabilities [T, C]
            target_phonemes: List of target phoneme indices
            boost_factor: How much to boost target phoneme probabilities

        Returns:
            Modified log probabilities
        """
        boosted_log_probs = log_probs.clone()

        # Get unique target phonemes (excluding blanks and padding)
        unique_targets = set(target_phonemes.tolist())
        unique_targets.discard(self.blank_id)
        unique_targets.discard(-100)  # Remove padding tokens

        for phoneme_idx in unique_targets:
            if phoneme_idx < log_probs.shape[1]:
                # Add boost to target phoneme probabilities
                boosted_log_probs[:, phoneme_idx] += boost_factor

        # Renormalize to maintain probability distribution
        boosted_log_probs = F.log_softmax(boosted_log_probs, dim=-1)

        return boosted_log_probs

    def _enforce_minimum_probabilities(self, log_probs, target_phonemes):
        """
        Ensure target phonemes have minimum probability at each frame.

        Args:
            log_probs: Log probabilities [T, C]
            target_phonemes: Target phoneme sequence

        Returns:
            Modified log probabilities
        """
        modified_log_probs = log_probs.clone()
        min_log_prob = torch.log(torch.tensor(self.min_phoneme_prob, device=log_probs.device))

        # Get unique target phonemes
        unique_targets = set(target_phonemes.tolist())
        unique_targets.discard(self.blank_id)
        unique_targets.discard(-100)

        for phoneme_idx in unique_targets:
            if phoneme_idx < log_probs.shape[1]:
                # Ensure minimum probability for target phonemes
                mask = modified_log_probs[:, phoneme_idx] < min_log_prob
                modified_log_probs[mask, phoneme_idx] = min_log_prob

        return modified_log_probs

    def _detect_silence_segments(self, log_probs, sil_prob_threshold=0.8, min_silence_frames=None, debug=False):
        """
        Detect segments of silence using softmax P(SIL) across all classes.

        Uses the full softmax probability of SIL in context with all other classes
        (including blank/noise). A sliding window averages the per-frame SIL
        probability, and windows where the average exceeds the threshold are
        marked as silence.

        Args:
            log_probs: Log probabilities tensor [T, C]
            sil_prob_threshold: Minimum average P(SIL) within a window to count
                as silence. Default 0.8 (80%).
            min_silence_frames: Minimum consecutive silent frames to keep.
                Defaults to ``self.silence_anchors``.

        Returns:
            List of (start_frame, end_frame) tuples for silence segments.
        """
        if min_silence_frames is None:
            min_silence_frames = self.silence_anchors

        sliding_frames = min_silence_frames
        num_frames = log_probs.shape[0]
        num_classes = log_probs.shape[1]

        if self.silence_id >= num_classes:
            return []
        if num_frames < sliding_frames:
            return []

        # Full softmax P(SIL) across all classes including blank
        probs = torch.exp(log_probs)  # [T, C]
        sil_prob = probs[:, self.silence_id]  # [T]

        # Sliding-window average for robustness
        if sliding_frames > 1:
            cumsum = torch.cumsum(sil_prob, dim=0)
            padded = torch.cat([torch.zeros(1, device=cumsum.device), cumsum])
            window_avg = (padded[sliding_frames:] - padded[:-sliding_frames]) / sliding_frames
        else:
            window_avg = sil_prob

        if debug: print(f"Silence detection: max_avg_prob={window_avg.max().item():.4f}, threshold={sil_prob_threshold}")

        # Threshold: which windows have >= threshold SIL probability?
        is_silent = window_avg >= sil_prob_threshold

        # Convert boolean mask to contiguous segments
        silence_segments = []
        in_silence = False
        silence_start = 0

        for i in range(len(is_silent)):
            if is_silent[i] and not in_silence:
                in_silence = True
                silence_start = i
            elif not is_silent[i] and in_silence:
                in_silence = False
                seg_end = i + sliding_frames - 1  # extend to cover the full window
                seg_end = min(seg_end, num_frames)
                if seg_end - silence_start >= min_silence_frames:
                    silence_segments.append((silence_start, seg_end))

        # Handle sequence ending in silence
        if in_silence:
            seg_end = num_frames
            if seg_end - silence_start >= min_silence_frames:
                silence_segments.append((silence_start, seg_end))

        return silence_segments

    def _find_target_sil_groups(self, true_sequence):
        """
        Find groups of consecutive SIL tokens in the target sequence.

        Args:
            true_sequence: Target phoneme tensor [S]

        Returns:
            List of (start_idx, end_idx) tuples (half-open intervals in target)
        """
        seq = true_sequence.tolist() if isinstance(true_sequence, torch.Tensor) else list(true_sequence)
        groups = []
        i = 0
        while i < len(seq):
            if seq[i] == self.silence_id:
                start = i
                while i < len(seq) and seq[i] == self.silence_id:
                    i += 1
                groups.append((start, i))
            else:
                i += 1
        return groups

    def _match_silences(self, sil_groups, audio_silences, target_len, num_frames, best_dist_threshold=0.3):
        """
        Match target SIL groups to audio silence segments by ordered proportional position.

        Args:
            sil_groups: List of (start, end) in target sequence
            audio_silences: List of (start_frame, end_frame) in audio
            target_len: Total length of target sequence
            num_frames: Total number of audio frames
            best_dist_threshold: Maximum allowed distance (as proportion of sequence) for a match
        Returns:
            List of (target_group_idx, audio_silence_idx) matched pairs (ordered)
        """
        if not sil_groups or not audio_silences or target_len == 0 or num_frames == 0:
            return []

        matches = []
        audio_idx = 0

        for tg_idx, (tg_start, tg_end) in enumerate(sil_groups):
            target_pos = (tg_start + tg_end) / 2.0 / target_len

            best_audio = None
            best_dist = float('inf')

            for ai in range(audio_idx, len(audio_silences)):
                as_start, as_end = audio_silences[ai]
                audio_pos = (as_start + as_end) / 2.0 / num_frames
                dist = abs(target_pos - audio_pos)

                if dist < best_dist:
                    best_dist = dist
                    best_audio = ai
                elif dist > best_dist:
                    break  # Getting farther, stop searching

            if best_audio is not None and best_dist < best_dist_threshold:
                matches.append((tg_idx, best_audio))
                audio_idx = best_audio + 1  # Enforce ordering

        return matches

    def _anchor_silence_in_log_probs(self, log_probs, silence_segments, boost=10.0):
        """
        Boost blank probability at detected silence frames so the Viterbi
        naturally stays on blank states during pauses, without splitting
        the target sequence.

        Args:
            log_probs: Log probabilities tensor [T, C]
            silence_segments: List of (start_frame, end_frame) tuples
            boost: How much to boost blank probability (in log space before re-normalization)

        Returns:
            Modified log probabilities
        """
        modified = log_probs.clone()
        for start, end in silence_segments:
            modified[start:end, self.blank_id] += boost
            modified[start:end] = F.log_softmax(modified[start:end], dim=-1)
        return modified

    def _segmented_viterbi_decode(self, log_probs, true_sequence, true_sequence_idx,
                                   boundary_pad=3, min_speech_frames=20, debug=False):
        """
        Segmented Viterbi: match target SIL groups to audio silences,
        split at matched points, run Viterbi on each segment.

        Args:
            log_probs: Modified log probabilities [T, C]
            true_sequence: Target phoneme tensor [S]
            true_sequence_idx: Target index tensor [S]
            boundary_pad: Number of frames to pad on each side of speech segments
                so that phonemes near silence edges aren't clipped.
            min_speech_frames: Minimum frames for a speech segment.  Segments
                shorter than this are merged with the nearest neighbor.

        Returns (frame_phonemes, frame_phonemes_idx) or None if segmentation
        is not possible (caller should fall back to single Viterbi).
        """
        device = log_probs.device
        num_frames = log_probs.shape[0]
        target_len = true_sequence.shape[0]

        # Step 1: find anchor points in both sequences
        sil_groups = self._find_target_sil_groups(true_sequence)

        if len(sil_groups) == 0:
            if debug: print("WARN: No target SIL groups detected in the target sequence. No punctuation detected. Use punctuated text or disable silence anchoring for better alignment.")
            return [], []  # No silences to anchor on, fallback to single Viterbi
        min_silence_frames = self.silence_anchors
        audio_silences = self._detect_silence_segments(log_probs, sil_prob_threshold=0.9, min_silence_frames=min_silence_frames, debug=debug)
        if len(audio_silences) == 0 and len(true_sequence) > 200:
            new_thresh = 1.0 - (0.09 * min_silence_frames)
            new_thresh = max(0.05, new_thresh)  # Ensure threshold stays positive

            if debug: print(f"WARN: No audio silences detected. Retrying with lower threshold {new_thresh:.2f}")
            audio_silences = self._detect_silence_segments(log_probs, sil_prob_threshold=new_thresh, min_silence_frames=min_silence_frames, debug=debug)
        if len(audio_silences) == 0 and len(true_sequence) > 200 and min_silence_frames > 3:
            if debug: print("WARN: No audio silences detected. Retrying with min_silence_frames=3 , previously=", min_silence_frames)
            min_silence_frames = 3
            audio_silences = self._detect_silence_segments(log_probs, sil_prob_threshold=0.9, min_silence_frames=min_silence_frames, debug=debug)

        if debug:
            print(f"Detected target SIL (text punctuation) at indices: {sil_groups}")
            print(f"Detected audio silences at frames: {audio_silences}")

        if len(audio_silences) == 0:
            if debug: print("WARN:  No audio silences detected in the audio. Skipping segmented alignment.")
            return [], []  # No silences to anchor on, fallback to single Viterbi

        if not sil_groups or not audio_silences:
            return [], []

        # Step 2: match target SIL groups to audio silences
        matches = self._match_silences(sil_groups, audio_silences, target_len, num_frames)
        if not matches:
            return [], []

        # Step 3: build ordered list of (audio_start, audio_end, target_start, target_end, is_silence)
        segments = []
        prev_audio = 0
        prev_target = 0

        for tg_idx, as_idx in matches:
            tg_start, tg_end = sil_groups[tg_idx]
            as_start, as_end = audio_silences[as_idx]

            # Speech segment before this silence
            if prev_audio < as_start and prev_target < tg_start:
                segments.append((prev_audio, as_start, prev_target, tg_start, False))
            elif prev_audio < as_start:
                # Audio gap but no target phonemes — fill with blank
                segments.append((prev_audio, as_start, prev_target, prev_target, False))

            # Silence segment
            segments.append((as_start, as_end, tg_start, tg_end, True))

            prev_audio = as_end
            prev_target = tg_end

        # Final speech segment after last matched silence
        if prev_audio < num_frames and prev_target < target_len:
            segments.append((prev_audio, num_frames, prev_target, target_len, False))
        elif prev_audio < num_frames:
            # Remaining audio with no target phonemes — fill with blank
            segments.append((prev_audio, num_frames, prev_target, prev_target, False))

        # Step 3b: merge speech segments that are too short
        merged = []
        for seg in segments:
            audio_start, audio_end, tgt_start, tgt_end, is_silence = seg
            n_frames = audio_end - audio_start
            n_phonemes = tgt_end - tgt_start

            if not is_silence and n_phonemes > 0 and n_frames < min_speech_frames and merged:
                # Merge with previous segment (extend its audio/target range)
                prev = merged[-1]
                merged[-1] = (prev[0], audio_end, prev[2], tgt_end, False)
            else:
                merged.append(seg)
        segments = merged

        # Step 4: process each segment
        all_frame_phonemes = []
        all_frame_phonemes_idx = []

        if debug: print(f"Segments after silence anchoring: {segments}")

        for audio_start, audio_end, tgt_start, tgt_end, is_silence in segments:
            n_frames_seg = audio_end - audio_start
            if n_frames_seg <= 0:
                continue

            if is_silence:
                # Fill silence frames with SIL, distribute target SIL indices evenly
                sil_phonemes = torch.full((n_frames_seg,), self.silence_id, dtype=torch.long, device=device)
                n_sils = tgt_end - tgt_start
                if n_sils > 0:
                    sil_idx = torch.full((n_frames_seg,), -1, dtype=torch.long, device=device)
                    frames_per_sil = n_frames_seg / n_sils
                    for k in range(n_sils):
                        f_start = int(k * frames_per_sil)
                        f_end = int((k + 1) * frames_per_sil)
                        sil_idx[f_start:f_end] = tgt_start + k
                else:
                    sil_idx = torch.full((n_frames_seg,), -1, dtype=torch.long, device=device)

                all_frame_phonemes.append(sil_phonemes)
                all_frame_phonemes_idx.append(sil_idx)
            else:
                # Apply boundary padding: extend the audio slice into
                # neighboring silence regions so edge phonemes aren't clipped.
                padded_start = max(0, audio_start - boundary_pad)
                padded_end = min(num_frames, audio_end + boundary_pad)
                pad_left = audio_start - padded_start  # frames added on the left

                seg_target = true_sequence[tgt_start:tgt_end]
                seg_target_idx = true_sequence_idx[tgt_start:tgt_end]
                seg_log_probs = log_probs[padded_start:padded_end]

                if seg_target.shape[0] == 0:
                    # No target phonemes for this audio — fill with blank
                    all_frame_phonemes.append(torch.full((n_frames_seg,), self.blank_id, dtype=torch.long, device=device))
                    all_frame_phonemes_idx.append(torch.full((n_frames_seg,), -1, dtype=torch.long, device=device))
                else:
                    # Boost blank at detected sub-silence frames within this segment
                    seg_log_probs = self._anchor_silence_in_log_probs(
                        seg_log_probs,
                        self._detect_silence_segments(seg_log_probs, min_silence_frames=min_silence_frames),
                        boost=5.0
                    )

                    # Build CTC path and run Viterbi
                    seg_seq_len = seg_target.shape[0]
                    stride = 4
                    if (stride * seg_seq_len + 1) > seg_log_probs.shape[0] * 0.9: stride = 3
                    if (stride * seg_seq_len + 1) > seg_log_probs.shape[0] * 0.8: stride = 2
                    expanded_len = stride * seg_seq_len + 1
                    if expanded_len > seg_log_probs.shape[0] * 1.2:
                        return [], []  # return empty results to trigger fallback to non-segmented Viterbi

                    ctc_path = torch.zeros(expanded_len, device=device, dtype=torch.long).fill_(self.blank_id)
                    ctc_path_true_idx = torch.zeros(expanded_len, device=device, dtype=torch.long).fill_(-1)
                    ctc_path[1::stride] = seg_target
                    ctc_path_true_idx[1::stride] = seg_target_idx
                    ctc_len = len(ctc_path)

                    band_width = max(ctc_len // 3, 30) if ctc_len > 60 else 0
                    seg_frames, seg_frames_idx = self._viterbi_decode(
                        seg_log_probs, ctc_path, ctc_len, ctc_path_true_idx, band_width=band_width
                    )

                    # Trim the padding back off so we return exactly n_frames_seg frames
                    seg_frames = seg_frames[pad_left: pad_left + n_frames_seg]
                    seg_frames_idx = seg_frames_idx[pad_left: pad_left + n_frames_seg]

                    all_frame_phonemes.append(seg_frames)
                    all_frame_phonemes_idx.append(seg_frames_idx)

        # Step 5: concatenate
        if not all_frame_phonemes:
            return [], []

        frame_phonemes = torch.cat(all_frame_phonemes, dim=0)
        frame_phonemes_idx = torch.cat(all_frame_phonemes_idx, dim=0)

        # Verify total frames match (pad/trim if off by rounding)
        if frame_phonemes.shape[0] < num_frames:
            pad = num_frames - frame_phonemes.shape[0]
            frame_phonemes = torch.cat([frame_phonemes, torch.full((pad,), self.blank_id, dtype=torch.long, device=device)])
            frame_phonemes_idx = torch.cat([frame_phonemes_idx, torch.full((pad,), -1, dtype=torch.long, device=device)])
        elif frame_phonemes.shape[0] > num_frames:
            frame_phonemes = frame_phonemes[:num_frames]
            frame_phonemes_idx = frame_phonemes_idx[:num_frames]

        return frame_phonemes, frame_phonemes_idx

    def decode_with_forced_alignment(self, log_probs, true_sequence, return_scores=False,
                                   boost_targets=True, enforce_minimum=True, anchor_pauses=True, debug=False):
        """
        Decode that ensures all target phonemes are aligned.

        Args:
            log_probs: Log probabilities tensor [T, C]
            true_sequence: Target phoneme sequence [S]
            return_scores: Whether to return alignment scores
            boost_targets: Whether to boost target phoneme probabilities
            enforce_minimum: Whether to enforce minimum probabilities
            anchor_pauses: Whether to use silence-anchored segmented alignment.
        Returns:
            frame_phonemes: Frame-level phoneme assignments
            frame_phonemes_idx: Frame-level target sequence indices
            alignment_score: Optional alignment score
        """
        if self.blank_id is None:
            raise ValueError("Blank ID not set. Call set_blank_id first.")

        seq_len = true_sequence.shape[0]
        num_frames = log_probs.shape[0]
        device = log_probs.device
        true_sequence_idx = torch.arange(seq_len, device=device)

        if seq_len == 0:
            frame_phonemes = torch.full((num_frames,), self.blank_id, dtype=torch.long, device=device)
            frame_phonemes_idx = torch.full((num_frames,), -1, dtype=torch.long, device=device)
            if return_scores:
                alignment_score = log_probs[:, self.blank_id].sum()
                return frame_phonemes, frame_phonemes_idx, alignment_score
            return frame_phonemes, frame_phonemes_idx, None

        # Apply probability modifications if requested
        modified_log_probs = log_probs.clone()

        if boost_targets:
            modified_log_probs = self._boost_target_phonemes(modified_log_probs, true_sequence)
            if debug: print("Applied target phoneme boosting.")

        if enforce_minimum:
            modified_log_probs = self._enforce_minimum_probabilities(modified_log_probs, true_sequence)
            if debug: print("Applied minimum probability enforcement.")

        # Try segmented alignment: match target SIL groups with audio silences,
        # split at matched points, run Viterbi on each correctly-assigned segment.
        if anchor_pauses and self.silence_id is not None:
            if debug: print("Attempting segmented Viterbi alignment with silence anchoring...")
            result = self._segmented_viterbi_decode(
                modified_log_probs, true_sequence, true_sequence_idx, debug=debug
            )
            if result is not None and len(result[0]) == num_frames:

                if debug: print("Segmented Viterbi alignment successful.")
                frame_phonemes, frame_phonemes_idx = result
                if return_scores:
                    alignment_score = self._calculate_alignment_score(log_probs, frame_phonemes)
                    return frame_phonemes, frame_phonemes_idx, alignment_score
                return frame_phonemes, frame_phonemes_idx, None

        if debug: print("Running standard Viterbi...")

        # Fallback: standard Viterbi (with band constraint for long sequences)
        stride = 4
        if (stride * seq_len + 1) > num_frames * 0.9: stride = 3
        if (stride * seq_len + 1) > num_frames * 0.8: stride = 2
        expanded_len = stride * seq_len + 1
        if expanded_len > num_frames:
            raise ValueError(f"Target sequence too long to align: expanded CTC path length {expanded_len} (for {seq_len} phonemes) exceeds number of frames {num_frames}. Consider reducing the target sequence or increasing the audio length.")

        ctc_path = torch.zeros(expanded_len, device=device, dtype=torch.long).fill_(self.blank_id)
        ctc_path_true_idx = torch.zeros(expanded_len, device=device, dtype=torch.long).fill_(-1)
        ctc_path[1::stride] = true_sequence
        ctc_path_true_idx[1::stride] = true_sequence_idx
        ctc_len = len(ctc_path)

        band_width = max(ctc_len // 4, 20) if ctc_len > 60 else 0
        frame_phonemes, frame_phonemes_idx = self._viterbi_decode(
            modified_log_probs, ctc_path, ctc_len, ctc_path_true_idx, band_width=band_width, debug=debug
        )

        if return_scores:
            alignment_score = self._calculate_alignment_score(log_probs, frame_phonemes)
            return frame_phonemes, frame_phonemes_idx, alignment_score

        return frame_phonemes, frame_phonemes_idx, None

    def _calculate_alignment_score(self, log_probs, frame_phonemes):
        """Calculate the total alignment score."""
        total_score = 0.0
        for t, phoneme_idx in enumerate(frame_phonemes):
            if phoneme_idx < log_probs.shape[1]:
                total_score += log_probs[t, phoneme_idx].item()
        return total_score

    # Modified assort_frames method for the ViterbiDecoder - 2026-02-10
    def assort_frames(self, frame_phonemes, frame_phonemes_idx, max_blanks=10):
        """
        Assort_frames that handles forced phoneme alignments better.
        """
        if len(frame_phonemes) == 0:
            return []
        
        device = frame_phonemes.device
        
        if not isinstance(frame_phonemes, torch.Tensor):
            frame_phonemes = torch.tensor(frame_phonemes, device=device)
        
        if not isinstance(frame_phonemes_idx, torch.Tensor):
            frame_phonemes_idx = torch.tensor(frame_phonemes_idx, device=device)
        
        # Find all non-blank segments
        #is_blank = frame_phonemes == self.blank_id
        
        # Find transition points: split when phoneme OR target index changes.
        # This ensures consecutive SIL tokens with different target positions
        # (e.g., from two punctuation-derived SILs) produce separate entries.
        transitions = torch.cat([
            torch.tensor([True], device=device),
            (frame_phonemes[1:] != frame_phonemes[:-1]) | (frame_phonemes_idx[1:] != frame_phonemes_idx[:-1])
        ])
        
        transition_indices = torch.where(transitions)[0]
        
        framestamps = []
        for i in range(len(transition_indices)):
            start_idx = transition_indices[i].item()
            end_idx = transition_indices[i + 1].item() if i + 1 < len(transition_indices) else len(frame_phonemes)
            
            segment_phoneme = frame_phonemes[start_idx].item()
            segment_phoneme_idx = frame_phonemes_idx[start_idx].item()
            if segment_phoneme_idx == -1:
                for idx in range(start_idx, end_idx):
                    if frame_phonemes_idx[idx].item() != -1:
                        segment_phoneme_idx = frame_phonemes_idx[idx].item()
                        break
        
            # Skip overly long blank segments
            if segment_phoneme == self.blank_id:
                segment_length = end_idx - start_idx
                if (self.ignore_noise):
                    if segment_length > max_blanks:
                        continue
                else:
                    # If ignore_noise is False, include long blank segments as noise
                    if segment_length > max_blanks:
                        framestamps.append((segment_phoneme, start_idx, end_idx, segment_phoneme_idx))  # -1 for target_seq_idx, to be filled later if needed
            
            # For non-blank segments, always include them (even if very short)
            if segment_phoneme != self.blank_id:
                framestamps.append((segment_phoneme, start_idx, end_idx, segment_phoneme_idx))  # -1 for target_seq_idx, to be filled later if needed
                
        
        return framestamps

class AlignmentUtils: # 2025-09-10
    """
    Alignment utilities with forced alignment fixes.
    """
    
    def __init__(self, blank_id, silence_id, silence_anchors=10, ignore_noise=True):
        '''
        Initialize AlignmentUtils.
        Args:
            blank_id: ID of the blank token
            silence_id: ID of the silence token
            silence_anchors: Number of silence frames to anchor pauses (slice at silences for easy alignment). Set `0` to disable. Default is `10`. Set a lower value to increase sensitivity to silences.
        '''
        self.blank_id = blank_id
        self.silence_id = silence_id
        self.silence_anchors = silence_anchors  # Number of silence frames to anchor pauses
        self.viterbi_decoder = ViterbiDecoder(blank_id, silence_id, silence_anchors=self.silence_anchors, ignore_noise=ignore_noise)
    
    

    def decode_alignments_simple(self, log_probs, true_seqs, pred_lens=None, true_seqs_lens=None):
        """
        Simple forced alignment: CTC path construction + Viterbi + assort.
        No boosting, no silence anchoring. Designed for easy C++ porting.

        Args:
            log_probs: Log probabilities tensor [B, T, C]
            true_seqs: Target phoneme sequences [B, S]
            pred_lens: Optional prediction lengths [B]
            true_seqs_lens: Target sequence lengths [B]

        Returns:
            List of frame-level alignments per batch item.
            Each is a list of (phoneme_id, start_frame, end_frame, target_seq_idx).
        """
        batch_size = log_probs.shape[0]
        device = log_probs.device
        blank_id = self.viterbi_decoder.blank_id

        assorted = []
        for i in range(batch_size):
            # Extract per-sample tensors
            T = pred_lens[i] if pred_lens is not None else log_probs.shape[1]
            lp = log_probs[i, :T]  # [T, C]

            S = true_seqs_lens[i] if true_seqs_lens is not None else true_seqs.shape[1]
            seq = true_seqs[i, :S]  # [S]
            seq_idx = torch.arange(S, device=device)

            # Build CTC path: blank-phoneme-blank-phoneme-...-blank
            # stride between phonemes chosen to fit within T frames
            stride = 4
            if stride * S + 1 > T * 0.9:
                stride = 3
            if stride * S + 1 > T * 0.8:
                stride = 2
            ctc_len = stride * S + 1

            ctc_path = torch.full((ctc_len,), blank_id, device=device, dtype=torch.long)
            ctc_path_idx = torch.full((ctc_len,), -1, device=device, dtype=torch.long)
            ctc_path[1::stride] = seq
            ctc_path_idx[1::stride] = seq_idx

            # Band constraint for long sequences
            band_width = max(ctc_len // 4, 20) if ctc_len > 60 else 0

            # Viterbi decode
            frame_phonemes, frame_phonemes_idx = self.viterbi_decoder._viterbi_decode(
                lp, ctc_path, ctc_len, ctc_path_idx, band_width=band_width
            )

            # Assort into (phoneme, start, end, target_idx) segments
            frames = self.viterbi_decoder.assort_frames(frame_phonemes, frame_phonemes_idx)
            assorted.append(frames)

        return assorted

    def decode_alignments(self, log_probs, true_seqs=None, pred_lens=None,
                         true_seqs_lens=None, forced_alignment=True,
                         boost_targets=True, enforce_minimum=True, debug=False):
        """
        Decode alignments with better forced alignment.

        Args:
            log_probs: Log probabilities tensor [B, T, C]
            true_seqs: Target phoneme sequences [B, S]
            pred_lens: Prediction lengths [B]
            true_seqs_lens: Target sequence lengths [B]
            forced_alignment: Whether to use forced alignment
            boost_targets: Whether to boost target phoneme probabilities
            enforce_minimum: Whether to enforce minimum probabilities
        Returns:
            List of frame-level alignments, tuples of (phoneme_idx, start_frame, end_frame, target_seq_idx)
            where target_seq_idx is the index of the phoneme in the true_seqs (-1 for blanks or unmatched)
        """
        batch_size = log_probs.shape[0]
        device = log_probs.device

        if forced_alignment:
            if (true_seqs is None) or (true_seqs_lens is None):
                raise ValueError("Phoneme sequences and lengths required for forced alignment")

            with torch.no_grad():
                # Using silence anchoring for forced alignment
                all_frame_phonemes = []

                for i in range(batch_size):
                    if pred_lens is not None:
                        log_probs_seq = log_probs[i][:pred_lens[i]]
                    else:
                        log_probs_seq = log_probs[i]

                    true_seq_len = true_seqs_lens[i]
                    true_seq = true_seqs[i, :true_seq_len]

                    if true_seq_len == 0:
                        # Empty sequence - fill with blanks
                        all_frame_phonemes.append((torch.tensor([], device=device, dtype=torch.long), torch.tensor([], device=device, dtype=torch.long)))
                        continue

                    # Decode with forced alignment and silence anchoring
                    frame_phonemes, frame_phonemes_idx, _ = self.viterbi_decoder.decode_with_forced_alignment(
                        log_probs_seq, true_seq, return_scores=False,
                        boost_targets=boost_targets, enforce_minimum=enforce_minimum, anchor_pauses=self.silence_anchors > 0,
                        debug=debug
                    )
                    all_frame_phonemes.append((frame_phonemes, frame_phonemes_idx))

                # Apply assort_frames and add target sequence indices
                assorted = [self.viterbi_decoder.assort_frames(fp, fpi) for fp, fpi in all_frame_phonemes]

                return assorted

        else:
            # Free decoding
            if pred_lens is not None:
                max_len = max(pred_lens)
                mask = torch.arange(max_len, device=device)[None, :] < torch.tensor(pred_lens, device=device)[:, None]
                pred_phonemes = torch.argmax(log_probs, dim=2)
                pred_phonemes = pred_phonemes * mask.long() + (~mask).long() * self.blank_id
                frame_phonemes = [pred_phonemes[i, :pred_lens[i]] for i in range(batch_size)]
            else:
                frame_phonemes = [torch.argmax(log_probs[i], dim=1).long() for i in range(batch_size)]

            assorted = []
            for frame_phonemes_seq in frame_phonemes:
                frames = self.viterbi_decoder.assort_frames(frame_phonemes_seq, torch.full_like(frame_phonemes_seq, -1))
                assorted.append(frames)
            return assorted


class PhonemeTimestampAligner:

    def __init__(self, preset="en-us", model_name=None, cupe_ckpt_path=None, lang='en-us', mapper="ph66", duration_max=10, output_frames_key="phoneme_idx", device="cpu", silence_anchors=10, boost_targets=True, enforce_minimum=True, enforce_all_targets=True, ignore_noise=True):
        self.device = device

        if cupe_ckpt_path is not None:
            cupe_ckpt_path = cupe_ckpt_path
        elif model_name is not None:
            # cupe_ckpt_path = self.download_model(model_name=model_name)
            print("Using model name:", model_name)
        else:
            raise ValueError("Either cupe_ckpt_path or model_name must be provided.")
        if cupe_ckpt_path is None or not os.path.exists(cupe_ckpt_path):
            raise ValueError("CUPE model checkpoint not found.", cupe_ckpt_path)
        self.extractor = CUPEONNXPredictor(cupe_ckpt_path)
        
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
            self.phonemizer = Phonemizer(language=lang,remove_noise_phonemes=True)

        self.phonemes_ipa_key = "ipa"
        self.phonemes_key = "ph66"
        self.phoneme_groups_key = "pg16"
        self.phoneme_id_to_label = {v: k for k, v in phoneme_mapped_index.items()}
        self.phoneme_label_to_id = {label: idx for idx, label in self.phoneme_id_to_label.items()}
        self.group_id_to_label = {v: k for k, v in phoneme_groups_index.items()}
        self.group_label_to_id = {label: idx for idx, label in self.group_id_to_label.items()}
        self.phoneme_id_to_group_id = phoneme_groups_mapper


        self.silence_anchors = silence_anchors
        self.boost_targets = boost_targets
        self.enforce_minimum = enforce_minimum
        self.enforce_all_targets = enforce_all_targets
        self.ignore_noise = ignore_noise
        self._setup_config()
        self._setup_decoders()
        
    def _setup_config(self, window_size_ms=120, stride_ms=80):
        """Setup configuration parameters."""
        self.window_size_ms = window_size_ms
        self.stride_ms = stride_ms
        self.sample_rate = 16000
        
        # Calculate window parameters
        self.window_size_wav = int(window_size_ms * self.sample_rate / 1000)
        self.stride_size_wav = int(stride_ms * self.sample_rate / 1000)
        
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

    def _cupe_prediction(self, audio_batch, wav_len, extract_embeddings=False):

        if audio_batch.dim() == 1:
            audio_batch = audio_batch.unsqueeze(0)
        print("audio shape:", audio_batch.shape)
        print("wav_len:", wav_len)
        # Window the audio
        windowed_audio = slice_windows(
            audio_batch.to(self.device), 
            self.sample_rate, 
            self.window_size_ms, 
            self.stride_ms
        )
        print("Windowed audio shape:", windowed_audio.shape)
        
        batch_size, num_windows, window_size = windowed_audio.shape
        windows_flat = windowed_audio.reshape(-1, window_size)

        print("Reshaped Window:", windows_flat.shape)
        # Get predictions
        if extract_embeddings:
            logits_class, logits_group, embeddings = self.extractor.predict(
                windows_flat
            )
        else:
            logits_class, logits_group, _ = self.extractor.predict(
                windows_flat, 
                return_embeddings=False
            )
            embeddings = None
        
        frames_per_window = logits_group.shape[1]
        
        # Reshape outputs
        logits_class = logits_class.reshape(batch_size, num_windows, frames_per_window, -1)
        logits_group = logits_group.reshape(batch_size, num_windows, frames_per_window, -1)
        print("Logits class reshape:", logits_class.shape, "\nLogits group reshape:", logits_group.shape)
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
        print("Logits class after stiching:", logits_class.shape, "\nLogits group after stiching:", logits_group.shape)

        '''
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
        '''
        # Calculate spectral length
        spectral_len = calc_spec_len_ext(
            torch.tensor([wav_len]), 
            self.window_size_ms, 
            self.stride_ms, 
            self.sample_rate, 
            torch.tensor(frames_per_window, dtype=torch.long)
        )[0].item()
        print("Calculated spectral length:", spectral_len)
        return logits_class, logits_group, embeddings, spectral_len
    
    def extract_timestamps_from_segment_simplified(self, wavs, wav_lens, phoneme_sequences, start_offset_times=0.0, debug=True):
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

        # CUPE forward pass (no embeddings) — single batch item at a time
        # For true batching, loop over items and collect results
        batch_size = wavs.shape[0]
        # reshape to (batch_size, 1, wav_len) for compatibility with _cupe_prediction
        wavs = wavs.unsqueeze(1)

        all_logits = []
        spectral_lens = []
        for i in range(batch_size):
            logits_class_i, _, _, spec_len_i = self._cupe_prediction(wavs[i:i+1], wav_lens[i])
            all_logits.append(logits_class_i)
            spectral_lens.append(spec_len_i)
        logits_class = torch.cat(all_logits, dim=0)

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
            frame_phonemes[b] = self.convert_to_ms(
                frame_phonemes[b], spectral_lens[b], offset, wav_lens[b], self.resampler_sample_rate,
            )

        timestamp_dicts = [{'phoneme_timestamps': frame_phonemes[b]} for b in range(len(frame_phonemes))]

        return timestamp_dicts

    def extract_timestamps_from_segment_advanced(self, wavs, wav_lens, phoneme_sequences, start_offset_times=0.0, debug=True):
        """
        Advanced version of extract_timestamps_from_segment using full forced alignment
        with target boosting, silence anchoring, and confidence scoring.

        Args:
            wavs: Audio tensors [B, samples]
            wav_lens: List of audio lengths in samples
            phoneme_sequences: List of phoneme sequences (each as list of phoneme indices)
            start_offset_times: Start time offsets in seconds (list or single value)
            debug: Enable debug output

        Returns:
            timestamp_dicts: List of dicts with 'phoneme_timestamps'
        """
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

        # CUPE forward pass — single batch item at a time
        batch_size = wavs.shape[0]
        wavs_input = wavs.unsqueeze(1) if wavs.dim() == 2 else wavs

        all_logits = []
        spectral_lens = []
        for i in range(batch_size):
            logits_class_i, _, _, spec_len_i = self._cupe_prediction(wavs_input[i:i+1], wav_lens[i])
            all_logits.append(logits_class_i)
            spectral_lens.append(spec_len_i)
        logits_class = torch.cat(all_logits, dim=0)

        # Log softmax
        log_probs_p = F.log_softmax(logits_class, dim=2)

        # Full alignment with boosting and silence anchoring
        ph_seqs = phoneme_sequences.to(self.device)
        ph_seq_lens_t = torch.tensor(ph_seq_lens, dtype=torch.long, device=self.device)
        spectral_lens_t = torch.tensor(spectral_lens, dtype=torch.long, device=self.device)

        frame_phonemes = self.alignment_utils_p.decode_alignments(
            log_probs_p,
            true_seqs=ph_seqs,
            pred_lens=spectral_lens_t,
            true_seqs_lens=ph_seq_lens_t,
            forced_alignment=True,
            boost_targets=self.boost_targets,
            enforce_minimum=self.enforce_minimum,
            debug=debug,
        )

        # Calculate confidence scores and convert to ms timestamps
        for b in range(len(frame_phonemes)):
            frame_phonemes[b] = self._calculate_confidences(log_probs_p[b], frame_phonemes[b])

            offset = start_offset_times[b] if isinstance(start_offset_times, (list, tuple)) else start_offset_times
            frame_phonemes[b] = self.convert_to_ms(
                frame_phonemes[b], spectral_lens[b], offset, wav_lens[b], self.resampler_sample_rate,
            )

            # Sort by start timestamp (ms) after any reordering
            frame_phonemes[b] = sorted(frame_phonemes[b], key=lambda x: x[4])

        timestamp_dicts = [{'phoneme_timestamps': frame_phonemes[b]} for b in range(len(frame_phonemes))]

        return timestamp_dicts



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

        return self.phonemizer.phonemize_sentence(text)

    def ceil(self, float_value): 
        return int(float_value) + (float_value % 1 > 0)

if __name__ == "__main__":

    # Configuration
    onnx_path = "models/en_libri1000_ua01c_e4_val_GER=0.2186.ckpt.onnx"
    audio_path = "examples/samples/audio/109867__timkahn__butterfly.wav"
    text_sentence = "butterfly"

    # Initialize aligner
    extractor = PhonemeTimestampAligner(
        cupe_ckpt_path=onnx_path,
        lang="en-us",
        duration_max=10,
        device="cpu",
    )

    # Load audio (16kHz mono)
    wav_float, sr = librosa.load(audio_path, sr=16000, mono=True)
    audio_wav = torch.from_numpy(wav_float).unsqueeze(0)  # [1, samples]

    # Phonemize
    ts_out = extractor.phonemize_sentence(text_sentence)
    phoneme_seq = ts_out[extractor.phonemes_key]
    ipa_labels = ts_out.get("eipa", [])
    print(f"Text: '{text_sentence}'")
    print(f"IPA:  {ipa_labels}")
    print(f"ph66: {phoneme_seq}")

    # Chop/pad audio
    wav, wav_len = extractor.chop_wav(audio_wav, 0, audio_wav.shape[1])
    wavs = wav.unsqueeze(0)  # [1, padded_samples]

    # Run simplified pipeline
    t0 = time.time()
    timestamp_dicts = extractor.extract_timestamps_from_segment_simplified(
        wavs, [wav_len],
        phoneme_sequences=[phoneme_seq],
        start_offset_times=[0.0],
        debug=False,
    )
    elapsed_ms = (time.time() - t0) * 1000

    # Print results
    phoneme_ts = timestamp_dicts[0]["phoneme_timestamps"]
    print(f"\nAligned {len(phoneme_ts)} phonemes in {elapsed_ms:.1f} ms:\n")
    for i, tup in enumerate(phoneme_ts):
        ph_id = tup[0]
        start_ms, end_ms = tup[4], tup[5]
        label = index_to_plabel.get(ph_id, f"UNK_{ph_id}")
        ipa = ipa_labels[i] if i < len(ipa_labels) else "?"
        print(f"  {i+1:>2}: {label:>6s} ({ipa:>3s})  {start_ms:7.1f} - {end_ms:7.1f} ms")

    ''' prints:
    Setting espeak backend for language: en-us
    Text: 'butterfly'
    IPA:  ['b', 'ʌ', 'ɾ', 'ɚ', 'f', 'l', 'aɪ']
    ph66: [29, 10, 58, 9, 43, 56, 23]
    audio shape: torch.Size([1, 1, 160000])
    wav_len: 20141
    Windowed audio shape: torch.Size([1, 124, 1920])
    Reshaped Window: torch.Size([124, 1920])
    Logits class reshape: torch.Size([1, 124, 10, 67]) 
    Logits group reshape: torch.Size([1, 124, 10, 17])
    Logits class after stiching: torch.Size([1, 620, 67]) 
    Logits group after stiching: torch.Size([1, 620, 17])
    wav_lens: tensor([20141])
    window_size_wav: 1920 stride_size_wav: 1280 frames_per_window: tensor(10)
    total_frames: tensor(75)
    spectral_lens: tensor([75])
    Calculated spectral length: 75

    Aligned 7 phonemes in 315.7 ms:

    1:      b (  b)     33.6 -    50.4 ms
    2:      ʌ (  ʌ)    100.7 -   117.5 ms
    3:      ɾ (  ɾ)    134.3 -   151.1 ms
    4:      ɚ (  ɚ)    285.3 -   302.1 ms
    5:      f (  f)    369.3 -   386.0 ms
    6:      l (  l)    520.3 -   553.9 ms
    7:     aɪ ( aɪ)    604.2 -   621.0 ms
    
    '''