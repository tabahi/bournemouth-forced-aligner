'''
Updated on 2025-08-07 to better forced phoneme alignment capabilities. All phonemes in the target sequence are ensured to be aligned, even when they have very low probabilities.
This module provides a Viterbi decoder that ensures all phonemes in the target sequence are aligned,
even when they have very low probabilities. It includes methods for boosting target phoneme probabilities,
enforcing minimum probabilities, and post-processing to ensure all target phonemes appear in the alignment.
'''

import torch
import torch.nn.functional as F

class ViterbiDecoder:
    """
    Viterbi decoder that ensures all phonemes in the target sequence are aligned,
    even when they have very low probabilities.
    """
    def __init__(self, blank_id, silence_id, silence_anchors=3,  min_phoneme_prob=1e-8, ignore_noise=True, truly_forced=False):
        self.blank_id = blank_id
        self.silence_id = silence_id
        self.silence_anchors = silence_anchors  # Number of silence frames to anchor pauses
        self.min_phoneme_prob = min_phoneme_prob  # Minimum probability floor for phonemes
        self.ignore_noise = ignore_noise
        self.truly_forced = truly_forced
        self._neg_inf = -1000.0
        
    def set_blank_id(self, blank_id):
        """Set the blank token ID"""
        self.blank_id = blank_id
    
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
        if (stride * seq_len + 1) > seq_len*0.9: stride = 3
        if (stride * seq_len + 1) > seq_len*0.8: stride = 2
        expanded_len = stride * seq_len + 1
        if expanded_len > num_frames: raise ValueError(f"Target sequence too long to align: expanded CTC path length {expanded_len} (for {seq_len} phonemes) exceeds number of frames {num_frames}. Consider reducing the target sequence or increasing the audio length. Ensure duration_max is set appropriately for the length of the audio and target sequence. And silent_anchors is set above 2 if the segment is too long..")

        
        
        
        ctc_path = torch.zeros(expanded_len, device=device, dtype=torch.long).fill_(self.blank_id)
        ctc_path_true_idx = torch.zeros(expanded_len, device=device, dtype=torch.long).fill_(-1)
        #ctc_path_true_idx[::2] = -1
        #ctc_path[::2] = self.blank_id
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

    # --- Segmented alignment: match target SIL groups to audio silences ---

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
        
        if len(sil_groups) == 0 :
            if debug: print("WARN: No target SIL groups detected in the target sequence. No punctuation detected. Use punctuated text or disable silence anchoring for better alignment.")
            return [], []  # No silences to anchor on, fallback to single Viterbi
        min_silence_frames = self.silence_anchors
        audio_silences = self._detect_silence_segments(log_probs, sil_prob_threshold=0.9,  min_silence_frames=min_silence_frames, debug=debug)
        if len(audio_silences) == 0 and len(true_sequence) > 200:
            #_thresh_by_nframes = {6: 0.4, 7: 0.3, 8: 0.2, 9: 0.1, 10: 0.05}
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
                    if (stride * seg_seq_len + 1) > seg_log_probs.shape[0]*0.9: stride = 3
                    if (stride * seg_seq_len + 1) > seg_log_probs.shape[0]*0.8: stride = 2
                    expanded_len = stride * seg_seq_len + 1
                    if expanded_len > seg_log_probs.shape[0] * 1.2:
                        # in this case, the SIL has caused the segment to become too short for the target phonemes.  This can happen when there are very long silences or very short speech segments. 
                        return [], [] # return empty results to trigger fallback to non-segmented Viterbi, which can still align all phonemes but may be less accurate around silences.  Alternatively, could raise an error here to force the caller to handle this case explicitly.
                        #raise ValueError(f"Segment audio too short for target phonemes: expected CTC path length {expanded_len} exceeds segment frames {seg_log_probs.shape[0]}. Check if the audio segment is too short, or there are no words in the text.")


                    ctc_path = torch.zeros(expanded_len, device=device, dtype=torch.long).fill_(self.blank_id)
                    ctc_path_true_idx = torch.zeros(expanded_len, device=device, dtype=torch.long).fill_(-1)
                    #ctc_path_true_idx[::2] = -1
                    #ctc_path[::2] = self.blank_id
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

        # Threshold: which windows have >= 80% SIL probability?
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
        

        if not self.truly_forced:
            

            # Find best final state
            final_scores = dp[num_frames-1]
            valid_mask = final_scores > self._neg_inf
            if valid_mask.any():
                valid_indices = torch.where(valid_mask)[0]
                final_state = valid_indices[torch.argmax(final_scores[valid_indices])]
            else:
                final_state = torch.argmax(final_scores)

        else:
            # Truly forced alignment: path must terminate at the last CTC state so
            # that every target phoneme is guaranteed to have been visited in order.
            # ctc_path always ends with a blank, so ctc_len-1 is a valid terminal.
            final_state = ctc_len - 1
            if dp[num_frames - 1, final_state] <= self._neg_inf and ctc_len >= 2:
                # Trailing blank unreachable — accept last-phoneme state as fallback
                final_state = ctc_len - 2
            if dp[num_frames - 1, final_state] <= self._neg_inf:
                # Shouldn't happen with a valid CTC path — use rightmost reachable state
                valid_mask = dp[num_frames - 1] > self._neg_inf
                if valid_mask.any():
                    final_state = torch.where(valid_mask)[0][-1]  # rightmost = closest to end
                else:
                    final_state = torch.tensor(ctc_len - 1, device=device)

        
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
    
    def _ensure_all_phonemes_aligned___deprecated(self, frame_phonemes, frame_phonemes_idx, target_sequence, log_probs):
        """
        Post-processing step to ensure all target phonemes appear in the alignment.
        If any phonemes are missing, assign them to the frames where they have highest probability.
        """
        #device = frame_phonemes.device
        num_frames = len(frame_phonemes)
        
        # Get unique target phonemes
        target_set = set(target_sequence.tolist())
        target_set.discard(-100)  # Remove padding
        
        # Find which phonemes are already aligned
        aligned_phonemes = set(frame_phonemes.tolist())
        aligned_phonemes.discard(self.blank_id)
        
        # Find missing phonemes
        missing_phonemes = target_set - aligned_phonemes
        
        if not missing_phonemes:
            return frame_phonemes, frame_phonemes_idx  # All phonemes already aligned
        
        #print(f"Warning: {len(missing_phonemes)} phonemes missing from alignment. Force-aligning them.")
        
        # For each missing phoneme, find the best frame(s) to assign it
        modified_alignment = frame_phonemes.clone()
        modified_alignment_idx = frame_phonemes_idx.clone()

        for missing_ph in missing_phonemes:
            if missing_ph >= log_probs.shape[1]:
                continue
                
            # Find frames with highest probability for this phoneme
            phoneme_probs = log_probs[:, missing_ph]
            
            # Find best frame (highest probability)
            best_frame = torch.argmax(phoneme_probs).item()
            
            # Check if we can assign this phoneme without breaking the sequence too much
            # Strategy 1: Replace a blank frame if possible
            if modified_alignment[best_frame] == self.blank_id:
                modified_alignment[best_frame] = missing_ph
                modified_alignment_idx[best_frame] = target_sequence.tolist().index(missing_ph)
            else:
                # Strategy 2: Find nearby blank frames
                search_radius = min(5, num_frames // 10)  # Search within small radius
                for offset in range(1, search_radius + 1):
                    # Check left
                    if best_frame - offset >= 0 and modified_alignment[best_frame - offset] == self.blank_id:
                        modified_alignment[best_frame - offset] = missing_ph
                        break
                    # Check right
                    if best_frame + offset < num_frames and modified_alignment[best_frame + offset] == self.blank_id:
                        modified_alignment[best_frame + offset] = missing_ph
                        break
                else:
                    # Strategy 3: Replace the original frame anyway (last resort)
                    print(f"Warning:: Force-replacing frame {best_frame} with missing phoneme {missing_ph}")
                    modified_alignment[best_frame] = missing_ph
        
        return modified_alignment
    
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
    
    def __init__(self, blank_id, silence_id, silence_anchors=10, ignore_noise=True, truly_forced=True):
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
        self.truly_forced = truly_forced
        self.viterbi_decoder = ViterbiDecoder(blank_id, silence_id, silence_anchors=self.silence_anchors, ignore_noise=ignore_noise, truly_forced=self.truly_forced)
    
    
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
                # Using silence anchoring for forced alignment - 2026-02-07
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
                assorted.extend(frames)
            return assorted



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