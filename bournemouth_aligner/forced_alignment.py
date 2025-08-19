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
    def __init__(self, blank_id=None, min_phoneme_prob=1e-8):
        self.blank_id = blank_id
        self.min_phoneme_prob = min_phoneme_prob  # Minimum probability floor for phonemes
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
                                   boost_targets=True, enforce_minimum=True):
        """
        Decode that ensures all target phonemes are aligned.
        
        Args:
            log_probs: Log probabilities tensor [T, C]
            true_sequence: Target phoneme sequence [S]
            return_scores: Whether to return alignment scores
            boost_targets: Whether to boost target phoneme probabilities
            enforce_minimum: Whether to enforce minimum probabilities
            
        Returns:
            frame_phonemes: Frame-level phoneme assignments
            alignment_score: Optional alignment score
        """
        if self.blank_id is None:
            raise ValueError("Blank ID not set. Call set_blank_id first.")
        
        seq_len = true_sequence.shape[0]
        num_frames = log_probs.shape[0]
        device = log_probs.device
        
        if seq_len == 0:
            frame_phonemes = torch.full((num_frames,), self.blank_id, dtype=torch.long, device=device)
            if return_scores:
                alignment_score = log_probs[:, self.blank_id].sum()
                return frame_phonemes, alignment_score
            return frame_phonemes
        
        # Apply probability modifications if requested
        modified_log_probs = log_probs.clone()
        
        if boost_targets:
            modified_log_probs = self._boost_target_phonemes(modified_log_probs, true_sequence)
        
        if enforce_minimum:
            modified_log_probs = self._enforce_minimum_probabilities(modified_log_probs, true_sequence)
        
        # Create CTC path: blank + phoneme + blank + phoneme + ... + blank
        ctc_path = torch.zeros(2 * seq_len + 1, device=device, dtype=torch.long)
        ctc_path[::2] = self.blank_id  # Even indices are blanks
        ctc_path[1::2] = true_sequence  # Odd indices are phonemes
        ctc_len = len(ctc_path)
        
        # Run standard Viterbi with modified probabilities
        frame_phonemes = self._viterbi_decode(modified_log_probs, ctc_path, ctc_len)
        
        # Post-process to ensure all target phonemes appear
        frame_phonemes = self._ensure_all_phonemes_aligned(
            frame_phonemes, true_sequence, modified_log_probs
        )
        
        if return_scores:
            alignment_score = self._calculate_alignment_score(log_probs, frame_phonemes)
            return frame_phonemes, alignment_score
        
        return frame_phonemes
    
    def _viterbi_decode(self, log_probs, ctc_path, ctc_len):
        """Standard Viterbi decoding implementation."""
        num_frames = log_probs.shape[0]
        device = log_probs.device
        
        # Initialize DP table
        dp = torch.full((num_frames, ctc_len), self._neg_inf, device=device, dtype=torch.float32)
        backpointers = torch.zeros((num_frames, ctc_len), dtype=torch.long, device=device)
        
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
        return frame_phonemes
    
    def _ensure_all_phonemes_aligned(self, frame_phonemes, target_sequence, log_probs):
        """
        Post-processing step to ensure all target phonemes appear in the alignment.
        If any phonemes are missing, assign them to the frames where they have highest probability.
        """
        device = frame_phonemes.device
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
            return frame_phonemes  # All phonemes already aligned
        
        #print(f"Warning: {len(missing_phonemes)} phonemes missing from alignment. Force-aligning them.")
        
        # For each missing phoneme, find the best frame(s) to assign it
        modified_alignment = frame_phonemes.clone()
        
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

    
    # Modified assort_frames method for the ViterbiDecoder - 2025-08-07
    def assort_frames(self, frame_phonemes, max_blanks=10):
        """
        Assort_frames that handles forced phoneme alignments better.
        """
        if len(frame_phonemes) == 0:
            return []
        
        device = frame_phonemes.device
        
        if not isinstance(frame_phonemes, torch.Tensor):
            frame_phonemes = torch.tensor(frame_phonemes, device=device)
        
        # Find all non-blank segments
        is_blank = frame_phonemes == self.blank_id
        
        # Find transition points
        transitions = torch.cat([
            torch.tensor([True], device=device),
            frame_phonemes[1:] != frame_phonemes[:-1]
        ])
        
        transition_indices = torch.where(transitions)[0]
        
        framestamps = []
        
        for i in range(len(transition_indices)):
            start_idx = transition_indices[i].item()
            end_idx = transition_indices[i + 1].item() if i + 1 < len(transition_indices) else len(frame_phonemes)
            
            segment_phoneme = frame_phonemes[start_idx].item()
            
            # Skip overly long blank segments
            if segment_phoneme == self.blank_id:
                segment_length = end_idx - start_idx
                if segment_length > max_blanks:
                    continue
            
            # For non-blank segments, always include them (even if very short)
            if segment_phoneme != self.blank_id:
                framestamps.append((segment_phoneme, start_idx, end_idx))
        
        return framestamps

class AlignmentUtils: # 2025-08-07
    """
    Alignment utilities with forced alignment fixes.
    """
    
    def __init__(self, blank_id):
        self.blank_id = blank_id
        self.viterbi_decoder = ViterbiDecoder(blank_id)
    
    def decode_alignments(self, log_probs, true_seqs=None, pred_lens=None, 
                         true_seqs_lens=None, forced_alignment=True, 
                         boost_targets=True, enforce_minimum=True):
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
            List of frame-level alignments
        """
        batch_size = log_probs.shape[0]
        device = log_probs.device
        
        if forced_alignment:
            if (true_seqs is None) or (true_seqs_lens is None):
                raise ValueError("Phoneme sequences and lengths required for forced alignment")
            
            with torch.no_grad():
                all_frame_phonemes = []
                
                for i in range(batch_size):
                    # Get sequence for this batch item
                    if pred_lens is not None:
                        log_probs_seq = log_probs[i][:pred_lens[i]]
                    else:
                        log_probs_seq = log_probs[i]
                    
                    true_seq_len = true_seqs_lens[i]
                    true_seq = true_seqs[i, :true_seq_len]
                    
                    if true_seq_len == 0:
                        frame_phonemes = torch.tensor([], device=device, dtype=torch.long)
                    else:
                        # Use enhanced decoder
                        frame_phonemes = self.viterbi_decoder.decode_with_forced_alignment(
                            log_probs_seq, 
                            true_seq,
                            boost_targets=boost_targets,
                            enforce_minimum=enforce_minimum
                        )
                    
                    all_frame_phonemes.append(frame_phonemes)
                
                # Apply assort_frames to get timestamps
                assorted = []
                for frame_phonemes in all_frame_phonemes:
                    assorted.append(self.viterbi_decoder.assort_frames(frame_phonemes))
                
                return assorted
        
        else:
            # Free decoding (unchanged)
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
                assorted.append(self.viterbi_decoder.assort_frames(frame_phonemes_seq))
            return assorted

