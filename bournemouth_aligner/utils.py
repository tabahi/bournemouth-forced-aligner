'''
Convert ts output dict to textgrid that can be imported into Praat
'''


import math

import torch

def weighted_pool_embeddings(embeddings, log_probs, framestamps):
    """
    Average Weighted by confidence embeddings over frame ranges for each phoneme timestamp.
    
    Args:
        embeddings: Tensor of shape [T, D] where T is number of frames and D is embedding dimension
        log_probs:  log_probs: Log probabilities [T, C], can pass either phoneme or group log_probs
        timestamps: List of tuples (phoneme_id, start_frame, end_frame, start_ms, end_ms)
    
    Returns:
        pooled_embeddings: Tensor of shape [N, D] where N is length of timestamps
    """
    if len(framestamps) == 0:
        return torch.empty(0, embeddings.shape[1], device=embeddings.device)
    
    assert embeddings.dim() == 2, "Embeddings should be of shape [T, D] remove the batch dim"
    assert log_probs.shape[0] == embeddings.shape[0], "Log probabilities and embeddings must have the same number of frames"

    probs = torch.exp(log_probs.to(embeddings.device))
    pooled_embeddings = []
    
    for framestamp in framestamps:
        phoneme_id, start_frame, end_frame,  = framestamp[:3]  # Ignore start_ms and end_ms for pooling
        # Clamp frame indices to valid range
        start_frame = max(0, int(start_frame))
        end_frame = min(embeddings.shape[0], int(end_frame))
        
        if start_frame < end_frame:
            # Get segment embeddings and confidence weights
            segment_embeddings = embeddings[start_frame:end_frame]  # Shape: [num_frames, D]
            confidence_weights = probs[start_frame:end_frame, phoneme_id]  # Shape: [num_frames]
            
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

def _calculate_confidences(log_probs, framestamps):
    """
    Per-phone confidence: the highest posterior the model puts on that phone anywhere inside
    its own segment.

    The PEAK, not the mean, because of the shape the model is trained to produce. p3cupe holds
    a phone's posterior at 1.0 across the interior of its segment and brings it down to exactly
    0.5 at each of its two boundaries, where the neighbouring phone rises through 0.5 - that
    crossing is how the boundary is represented at all. So the frames at both ends of every
    segment are uncertain BY DESIGN, and an average over the segment measures those ramps as
    much as it measures the phone: it reads a short phone, which is nearly all ramp, as far
    less confident than a long one of identical quality. The peak is the quantity the training
    target actually pins at 1.0.

    Note that these numbers are NOT comparable with a CTC model's. CTC produces a near-one-hot
    spike per phone, so its confidences sit close to 1.0; this loss deliberately does not
    saturate (its target distribution has entropy), so a well-aligned phone can read well below
    0.5. Compare against other segments from the same checkpoint, not against an absolute.

    Args:
        log_probs: Log probabilities [T, C]
        framestamps: List of (phoneme_id, start_frame, end_frame, target_seq_idx, is_estimated)
            tuples. Frames may be fractional.

    Returns:
        The same tuples with the confidence appended.
    """
    probs = torch.exp(log_probs)
    n_frames, n_classes = probs.shape
    updated_tuples = []

    for phoneme_id, start_frame, end_frame, target_seq_idx, is_estimated in framestamps:
        start = max(0, min(int(start_frame), n_frames - 1))
        end = max(start + 1, min(int(math.ceil(float(end_frame))), n_frames))
        if phoneme_id < n_classes:
            confidence = float(probs[start:end, phoneme_id].max())
        else:
            confidence = 0.0
        updated_tuples.append((phoneme_id, start_frame, end_frame, target_seq_idx,
                               is_estimated, confidence))

    return updated_tuples

def convert_to_ms(framestamps, spectral_length, start_offset_time, wav_len, sample_rate,
                  frame_stride_samples=None):
    '''
    Args:
        framestamps: List of tuples (phoneme_id, start_frame, end_frame, target_seq_idx, is_estimated, avg_confidence)
        spectral_length: Number of spectral frames (int)
        start_time: Start time of the segment in seconds, used to offset the timestamps
        wav_len: Length of the audio segment in samples, used to estimate the duration per spectral-frame
        sample_rate: Sample rate of the audio, used to convert frames to milliseconds
    Returns:
        updated_tuples: List of tuples (phoneme_id, start_frame, end_frame, target_seq_idx, is_estimated, avg_confidence, start_ms, end_ms)
    '''
    # Frames advance at a FIXED rate: the feature extractor's total conv stride (80 samples
    # = 5 ms for p3cupe). Dividing the clip duration by the frame count is not the same
    # thing - the windows cover slightly less than the whole clip, so that form drifts by
    # tens of ms over a 10 s clip. Pass `frame_stride_samples` to get the exact mapping.
    if frame_stride_samples:
        duration_per_frame = frame_stride_samples / sample_rate
    else:
        duration_in_seconds = wav_len / sample_rate
        duration_per_frame = duration_in_seconds / spectral_length if spectral_length > 0 else 0

    updated_tuples = []
    for tup in framestamps:
        if len(tup) == 6:
            phoneme_id, start_frame, end_frame, target_seq_idx, is_estimated, avg_confidence = tup
        else:
            # fallback for tuples with different length
            phoneme_id, start_frame, end_frame = tup[:3]
            target_seq_idx = tup[3] if len(tup) > 3 else -1
            is_estimated = tup[4] if len(tup) > 4 else False
            avg_confidence = tup[5] if len(tup) > 5 else 0.0

        # Calculate start and end times in seconds
        start_sec = start_offset_time + (start_frame * duration_per_frame)
        end_sec = start_offset_time + (end_frame * duration_per_frame)
        # Convert to milliseconds
        start_ms = start_sec * 1000
        end_ms = end_sec * 1000

        updated_tuples.append((phoneme_id, start_frame, end_frame, target_seq_idx, is_estimated, avg_confidence, start_ms, end_ms))

    return updated_tuples
    
    
def dict_to_textgrid(data, output_file=None, include_confidence=False):
    """
    Convert a dictionary with phoneme and group timing data to TextGrid format.
    
    Args:
        data (dict): Dictionary containing segments with phoneme_ts and group_ts
        output_file (str, optional): Output file path. If None, returns string.
    
    Returns:
        str: TextGrid content if output_file is None, otherwise writes to file
    """
    if include_confidence: return dict_to_textgrid_with_confidence(data, output_file=output_file, include_confidence=True)

    # Extract timing information
    segments = data['segments']
    if not segments:
        raise ValueError("No segments found in data")
    
    segment = segments[0]  # Assuming single segment for now
    phoneme_data = segment.get('phoneme_ts', [])
    group_data = segment.get('group_ts', [])
    words_data = segment.get('words_ts', [])
    
    # Calculate total duration
    if phoneme_data or group_data or words_data:
        # Find the maximum end time from phoneme, group, and words data
        max_phoneme_end = max([p['end_ms'] for p in phoneme_data]) / 1000.0 if phoneme_data else 0
        max_group_end = max([g['end_ms'] for g in group_data]) / 1000.0 if group_data else 0
        max_words_end = max([w['end_ms'] for w in words_data]) / 1000.0 if words_data else 0
        xmax = max(max_phoneme_end, max_group_end, max_words_end, segment.get('end', 0))
    else:
        xmax = segment.get('end', 1.0)
    
    # Start building TextGrid content
    textgrid_content = []
    textgrid_content.append('File type = "ooTextFile"')
    textgrid_content.append('Object class = "TextGrid"')
    textgrid_content.append('')
    textgrid_content.append('xmin = 0')
    textgrid_content.append(f'xmax = {xmax}')
    textgrid_content.append('tiers? <exists>')
    
    # Count tiers (phonemes, groups, and words)
    tier_count = 0
    if phoneme_data:
        tier_count += 1
    if group_data:
        tier_count += 1
    if words_data:
        tier_count += 1
    
    textgrid_content.append(f'size = {tier_count}')
    textgrid_content.append('item []:')
    
    tier_num = 1
    
    # Add phoneme tier
    if phoneme_data:
        textgrid_content.append(f'    item [{tier_num}]:')
        textgrid_content.append('        class = "IntervalTier"')
        textgrid_content.append('        name = "phonemes"')
        textgrid_content.append('        xmin = 0')
        textgrid_content.append(f'        xmax = {xmax}')
        textgrid_content.append(f'        intervals: size = {len(phoneme_data)}')
        
        for i, phoneme in enumerate(phoneme_data, 1):
            start_time = phoneme['start_ms'] / 1000.0
            end_time = phoneme['end_ms'] / 1000.0
            label = phoneme['ipa_label']
            
            textgrid_content.append(f'        intervals [{i}]:')
            textgrid_content.append(f'            xmin = {start_time}')
            textgrid_content.append(f'            xmax = {end_time}')
            textgrid_content.append(f'            text = "{label}"')
        
        tier_num += 1
    
    # Add words tier
    if words_data:
        textgrid_content.append(f'    item [{tier_num}]:')
        textgrid_content.append('        class = "IntervalTier"')
        textgrid_content.append('        name = "words"')
        textgrid_content.append('        xmin = 0')
        textgrid_content.append(f'        xmax = {xmax}')
        textgrid_content.append(f'        intervals: size = {len(words_data)}')
        
        for i, word in enumerate(words_data, 1):
            start_time = word['start_ms'] / 1000.0
            end_time = word['end_ms'] / 1000.0
            label = word['word']
            
            textgrid_content.append(f'        intervals [{i}]:')
            textgrid_content.append(f'            xmin = {start_time}')
            textgrid_content.append(f'            xmax = {end_time}')
            textgrid_content.append(f'            text = "{label}"')
        
        tier_num += 1
    
    # Add group tier
    if group_data:
        textgrid_content.append(f'    item [{tier_num}]:')
        textgrid_content.append('        class = "IntervalTier"')
        textgrid_content.append('        name = "groups"')
        textgrid_content.append('        xmin = 0')
        textgrid_content.append(f'        xmax = {xmax}')
        textgrid_content.append(f'        intervals: size = {len(group_data)}')
        
        for i, group in enumerate(group_data, 1):
            start_time = group['start_ms'] / 1000.0
            end_time = group['end_ms'] / 1000.0
            label = group['group_label']
            
            textgrid_content.append(f'        intervals [{i}]:')
            textgrid_content.append(f'            xmin = {start_time}')
            textgrid_content.append(f'            xmax = {end_time}')
            textgrid_content.append(f'            text = "{label}"')
    
    # Join all content
    result = '\n'.join(textgrid_content)
    
    # Write to file or return string
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"TextGrid saved to {output_file}")
    else:
        return result

def dict_to_textgrid_with_confidence(data, output_file=None, include_confidence=True):
    """
    Simple version that optionally includes confidence scores in labels.
    
    Args:
        data (dict): Dictionary containing segments with phoneme_ts and group_ts
        output_file (str, optional): Output file path
        include_confidence (bool): Whether to include confidence scores in labels
    
    Returns:
        str: TextGrid content if output_file is None
    """
    
    segments = data['segments']
    if not segments:
        raise ValueError("No segments found in data")
    
    segment = segments[0]
    phoneme_data = segment.get('phoneme_ts', [])
    group_data = segment.get('group_ts', [])
    words_data = segment.get('words_ts', [])
    
    # Calculate total duration
    if phoneme_data or group_data or words_data:
        max_phoneme_end = max([p['end_ms'] for p in phoneme_data]) / 1000.0 if phoneme_data else 0
        max_group_end = max([g['end_ms'] for g in group_data]) / 1000.0 if group_data else 0
        max_words_end = max([w['end_ms'] for w in words_data]) / 1000.0 if words_data else 0
        xmax = max(max_phoneme_end, max_group_end, max_words_end, segment.get('end', 0))
    else:
        xmax = segment.get('end', 1.0)
    
    # Build TextGrid content
    textgrid_content = []
    textgrid_content.append('File type = "ooTextFile"')
    textgrid_content.append('Object class = "TextGrid"')
    textgrid_content.append('')
    textgrid_content.append('xmin = 0')
    textgrid_content.append(f'xmax = {xmax}')
    textgrid_content.append('tiers? <exists>')
    
    # Count tiers
    tier_count = 0
    if words_data:
        tier_count += 1
    if phoneme_data:
        tier_count += 1
    if group_data:
        tier_count += 1
    
    textgrid_content.append(f'size = {tier_count}')
    textgrid_content.append('item []:')
    
    tier_num = 1
    
    # Add words tier
    if words_data:
        textgrid_content.append(f'    item [{tier_num}]:')
        textgrid_content.append('        class = "IntervalTier"')
        textgrid_content.append('        name = "words"')
        textgrid_content.append('        xmin = 0')
        textgrid_content.append(f'        xmax = {xmax}')
        textgrid_content.append(f'        intervals: size = {len(words_data)}')
        
        for i, word in enumerate(words_data, 1):
            start_time = word['start_ms'] / 1000.0
            end_time = word['end_ms'] / 1000.0
            label = word['word']
            if include_confidence:
                confidence = word.get('confidence', 0)
                label += f" ({confidence:.2f})"
            
            textgrid_content.append(f'        intervals [{i}]:')
            textgrid_content.append(f'            xmin = {start_time}')
            textgrid_content.append(f'            xmax = {end_time}')
            textgrid_content.append(f'            text = "{label}"')
        
        tier_num += 1
    
    # Add phoneme tier
    if phoneme_data:
        textgrid_content.append(f'    item [{tier_num}]:')
        textgrid_content.append('        class = "IntervalTier"')
        textgrid_content.append('        name = "phonemes"')
        textgrid_content.append('        xmin = 0')
        textgrid_content.append(f'        xmax = {xmax}')
        textgrid_content.append(f'        intervals: size = {len(phoneme_data)}')
        
        for i, phoneme in enumerate(phoneme_data, 1):
            start_time = phoneme['start_ms'] / 1000.0
            end_time = phoneme['end_ms'] / 1000.0
            label = phoneme['ipa_label']
            if include_confidence:
                confidence = phoneme.get('confidence', 0)
                label += f" ({confidence:.2f})"
            
            textgrid_content.append(f'        intervals [{i}]:')
            textgrid_content.append(f'            xmin = {start_time}')
            textgrid_content.append(f'            xmax = {end_time}')
            textgrid_content.append(f'            text = "{label}"')
        
        tier_num += 1
    
    # Add group tier
    if group_data:
        textgrid_content.append(f'    item [{tier_num}]:')
        textgrid_content.append('        class = "IntervalTier"')
        textgrid_content.append('        name = "groups"')
        textgrid_content.append('        xmin = 0')
        textgrid_content.append(f'        xmax = {xmax}')
        textgrid_content.append(f'        intervals: size = {len(group_data)}')
        
        for i, group in enumerate(group_data, 1):
            start_time = group['start_ms'] / 1000.0
            end_time = group['end_ms'] / 1000.0
            label = group['group_label']
            if include_confidence:
                confidence = group.get('confidence', 0)
                label += f" ({confidence:.2f})"
            
            textgrid_content.append(f'        intervals [{i}]:')
            textgrid_content.append(f'            xmin = {start_time}')
            textgrid_content.append(f'            xmax = {end_time}')
            textgrid_content.append(f'            text = "{label}"')
    
    result = '\n'.join(textgrid_content)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"TextGrid saved to {output_file}")
    else:
        return result



def detect_device_automatically():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


# ---------------------------------------------------------------------------
# Mel-spectrogram + phoneme alignment visualisation
# ---------------------------------------------------------------------------

def plot_mel_alignment(mel, phoneme_ts, save_path="mel_alignment.png",
                       segment_start_ms=0.0, title=None, width=20, height=5):
    """
    Plot a mel spectrogram with phoneme boundary markers and IPA labels.

    Args:
        mel (Tensor): Mel spectrogram of shape (T, mel_bins).
        phoneme_ts (list[dict]): Phoneme timestamp list from the alignment output,
            each dict having at least ``ipa_label``, ``start_ms``, ``end_ms``,
            and ``confidence``.
        save_path (str): Destination file path (.png or any matplotlib-supported format).
        segment_start_ms (float): Offset in ms to subtract from timestamps so they
            are relative to the mel tensor start.  Pass
            ``segment['start'] * 1000`` when the timestamps are absolute.
        title (str | None): Optional figure title.  Auto-generated when None.

    Returns:
        str: The path the figure was saved to.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    assert mel.dim() == 2, f"Expected 2D mel tensor (T, mel_bins), got shape {tuple(mel.shape)}"

    T, mel_bins = mel.shape
    mel_np = mel.cpu().numpy().T  # (mel_bins, T) — imshow convention

    # Derive frames-per-ms from the tensor length and the timestamp span
    if phoneme_ts:
        span_ms = phoneme_ts[-1]['end_ms'] - segment_start_ms
        if span_ms > 0:
            frames_per_ms = T / span_ms
        else:
            frames_per_ms = T / max(phoneme_ts[-1]['end_ms'], 1.0)
    else:
        frames_per_ms = 1.0

    def ms_to_frame(ms):
        return (ms - segment_start_ms) * frames_per_ms

    fig, ax = plt.subplots(figsize=(max(width, T // 40), height))

    im = ax.imshow(mel_np, aspect='auto', origin='lower',
                   cmap='Spectral', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Log magnitude')
    ax.set_ylabel('Mel bin')
    ax.set_xlabel('Frame')

    if title is None:
        n_ph = len(phoneme_ts)
        title = f"Mel spectrogram with phoneme alignment  ({n_ph} phonemes)"
    ax.set_title(title)

    # ---- draw boundaries and labels ----------------------------------------
    prev_end_frame = None
    for ph in phoneme_ts:
        start_frame = ms_to_frame(ph['start_ms'])
        end_frame   = ms_to_frame(ph['end_ms'])
        label       = ph.get('ipa_label') or ph.get('phoneme_label', '?')
        confidence  = ph.get('confidence', 1.0)
        estimated   = ph.get('is_estimated', False)

        # Boundary line at phoneme start (skip very first)
        if prev_end_frame is not None:
            ax.axvline(x=start_frame, color="#b10000", linewidth=1.2, alpha=0.85)

        # Width of this phoneme in frames
        width = max(end_frame - start_frame, 1)
        centre = start_frame + width / 2

        # Confidence-based background / font colour
        if confidence > 0.9:
            facecolor = '#000000'   # black
            fontcolor = 'white'
        elif confidence > 0.6:
            facecolor = '#404040'   # dark gray
            fontcolor = 'white'
        elif confidence > 0.1:
            facecolor = '#c0c0c0'   # light gray
            fontcolor = 'black'
        else:
            facecolor = '#ffffff'   # white
            fontcolor = 'black'

        # Only annotate if the segment is at least 3 frames wide
        if width >= 3:
            ax.text(centre, mel_bins * 0.9, label,
                    ha='center', va='top', fontsize=9, fontweight='bold',
                    color=fontcolor,
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor=facecolor, alpha=0.9, linewidth=0))
            # Mark estimated phonemes with a small asterisk above the label
            if estimated:
                ax.text(centre, mel_bins * 0.95, '*',
                        ha='center', va='bottom', fontsize=10, fontweight='bold',
                        color='#ff6600', alpha=0.95)
        else:
            # Narrow phoneme — place label above the top of the plot
            ax.text(centre, mel_bins * 0.92, label,
                    ha='center', va='bottom', fontsize=7, fontweight='bold',
                    color=fontcolor,
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor=facecolor, alpha=0.9, linewidth=0))
            if estimated:
                ax.text(centre, mel_bins * 0.96, '*',
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                        color='#ff6600', alpha=0.95)

        prev_end_frame = end_frame

    # ---- legend ------------------------------------------------------------
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#000000', label='>0.9 confidence'),
        Patch(facecolor='#404040', label='>0.6 confidence'),
        Patch(facecolor='#c0c0c0', label='>0.1 confidence', edgecolor='gray'),
        Patch(facecolor='#ffffff', label='≤0.1 confidence', edgecolor='gray'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
              framealpha=0.75, title='* = estimated')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return save_path