'''
Convert ts output dict to textgrid that can be imported into Praat
'''


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
    Calculate confidence scores for each timestamped phoneme/group.

    Args:
        log_probs: Log probabilities [T, C]
        framestamps: List of (phoneme_id, start_frame, end_frame, target_seq_idx) tuples

    Returns:
        List of (phoneme_id, start_frame, end_frame, target_seq_idx, avg_confidence) tuples
    """
    probs = torch.exp(log_probs)
    updated_tuples = []

    for phoneme_id, start_frame, end_frame, target_seq_idx, is_estimated in framestamps:
        # Clamp to valid range
        start_frame = max(0, int(start_frame))
        end_frame = min(log_probs.shape[0], int(end_frame))
        if not is_estimated or (start_frame < probs.shape[0] and end_frame <= log_probs.shape[0]):
            avg_confidence = probs[start_frame, phoneme_id]
        else:
            raise ValueError(f"Invalid frame range for estimated timestamp: start_frame={start_frame}, end_frame={end_frame}, log_probs shape={log_probs.shape}, is_estimated={is_estimated}, phoneme_id={phoneme_id}")

        if start_frame < end_frame and phoneme_id < log_probs.shape[1]:

            half_confidence = avg_confidence/2
            total_good_frames = 1

            # since there can be blanks after the first one, we only take probablities if at least prob > 0.5 compared to the first frame to avoid for-sure blanks
            for f in range(start_frame+1, end_frame):
                frame_prob = probs[f, phoneme_id]
                if (frame_prob > half_confidence) or (frame_prob > 0.1):
                    avg_confidence += frame_prob
                    total_good_frames += 1
            if total_good_frames > 1:
                avg_confidence /= total_good_frames

                max_confidence = probs[start_frame:end_frame, phoneme_id].max()
                if avg_confidence < max_confidence/2:
                    avg_confidence = max_confidence
            
        updated_tuples.append((phoneme_id, start_frame, end_frame, target_seq_idx, is_estimated, avg_confidence.item()))

    return updated_tuples

def convert_to_ms(framestamps, spectral_length, start_offset_time, wav_len, sample_rate):
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

