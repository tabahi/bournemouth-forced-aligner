'''
Convert ts output dict to textgrid that can be imported into Praat
'''

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
            label = phoneme['phoneme_label']
            
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
            label = phoneme['phoneme_label']
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

