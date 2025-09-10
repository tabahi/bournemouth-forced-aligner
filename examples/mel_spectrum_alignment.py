# dependency: pip install librosa
import torch
import time
from bournemouth_aligner import PhonemeTimestampAligner



def plot_mel_phonemes(mel, compress_framesed, save_path="mel_phonemes.png"):
    """
    Plot mel spectrogram with phoneme IDs overlaid directly on the spectrogram
    
    Args:
        mel: Mel spectrogram tensor [frames, mel_bins]
        compress_framesed: List of [phoneme_id, count] pairs representing phoneme alignment per frame
        save_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    
    assert mel.dim() == 2, f"Expected 2D mel tensor, got {mel.dim()}D"
    phn_frame_ids = [phoneme_id for phoneme_id, _ in compress_framesed]
    phn_frame_counts = [count for _, count in compress_framesed]

    # Create single plot - make it twice as wide
    fig, ax = plt.subplots(1, 1, figsize=(30, 8))
    
    # Convert mel to numpy for plotting
    mel_np = mel.cpu().numpy() if isinstance(mel, torch.Tensor) else mel
    
    # Add statistics to title instead of overlaying on spectrum
    unique_phonemes = len(set(phn_frame_ids))
    stats_text = f"Frames: {sum(phn_frame_counts)} | Unique Phonemes: {unique_phonemes} | Mel Bins: {mel.shape[1]}"
    title_text = f'Mel Spectrogram with Phoneme Alignment\n{stats_text}'
    
    # Plot mel spectrogram
    im = ax.imshow(mel_np.T, aspect='auto', origin='lower', 
                    cmap='viridis', interpolation='nearest')
    ax.set_ylabel('Mel Bins')
    ax.set_xlabel('Frame Index')
    ax.set_title(title_text)
    plt.colorbar(im, ax=ax, label='Magnitude')
    
    # Overlay phoneme information
    frame_pos = 0
    for phn_id, count in zip(phn_frame_ids, phn_frame_counts):
        # Draw vertical boundary lines (except for first segment)
        if frame_pos > 0:
            ax.axvline(x=frame_pos-0.5, color='red', linestyle='-', alpha=0.8, linewidth=2)
        
        # Add phoneme ID text at the top of the spectrogram
        if count > 1:  # Only add text if segment is wide enough
            text_x = frame_pos + count/2
            text_y = mel.shape[1] - 2  # Near the top of the mel bins
            
            # Add text with background for visibility
            ax.text(text_x, text_y, str(phn_id), 
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    color='white', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8))
        
        frame_pos += count
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Mel-phoneme alignment plot saved to: {save_path}")
    
    return save_path


def example_audio_timestamps():

    text_sentences = "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition"
    audio_path = "examples/samples/LJSpeech/LJ001-0001.wav"
    
    model_name = "en_libri1000_uj01d_e199_val_GER=0.2307.ckpt" 
    extractor = PhonemeTimestampAligner(model_name=model_name, lang='en-us', duration_max=10, device='cpu', enforce_all_targets=False)

    full_clip_wav = extractor.load_audio(audio_path) # can replace it with custom audio source


    timestamps = extractor.process_sentence(text_sentences, full_clip_wav, ts_out_path=None, extract_embeddings=False, vspt_path=None, do_groups=False, debug=False)


    
    # Extract Mel-Spectrum
    # Use the same config to generate mel-spectrum as bigvan vocoder model, so that the mel-spectrum can be converted back to audio easily.
    vocoder_config = {'num_mels': 80, 'num_freq': 1025, 'n_fft': 1024, 'hop_size': 256, 'win_size': 1024, 'sampling_rate': 22050, 'fmin': 0, 'fmax': 8000, 'model': 'nvidia/bigvgan_v2_22khz_80band_fmax8k_256x'}

    # we just need to extract the mel-spectrogram for the segment 0
    segment_start_frame = int(timestamps['segments'][0]['start'] * extractor.resampler_sample_rate)
    segment_end_frame = int(timestamps['segments'][0]['end'] * extractor.resampler_sample_rate)
    segment_wav = full_clip_wav[segment_start_frame:segment_end_frame]
    mel_spec = extractor.extract_mel_spectrum(segment_wav, wav_sample_rate=extractor.resampler_sample_rate, vocoder_config=vocoder_config)


    # Assort phonemes into frame steps to match the mel-spectrogram hop-size
    segment_duration = timestamps['segments'][0]['end'] - timestamps['segments'][0]['start']

    
    
    total_frames = mel_spec.shape[0]
    frames_per_second = total_frames / segment_duration
    frames_assorted = extractor.framewise_assortment(aligned_ts=timestamps['segments'][0]['phoneme_ts'], total_frames=total_frames, frames_per_second=frames_per_second, gap_contraction=5, select_key="phoneme_idx")
    # convert phoneme IDs to labels
    frames_assorted = [extractor.phoneme_id_to_label[phoneme_id] for phoneme_id in frames_assorted]

    compress_framesed = extractor.compress_frames(frames_assorted)


    plot_mel_phonemes(mel_spec, compress_framesed, save_path="mel_phonemes.png")

if __name__ == "__main__":
    torch.random.manual_seed(42)
    example_audio_timestamps()




