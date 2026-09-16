
import torch

import math


def frames_geometry(window_size_ms, stride_ms, sample_rate, frames_per_window,
                    frame_stride_samples, strict=True):
    '''
    Resolve how many OUTPUT frames one window stride advances.

    ## Why this exists (2026-08-02)

    `stich_window_predictions` used to hard-code `stride_frames = frames_per_window // 2`,
    which is only correct when `stride_ms == window_size_ms / 2`. The shipped config was
    `window_size_ms=120, stride_ms=80` - a 2/3 advance - so the stitch placed window `i`
    frame 7 and window `i+1` frame 2 at the SAME global index and averaged them, even
    though (measured from the conv stack by gradient centroid) those two frames describe
    moments **14.4 ms apart**.

    Every stitched frame was therefore a blend of two predictions about different instants:
    a ~14 ms smear applied unconditionally, which no acoustic model can undo. It explains
    why `abs_err_mean` sat at 27.7 ms while `signed_err_median` stayed near +2 ms (a
    symmetric smear inflates the mean error without shifting the median), and why three
    rounds of loss work plus a 30% PER improvement moved alignment by nothing.

    The advance must be `stride_samples / frame_stride_samples`, where `frame_stride_samples`
    is the product of the feature extractor's conv strides (7*5*3*2 = 210 here) - NOT
    `window_size_samples / frames_per_window` (1920/10 = 192), which is the average frame
    density including the padding-compressed edge frames, not the true spacing.

    For that ratio to be a whole number the stride must be a multiple of the frame stride:
        stride_ms = 5 * 210 / 16000 * 1000 = 65.625 ms   -> 5 frames, interior error 0.0 ms
        stride_ms = 6 * 210 / 16000 * 1000 = 78.75 ms    -> 6 frames

    Returns:
        (stride_frames, stride_samples, window_size_samples)
    '''
    window_size_samples = int(window_size_ms * sample_rate / 1000)
    stride_samples = int(stride_ms * sample_rate / 1000)
    exact = stride_samples / float(frame_stride_samples)
    stride_frames = int(round(exact))

    if abs(exact - stride_frames) > 1e-6:
        msg = (f"windowing geometry is inconsistent: stride {stride_samples} samples "
               f"({stride_ms} ms) is {exact:.4f} output frames, not a whole number "
               f"(frame stride {frame_stride_samples} samples). The stitch would average "
               f"frames describing moments {abs(exact - stride_frames) * frame_stride_samples / sample_rate * 1000:.1f} ms "
               f"apart. Use a stride_ms that is a multiple of "
               f"{1000.0 * frame_stride_samples / sample_rate:g} ms.")
        if strict:
            raise ValueError(msg)
        print("WARN:", msg)
    if stride_frames < 1:
        raise ValueError(f"stride_frames must be >= 1, got {stride_frames}")
    if stride_frames > frames_per_window:
        raise ValueError(f"stride_frames ({stride_frames}) > frames_per_window "
                         f"({frames_per_window}): windows would not overlap or would leave gaps")
    return stride_frames, stride_samples, window_size_samples


def total_stitched_frames(num_windows, frames_per_window, stride_frames):
    '''
    Frames produced by stitching `num_windows` windows.

    The old expression was `num_windows * frames_per_window // 2`, which both hard-coded
    the 50% overlap and silently dropped the tail of the last window. The correct count is
    the position of the last window's last frame, plus one.
    '''
    return (max(int(num_windows), 1) - 1) * int(stride_frames) + int(frames_per_window)


def slice_windows(audio_batch: torch.Tensor,
                        sample_rate: int = 16000,
                        window_size_ms: int = 160,
                        stride_ms: int = 80) -> torch.Tensor:
    """
    Create fixed-size windows with overlap from a batch of audio sequences using vectorized operations.
    
    Args:
        audio_batch: Input audio of shape [batch_size, 1, max_audio_length]
        sample_rate: Audio sample rate in Hz
        window_size_ms: Window size in milliseconds
        stride_ms: Stride size in milliseconds
    
    Returns:
        Tensor of shape [batch_size, num_windows, window_size]
    """
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

# Optional: If you need unfold-based implementation for very large audio
def large_windows_unfold(audio_batch: torch.Tensor,
                              sample_rate: int = 16000,
                              window_size_ms: int = 3000,
                              stride_ms: int = 250) -> torch.Tensor:
    """
    Alternative implementation using unfold operation for potentially better memory efficiency
    on very large audio files.
    Args:
        audio_batch: Input audio of shape [batch_size, 1, max_audio_length]
        sample_rate: Audio sample rate in Hz
        window_size_ms: Window size in milliseconds
        stride_ms: Stride size in milliseconds
    Returns:
        Tensor of shape [batch_size, num_windows, window_size]

    """
    audio_batch = audio_batch.squeeze(1)  # [batch_size, max_audio_length]
    #batch_size = audio_batch.shape[0]
    
    window_size = int(window_size_ms * sample_rate / 1000)
    stride = int(stride_ms * sample_rate / 1000)
    
    # Use unfold to create windows
    windows = audio_batch.unfold(dimension=1, size=window_size, step=stride)
    
    return windows  # [batch_size, num_windows, window_size]


def large_windows_fold(window_logits):
    """
    UNDER CONSTRUCTION
    Combines predictions from segmented windows using the unfold-based implementation.
    Args:
        window_logits: Input audio of shape [batch_size, num_windows, frames, num_phonemes]
    Returns:
        Tensor of shape [batch_size, num_windows, window_size]

    """
    audio_batch = audio_batch.squeeze(1)  # [batch_size, max_audio_length]
    #batch_size = audio_batch.shape[0]
    
    window_size = int(window_size_ms * sample_rate / 1000)
    stride = int(stride_ms * sample_rate / 1000)
    
    # Use unfold to create windows
    windows = audio_batch.unfold(dimension=1, size=window_size, step=stride)
    
    return windows  # [batch_size, num_windows, window_size]

def stich_window_predictions(window_logits: torch.Tensor,
                             original_audio_length: int,
                             cnn_output_size: int,
                             sample_rate: int = 16000,
                             window_size_ms: int = 160,
                             stride_ms: int = 80,
                             stride_frames: int = None) -> torch.Tensor:
    """
    Efficiently combines predictions from overlapping windows while maintaining the original behavior. Can be used for phoneme logits, embeddings, or CNN outputs features.
    
    Args:
        window_logits: Shape [batch_size, num_windows, frames_per_window, output_dim]
        original_audio_length: Original audio length in samples
        cnn_output_size: Number of frames output by CNN for each window
        sample_rate: Audio sample rate (default 16kHz)
        window_size_ms: Window size in milliseconds
        stride_ms: Stride size in milliseconds
    Returns:
        Tensor of shape [batch_size, total_frames, output_dim]
    """
    device = window_logits.device
    batch_size, num_windows, frames_per_window, num_phonemes = window_logits.shape
    
    # Pre-compute constants
    window_size_samples = int(window_size_ms * sample_rate / 1000)
    stride_samples = int(stride_ms * sample_rate / 1000)
    num_windows_total = ((original_audio_length - window_size_samples) // stride_samples) + 1
    if stride_frames is None:
        stride_frames = frames_per_window // 2
    stride_frames = int(stride_frames)
    total_frames = total_stitched_frames(num_windows_total, cnn_output_size, stride_frames)
    
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

def stich_window_predictions____non_vectorized(window_logits: torch.Tensor,
                             original_audio_length: int,
                             cnn_output_size,
                             sample_rate: int = 16000,
                             window_size_ms: int = 160,
                             stride_ms: int = 80) -> torch.Tensor:
    device = window_logits.device
    batch_size, num_windows, frames_per_window, num_phonemes = window_logits.shape
    
    window_size_samples = int(window_size_ms * sample_rate / 1000)
    stride_samples = int(stride_ms * sample_rate / 1000)
    
    # Calculate number of windows based on original audio length
    num_windows_total = ((original_audio_length - window_size_samples) // stride_samples) + 1
    
    # Use calculate_layer_sizes to get the output size after CNN layers
    frames_per_window_full = cnn_output_size # model.calculate_layer_sizes(torch.tensor([window_size_samples]))[0]
    total_frames = ((num_windows_total * frames_per_window_full) // 2)
    
    window_weights = torch.cos(torch.linspace(-math.pi/2, math.pi/2, frames_per_window))
    window_weights = window_weights.to(device).view(1, frames_per_window, 1)
    
    combined = torch.zeros(batch_size, total_frames, num_phonemes, device=device)
    weight_sum = torch.zeros(batch_size, total_frames, 1, device=device)
    
    stride_frames = frames_per_window // 2
    
    for i in range(num_windows):
        start_frame = i * stride_frames
        end_frame = start_frame + frames_per_window
        
        if end_frame > total_frames:
            frames_to_use = total_frames - start_frame
            window_logits_slice = window_logits[:, i, :frames_to_use]
            weights = window_weights[:, :frames_to_use]
        else:
            window_logits_slice = window_logits[:, i]
            weights = window_weights
        
        combined[:, start_frame:end_frame] += window_logits_slice * weights
        weight_sum[:, start_frame:end_frame] += weights
    
    combined = combined / (weight_sum + 1e-8)
    return combined




def calc_spec_len_ext(wav_lens, window_size_ms, stride_ms, sample_rate, frames_per_window, disable_windowing=False, wav_len_max=1*16000, stride_frames=None):
    """
    Calculate the total number of frames for the whole audio clip, for each clip in the batch.
    When `disable_windowing=False` then there are two level of windowing, one by the window slicing process and other by the CNN.
    Input:
        wav_lens: tensor of real lengths of the audio clips in samples. Shape: [batch_size]
    Returns:
        spectral_lens: tensor of total number of frames for each audio clip. Shape: [batch_size]
    """

    if (not disable_windowing):
        #window_size_samples = int(self.window_size_ms * self.sample_rate / 1000)
        #stride_samples = int(self.stride_ms * self.sample_rate / 1000)
        
        # move self.frames_per_window to the same device if not already:
        frames_per_window = frames_per_window.to(wav_lens.device)
        window_size_wav = int(window_size_ms * sample_rate / 1000)  # 1920
        stride_size_wav = int(stride_ms * sample_rate / 1000)    # 1280
        spectral_lens = []
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
                sf = int(stride_frames) if stride_frames is not None else int(frames_per_window) // 2
                total_frames = (num_windows - 1) * sf + frames_per_window
            
            if (total_frames < 2):
                raise Exception("WARN: spectral_len < 2, wav_lens:", wav_len.item(), "output frames:", total_frames.item(), "num_windows:", num_windows.item(), "Expected at least", window_size_ms, "ms", "got", (1000*wav_len.item()/sample_rate), "ms")
            spectral_lens.append(total_frames)
        
        spectral_lens = torch.tensor(spectral_lens, device=wav_lens.device, dtype=torch.long)


    else:
        # Given that there are 149 frames per 3 seconds,  49 frames per 1 seconds, we can calculate the number of frames for the whole audio clip
            
        #max_seconds = self.wav_len_max / self.sample_rate
        #max_frames = int(max_seconds * 50) # 49 frames per second, 20ms per frame
        

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
            
    return spectral_lens


def calc_spec_len_ext_v1(wav_lens, window_size_ms, stride_ms, sample_rate, frames_per_window, disable_windowing=False, wav_len_max=1*16000):
    """
    Calculate the total number of frames for the whole audio clip, for each clip in the batch.
    Input:
        wav_lens: tensor of real lengths of the audio clips in samples. Shape: [batch_size]
    Returns:
        spectral_lens: tensor of total number of frames for each audio clip. Shape: [batch_size]
    """
    
    if (not disable_windowing):
        window_size_samples = int(window_size_ms * sample_rate / 1000)  # 2560
        stride_samples = int(stride_ms * sample_rate / 1000)    # 1280

        # move self.frames_per_window to the same device if not already:
        frames_per_window = frames_per_window.to(wav_lens.device)
        
        spectral_lens = []
        for wav_len in wav_lens:
            # Calculate number of windows
            num_windows = ((wav_len - window_size_samples) // stride_samples) + 1
            # Calculate CNN output size for one window
            
            # Calculate total frames after combining windows
            total_frames = ((num_windows * frames_per_window) // 2)  # divide by 2 due to window overlap
        
            spectral_lens.append(total_frames)
        
        spectral_lens = torch.tensor(spectral_lens, device=wav_lens.device)


    else:
        # Given that there are 149 frames per 3 seconds,  49 frames per 1 seconds, we can calculate the number of frames for the whole audio clip
            
        #max_seconds = self.wav_len_max / self.sample_rate
        #max_frames = int(max_seconds * 50) # 49 frames per second, 20ms per frame
        

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
            
    return spectral_lens


