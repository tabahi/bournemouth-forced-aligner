'''
Windowing and stitching for p3cupe.

The model is deliberately context-free: it never sees more than `window_size_ms` of audio at
once. A clip is sliced into overlapping windows, every window is encoded independently, and the
per-window outputs are stitched back onto one clip-long frame grid. That is the whole mechanism
behind the multilingual claim - nothing in the model can learn a language's phonotactics,
because nothing in the model sees far enough to have any.

## The one invariant that matters

    stitched frame f is centred on input sample f * frame_stride_samples

This holds only if the window stride advances a WHOLE number of output frames, i.e. if
`stride_samples % frame_stride_samples == 0`. `frames_geometry` enforces it and raises
otherwise. In p2cupe the stitch hard-coded an advance of `frames_per_window // 2`, which was
wrong for every stride except exactly half a window, and it silently averaged frames describing
moments ~13 ms apart - a smear at the scale of the metric being optimised. Do not relax the
check: a warning would be read once and then ignored for a whole run.

With this module's geometry (80-sample frame stride = 5 ms, 1280-sample window stride = 80 ms)
the advance is exactly 16 frames, so the stitched grid IS the native conv grid and

    time_ms(frame f) = f * 1000 * frame_stride_samples / sample_rate = f * 5.0

with no correction term anywhere. Every timestamp this project produces comes from that line.
'''

import math

import torch


def frames_geometry(window_size_ms, stride_ms, sample_rate, frames_per_window,
                    frame_stride_samples):
    '''
    Resolve how many OUTPUT frames one window stride advances.

    Raises on a geometry that does not divide evenly, because the alternative is a silent
    sub-frame smear on every stitched frame.

    Returns:
        (stride_frames, stride_samples, window_size_samples)
    '''
    window_size_samples = int(round(window_size_ms * sample_rate / 1000))
    stride_samples = int(round(stride_ms * sample_rate / 1000))
    exact = stride_samples / float(frame_stride_samples)
    stride_frames = int(round(exact))

    if abs(exact - stride_frames) > 1e-6:
        raise ValueError(
            f"windowing geometry is inconsistent: a stride of {stride_samples} samples "
            f"({stride_ms} ms) is {exact:.4f} output frames, not a whole number "
            f"(frame stride {frame_stride_samples} samples). The stitch would average frames "
            f"describing moments "
            f"{abs(exact - stride_frames) * frame_stride_samples / sample_rate * 1000:.2f} ms "
            f"apart. Use a stride_ms that is a multiple of "
            f"{1000.0 * frame_stride_samples / sample_rate:g} ms.")
    if stride_frames < 1:
        raise ValueError(f"stride_frames must be >= 1, got {stride_frames}")
    if stride_frames > frames_per_window:
        raise ValueError(f"stride_frames ({stride_frames}) > frames_per_window "
                         f"({frames_per_window}): windows would leave gaps")
    return stride_frames, stride_samples, window_size_samples


def num_windows_for(audio_length_samples, window_size_samples, stride_samples):
    '''Number of windows `slice_windows` produces for a clip of this length (at least 1).'''
    n = ((int(audio_length_samples) - int(window_size_samples)) // int(stride_samples)) + 1
    return max(n, 1)


def total_stitched_frames(num_windows, frames_per_window, stride_frames):
    '''Number of stitched output frames for `num_windows` windows.'''
    return (max(int(num_windows), 1) - 1) * int(stride_frames) + int(frames_per_window)


def slice_windows(audio_batch, sample_rate=16000, window_size_ms=120, stride_ms=80):
    """
    Cut a padded batch of waveforms into overlapping fixed-size windows.

    Args:
        audio_batch: [B, 1, L] or [B, L]
    Returns:
        [B, num_windows, window_size_samples]
    """
    if audio_batch.dim() == 3:
        audio_batch = audio_batch.squeeze(1)
    batch_size, max_audio_length = audio_batch.shape

    window_size = int(round(window_size_ms * sample_rate / 1000))
    stride = int(round(stride_ms * sample_rate / 1000))
    num_windows = num_windows_for(max_audio_length, window_size, stride)

    offsets = torch.arange(window_size, device=audio_batch.device)
    starts = torch.arange(num_windows, device=audio_batch.device) * stride
    indices = starts.unsqueeze(1) + offsets.unsqueeze(0)          # [W, window_size]

    valid = indices < max_audio_length
    indices = indices.clamp(max=max_audio_length - 1)

    batch_indices = torch.arange(batch_size, device=audio_batch.device)[:, None, None]
    windows = audio_batch[batch_indices, indices]
    return windows * valid.to(windows.dtype)


def stitch_windows(window_outputs, num_windows_total, frames_per_window, stride_frames):
    """
    Overlap-add the per-window outputs onto the clip-long frame grid.

    Windows are blended with a cosine taper, so a frame is taken mostly from the window whose
    CENTRE it sits nearest. That matters more here than in a contextual model: a frame at a
    window edge was computed from a receptive field half of which is zero padding, and the
    neighbouring window holds the same moment computed from real audio on both sides.

    Args:
        window_outputs: [B, W, frames_per_window, dim]
        num_windows_total: windows implied by the padded clip length (== W here; passed
            explicitly so callers can assert it rather than assume it)
        frames_per_window, stride_frames: from `frames_geometry`
    Returns:
        [B, total_frames, dim]
    """
    device = window_outputs.device
    batch_size, num_windows, fpw, dim = window_outputs.shape
    total_frames = total_stitched_frames(num_windows_total, frames_per_window, stride_frames)

    # float32 accumulation even under bf16 autocast. bf16 carries 8 mantissa bits, i.e. a
    # resolution of ~0.06 on a logit of 10 - and the boundary readout is a linear
    # interpolation of the DIFFERENCE between adjacent frames' logits, so that rounding lands
    # directly on the quantity this model exists to produce.
    weights = torch.cos(torch.linspace(-math.pi / 2, math.pi / 2, fpw, device=device))
    weights = weights.clamp(min=1e-3).view(1, fpw, 1)
    x = window_outputs.float()

    combined = torch.zeros(batch_size, total_frames, dim, device=device)
    weight_sum = torch.zeros(batch_size, total_frames, 1, device=device)

    for i in range(num_windows):
        start = i * stride_frames
        end = min(start + fpw, total_frames)
        if end <= start:
            break
        n = end - start
        combined[:, start:end] += x[:, i, :n] * weights[:, :n]
        weight_sum[:, start:end] += weights[:, :n]

    return combined / weight_sum.clamp(min=1e-6)


def spectral_lengths(wav_lens, window_size_samples, stride_samples,
                     frames_per_window, stride_frames):
    """
    Valid stitched frames per clip, from the real (unpadded) sample lengths.

    Args:
        wav_lens: [B] real lengths in samples
    Returns:
        [B] long
    """
    n_windows = ((wav_lens.long() - int(window_size_samples)) // int(stride_samples)) + 1
    n_windows = n_windows.clamp(min=1)
    lens = (n_windows - 1) * int(stride_frames) + int(frames_per_window)

    # A clip shorter than one window still produces one full window of frames, but only the
    # leading `wav_len / frame_stride` of them saw real audio. Cut them back so the loss is not
    # supervised on frames that are pure zero padding.
    short = wav_lens.long() < int(window_size_samples)
    if short.any():
        frame_stride = int(stride_samples // stride_frames)
        short_lens = torch.ceil(wav_lens.float() / frame_stride).long()
        lens = torch.where(short, short_lens, lens)

    return lens.clamp(min=1)
