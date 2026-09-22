'''
Chunked alignment for very long segments.

## When this runs, and why it is not the main path

The segmental decoder in `p3cupe/aligner.py` does not collapse on long audio: it has no band
constraint and no lattice to compress, and every phone is placed in one global search. What it
does have is a python loop over frames, so a segment costs time and memory in proportion to its
length. `core.PhonemeTimestampAligner` therefore decodes whole segments up to
`segment_max_seconds` and only hands longer ones to this module. Chunking is strictly worse
when it is not needed - a cut can land mid-phone - so the threshold is a resource bound, not a
quality one.

## What it does

Walk the segment forward in overlapping windows of `chunk_seconds` of audio, giving each window
a slice of the remaining phonemes sized from its SPEECH seconds:

  1. estimate a phoneme budget from `phonemes_per_second` over the window's speech seconds,
     capped at `max_phonemes` and at what the window's frames can hold at the duration floor;
  2. align the window;
  3. decide where the window stops being trustworthy (`find_derailment`);
  4. keep everything before that, and restart the next window at the end of the last kept
     phoneme, with the phonemes that were not kept.

Because step 4 restarts at a *phoneme* boundary rather than at a fixed offset, the windows
overlap by however much of the previous window was untrustworthy, and no audio between two kept
phonemes is ever skipped.

## What "derailment" means for this decoder

CONFIDENCE is the signal that matters here. The old CTC lattice announced a window whose text
outran its audio by cramming the surplus into one frame apiece - a loud duration signal - but
this decoder enforces a minimum phone duration and simply reports the alignment as infeasible
instead, which `core._align_chunk` catches and retries with a smaller budget. So a window that
goes wrong here goes wrong quietly: it drifts onto the wrong audio, and its confidences collapse
while its durations stay plausible. The duration rule is kept because it costs nothing and still
catches a decoder that has been forced down to a one-frame floor.

## What the VAD is for

`phonemes_per_second` is a rate over SPEECH, not over wall-clock time. A window holding 4 s of
speech and 6 s of pause contains ~60 phonemes, not ~150, and over-budgeting it pushes phonemes
into audio that does not contain them. `detect_speech_segments` is a plain energy VAD whose only
jobs are to measure speech seconds per window and to let `snap_cut_to_silence` move a cut into a
nearby pause. It never splits the target sequence and never touches the posteriors.
'''

import torch


# --- defaults -------------------------------------------------------------------------
# Window length. 10 s is comfortably inside the range the decoder handles well and leaves
# room for a generous phoneme budget.
DEFAULT_CHUNK_SECONDS = 10.0

# Phonemes per second of SPEECH used to size a window's budget. English runs ~10-13 /s in
# normal delivery, so 15 leaves ~20-50% of headroom.
#
# Over-budgeting is not free: the forced path must place EVERY target it is given inside the
# window, so surplus targets cannot be declined and take frames from the phonemes that are
# genuinely there. Under-budgeting is the safe direction and is already handled - a window with
# no derailment and audio still to come takes the `NO_DERAIL_TRIM_FRACTION` path, drops a short
# tail, and the next window re-decodes those phonemes.
DEFAULT_PHONEMES_PER_SECOND = 15.0

# Upper bound on the phoneme budget for ONE window. Deliberately independent of
# `DEFAULT_PHONEMES_PER_SECOND`: tying the two together (max = pps * seconds) would make it
# unreachable by construction, since the budget is `pps * speech_seconds` and speech_seconds
# never exceeds chunk_seconds. A cap that can never bind is not a cap.
#
# `core.align_segment_chunked` applies a second, tighter bound that this one cannot express,
# because it depends on the loaded checkpoint: a window's frames divided by the duration floor,
# i.e. the most phones the audio can physically hold. That is the one that keeps a window
# feasible; this is a blunt ceiling on memory and time.
DEFAULT_MAX_PHONEMES = 200

# A phoneme this short (in frames) is not a phoneme, it is the decoder running out of audio.
# The decoder's own duration floor normally keeps phones well above this, so the rule only
# fires when that floor has been reduced to fit a clip whose text outruns its audio.
DEFAULT_DERAIL_FRAMES = 1
# How many consecutive short phonemes make a derailment rather than one genuinely brief stop.
DEFAULT_DERAIL_RUN = 3
# Phonemes dropped before the detected derailment. The alignment starts drifting slightly
# before it visibly collapses, so the last couple of phonemes before the run are not
# trustworthy. This is the MINIMUM backoff; `find_derailment` extends it backwards over any
# contiguous run of near-zero-confidence phonemes immediately preceding it (see `conf_floor`).
DEFAULT_BACKOFF = 2
# Confidence below this counts as "the decoder is not actually seeing this phoneme". It is
# the primary derailment signal for this decoder (see the module docstring), but never on a
# single phoneme: real phonemes score near zero in perfectly good regions, because the model
# is confident about a neighbour instead. Only a sustained run of it means anything.
DEFAULT_CONF_FLOOR = 0.02
# How many consecutive near-zero-confidence phonemes count as a derailment on their own,
# when durations still look plausible. Much longer than `derail_run`, precisely because
# confidence is the noisier of the two signals.
DEFAULT_WEAK_RUN = 12
# Cap on how far the confidence-driven backoff may walk back from a detected derailment, as a
# fraction of the window's budget. Without it, a window that was weak throughout would
# back off to nothing and stall the walk.
MAX_BACKOFF_FRACTION = 0.35
# Fraction of the budget that must be kept from every window, so the walk always makes
# progress even when a window's alignment is bad from the start.
MIN_KEEP_FRACTION = 0.15
# When no derailment is found and audio remains, a window is trusted only up to this fraction
# of its own audio: every phoneme the aligner placed past that point is handed back to the next
# window, which sees it with real audio on both sides instead of at its edge.
#
# A TIME fraction, not a count fraction, and applied as a CAP on top of `find_derailment`
# rather than as a fallback when it finds nothing. That distinction is the whole point. The
# old rule dropped 5% of the phoneme COUNT, which says nothing about whether the window's
# phonemes actually reached the end of its audio. With a decoder that cannot cram - this one
# enforces a minimum phone duration and reports infeasibility instead - an over-budgeted
# window places its surplus phonemes early rather than announcing itself, and keeps nearly all
# of them, so the walk runs AHEAD of the audio and the error compounds window after window.
#
# MEASURED on 30s.wav (31.95 s, 356 phonemes) forced through the chunker at 10 s windows:
#
#                        last phoneme ends   mean confidence   flagged bad
#   count trim only           30.31 s             0.217            yes
#   time cap                  31.87 s             0.317            no
#   (whole-segment decode)    31.88 s             0.319            no
#
# i.e. the count rule finished the transcript 1.6 s before the end of the clip, and the time
# cap brings the chunked result back to within 10 ms of the whole-segment one.
TRUST_WINDOW_FRACTION = 0.75
# Floor on how much of a window survives the time trim, as a fraction of its budget, so a
# window whose audio the aligner barely used still makes progress.
NO_DERAIL_TRIM_FRACTION = 0.05
# How far a cut may be moved to land in a pause instead of mid-word.
DEFAULT_SNAP_TO_SILENCE_SEC = 0.25


# --- VAD ------------------------------------------------------------------------------

def detect_speech_segments(wav, sample_rate, frame_ms=25.0, hop_ms=10.0,
                           noise_margin_db=10.0, dynamic_range_db=40.0,
                           min_speech_ms=120.0, min_silence_ms=180.0, pad_ms=30.0):
    '''
    Energy VAD. Returns speech spans as a list of `(start_sec, end_sec)`.

    The threshold is picked per clip from the frame-energy distribution rather than being
    a fixed dB value, because the caller may hand us RMS-normalised audio, raw audio, or a
    quiet field recording:

        floor  = 10th percentile of frame energy   (the clip's own noise floor)
        peak   = 95th percentile                   (its own speech level)
        thresh = max(floor + noise_margin_db, peak - dynamic_range_db), and never above
                 `peak - 6 dB` so a clip that is nearly all speech cannot threshold its
                 own speech away.

    Then two smoothing passes, in this order: silence gaps shorter than `min_silence_ms`
    are filled (a stop closure inside a word is not a pause), and speech islands shorter
    than `min_speech_ms` are dropped (a click is not a word). Finally each span is padded
    by `pad_ms` on both sides, since energy rises after a phone's true onset.

    Args:
        wav: 1-D waveform tensor, or (C, T) - channels are averaged.
        sample_rate: samples per second.
    Returns:
        List of (start_sec, end_sec), non-overlapping and in order. An empty list means
        "no speech found"; callers should treat that as "assume all of it is speech"
        rather than as "there are zero phonemes here".
    '''
    if wav is None:
        return []
    if wav.dim() > 1:
        wav = wav.mean(dim=0)
    wav = wav.detach().float().cpu()

    n = wav.shape[0]
    total_sec = n / float(sample_rate)
    frame_len = max(1, int(frame_ms * sample_rate / 1000.0))
    hop_len = max(1, int(hop_ms * sample_rate / 1000.0))
    if n < frame_len * 2:
        return [(0.0, total_sec)] if total_sec > 0 else []

    frames = wav.unfold(0, frame_len, hop_len)                  # [F, frame_len]
    energy = frames.pow(2).mean(dim=1).clamp_min(1e-12)
    db = 10.0 * torch.log10(energy)

    floor_db = torch.quantile(db, 0.10).item()
    peak_db = torch.quantile(db, 0.95).item()
    thresh = max(floor_db + noise_margin_db, peak_db - dynamic_range_db)
    thresh = min(thresh, peak_db - 6.0)

    is_speech = (db >= thresh).tolist()
    spans = _mask_to_spans(is_speech)

    # frame index -> seconds. A frame covers [i*hop, i*hop + frame_len).
    def _t0(i):
        return i * hop_len / float(sample_rate)

    def _t1(i):
        return min(total_sec, (i * hop_len + frame_len) / float(sample_rate))

    spans = [(_t0(a), _t1(b - 1)) for a, b in spans]

    spans = _fill_short_gaps(spans, min_silence_ms / 1000.0)
    spans = [s for s in spans if (s[1] - s[0]) >= (min_speech_ms / 1000.0)]

    pad = pad_ms / 1000.0
    spans = [(max(0.0, a - pad), min(total_sec, b + pad)) for a, b in spans]
    return _fill_short_gaps(spans, 0.0)     # merge anything the padding made contiguous


def _mask_to_spans(mask):
    '''Boolean list -> list of half-open [start, end) index spans of True.'''
    spans = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def _fill_short_gaps(spans, min_gap_sec):
    '''Merge consecutive spans separated by less than `min_gap_sec`.'''
    if not spans:
        return []
    merged = [list(spans[0])]
    for a, b in spans[1:]:
        if a - merged[-1][1] <= min_gap_sec:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def speech_seconds(speech_spans, t0, t1):
    '''
    Seconds of speech inside `[t0, t1)`.

    With no VAD result at all the whole interval counts as speech - budgeting too many
    phonemes is recoverable (the derailment detector cuts the window short), budgeting too
    few is not (the surplus audio silently stretches the last kept phoneme).
    '''
    if t1 <= t0:
        return 0.0
    if not speech_spans:
        return t1 - t0
    total = 0.0
    for a, b in speech_spans:
        if b <= t0:
            continue
        if a >= t1:
            break
        total += min(b, t1) - max(a, t0)
    return total


def silence_gaps(speech_spans, total_sec):
    '''Complement of `speech_spans` over `[0, total_sec)` - i.e. the pauses.'''
    if total_sec <= 0:
        return []
    if not speech_spans:
        return []
    gaps = []
    prev = 0.0
    for a, b in speech_spans:
        if a > prev:
            gaps.append((prev, a))
        prev = max(prev, b)
    if prev < total_sec:
        gaps.append((prev, total_sec))
    return gaps


def snap_cut_to_silence(cut_sec, speech_spans, total_sec, tolerance_sec=DEFAULT_SNAP_TO_SILENCE_SEC):
    '''
    Move a cut point to the start of a nearby pause, if there is one within
    `tolerance_sec`. Restarting a window on a pause rather than mid-word gives the next
    window's first phonemes a clean onset to latch onto.

    Never moves the cut backwards past `cut_sec - tolerance_sec`, so it cannot undo the
    walk's forward progress.
    '''
    if tolerance_sec <= 0:
        return cut_sec
    best = cut_sec
    best_dist = tolerance_sec
    for a, b in silence_gaps(speech_spans, total_sec):
        if b <= cut_sec - tolerance_sec:
            continue
        if a > cut_sec + tolerance_sec:
            break
        dist = abs(a - cut_sec)
        if dist < best_dist:
            best_dist = dist
            best = a
    return best


# --- derailment detection -------------------------------------------------------------

def find_derailment(durations, confidences=None, min_keep=1,
                    derail_frames=DEFAULT_DERAIL_FRAMES, derail_run=DEFAULT_DERAIL_RUN,
                    conf_floor=DEFAULT_CONF_FLOOR, weak_run=DEFAULT_WEAK_RUN,
                    backoff=DEFAULT_BACKOFF):
    '''
    Index of the first phoneme this window can no longer be trusted at.

    Two signals, in priority order, because they fail in different ways:

      1. **Duration** - a run of `derail_run` phonemes at `<= derail_frames` frames each: the
         decoder emitting one frame per leftover target because it ran out of audio.
         Unambiguous when it fires, but the segmental decoder's duration floor normally keeps
         phones well clear of it, so it rarely does.
      2. **Confidence** - a run of `weak_run` phonemes below `conf_floor`. This is the signal
         that actually catches a bad window here: it drifts onto the wrong audio while its
         durations stay plausible, and the confidences go to zero first. `weak_run` is
         deliberately much longer than `derail_run`, because isolated real phonemes score
         0.000 in healthy regions too.

    The cut is then walked BACKWARDS from whichever fired, over any contiguous run of
    below-`conf_floor` phonemes immediately preceding it, and at least `backoff` phonemes
    in any case. The alignment does not snap from good to derailed in one phoneme; it
    drifts first, and the drift shows up as confidence decaying before the durations
    collapse. The walk-back is capped at `MAX_BACKOFF_FRACTION` of the window.

    Args:
        durations: per-target-index duration in FRAMES, or None where the target was not
            aligned at all. `None` counts as derailed - an unaligned target in the middle
            of a window means the decoder had already lost the thread.
        confidences: per-target-index mean confidence, or None to use durations only.
        min_keep: never report a cut before this index. A window whose very first
            phonemes look derailed is a window with a bad start, not a window whose budget
            ran out, and cutting there would stall the walk.
        derail_frames: a duration `<= this` (in frames) is impossibly short.
        derail_run: how many consecutive short phonemes constitute a derailment.
        conf_floor: confidence below this counts as "not actually seeing this phoneme".
        weak_run: how many consecutive weak phonemes derail a window on their own.
        backoff: minimum phonemes to drop before the detected point.
    Returns:
        `(cut_index, reason)` where reason is 'duration' or 'confidence', or
        `(None, None)` if the window looks healthy all the way through.
    '''
    n = len(durations)
    if n == 0:
        return None, None
    short = [(d is None) or (d <= derail_frames) for d in durations]
    if confidences is None:
        weak = [False] * n
    else:
        weak = [(c is None) or (c < conf_floor) for c in confidences]

    hit = _first_run(short, derail_run, min_keep)
    reason = 'duration'
    if hit is None:
        hit = _first_run(weak, weak_run, min_keep)
        reason = 'confidence'
    if hit is None:
        return None, None

    # Walk back over the drift: the confidences decay before the durations collapse.
    cut = hit - backoff
    limit = max(min_keep, hit - int(MAX_BACKOFF_FRACTION * n))
    while cut > limit and weak[cut - 1]:
        cut -= 1
    cut = max(cut, min_keep)
    return (cut if cut < n else None), (reason if cut < n else None)


def _first_run(flags, run_len, min_keep):
    '''
    Start index of the first run of `run_len` consecutive True flags beginning at or after
    `min_keep`. A run that reaches the end of the list counts even when shorter than
    `run_len` - there is no further evidence coming. Runs starting before `min_keep` are
    skipped over rather than reported.
    '''
    n = len(flags)
    run_start = None
    for i in range(n):
        if flags[i]:
            if run_start is None:
                run_start = i
            if ((i - run_start + 1) >= run_len or i == n - 1) and run_start >= min_keep:
                return run_start
        else:
            run_start = None
    return None
