'''
Adaptive chunking for long-form forced alignment.

## The problem this replaces

Viterbi forced alignment over a whole 30 s clip with ~350 target phonemes does not
degrade gracefully - it collapses. On `30s.wav` (32 s, 353 phonemes) the decoder packed
every phoneme into the first 18.3 s and left 14 s of audio unused; the last ~200 phonemes
came out at exactly one frame (13.125 ms) each with confidence 0.000.

The previous mitigation was *silence anchoring*: find SIL groups in the phonemized text
(i.e. punctuation), find silence-looking regions in the posteriors, match them up by
proportional position, and decode each span separately. It fails whenever the text's
punctuation and the speaker's pauses do not correspond one-to-one, which is most of the
time - on the clip above it produced two spans, one of which still held 303 phonemes.

## What this does instead

Walk the clip forward in overlapping windows of `chunk_seconds` of audio, feeding each
window a DELIBERATELY OVER-PROVISIONED slice of the remaining phonemes. Over-provisioning
is the whole trick: when a window is handed more phonemes than its audio actually
contains, the forced decoder aligns the ones that are really there and then crams the
surplus into the final frames at one frame apiece. That cram is a loud, unambiguous
signal, and it marks exactly where the window's trustworthy alignment ends.

So each iteration:

  1. estimate a phoneme budget for this window from its SPEECH seconds (see the VAD
     below) at `phonemes_per_second`, capped at `max_phonemes`;
  2. align the window;
  3. find the first run of `derail_run` consecutive phonemes whose duration is
     `<= derail_frames` frames - the cram - and back off from it over the preceding
     near-zero-confidence drift (confidence collapses alongside duration, and slightly
     earlier, so it is a second clue for the same event; see `find_derailment`);
  4. keep everything before that, and restart the next window at the end of the last
     kept phoneme, with the phonemes that were not kept.

Because step 4 restarts at a *phoneme* boundary rather than at a fixed offset, the
windows overlap by however much of the previous window was untrustworthy, and no audio
between two kept phonemes is ever skipped.

## What the VAD is for

`phonemes_per_second` (15 /s is a safe upper bound for English) is a rate over SPEECH, not
over wall-clock time. A window holding 4 s of speech and 6 s of pause contains ~60
phonemes, not ~150, and budgeting 150 for it would push the cram point so far past the end
of the real content that step 3 cuts in the wrong place. `detect_speech_segments` is a
plain energy VAD whose only jobs are to measure speech seconds per window and to let
`snap_cut_to_silence` move a cut into a nearby pause. It is deliberately NOT the silence
anchoring described above: it never splits the target sequence and never touches the
posteriors.
'''

import torch


# --- defaults -------------------------------------------------------------------------
# Window length. 10 s is comfortably inside the range the decoder handles well and leaves
# room for a generous phoneme budget.
DEFAULT_CHUNK_SECONDS = 10.0

# Phonemes per second of SPEECH used to size a window's budget. English runs ~10-13 /s in
# normal delivery, so 15 over-provisions by ~20-50% - which is the point (see module docs).
#
# Over-provisioning is NOT free, and more is not better. MEASURED against Buckeye
# ground-truth phone boundaries (s0103a, 2267 boundaries, median absolute onset error and
# the share of boundaries within 25 ms):
#
#   phonemes/s    12       15       18       20
#   median err  16.4ms   16.8ms   17.8ms   17.8ms
#   within 25ms  64.6%    63.2%    60.3%    59.9%
#   worst decile  19ms     21ms     42ms     31ms
#
# Monotonic: the less a window is over-provisioned, the better it aligns. The same shows up
# far more violently on a 36.5 s LJSpeech concatenation, where 15 -> 20 took mean confidence
# from 0.417 to 0.212, audio coverage from 99% to 88%, and DOUBLED the share of one-frame
# phonemes (16.7% -> 35.8%).
#
# THE MECHANISM IS NOT THE CTC LATTICE. That was the first guess and it is wrong: at 20 /s
# the LJ budget is ~186 phonemes in a 761-frame window, so `4*S+1 = 745 <= 761` and the
# decode stays at stride 4 in both configurations. What actually happens is simpler and
# worse. `enforce_all_targets` means the forced path must emit EVERY target inside the
# window. Surplus targets cannot be declined; they take frames from the phonemes that are
# genuinely there. So over-provisioning does not merely append a detectable tail - it
# distorts the whole window, and the distortion grows with the surplus.
#
# That is the tension the whole design sits on: the cram has to be big enough to SEE (it is
# the cut signal) and small enough not to corrupt what precedes it. 15 /s is the compromise;
# 12 /s measured marginally better on the one conversational file above, which is not enough
# evidence to move a default that has to work on read speech too.
#
# Under-provisioning is the safe direction and is already handled: a window with no cram
# and audio still to come takes the `NO_DERAIL_TRIM_FRACTION` path, drops a short tail, and
# the next window re-decodes those phonemes.
DEFAULT_PHONEMES_PER_SECOND = 15.0

# Upper bound on the phoneme budget for ONE window, derived rather than chosen.
#
# `ViterbiDecoder.decode_with_forced_alignment` builds a CTC lattice of `stride * S + 1`
# states and walks `stride` down from 4 until it fits the window's frames. Stride 4 is what
# leaves room for blanks between phones; below it the lattice is compressed and placement
# degrades. So the largest budget a window can take WITHOUT forcing that compression is
#
#     S_max = (frames_in_window - 1) // 4,    frames = chunk_seconds * 16000 / 210
#
# 190 for a 10 s window at the 210-sample stride; the previous hand-picked 200 sat just
# over the line. A DERIVED bound beats a round number here because it tracks
# `chunk_seconds`: an 8 s window only holds 152, a 12 s window 228, and nothing warns you
# when a shortened window quietly crosses over. `stride4_max_phonemes` is that arithmetic,
# and `PhonemeTimestampAligner._warn_if_budget_overruns_lattice` checks it on every
# construction and every `set_chunking` call.
#
# HONEST SCOPE. This is a guard, not the fix for anything measured so far. The stride drop
# it prevents is real arithmetic, but it did not occur in any configuration tested: on the
# Buckeye ground-truth sweep, cap 190 and cap 210 gave byte-identical results at 20 /s
# (17.8 ms median, 59.9% within 25 ms) because the rate, not the cap, was binding. What
# actually degrades a window at a high rate is described on DEFAULT_PHONEMES_PER_SECOND.
#
# This is a CAP, deliberately independent of `DEFAULT_PHONEMES_PER_SECOND`: tying the two
# together (max = pps * seconds) makes it unreachable by construction, since the budget is
# `pps * speech_seconds` and speech_seconds <= chunk_seconds. A cap that can never bind is
# not a cap.
_FRAMES_PER_SECOND = 16000 / 210.0          # the stride every released checkpoint uses
DEFAULT_MAX_PHONEMES = (int(DEFAULT_CHUNK_SECONDS * _FRAMES_PER_SECOND) - 1) // 4    # 190


def stride4_max_phonemes(chunk_seconds, frames_per_second=_FRAMES_PER_SECOND):
    '''
    Largest phoneme budget a `chunk_seconds` window can hold while the CTC lattice still
    fits at stride 4 - the cap explained on `DEFAULT_MAX_PHONEMES`.

    Use it when changing `chunk_seconds`, because the two are not independent: an 8 s window
    only supports 152 phonemes, a 12 s window 228. Pass `frames_per_second` for a model
    whose feature extractor does not use the 210-sample stride.
    '''
    return max(1, (int(chunk_seconds * frames_per_second) - 1) // 4)

# A phoneme this short (in frames) is not a phoneme, it is the decoder running out of
# audio. 1 frame = 13.125 ms at the 210-sample stride; the shortest real phones (stop
# bursts) measure 2-3 frames.
DEFAULT_DERAIL_FRAMES = 1
# How many consecutive short phonemes make a cram rather than one genuinely brief stop.
DEFAULT_DERAIL_RUN = 3
# Phonemes dropped before the detected cram. The alignment starts drifting slightly before
# it visibly collapses, so the last couple of phonemes before the run are not trustworthy.
# This is the MINIMUM backoff; `find_derailment` extends it backwards over any contiguous
# run of near-zero-confidence phonemes immediately preceding the cram (see `conf_floor`).
DEFAULT_BACKOFF = 2
# Confidence below this counts as "the decoder is not actually seeing this phoneme".
# Confidence collapses alongside duration when a window derails - in the 30s.wav baseline
# the crammed tail ran at 0.000-0.004 while the healthy head averaged 0.28 - so it is a
# second, independent clue for the same event. It is NOT used on its own to trigger a cut:
# individual real phonemes score 0.000 in perfectly good regions (the decoder is confident
# about a neighbour instead), so only a sustained run of it means anything.
DEFAULT_CONF_FLOOR = 0.02
# How many consecutive near-zero-confidence phonemes count as a derailment on their own,
# when durations still look plausible. Much longer than `derail_run`, precisely because
# confidence is the noisier of the two signals.
DEFAULT_WEAK_RUN = 12
# Cap on how far the confidence-driven backoff may walk back from a detected cram, as a
# fraction of the window's budget. Without it, a window that was weak throughout would
# back off to nothing and stall the walk.
MAX_BACKOFF_FRACTION = 0.35
# Fraction of the budget that must be kept from every window, so the walk always makes
# progress even when a window's alignment is bad from the start.
MIN_KEEP_FRACTION = 0.15
# When no cram is found and audio remains, the budget was too SMALL for the window: its
# last phonemes were stretched to fill the audio. Drop this fraction of the tail.
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
    phonemes is recoverable (the cram detector cuts the window short), budgeting too few
    is not (the surplus audio silently stretches the last kept phoneme).
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

      1. **Duration** - a run of `derail_run` phonemes at `<= derail_frames` frames each.
         This is the cram: the decoder ran out of audio and is emitting one frame per
         leftover target. Unambiguous, and it is what the walk is normally cut by.
      2. **Confidence** - a run of `weak_run` phonemes below `conf_floor`. Weaker
         evidence, used only when (1) finds nothing: a window can drift onto the wrong
         audio without the durations degenerating, and when it does the confidences go to
         zero first. `weak_run` is deliberately much longer than `derail_run`, because
         isolated real phonemes score 0.000 in healthy regions too.

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
        derail_run: how many consecutive short phonemes constitute the cram.
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
