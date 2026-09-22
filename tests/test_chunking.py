# File: tests/test_chunking.py
"""
Tests for the adaptive long-segment chunker (`bournemouth_aligner.chunking`).

These cover the two decisions the chunker makes on its own - where a window derailed, and
how much speech a window holds - without loading a model. The end-to-end behaviour is
covered by `test_core.py`.
"""

import math

import pytest
import torch

import sys
sys.path.append('.')
from bournemouth_aligner import chunking


class TestFindDerailment:
    """`find_derailment` - where a window stopped being trustworthy."""

    def test_healthy_window_is_not_cut(self):
        durations = [5] * 60
        confidences = [0.7] * 60
        cut, reason = chunking.find_derailment(durations, confidences, min_keep=5)
        assert cut is None
        assert reason is None

    def test_cram_at_tail_is_found(self):
        # 40 real phonemes, then the decoder runs out of audio and emits one frame each.
        durations = [4] * 40 + [1] * 60
        confidences = [0.7] * 40 + [0.0] * 60
        cut, reason = chunking.find_derailment(durations, confidences, min_keep=5, backoff=2)
        assert reason == 'duration'
        # The cram starts at 40; backoff drops 2 more.
        assert cut == 38

    def test_confidence_drift_extends_the_backoff(self):
        # Confidence decays for 6 phonemes BEFORE the durations collapse - the walk-back
        # should follow it rather than stopping at the fixed backoff.
        durations = [4] * 40 + [1] * 60
        confidences = [0.7] * 34 + [0.0] * 6 + [0.0] * 60
        cut, _ = chunking.find_derailment(durations, confidences, min_keep=5, backoff=2)
        assert cut == 34

    def test_backoff_is_capped(self):
        # A window that is weak throughout must not back off to nothing.
        n = 100
        durations = [4] * 60 + [1] * 40
        confidences = [0.0] * n
        cut, _ = chunking.find_derailment(durations, confidences, min_keep=5, backoff=2)
        assert cut >= 60 - int(chunking.MAX_BACKOFF_FRACTION * n)

    def test_confidence_run_alone_can_derail(self):
        # Durations stay plausible but the decoder has drifted onto the wrong audio.
        durations = [4] * 100
        confidences = [0.7] * 50 + [0.0] * 50
        cut, reason = chunking.find_derailment(durations, confidences, min_keep=5,
                                               weak_run=12, backoff=2)
        assert reason == 'confidence'
        assert cut == 48

    def test_isolated_short_phoneme_does_not_cut(self):
        # A single 1-frame stop in the middle of a healthy window is a real phoneme.
        durations = [4] * 30 + [1] + [4] * 30
        confidences = [0.7] * 61
        cut, _ = chunking.find_derailment(durations, confidences, min_keep=5, derail_run=3)
        assert cut is None

    def test_run_before_min_keep_is_skipped(self):
        # A window that starts badly is a bad window, not a window whose budget ran out;
        # cutting at index 1 would stall the walk.
        durations = [1, 1, 1] + [4] * 50 + [1] * 20
        confidences = [0.7] * 73
        cut, _ = chunking.find_derailment(durations, confidences, min_keep=10, backoff=2)
        assert cut == 51

    def test_unaligned_target_counts_as_derailed(self):
        durations = [4] * 40 + [None] * 10
        confidences = [0.7] * 50
        cut, reason = chunking.find_derailment(durations, confidences, min_keep=5, backoff=0)
        assert reason == 'duration'
        assert cut == 40

    def test_empty_input(self):
        assert chunking.find_derailment([], [], min_keep=1) == (None, None)


class TestKeepCount:
    """
    `_chunk_keep_count`'s policy, exercised without a model.

    The method only reads `self.chunk_*` thresholds, so a bare namespace standing in for the
    aligner is enough and keeps these tests off the network.
    """

    @staticmethod
    def _keeper():
        from bournemouth_aligner.core import PhonemeTimestampAligner
        obj = PhonemeTimestampAligner.__new__(PhonemeTimestampAligner)
        obj.chunk_derail_frames = chunking.DEFAULT_DERAIL_FRAMES
        obj.chunk_derail_run = chunking.DEFAULT_DERAIL_RUN
        obj.chunk_backoff = chunking.DEFAULT_BACKOFF
        obj.chunk_conf_floor = chunking.DEFAULT_CONF_FLOOR
        obj.chunk_weak_run = chunking.DEFAULT_WEAK_RUN
        return obj

    @staticmethod
    def _stamps(durations, confidences):
        """Minimal 8-tuples: (id, start_frame, end_frame, target_idx, est, conf, s_ms, e_ms)."""
        out, f = [], 0
        for i, (d, c) in enumerate(zip(durations, confidences)):
            out.append((1, f, f + d, i, False, c, f * 5.0, (f + d) * 5.0))
            f += d
        return out

    def test_cut_applies_when_more_audio_follows(self):
        n = 100
        stamps = self._stamps([4] * 60 + [1] * 40, [0.7] * 60 + [0.0] * 40)
        keep, reason = self._keeper()._chunk_keep_count(stamps, n, final=False)
        assert reason == 'duration'
        assert keep < n

    def test_final_window_keeps_everything(self):
        # A cut defers phonemes to the next window; on the final window it would DELETE them.
        # This is the 30s.wav regression: the last 3 phonemes of "find" went missing.
        n = 100
        stamps = self._stamps([4] * 60 + [1] * 40, [0.7] * 60 + [0.0] * 40)
        keep, reason = self._keeper()._chunk_keep_count(stamps, n, final=True)
        assert keep == n
        assert 'final' in reason      # still reported, just not acted on

    def test_final_window_keeps_everything_on_a_confidence_cut_too(self):
        n = 33
        stamps = self._stamps([4] * 33, [0.7] * 20 + [0.0] * 13)
        keep, _ = self._keeper()._chunk_keep_count(stamps, n, final=True)
        assert keep == n

    def test_healthy_non_final_window_trims_a_short_tail(self):
        # No cram and audio still to come means the budget was too small, not too big.
        n = 100
        stamps = self._stamps([4] * n, [0.7] * n)
        keep, reason = self._keeper()._chunk_keep_count(stamps, n, final=False)
        assert reason == 'trim'
        assert 0 < keep < n

    def test_keep_never_falls_below_the_floor(self):
        n = 100
        stamps = self._stamps([1] * n, [0.0] * n)
        keep, _ = self._keeper()._chunk_keep_count(stamps, n, final=False)
        assert keep >= max(1, int(chunking.MIN_KEEP_FRACTION * n))


class TestPhonemeCap:
    """
    `chunk_max_phonemes` is a blunt ceiling on one window's budget. The bound that actually
    keeps a window feasible is applied in `core.align_segment_chunked` and depends on the
    loaded checkpoint (the window's frames divided by the duration floor), so it cannot live
    here.
    """

    def test_default_rate_stays_under_the_cap(self):
        # A full window at the default rate must not need the cap to rescue it.
        ask = chunking.DEFAULT_PHONEMES_PER_SECOND * chunking.DEFAULT_CHUNK_SECONDS
        assert ask <= chunking.DEFAULT_MAX_PHONEMES

    def test_cap_is_independent_of_the_rate(self):
        # Tying max = pps * seconds makes the cap unreachable, since the budget is
        # pps * speech_seconds and speech_seconds <= chunk_seconds. It must be able to bind.
        assert chunking.DEFAULT_MAX_PHONEMES != int(
            chunking.DEFAULT_PHONEMES_PER_SECOND * chunking.DEFAULT_CHUNK_SECONDS)


class TestNeedsChunking:
    """
    Which segments take the chunked path.

    Length, and nothing else. The segmental decoder does not collapse on a long segment or on
    a dense one, so chunking exists only to bound the DP's time and memory: it engages above
    `segment_max_seconds` and is otherwise strictly worse, because a cut can land mid-phone.
    """

    @staticmethod
    def _aligner(**kw):
        from bournemouth_aligner.core import PhonemeTimestampAligner
        obj = PhonemeTimestampAligner.__new__(PhonemeTimestampAligner)
        obj.chunk_audio = kw.get('chunk_audio', True)
        obj.chunk_seconds = kw.get('chunk_seconds', 10.0)
        obj.chunk_max_phonemes = kw.get('chunk_max_phonemes', 200)
        obj.segment_max_seconds = kw.get('segment_max_seconds', 120.0)
        return obj

    def test_exactly_at_the_threshold_stays_single_shot(self):
        assert self._aligner()._needs_chunking(120.0, 1800) is False

    def test_just_over_the_threshold_is_chunked(self):
        assert self._aligner()._needs_chunking(120.5, 1800) is True

    def test_phoneme_count_alone_does_not_trigger_it(self):
        # A dense segment is not a reason to chunk: the decoder places every phone in one
        # global search, and shortening the duration floor is how it copes when text
        # outruns audio.
        assert self._aligner()._needs_chunking(9.0, 2400) is False

    def test_disabled_means_never(self):
        a = self._aligner(chunk_audio=False)
        assert a._needs_chunking(600.0, 9000) is False

    def test_threshold_follows_the_setting(self):
        a = self._aligner(segment_max_seconds=30.0)
        assert a._needs_chunking(30.5, 400) is True
        assert a._needs_chunking(29.5, 400) is False


class TestWindowTrustCap:
    """
    `_chunk_keep_count` - how much of a window is trusted.

    The rule that matters for this decoder is the TIME cap: a window is never trusted past
    `TRUST_WINDOW_FRACTION` of its own audio, whatever the derailment detector says. Without
    it an over-budgeted window keeps phonemes it placed early, and the walk runs ahead of the
    audio (measured: 1.6 s of drift over a 32 s clip).
    """

    @staticmethod
    def _aligner():
        from bournemouth_aligner.core import PhonemeTimestampAligner
        obj = PhonemeTimestampAligner.__new__(PhonemeTimestampAligner)
        obj.chunk_derail_frames = chunking.DEFAULT_DERAIL_FRAMES
        obj.chunk_derail_run = chunking.DEFAULT_DERAIL_RUN
        obj.chunk_conf_floor = chunking.DEFAULT_CONF_FLOOR
        obj.chunk_weak_run = chunking.DEFAULT_WEAK_RUN
        obj.chunk_backoff = chunking.DEFAULT_BACKOFF
        return obj

    @staticmethod
    def _stamps(n, window_ms, span_ms, conf=0.7):
        """n healthy phonemes evenly filling `span_ms` of a window starting at window_ms[0]."""
        a = window_ms[0]
        step = span_ms / n
        return [(1, i * 2.0, (i + 1) * 2.0, i, False, conf, a + i * step, a + (i + 1) * step)
                for i in range(n)]

    def test_time_cap_trims_a_window_that_filled_its_audio(self):
        a = self._aligner()
        window = (0.0, 10000.0)
        stamps = self._stamps(100, window, 10000.0)
        keep, reason = a._chunk_keep_count(stamps, 100, final=False, window_ms=window)
        assert reason == 'trim'
        # 75% of the window's audio holds 75 of the 100 evenly spaced phonemes.
        assert keep == 75

    def test_time_cap_binds_even_when_a_derailment_is_detected_later(self):
        # The window is healthy for 90 phonemes and only then goes weak, so the confidence
        # rule would cut at ~88. The time cap at 75 must still win: it is a cap, not a
        # fallback for when the detector finds nothing.
        a = self._aligner()
        window = (0.0, 10000.0)
        stamps = self._stamps(100, window, 10000.0)
        stamps = [t if i < 90 else (*t[:5], 0.0, *t[6:]) for i, t in enumerate(stamps)]
        cut, reason = chunking.find_derailment(
            [2.0] * 100, [0.7] * 90 + [0.0] * 10, min_keep=15,
            weak_run=chunking.DEFAULT_WEAK_RUN, backoff=chunking.DEFAULT_BACKOFF)
        assert cut is not None and cut > 75, "precondition: the detector cuts after the cap"
        keep, _ = a._chunk_keep_count(stamps, 100, final=False, window_ms=window)
        assert keep == 75

    def test_final_window_keeps_everything(self):
        a = self._aligner()
        window = (0.0, 10000.0)
        stamps = self._stamps(100, window, 10000.0)
        keep, _ = a._chunk_keep_count(stamps, 100, final=True, window_ms=window)
        assert keep == 100

    def test_no_timing_falls_back_to_a_count_trim(self):
        a = self._aligner()
        stamps = self._stamps(100, (0.0, 10000.0), 10000.0)
        keep, reason = a._chunk_keep_count(stamps, 100, final=False, window_ms=None)
        assert reason == 'trim'
        assert keep == 95


class TestWindowLayout:
    """
    The window grid, in isolation from the model.

    Reproduces the loop's own arithmetic: a window runs `chunk_seconds` from the cursor,
    is extended to the end only when what remains could not be aligned at all, and the next
    window restarts at the DERAILMENT CUT rather than at the window's end.
    """

    @staticmethod
    def _layout(seg_dur, chunk_seconds=10.0, seg_duration_min=0.05, cut_frac=0.75):
        t, wins = 0.0, []
        for _ in range(12):
            ce = min(t + chunk_seconds, seg_dur)
            if (seg_dur - ce) < seg_duration_min:
                ce = seg_dur
            wins.append((round(t, 3), round(ce, 3)))
            if ce >= seg_dur - 1e-9:
                break
            t = t + (ce - t) * cut_frac
        return wins

    def test_short_segment_is_one_window(self):
        assert len(self._layout(8.0)) == 1

    def test_dead_zone_segment_actually_splits(self):
        # Used to collapse to a single window - the old rule extended the window whenever
        # under a quarter of `chunk_seconds` remained, which is the wrong quantity because
        # the next window starts at the cut, not at the window's end.
        for seg_dur in (10.2, 11.0, 12.0, 12.4):
            wins = self._layout(seg_dur)
            assert len(wins) >= 2, f"{seg_dur}s collapsed to {wins}"

    def test_no_window_is_degenerate(self):
        for seg_dur in (10.2, 12.0, 20.0, 35.0):
            for a, b in self._layout(seg_dur):
                assert b - a > 0.5, f"{seg_dur}s produced a {b-a:.3f}s window"

    def test_windows_cover_the_segment_and_overlap(self):
        wins = self._layout(20.0)
        assert wins[0][0] == 0.0
        assert wins[-1][1] == 20.0
        for (a0, b0), (a1, _b1) in zip(wins, wins[1:]):
            assert a1 < b0, "consecutive windows must overlap"
            assert a1 > a0, "the cursor must advance"


class TestVAD:
    """`detect_speech_segments` and the speech-seconds accounting built on it."""

    @staticmethod
    def _tone(seconds, sr=16000, amp=0.3, freq=220.0):
        t = torch.arange(int(seconds * sr)) / sr
        return amp * torch.sin(2 * math.pi * freq * t)

    @staticmethod
    def _silence(seconds, sr=16000, amp=1e-4):
        return amp * torch.randn(int(seconds * sr))

    def test_finds_speech_between_silences(self):
        sr = 16000
        wav = torch.cat([self._silence(1.0), self._tone(2.0), self._silence(1.0),
                         self._tone(1.5), self._silence(1.0)])
        spans = chunking.detect_speech_segments(wav, sr)
        assert len(spans) == 2
        # Spans are padded by pad_ms on each side, so allow ~100 ms of slack.
        assert spans[0][0] == pytest.approx(1.0, abs=0.12)
        assert spans[0][1] == pytest.approx(3.0, abs=0.12)
        assert spans[1][0] == pytest.approx(4.0, abs=0.12)
        assert spans[1][1] == pytest.approx(5.5, abs=0.12)

    def test_short_gap_inside_speech_is_not_a_pause(self):
        # A 60 ms dip (a stop closure) must not split a word in two.
        sr = 16000
        wav = torch.cat([self._silence(0.5), self._tone(1.0), self._silence(0.06),
                         self._tone(1.0), self._silence(0.5)])
        spans = chunking.detect_speech_segments(wav, sr)
        assert len(spans) == 1

    def test_stereo_is_averaged(self):
        sr = 16000
        mono = torch.cat([self._silence(0.5), self._tone(1.0), self._silence(0.5)])
        stereo = torch.stack([mono, mono], dim=0)
        assert chunking.detect_speech_segments(stereo, sr) == \
               chunking.detect_speech_segments(mono, sr)

    def test_speech_seconds_within_window(self):
        spans = [(0.0, 2.0), (5.0, 8.0)]
        assert chunking.speech_seconds(spans, 0.0, 10.0) == pytest.approx(5.0)
        assert chunking.speech_seconds(spans, 1.0, 6.0) == pytest.approx(2.0)
        assert chunking.speech_seconds(spans, 2.0, 5.0) == pytest.approx(0.0)
        assert chunking.speech_seconds(spans, 5.0, 5.0) == pytest.approx(0.0)

    def test_no_vad_result_means_assume_all_speech(self):
        # Budgeting too many phonemes is recoverable (the cram detector cuts the window
        # short); budgeting too few silently stretches the last kept phoneme.
        assert chunking.speech_seconds([], 0.0, 10.0) == pytest.approx(10.0)

    def test_silence_gaps_is_the_complement(self):
        spans = [(0.0, 2.0), (5.0, 8.0)]
        assert chunking.silence_gaps(spans, 10.0) == [(2.0, 5.0), (8.0, 10.0)]

    def test_snap_cut_moves_into_a_nearby_pause(self):
        spans = [(0.0, 2.0), (5.0, 8.0)]
        # A cut at 2.1 s sits just inside the pause that starts at 2.0 s.
        assert chunking.snap_cut_to_silence(2.1, spans, 10.0, 0.25) == pytest.approx(2.0)
        # Too far away to snap.
        assert chunking.snap_cut_to_silence(3.5, spans, 10.0, 0.25) == pytest.approx(3.5)
        # Disabled.
        assert chunking.snap_cut_to_silence(2.1, spans, 10.0, 0.0) == pytest.approx(2.1)
