# File: tests/test_segmental.py
"""
Tests for the segmental decoder (`bournemouth_aligner.segmental`) on synthetic posteriors,
so they run without a model. Accuracy on real speech is measured by `tests/test_buckeye.py`.
"""

import numpy as np
import pytest

import sys
sys.path.append('.')
from bournemouth_aligner.segmental import decode_segmental, boundary_bonus

BLANK, SIL = 66, 0
C = 67


def posteriors(spans, T, floor=1e-4):
    """spans: [(class_id, start, end)]. Frames outside every span are SIL, which is how CUPE
    marks a pause - its blank fires INSIDE speech, between CTC spikes, so the decoder
    deliberately does not read blank as silence (see `_emissions`)."""
    P = np.full((T, C), floor)
    P[:, SIL] = 1.0
    for c, a, b in spans:
        P[a:b, SIL] = floor
        P[a:b, c] = 1.0
    P /= P.sum(1, keepdims=True)
    return np.log(P)


def onsets(stamps):
    return [s[1] for s in stamps]


class TestForced:
    def test_places_each_phone_on_its_span(self):
        spans = [(10, 5, 15), (30, 15, 22), (50, 22, 40)]
        stamps = decode_segmental(posteriors(spans, 50), [10, 30, 50], BLANK, SIL)
        assert [s[0] for s in stamps] == [10, 30, 50]
        assert [s[3] for s in stamps] == [0, 1, 2]
        assert onsets(stamps) == [5, 15, 22]

    def test_every_target_aligned_even_without_evidence(self):
        # 30 has no posterior support at all; forced alignment still gives it frames.
        spans = [(10, 5, 15), (50, 22, 40)]
        stamps = decode_segmental(posteriors(spans, 50), [10, 30, 50], BLANK, SIL)
        assert [s[3] for s in stamps] == [0, 1, 2]
        assert all(e - s >= 1 for _, s, e, _ in stamps)

    def test_minimum_duration(self):
        # A 1-frame spike still gets the 2-frame minimum.
        spans = [(10, 5, 15), (30, 15, 16), (50, 16, 30)]
        stamps = decode_segmental(posteriors(spans, 40), [10, 30, 50], BLANK, SIL)
        assert stamps[1][2] - stamps[1][1] >= 2

    def test_phones_exactly_filling_the_audio(self):
        # 23 phones x 2 frames = 46 frames: the path has no slack for skipping soft states or
        # pauses one frame at a time. Regression for the "no complete path" failure.
        ids = [10 + (i % 40) for i in range(23)]
        words = [i // 3 for i in range(23)]
        spans = [(p, 2 * i, 2 * i + 2) for i, p in enumerate(ids)]
        stamps = decode_segmental(posteriors(spans, 46), ids, BLANK, SIL, word_idx=words)
        assert len(stamps) == 23
        assert onsets(stamps) == [2 * i for i in range(23)]

    def test_min_duration_relaxes_for_fast_speech(self):
        ids = [10 + i for i in range(30)]
        spans = [(p, i, i + 1) for i, p in enumerate(ids)]
        stamps = decode_segmental(posteriors(spans, 30), ids, BLANK, SIL)
        assert len(stamps) == 30

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            decode_segmental(posteriors([], 5), [10, 11, 12, 13, 14, 15], BLANK, SIL)

    def test_target_silence_is_optional(self):
        # Punctuation SIL where the speaker did not pause: it must not steal frames.
        spans = [(10, 5, 15), (30, 15, 30)]
        stamps = decode_segmental(posteriors(spans, 35), [10, SIL, 30], BLANK, SIL,
                                  word_idx=[0, 1, 2])
        speech = [s for s in stamps if s[0] != SIL]
        assert onsets(speech) == [5, 15]

    def test_pause_between_words(self):
        spans = [(10, 5, 15), (30, 25, 35)]   # 10 frames of silence between the words
        lp = posteriors(spans, 40)
        stamps = decode_segmental(lp, [10, 30], BLANK, SIL, word_idx=[0, 1])
        assert onsets(stamps) == [5, 25]
        assert stamps[0][2] <= 16             # the first word does not swallow the pause

    def test_boundary_bonus_moves_ambiguous_onset(self):
        # Two phones with identical, flat evidence over [5, 25): the posteriors cannot place
        # the boundary, the change signal can.
        T = 30
        P = np.full((T, C), 1e-4)
        P[:, SIL] = 1.0
        P[5:25, SIL] = 1e-4
        P[5:25, 10] = 0.5
        P[5:25, 30] = 0.5
        lp = np.log(P / P.sum(1, keepdims=True))
        change = np.zeros(T - 1)
        change[17] = 5.0                       # frames 17 -> 18 differ
        stamps = decode_segmental(lp, [10, 30], BLANK, SIL, change=change)
        assert stamps[1][1] == 18


class TestNotForced:
    def test_skips_word_not_in_audio(self):
        # Text: w0 w1 w2, audio: only w0 and w2.
        ids = [10, 11, 20, 21, 30, 31]
        words = [0, 0, 1, 1, 2, 2]
        spans = [(10, 2, 10), (11, 10, 18), (30, 25, 33), (31, 33, 41)]
        lp = posteriors(spans, 45)
        forced = decode_segmental(lp, ids, BLANK, SIL, word_idx=words, enforce=True)
        free = decode_segmental(lp, ids, BLANK, SIL, word_idx=words, enforce=False)
        assert sorted(s[3] for s in forced) == [0, 1, 2, 3, 4, 5]
        assert sorted(s[3] for s in free) == [0, 1, 4, 5]
        assert onsets(free) == [2, 10, 25, 33]

    def test_skips_words_at_segment_edges(self):
        # Clipped audio: the first and last words of the text are not in the segment.
        ids = [40, 41, 10, 11, 50, 51]
        words = [0, 0, 1, 1, 2, 2]
        spans = [(10, 5, 15), (11, 15, 25)]
        free = decode_segmental(posteriors(spans, 30), ids, BLANK, SIL, word_idx=words,
                                enforce=False)
        assert [s[3] for s in free] == [2, 3]
        assert onsets(free) == [5, 15]

    def test_clean_text_unchanged(self):
        spans = [(10, 5, 15), (30, 15, 22), (50, 22, 40)]
        lp = posteriors(spans, 50)
        a = decode_segmental(lp, [10, 30, 50], BLANK, SIL, word_idx=[0, 1, 2], enforce=True)
        b = decode_segmental(lp, [10, 30, 50], BLANK, SIL, word_idx=[0, 1, 2], enforce=False)
        assert a == b


def test_boundary_bonus_indexing():
    change = np.zeros(9)
    change[3] = 1.0                            # frames 3 -> 4
    assert int(np.argmax(boundary_bonus(change, 10))) == 4
    score = np.zeros(10)
    score[4] = 1.0
    assert int(np.argmax(boundary_bonus(score, 10, per_frame=True))) == 4

