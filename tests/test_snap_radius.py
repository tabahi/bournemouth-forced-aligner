# File: tests/test_snap_radius.py
"""
Tests for onset-snap configuration: the radius (now expressed in FRAMES) and what
`boundary_snap_source='auto'` resolves to.

These construct the aligner with `preset=None`, so no checkpoint is downloaded. The frame
stride then falls back to 210 samples (13.125 ms/frame), which is what every released
checkpoint uses, and `auto` resolves via the model2i branch since no model is loaded.
"""

import pytest

import sys
sys.path.append('.')
from bournemouth_aligner.core import (PhonemeTimestampAligner,
                                      DEFAULT_SNAP_RADIUS_FRAMES,
                                      DEFAULT_SNAP_RADIUS_FRAMES_FALLBACK,
                                      DEFAULT_SNAP_RADIUS_MS,
                                      DEFAULT_SNAP_RADIUS_MS_FALLBACK,
                                      AUTO_SNAP_MODEL2I, AUTO_SNAP_MODEL3A)

FRAME_MS = 13.125   # 210 samples at 16 kHz


def make(**kw):
    kw.setdefault('warn_level', 0)
    return PhonemeTimestampAligner(preset=None, **kw)


class TestRadiusDefaults:

    def test_default_is_the_frame_fallback(self):
        a = make()
        assert a.boundary_snap_radius == DEFAULT_SNAP_RADIUS_FRAMES_FALLBACK == 4
        assert a.boundary_snap_radius_ms == pytest.approx(4 * FRAME_MS)

    def test_per_source_default(self):
        a = make(boundary_snap_source='boundary_head')
        # No extractor is loaded, so the source is taken at face value here.
        assert a.boundary_snap_radius == DEFAULT_SNAP_RADIUS_FRAMES['boundary_head'] == 2

    def test_ms_aliases_are_derived_from_the_frame_defaults(self):
        assert DEFAULT_SNAP_RADIUS_MS_FALLBACK == pytest.approx(
            DEFAULT_SNAP_RADIUS_FRAMES_FALLBACK * FRAME_MS)
        assert DEFAULT_SNAP_RADIUS_MS['boundary_head'] == pytest.approx(
            DEFAULT_SNAP_RADIUS_FRAMES['boundary_head'] * FRAME_MS)


class TestRadiusPrecedence:

    def test_frames_are_taken_as_given(self):
        assert make(boundary_snap_radius=6).boundary_snap_radius == 6

    def test_frames_win_over_ms(self):
        a = make(boundary_snap_radius=2, boundary_snap_radius_ms=52.5)
        assert a.boundary_snap_radius == 2

    def test_ms_is_rounded_to_frames(self):
        assert make(boundary_snap_radius_ms=52.5).boundary_snap_radius == 4
        assert make(boundary_snap_radius_ms=26.25).boundary_snap_radius == 2

    def test_nearby_ms_values_are_the_same_search(self):
        # The reason frames are the unit of record: these name one 4-frame search.
        assert (make(boundary_snap_radius_ms=50.0).boundary_snap_radius ==
                make(boundary_snap_radius_ms=55.0).boundary_snap_radius == 4)

    def test_reported_ms_describes_the_resolved_frames_not_the_request(self):
        a = make(boundary_snap_radius_ms=55.0)
        assert a.boundary_snap_radius_ms == pytest.approx(4 * FRAME_MS)   # not 55.0

    def test_a_positive_ms_request_never_collapses_to_disabled(self):
        # 1 ms rounds to 0 frames, which would mean "no snapping" - clamp to 1 instead.
        assert make(boundary_snap_radius_ms=1.0).boundary_snap_radius == 1

    def test_zero_disables(self):
        assert make(boundary_snap_radius=0).boundary_snap_radius == 0
        assert make(boundary_snap_radius_ms=0).boundary_snap_radius == 0

    def test_source_none_overrides_an_explicit_radius(self):
        a = make(boundary_snap_source='none', boundary_snap_radius=4)
        assert a.boundary_snap_radius == 0


class TestSetBoundarySnap:

    def test_frames_override_a_live_ms_value(self):
        a = make(boundary_snap_radius_ms=52.5)
        assert a.set_boundary_snap(radius_frames=6, verbose=False) == 6

    def test_clear_restores_the_per_source_default(self):
        a = make(boundary_snap_radius=6)
        assert a.set_boundary_snap(radius_frames='clear', verbose=False) == \
            DEFAULT_SNAP_RADIUS_FRAMES_FALLBACK

    def test_ms_applies_once_frames_are_cleared(self):
        a = make(boundary_snap_radius=6)
        assert a.set_boundary_snap(radius_frames='clear', radius_ms=26.25, verbose=False) == 2

    def test_none_holds_the_radius_at_zero(self):
        a = make()
        a.set_boundary_snap(source='none', verbose=False)
        assert a.set_boundary_snap(radius_frames=4, verbose=False) == 0

    def test_round_trip_back_to_defaults(self):
        a = make()
        a.set_boundary_snap(source='none', radius_frames=2, verbose=False)
        a.set_boundary_snap(source='auto', radius_frames='clear', radius_ms='default',
                            verbose=False)
        assert a.boundary_snap_source == AUTO_SNAP_MODEL2I
        assert a.boundary_snap_radius == DEFAULT_SNAP_RADIUS_FRAMES_FALLBACK

    def test_rejects_an_unknown_source(self):
        with pytest.raises(ValueError):
            make().set_boundary_snap(source='not_a_source', verbose=False)


class TestAutoSource:

    def test_auto_branches_on_architecture(self):
        # conclusions.md §10: the two families do not rank the sources the same way.
        assert AUTO_SNAP_MODEL3A == 'spectral_stream'
        assert AUTO_SNAP_MODEL2I == 'pre_transformer'
        assert AUTO_SNAP_MODEL3A != AUTO_SNAP_MODEL2I

    def test_auto_never_resolves_to_a_boundary_head(self):
        # A trained head measures below its own control, so `auto` must not select it.
        assert 'boundary_head' not in (AUTO_SNAP_MODEL2I, AUTO_SNAP_MODEL3A)

    def test_auto_resolves_to_model2i_without_a_loaded_model(self):
        a = make()
        assert a.boundary_snap_source_requested == 'auto'
        assert a.boundary_snap_source == AUTO_SNAP_MODEL2I

    def test_explicit_source_is_never_rewritten(self):
        assert make(boundary_snap_source='embeddings').boundary_snap_source == 'embeddings'
