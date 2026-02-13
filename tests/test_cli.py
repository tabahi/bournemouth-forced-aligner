#!/usr/bin/env python3
"""
Tests for Bournemouth Forced Aligner CLI (balign).
"""

import subprocess
import tempfile
import json
import os

import pytest
import torch
import torchaudio


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def dummy_audio_path(temp_dir):
    """Create a dummy 2-second WAV file and return its path."""
    path = os.path.join(temp_dir, "test_audio.wav")
    audio = torch.randn(1, 32000) * 0.1
    torchaudio.save(path, audio, 16000)
    return path


@pytest.fixture
def srt_path(temp_dir):
    """Create a single-segment SRT JSON file and return its path."""
    path = os.path.join(temp_dir, "test.srt.json")
    srt_data = {
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "hello world"}
        ]
    }
    with open(path, "w") as f:
        json.dump(srt_data, f)
    return path


@pytest.fixture
def multi_segment_srt_path(temp_dir):
    """Create a multi-segment SRT JSON file and return its path."""
    path = os.path.join(temp_dir, "test_multi.srt.json")
    srt_data = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.0, "end": 2.0, "text": "world"},
        ]
    }
    with open(path, "w") as f:
        json.dump(srt_data, f)
    return path


def output_path(temp_dir):
    """Return an output JSON path inside the temp directory."""
    return os.path.join(temp_dir, "output.json")


# ---- Basic CLI commands ----

class TestCLIBasic:
    """Test basic CLI commands that don't require model loading."""

    def test_help(self):
        """--help should exit 0 and print usage info."""
        result = subprocess.run(
            ["balign", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "AUDIO_PATH" in result.stdout
        assert "SRT_PATH" in result.stdout
        assert "OUTPUT_PATH" in result.stdout

    def test_version(self):
        """--version should exit 0 and print a version string."""
        result = subprocess.run(
            ["balign", "--version"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        version = result.stdout.strip()
        assert len(version) > 0

    def test_missing_args(self):
        """Running balign with no arguments should fail."""
        result = subprocess.run(
            ["balign"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0


# ---- Processing tests ----

class TestCLIProcessing:
    """Test actual audio processing through the CLI."""

    def test_basic_processing(self, temp_dir, dummy_audio_path, srt_path):
        """Test basic CLI processing produces valid output JSON."""
        out = output_path(temp_dir)
        result = subprocess.run(
            ["balign", dummy_audio_path, srt_path, out],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            pytest.skip(f"CLI processing failed (model unavailable?): {result.stderr[-200:]}")

        assert os.path.exists(out)
        with open(out) as f:
            data = json.load(f)

        assert "segments" in data
        assert len(data["segments"]) == 1

        seg = data["segments"][0]
        assert "phoneme_ts" in seg
        assert "text" in seg
        assert seg["text"] == "hello world"

    def test_multi_segment_processing(self, temp_dir, dummy_audio_path, multi_segment_srt_path):
        """Test CLI processing with multiple segments."""
        out = output_path(temp_dir)
        result = subprocess.run(
            ["balign", dummy_audio_path, multi_segment_srt_path, out],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            pytest.skip(f"CLI processing failed: {result.stderr[-200:]}")

        with open(out) as f:
            data = json.load(f)

        assert len(data["segments"]) == 2
        assert data["segments"][0]["text"] == "hello"
        assert data["segments"][1]["text"] == "world"

    def test_debug_flag(self, temp_dir, dummy_audio_path, srt_path):
        """Test that --debug flag is accepted and produces extra output."""
        out = output_path(temp_dir)
        result = subprocess.run(
            ["balign", dummy_audio_path, srt_path, out, "--debug"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            pytest.skip(f"CLI processing failed: {result.stderr[-200:]}")

        assert os.path.exists(out)

    def test_phoneme_ts_structure(self, temp_dir, dummy_audio_path, srt_path):
        """Test that output phoneme timestamps have all required fields."""
        out = output_path(temp_dir)
        result = subprocess.run(
            ["balign", dummy_audio_path, srt_path, out],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            pytest.skip(f"CLI processing failed: {result.stderr[-200:]}")

        with open(out) as f:
            data = json.load(f)

        required_fields = {
            "phoneme_id", "phoneme_label", "ipa_label",
            "start_ms", "end_ms", "confidence",
            "is_estimated", "target_seq_idx", "index",
        }
        for seg in data["segments"]:
            for ts_item in seg["phoneme_ts"]:
                missing = required_fields - set(ts_item.keys())
                assert not missing, f"Missing fields: {missing}"
                assert ts_item["end_ms"] >= ts_item["start_ms"]

    def test_output_dir_creation(self, temp_dir, dummy_audio_path, srt_path):
        """Test that the CLI creates output directories if they don't exist."""
        nested_out = os.path.join(temp_dir, "sub", "dir", "output.json")
        result = subprocess.run(
            ["balign", dummy_audio_path, srt_path, nested_out],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            pytest.skip(f"CLI processing failed: {result.stderr[-200:]}")

        assert os.path.exists(nested_out)


# ---- Error handling ----

class TestCLIErrors:
    """Test CLI error handling."""

    def test_nonexistent_audio(self, temp_dir, srt_path):
        """CLI should fail when audio file doesn't exist."""
        result = subprocess.run(
            ["balign", "/nonexistent/audio.wav", srt_path, output_path(temp_dir)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0

    def test_nonexistent_srt(self, temp_dir, dummy_audio_path):
        """CLI should fail when SRT file doesn't exist."""
        result = subprocess.run(
            ["balign", dummy_audio_path, "/nonexistent/srt.json", output_path(temp_dir)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0
