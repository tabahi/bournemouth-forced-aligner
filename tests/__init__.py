# tests/__init__.py
"""
Test suite for Bournemouth Forced Aligner.

This package contains comprehensive tests for all components of the
Bournemouth Forced Aligner library.
"""

import os
import sys
import tempfile
import torch
import torchaudio
import json
from pathlib import Path

# Test configuration
TEST_CONFIG = {
    "sample_rate": 16000,
    "test_duration": 2.0,
    "device": "cpu",
    "model_name": "en_libri1000_uj01d_e199_val_GER=0.2307.ckpt",
    "lang": "en-us",
}

def create_dummy_audio(duration=2.0, sample_rate=16000):
    """Create dummy audio for testing."""
    samples = int(duration * sample_rate)
    return torch.randn(1, samples) * 0.1

def create_test_srt_data(text="hello world", start=0.0, end=2.0):
    """Create test SRT data."""
    return {
        "segments": [
            {
                "start": start,
                "end": end, 
                "text": text
            }
        ]
    }

def create_temp_audio_file(duration=2.0, sample_rate=16000):
    """Create temporary audio file."""
    audio = create_dummy_audio(duration, sample_rate)
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    torchaudio.save(temp_file.name, audio, sample_rate)
    return temp_file.name

def create_temp_srt_file(text="hello world", start=0.0, end=2.0):
    """Create temporary SRT file."""
    srt_data = create_test_srt_data(text, start, end)
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=".srt", delete=False)
    json.dump(srt_data, temp_file, indent=2)
    temp_file.close()
    return temp_file.name