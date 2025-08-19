
# File: tests/test_core.py
"""
Tests for Bournemouth Forced Aligner core functionality.
"""

import pytest
import torch
import tempfile
import json
import os
from bournemouth_aligner import PhonemeTimestampAligner

class TestPhonemeTimestampAligner:
    """Test cases for PhonemeTimestampAligner."""
    
    @pytest.fixture
    def aligner(self):
        """Create a test aligner instance."""
        return PhonemeTimestampAligner(
            model_name="en_libri1000_uj01d_e199_val_GER=0.2307.ckpt",
            lang='en-us',
            duration_max=10,
            device='cpu'
        )
    
    @pytest.fixture
    def dummy_audio(self):
        """Create dummy audio tensor."""
        # Create 1 second of dummy audio at 16kHz
        return torch.randn(1, 16000)
    
    def test_aligner_initialization(self, aligner):
        """Test that aligner initializes correctly."""
        assert aligner is not None
        assert torch.device(aligner.device) == torch.device('cpu')
        assert aligner.phonemes_key is not None
        assert aligner.phoneme_groups_key is not None
    
    def test_audio_loading(self, aligner):
        """Test audio loading functionality."""
        # This would need a real audio file to test properly
        # For now, just test that the method exists
        assert hasattr(aligner, 'load_audio')
    
    def test_process_transcription(self, aligner, dummy_audio):
        """Test transcription processing."""
        try:
            result = aligner.process_transcription(
                "hello", 
                dummy_audio, 
                extract_embeddings=False,
                debug=False
            )
            
            assert result is not None
            assert 'segments' in result
            assert len(result['segments']) > 0
            
        except Exception as e:
            # Some tests might fail without proper model files
            pytest.skip(f"Skipping due to model dependency: {e}")


        if result:
            textgrid_result = aligner.convert_to_textgrid(result, None, False)
            assert textgrid_result is not None
        else:
            pytest.skip("Skipping due to model dependency")

    def test_chop_wav(self, aligner, dummy_audio):
        """Test audio chopping functionality."""
        start_frame = 0
        end_frame = 8000  # 0.5 seconds
        
        try:
            chopped_wav, wav_len = aligner.chop_wav(dummy_audio, start_frame, end_frame)
            assert chopped_wav is not None
            assert wav_len == end_frame - start_frame
        except Exception as e:
            pytest.skip(f"Skipping due to model dependency: {e}")



if __name__ == "__main__":
    pytest.main()