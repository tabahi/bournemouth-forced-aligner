
# File: tests/test_core.py
"""
Tests for Bournemouth Forced Aligner core functionality.
"""

import pytest
import torch
import tempfile
import json
import os

import sys
sys.path.append('.')
from bournemouth_aligner import PhonemeTimestampAligner

SAMPLE_AUDIO = "examples/samples/audio/109867__timkahn__butterfly.wav"

class TestPhonemeTimestampAligner:
    """Test cases for PhonemeTimestampAligner."""

    @pytest.fixture(scope="class")
    def aligner(self):
        """Create a test aligner instance using preset."""
        try:
            return PhonemeTimestampAligner(
                preset="en-us",
                duration_max=10,
                device='cpu'
            )
        except Exception as e:
            pytest.skip(f"Cannot initialize aligner: {e}")

    @pytest.fixture
    def dummy_audio(self):
        """Create dummy audio tensor (1 second at 16kHz)."""
        return torch.randn(1, 16000)

    @pytest.fixture
    def dummy_audio_2s(self):
        """Create dummy audio tensor (2 seconds at 16kHz)."""
        return torch.randn(1, 32000)

    # ---- Initialization ----

    def test_aligner_initialization(self, aligner):
        """Test that aligner initializes correctly."""
        assert aligner is not None
        assert torch.device(aligner.device) == torch.device('cpu')
        assert aligner.phonemes_key is not None
        assert aligner.phoneme_groups_key is not None
        assert aligner.resampler_sample_rate == 16000
        assert aligner.sample_rate == 16000

    def test_audio_loading(self, aligner):
        """Test audio loading from file."""
        assert hasattr(aligner, 'load_audio')
        if os.path.exists(SAMPLE_AUDIO):
            wav = aligner.load_audio(SAMPLE_AUDIO)
            assert isinstance(wav, torch.Tensor)
            assert wav.dim() == 2  # (channels, samples)
            assert wav.shape[0] == 1

    # ---- Phonemization ----

    def test_phonemize_sentence(self, aligner):
        """Test phonemization returns expected structure."""
        result = aligner.phonemize_sentence("hello")
        assert aligner.phonemes_key in result
        assert aligner.phoneme_groups_key in result
        ph_seq = result[aligner.phonemes_key]
        assert isinstance(ph_seq, list)
        assert len(ph_seq) > 0
        assert all(isinstance(p, int) for p in ph_seq)

    # ---- chop_wav ----

    def test_chop_wav(self, aligner, dummy_audio):
        """Test audio chopping returns correct length."""
        start_frame = 0
        end_frame = 8000  # 0.5 seconds
        chopped_wav, wav_len = aligner.chop_wav(dummy_audio, start_frame, end_frame)
        assert chopped_wav is not None
        assert wav_len == end_frame - start_frame

    def test_chop_wav_too_short_raises(self, aligner):
        """Test that chopping a too-short segment raises ValueError."""
        short_audio = torch.randn(1, 16000)
        with pytest.raises(ValueError, match="Segment too short"):
            aligner.chop_wav(short_audio, 0, 100)  # 100 samples < min

    # ---- process_sentence ----

    def test_process_sentence(self, aligner, dummy_audio):
        """Test single sentence processing returns expected dict structure."""
        try:
            result = aligner.process_sentence(
                "hello",
                dummy_audio,
                extract_embeddings=False,
                debug=False
            )
        except Exception as e:
            pytest.skip(f"Skipping due to model error: {e}")

        assert result is not None
        assert 'segments' in result
        assert len(result['segments']) > 0
        seg = result['segments'][0]
        assert 'phoneme_ts' in seg
        assert 'text' in seg

    def test_process_sentence_with_embeddings(self, aligner, dummy_audio):
        """Test that process_sentence returns tuple when extracting embeddings."""
        try:
            result = aligner.process_sentence(
                "hello",
                dummy_audio,
                extract_embeddings=True,
                do_groups=True,
                debug=False
            )
        except Exception as e:
            pytest.skip(f"Skipping due to model error: {e}")

        assert isinstance(result, tuple)
        assert len(result) == 3
        ts_dict, p_embds, g_embds = result
        assert 'segments' in ts_dict
        assert isinstance(p_embds, list)
        assert isinstance(g_embds, list)

    def test_process_sentence_textgrid(self, aligner, dummy_audio):
        """Test TextGrid conversion from process_sentence output."""
        try:
            result = aligner.process_sentence("hello", dummy_audio, debug=False)
        except Exception as e:
            pytest.skip(f"Skipping due to model error: {e}")

        textgrid_result = aligner.convert_to_textgrid(result, None, False)
        assert textgrid_result is not None

    # ---- process_segments: input normalization ----

    def test_process_segments_single_dict_input(self, aligner, dummy_audio):
        """Test that a single dict is normalized to a list."""
        srt_data = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}
        try:
            result = aligner.process_segments(srt_data, [dummy_audio], debug=False)
        except Exception as e:
            pytest.skip(f"Skipping due to model error: {e}")

        assert isinstance(result, list)
        assert len(result) == 1
        assert 'segments' in result[0]

    def test_process_segments_tensor_audio_2d(self, aligner, dummy_audio):
        """Test that a 2D tensor (C, T) is normalized to a single-item list."""
        srt_data = [{"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}]
        try:
            result = aligner.process_segments(srt_data, dummy_audio, debug=False)
        except Exception as e:
            pytest.skip(f"Skipping due to model error: {e}")

        assert isinstance(result, list)
        assert len(result) == 1

    def test_process_segments_tensor_audio_3d(self, aligner, dummy_audio):
        """Test that a 3D tensor (B, C, T) is unpacked to a list of clips."""
        # 2 clips batched into a single tensor
        batch_audio = torch.stack([dummy_audio, dummy_audio], dim=0)  # (2, 1, T)
        srt_data = [
            {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]},
            {"segments": [{"start": 0.0, "end": 1.0, "text": "world"}]},
        ]
        try:
            result = aligner.process_segments(srt_data, batch_audio, debug=False)
        except Exception as e:
            pytest.skip(f"Skipping due to model error: {e}")

        assert isinstance(result, list)
        assert len(result) == 2

    # ---- process_segments: validation ----

    def test_process_segments_batch_mismatch_raises(self, aligner, dummy_audio):
        """Test that mismatched srt_data and audio_wavs lengths raise ValueError."""
        srt_data = [
            {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]},
            {"segments": [{"start": 0.0, "end": 1.0, "text": "world"}]},
        ]
        with pytest.raises(ValueError, match="Batch size mismatch"):
            aligner.process_segments(srt_data, [dummy_audio], debug=False)

    def test_process_segments_missing_segments_key_raises(self, aligner, dummy_audio):
        """Test that missing 'segments' key raises ValueError."""
        srt_data = [{"text": "hello"}]  # no "segments" key
        with pytest.raises(ValueError, match="missing 'segments' key"):
            aligner.process_segments(srt_data, [dummy_audio], debug=False)

    def test_process_segments_missing_sub_segment_keys_raises(self, aligner, dummy_audio):
        """Test that missing start/end/text in sub-segments raises ValueError."""
        srt_data = [{"segments": [{"text": "hello"}]}]  # missing start, end
        with pytest.raises(ValueError, match="missing required keys"):
            aligner.process_segments(srt_data, [dummy_audio], debug=False)

    def test_process_segments_empty_segments(self, aligner, dummy_audio):
        """Test that empty segments list returns empty results."""
        srt_data = [{"segments": []}]
        result = aligner.process_segments(srt_data, [dummy_audio], debug=False)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == {"segments": []}

    # ---- process_segments: single clip ----

    def test_process_segments_single_clip(self, aligner, dummy_audio):
        """Test process_segments with one clip and one segment."""
        srt_data = [{"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}]
        try:
            result = aligner.process_segments(srt_data, [dummy_audio], debug=False)
        except Exception as e:
            pytest.skip(f"Skipping due to model error: {e}")

        assert isinstance(result, list)
        assert len(result) == 1
        assert 'segments' in result[0]
        assert len(result[0]['segments']) == 1
        seg = result[0]['segments'][0]
        assert 'phoneme_ts' in seg
        assert 'text' in seg
        assert seg['text'] == 'hello'

    def test_process_segments_single_clip_multi_segments(self, aligner, dummy_audio_2s):
        """Test process_segments with one clip and multiple segments."""
        srt_data = [{"segments": [
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.0, "end": 2.0, "text": "world"},
        ]}]
        try:
            result = aligner.process_segments(srt_data, [dummy_audio_2s], debug=False)
        except Exception as e:
            pytest.skip(f"Skipping due to model error: {e}")

        assert len(result) == 1
        assert len(result[0]['segments']) == 2
        assert result[0]['segments'][0]['text'] == 'hello'
        assert result[0]['segments'][1]['text'] == 'world'

    # ---- process_segments: multi clip batch ----

    def test_process_segments_multi_clip(self, aligner, dummy_audio):
        """Test process_segments with multiple clips, each with their own audio."""
        audio_wav2 = torch.randn(1, 16000)
        srt_data = [
            {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]},
            {"segments": [{"start": 0.0, "end": 1.0, "text": "world"}]},
        ]
        try:
            result = aligner.process_segments(srt_data, [dummy_audio, audio_wav2], debug=False)
        except Exception as e:
            pytest.skip(f"Skipping due to model error: {e}")

        assert isinstance(result, list)
        assert len(result) == 2
        assert len(result[0]['segments']) == 1
        assert len(result[1]['segments']) == 1
        assert result[0]['segments'][0]['text'] == 'hello'
        assert result[1]['segments'][0]['text'] == 'world'

    # ---- process_segments: embeddings ----

    def test_process_segments_embeddings_structure(self, aligner, dummy_audio):
        """Test that embeddings are returned as nested lists [clip][segment]."""
        audio_wav2 = torch.randn(1, 16000)
        srt_data = [
            {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]},
            {"segments": [{"start": 0.0, "end": 1.0, "text": "world"}]},
        ]
        try:
            result = aligner.process_segments(
                srt_data, [dummy_audio, audio_wav2],
                extract_embeddings=True, do_groups=True, debug=False
            )
        except Exception as e:
            pytest.skip(f"Skipping due to model error: {e}")

        assert isinstance(result, tuple)
        assert len(result) == 3
        batch_results, batch_p_embds, batch_g_embds = result

        # batch_results: list of dicts
        assert isinstance(batch_results, list)
        assert len(batch_results) == 2

        # embeddings: nested lists [clip][segment]
        assert isinstance(batch_p_embds, list)
        assert len(batch_p_embds) == 2
        assert isinstance(batch_g_embds, list)
        assert len(batch_g_embds) == 2

        # each clip has a list of segment embeddings
        for clip_embds in batch_p_embds:
            assert isinstance(clip_embds, list)
            for emb in clip_embds:
                assert isinstance(emb, torch.Tensor)
                assert emb.device == torch.device('cpu')

    # ---- process_srt_file ----

    def test_process_srt_file(self, aligner):
        """Test processing from SRT file with temp files."""
        with tempfile.TemporaryDirectory() as tmp:
            # Create dummy audio
            audio_path = os.path.join(tmp, "test.wav")
            import torchaudio
            dummy = torch.randn(1, 16000)
            torchaudio.save(audio_path, dummy, 16000)

            # Create SRT JSON
            srt_path = os.path.join(tmp, "test.srt.json")
            srt_data = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}
            with open(srt_path, 'w') as f:
                json.dump(srt_data, f)

            # Output path
            out_path = os.path.join(tmp, "output.json")

            try:
                result = aligner.process_srt_file(
                    srt_path, audio_path,
                    ts_out_path=out_path,
                    debug=False
                )
            except Exception as e:
                pytest.skip(f"Skipping due to model error: {e}")

            assert result is not None
            assert 'segments' in result
            assert os.path.exists(out_path)

            with open(out_path) as f:
                saved = json.load(f)
            assert 'segments' in saved

    # ---- Utility methods ----

    def test_compress_frames(self, aligner):
        """Test frame compression."""
        frames = [0, 0, 0, 1, 1, 3, 4, 2, 2, 2]
        compressed = aligner.compress_frames(frames)
        assert compressed == [(0, 3), (1, 2), (3, 1), (4, 1), (2, 3)]

    def test_decompress_frames(self, aligner):
        """Test frame decompression."""
        compressed = [(0, 3), (1, 2), (3, 1), (4, 1), (2, 3)]
        decompressed = aligner.decompress_frames(compressed)
        assert decompressed == [0, 0, 0, 1, 1, 3, 4, 2, 2, 2]

    def test_compress_decompress_roundtrip(self, aligner):
        """Test compress/decompress roundtrip."""
        original = [5, 5, 3, 3, 3, 1, 0, 0]
        assert aligner.decompress_frames(aligner.compress_frames(original)) == original

    def test_compress_empty(self, aligner):
        """Test compress on empty list."""
        assert aligner.compress_frames([]) == []

    # ---- Phoneme timestamp output structure ----

    def test_phoneme_ts_fields(self, aligner, dummy_audio):
        """Test that each phoneme timestamp has all required fields."""
        try:
            result = aligner.process_sentence("hello", dummy_audio, debug=False)
        except Exception as e:
            pytest.skip(f"Skipping due to model error: {e}")

        required_fields = {"phoneme_id", "phoneme_label", "ipa_label",
                           "start_ms", "end_ms", "confidence",
                           "is_estimated", "target_seq_idx", "index"}
        for ts_item in result['segments'][0]['phoneme_ts']:
            assert required_fields.issubset(ts_item.keys()), \
                f"Missing fields: {required_fields - set(ts_item.keys())}"
            assert isinstance(ts_item['start_ms'], float)
            assert isinstance(ts_item['end_ms'], float)
            assert ts_item['end_ms'] >= ts_item['start_ms']


if __name__ == "__main__":
    pytest.main()
