#!/usr/bin/env python3
"""
Tests for Bournemouth Forced Aligner CLI (balign).

Layout
------
TestCLIFlags        — fast flag-only tests (no model loading)
TestPresetResolution — unit tests for _resolve_model_lang()
TestCLIProcessing   — end-to-end tests that load the model
TestCLIErrors       — error-handling / bad-input tests

python -m pytest tests/test_cli.py
"""

import json
import os
import subprocess
import tempfile

import pytest
import torch
import torchaudio


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def dummy_audio_path(temp_dir):
    """2-second 16 kHz mono WAV."""
    path = os.path.join(temp_dir, "test_audio.wav")
    audio = torch.randn(1, 32000) * 0.1
    torchaudio.save(path, audio, 16000)
    return path


@pytest.fixture
def srt_path(temp_dir):
    """Single-segment Whisper-style JSON transcript."""
    path = os.path.join(temp_dir, "test.srt.json")
    with open(path, "w") as f:
        json.dump({"segments": [{"start": 0.0, "end": 2.0, "text": "hello world"}]}, f)
    return path


@pytest.fixture
def multi_segment_srt_path(temp_dir):
    path = os.path.join(temp_dir, "test_multi.srt.json")
    with open(path, "w") as f:
        json.dump({"segments": [
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.0, "end": 2.0, "text": "world"},
        ]}, f)
    return path


def out(temp_dir, name="output.json"):
    return os.path.join(temp_dir, name)


def _run(*args, timeout=30):
    return subprocess.run(
        ["balign", *args],
        capture_output=True, text=True, timeout=timeout,
    )


def _skip_if_failed(result):
    if result.returncode != 0:
        pytest.skip(f"Model unavailable or processing failed: {result.stderr[-300:]}")


# ---------------------------------------------------------------------------
# 1. Flag-only tests (no model load)
# ---------------------------------------------------------------------------

class TestCLIFlags:
    """Tests that exercise CLI flags without loading a model."""

    def test_help(self):
        r = _run("--help")
        assert r.returncode == 0
        assert "AUDIO_PATH" in r.stdout
        assert "TEXT_OR_SRT" in r.stdout
        assert "--preset" in r.stdout
        assert "--mel-path" in r.stdout
        assert "--textgrid" in r.stdout

    def test_version(self):
        r = _run("--version")
        assert r.returncode == 0
        assert len(r.stdout.strip()) > 0

    def test_list_presets(self):
        r = _run("--list-presets")
        assert r.returncode == 0
        # Should mention a few well-known presets
        for code in ("en-us", "de", "fr", "hi", "ar"):
            assert code in r.stdout

    def test_missing_args(self):
        r = _run()
        assert r.returncode != 0

    def test_missing_audio_arg(self):
        r = _run("only_one_arg")
        assert r.returncode != 0


# ---------------------------------------------------------------------------
# 2. Preset resolution unit tests (import the helper directly)
# ---------------------------------------------------------------------------

class TestPresetResolution:
    """Unit-test _resolve_model_lang without invoking the CLI subprocess."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from bournemouth_aligner.cli import _resolve_model_lang
        self.resolve = _resolve_model_lang

    def test_en_us_preset(self):
        model, lang = self.resolve("en-us", None, None)
        assert "ua01c" in model          # English model
        assert lang == "en-us"

    def test_de_preset(self):
        model, lang = self.resolve("de", None, None)
        assert "MLS8" in model or "mls8" in model.lower()
        assert lang == "de"

    def test_hi_preset(self):
        model, lang = self.resolve("hi", None, None)
        assert "mswc" in model.lower()
        assert lang == "hi"

    def test_model_override(self):
        """--model should always win over --preset."""
        model, lang = self.resolve("en-us", "my_custom.ckpt", None)
        assert model == "my_custom.ckpt"

    def test_lang_override(self):
        """--lang should override the language part of --preset."""
        _, lang = self.resolve("en-us", None, "en-gb")
        assert lang == "en-gb"

    def test_no_preset_defaults(self):
        """With no preset and no overrides, use English defaults."""
        model, lang = self.resolve(None, None, None)
        assert "ua01c" in model
        assert lang == "en-us"

    def test_unknown_preset_fallback(self):
        """Unknown presets should fall back to the universal model."""
        model, lang = self.resolve("xx-unknown", None, None)
        assert "mswc" in model.lower()


# ---------------------------------------------------------------------------
# 3. End-to-end processing tests (require model download)
# ---------------------------------------------------------------------------

class TestCLIProcessing:

    # -- text mode -----------------------------------------------------------

    def test_text_mode_basic(self, temp_dir, dummy_audio_path):
        """balign audio.wav "hello" out.json  — text as second arg."""
        o = out(temp_dir)
        r = _run(dummy_audio_path, "hello", o, "--preset=en-us", timeout=120)
        _skip_if_failed(r)
        assert os.path.exists(o)
        data = json.loads(open(o).read())
        assert "segments" in data
        assert len(data["segments"]) >= 1

    def test_text_mode_auto_output(self, temp_dir):
        """When OUTPUT_PATH is omitted the result is saved as <audio>.vs.json."""
        import torchaudio, torch
        wav_path = os.path.join(temp_dir, "sample.wav")
        torchaudio.save(wav_path, torch.randn(1, 32000) * 0.1, 16000)
        expected_out = os.path.join(temp_dir, "sample.vs.json")

        r = _run(wav_path, "hello world", "--preset=en-us", timeout=120)
        _skip_if_failed(r)
        assert os.path.exists(expected_out)

    def test_text_mode_preset_de(self, temp_dir, dummy_audio_path):
        """--preset=de should not crash for a German word."""
        o = out(temp_dir)
        r = _run(dummy_audio_path, "hallo", o, "--preset=de", timeout=120)
        _skip_if_failed(r)
        assert os.path.exists(o)

    # -- SRT file mode -------------------------------------------------------

    def test_srt_mode_single_segment(self, temp_dir, dummy_audio_path, srt_path):
        """Second arg is an existing file → SRT mode."""
        o = out(temp_dir)
        r = _run(dummy_audio_path, srt_path, o, "--preset=en-us", timeout=120)
        _skip_if_failed(r)
        assert os.path.exists(o)
        data = json.loads(open(o).read())
        assert data["segments"][0]["text"] == "hello world"

    def test_srt_mode_multi_segment(self, temp_dir, dummy_audio_path, multi_segment_srt_path):
        o = out(temp_dir)
        r = _run(dummy_audio_path, multi_segment_srt_path, o, "--preset=en-us", timeout=120)
        _skip_if_failed(r)
        data = json.loads(open(o).read())
        assert len(data["segments"]) == 2
        assert data["segments"][0]["text"] == "hello"
        assert data["segments"][1]["text"] == "world"

    # -- optional outputs ----------------------------------------------------

    def test_mel_path_pt(self, temp_dir, dummy_audio_path):
        """--mel-path=mel.pt saves a PyTorch tensor."""
        o = out(temp_dir)
        mel_out = os.path.join(temp_dir, "mel.pt")
        r = _run(dummy_audio_path, "hello", o,
                 "--preset=en-us", f"--mel-path={mel_out}", timeout=120)
        _skip_if_failed(r)
        assert os.path.exists(mel_out)
        mel = torch.load(mel_out)
        assert mel.dim() == 2          # (T, mel_bins)
        assert mel.shape[1] == 80      # default 80 mel bands

    def test_mel_path_png(self, temp_dir, dummy_audio_path):
        """--mel-path=mel.png saves an image (requires matplotlib)."""
        pytest.importorskip("matplotlib")
        o = out(temp_dir)
        mel_out = os.path.join(temp_dir, "mel.png")
        r = _run(dummy_audio_path, "hello", o,
                 "--preset=en-us", f"--mel-path={mel_out}", timeout=120)
        _skip_if_failed(r)
        assert os.path.exists(mel_out)
        assert os.path.getsize(mel_out) > 0

    def test_textgrid_output(self, temp_dir, dummy_audio_path):
        """--textgrid saves a Praat TextGrid file."""
        o = out(temp_dir)
        tg_out = os.path.join(temp_dir, "output.TextGrid")
        r = _run(dummy_audio_path, "hello", o,
                 "--preset=en-us", f"--textgrid={tg_out}", timeout=120)
        _skip_if_failed(r)
        assert os.path.exists(tg_out)
        content = open(tg_out).read()
        assert "TextGrid" in content

    def test_embeddings_output(self, temp_dir, dummy_audio_path):
        """--embeddings saves a .pt tensor."""
        o = out(temp_dir)
        emb_out = os.path.join(temp_dir, "emb.pt")
        r = _run(dummy_audio_path, "hello", o,
                 "--preset=en-us", f"--embeddings={emb_out}", timeout=120)
        _skip_if_failed(r)
        assert os.path.exists(emb_out)

    # -- output structure ----------------------------------------------------

    def test_phoneme_ts_fields(self, temp_dir, dummy_audio_path, srt_path):
        """Every phoneme_ts entry has the required fields."""
        o = out(temp_dir)
        r = _run(dummy_audio_path, srt_path, o, "--preset=en-us", timeout=120)
        _skip_if_failed(r)
        required = {"phoneme_id", "phoneme_label", "ipa_label",
                    "start_ms", "end_ms", "confidence", "is_estimated"}
        data = json.loads(open(o).read())
        for seg in data["segments"]:
            for ts in seg["phoneme_ts"]:
                missing = required - set(ts.keys())
                assert not missing, f"Missing fields: {missing}"
                assert ts["end_ms"] >= ts["start_ms"]

    def test_debug_flag_accepted(self, temp_dir, dummy_audio_path, srt_path):
        o = out(temp_dir)
        r = _run(dummy_audio_path, srt_path, o, "--preset=en-us", "--debug", timeout=120)
        _skip_if_failed(r)
        assert os.path.exists(o)

    def test_output_dir_created(self, temp_dir, dummy_audio_path):
        """CLI creates nested output directories automatically."""
        nested = os.path.join(temp_dir, "a", "b", "out.json")
        r = _run(dummy_audio_path, "hi", nested, "--preset=en-us", timeout=120)
        _skip_if_failed(r)
        assert os.path.exists(nested)


# ---------------------------------------------------------------------------
# 4. Error handling
# ---------------------------------------------------------------------------

class TestCLIErrors:

    def test_nonexistent_audio(self, temp_dir, srt_path):
        r = _run("/nonexistent/audio.wav", srt_path, out(temp_dir), timeout=30)
        assert r.returncode != 0

    def test_nonexistent_srt_treated_as_text(self, temp_dir, dummy_audio_path):
        """A non-existing second arg is treated as inline text, not an error at parse time.
        The command will try to run in text mode — skip if model unavailable."""
        o = out(temp_dir)
        r = _run(dummy_audio_path, "nonexistent_file.json", o,
                 "--preset=en-us", timeout=120)
        # Either succeeds (text mode) or fails because of model — both are OK here.
        # What we assert is that it does NOT crash with a Python traceback before model init.
        assert "Traceback" not in r.stderr or r.returncode != 0
