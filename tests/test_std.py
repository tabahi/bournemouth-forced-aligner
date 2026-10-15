# File: tests/test_std.py
"""
Tests for the std phoneme set and the p4mbfa model family.

Nothing here downloads anything. The G2P model (~1.2 GB) and the released models are
replaced by what the code path under test actually needs:

  - a p4mbfa model with RANDOM weights, written to a temp directory in the released layout
    (config.json + model.safetensors) or to a file as a training checkpoint, so loading, the
    inventory check and the whole alignment pipeline run for real;
  - segments that are already phonemized to gold indices (`gold_ph`), which is the one input
    the std set aligns without calling the G2P.

Alignments from random weights are meaningless as alignments, and the tests do not look at
where a boundary lands - only at structure: every target placed once, in order, inside the
audio, labelled in the right inventory.

    python -m pytest tests/test_std.py
"""

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from bournemouth_aligner import BFA, hub, presets
from bournemouth_aligner.extractors import load_extractor
from bournemouth_aligner.p4mbfa import MbfaCNN
from bournemouth_aligner.phonemizers.std import StdPhonemizer, std_inventory
from bournemouth_aligner.segmental import checkpoint_family
from bournemouth_aligner.standard_g2p import lang_codes as lc
from bournemouth_aligner.standard_g2p import lang_group_inventory as LGI
from bournemouth_aligner.standard_g2p import phoneme_inventory_gold as PI

# What a p4mbfa run's yaml puts in `hyper_parameters`, as far as the decoder reads it.
DECODER_HP = dict(min_phone_frames=6, soft_phone_frames=10, soft_frame_cost=1.0,
                  sil_skippable=True, optional_sil=True, class_prior_norm=True,
                  boundary_bonus_weight=30.0, boundary_contrast_frames=17,
                  boundary_flux_source='mel', sample_rate=16000, window_size_ms=120,
                  window_stride_ms=80)


GEOMETRY_KEYS = ('sample_rate', 'window_size_ms', 'window_stride_ms')


def _random_model(lang_group):
    torch.manual_seed(0)
    return MbfaCNN(input_wav_length=1920, n_phg=15, lang_group=lang_group,
                   n_tokens=LGI.n_tokens(lang_group), n_tones=22)


def make_checkpoint(path, lang_group='latin'):
    """A random-weight p4mbfa TRAINING checkpoint: weights and hyperparameters, no labels."""
    model = _random_model(lang_group)
    torch.save({
        'model_config': model.save_config_state_dict(),
        'hyper_parameters': dict(DECODER_HP, lang_group=lang_group,
                                 n_tokens=LGI.n_tokens(lang_group)),
        'state_dict': {f'model.{k}': v for k, v in model.state_dict().items()},
    }, path)
    return path


def make_release(directory, lang_group='latin', tokens=None, **config_overrides):
    """
    A random-weight p4mbfa model in the RELEASED layout the Hub repositories hold
    (`p4mbfa/export_hf.py` in the models repository): config.json + model.safetensors.
    """
    from safetensors.torch import save_file
    model = _random_model(lang_group)
    os.makedirs(directory, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()},
              os.path.join(directory, 'model.safetensors'), metadata={'format': 'pt'})
    config = {
        'format': 'p4mbfa', 'format_version': 1,
        'model_config': model.save_config_state_dict(),
        'geometry': {k: DECODER_HP[k] for k in GEOMETRY_KEYS},
        'decoder': {k: v for k, v in DECODER_HP.items() if k not in GEOMETRY_KEYS},
        'labels': {'lang_group': lang_group,
                   'tokens': list(tokens if tokens is not None else LGI.tokens(lang_group)),
                   'gold_fingerprint': LGI.gold_fingerprint()},
        'training': {'experiment': 'test'},
    }
    config.update(config_overrides)
    with open(os.path.join(directory, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False)
    return directory


def gold_segment(words_ipa, start=0.0, end=2.0):
    """A `.gs.json`-style segment from `[(word, [ipa, ...]), ...]`: gold indices, no G2P."""
    phonemes, word_num = [], []
    for wi, (_word, ipa) in enumerate(words_ipa):
        phonemes += ipa
        word_num += [wi] * len(ipa)
    return {"start": start, "end": end, "text": " ".join(w for w, _ in words_ipa),
            "words": [w for w, _ in words_ipa], "phonemes": phonemes, "word_num": word_num,
            "tone": [0] * len(phonemes), "stress": [0] * len(phonemes),
            "length": [0] * len(phonemes),
            "gold_ph": [PI.TOKEN_INDEX[p] for p in phonemes],
            "gold_ph_count": [1] * len(phonemes)}


HELLO_WORLD = [("hello", ["h", "ɛ", "l", "o"]), ("world", ["w", "ɝ", "l", "d"])]


@pytest.fixture(scope="module")
def latin_ckpt(tmp_path_factory):
    return make_release(str(tmp_path_factory.mktemp("models") / "mbfa-latin-test"))


@pytest.fixture(scope="module")
def aligner(latin_ckpt):
    return BFA(ckpt_path=latin_ckpt, lang="en", device="cpu",
                                   duration_max=5)


# ---------------------------------------------------------------------------------------------
# presets: language -> canonical code -> language group -> checkpoint
# ---------------------------------------------------------------------------------------------

class TestStdPresets:

    @pytest.mark.parametrize("asked, code", [
        ("en-us", "en"), ("en", "en"), ("EN-US", "en"), ("en-gb", "en-GB"),
        ("pt-br", "pt-BR"), ("cmn", "zh"), ("zh-tw", "zh-Hant"), ("deu", "de"),
        ("fr-be", "fr"), ("en-gb-x-rp", "en-GB"),
    ])
    def test_spellings_normalise(self, asked, code):
        assert presets.normalize_std_lang(asked)[0] == code

    def test_every_canonical_code_is_routable(self):
        """A code `normalize_std_lang` returns must be one the inventory tables know."""
        for tag in lc.TAG_TO_ISO:
            code = lc.tag_to_bcp47(tag)
            assert presets.normalize_std_lang(code) == (code, None)
            assert LGI.lang_group_of(code) == presets.std_lang_group(code)

    def test_unknown_region_falls_back_with_a_note(self):
        code, note = presets.normalize_std_lang("de-at")
        assert code == "de"
        assert note and "de-at" in note

    def test_language_decides_the_checkpoint(self):
        for lang, group in (("en-us", "latin"), ("de", "latin"), ("ru", "cyrillic"),
                            ("zh", "cjk"), ("yue", "cjk"), ("ja", "japanese")):
            model_name, _code, got, _note = presets.get_std_preset(lang)
            assert got == group
            assert model_name == presets.P4MBFA_MODELS[group]

    def test_uncovered_language_names_the_way_out(self):
        with pytest.raises(presets.UnsupportedLanguageError, match="ph66"):
            presets.get_std_preset("bn")          # the G2P was never trained on Bengali

    def test_group_without_a_checkpoint_raises(self):
        group = presets.std_lang_group("hi")
        if group in presets.P4MBFA_MODELS:
            pytest.skip(f"'{group}' has a checkpoint now")
        with pytest.raises(presets.UnsupportedLanguageError, match=group):
            presets.get_std_preset("hi")

    def test_excluded_language_raises(self):
        with pytest.raises(presets.UnsupportedLanguageError, match="excluded"):
            presets.get_std_preset("km")

    def test_phoneme_set(self):
        assert presets.resolve_phoneme_set() == presets.DEFAULT_PHONEME_SET == "std"
        assert presets.resolve_phoneme_set("ph66") == "ph66"
        with pytest.raises(ValueError):
            presets.resolve_phoneme_set("ph67")
        with pytest.raises(TypeError):
            BFA(preset=None, mapper="ph66")


# ---------------------------------------------------------------------------------------------
# the inventory and the phonemizer (no G2P model involved)
# ---------------------------------------------------------------------------------------------

class TestStdInventory:

    def test_specials_sit_where_the_models_were_trained_with_them(self):
        for group in LGI.list_all_lang_groups:
            inv = std_inventory(group)
            assert inv.tokens[:4] == ['<blank>', 'SIL', 'noise', '<unk>']
            assert (inv.pad_id, inv.sil_id) == (0, 1)
            assert inv.n_classes == LGI.n_tokens(group)
            assert inv.n_groups == 15
            assert inv.name == f"std/{group}"

    def test_every_token_has_a_group(self):
        inv = std_inventory("latin")
        assert set(inv.phoneme_id_to_group_id) == set(range(inv.n_classes))
        assert all(0 <= g < inv.n_groups for g in inv.phoneme_id_to_group_id.values())

    def test_no_silence_at_punctuation_by_default(self):
        p = StdPhonemizer("en")
        assert p._split("Hello, world. Now!") == ["hello", "world", "now"]

    def test_punctuation_sil(self):
        p = StdPhonemizer("en", punctuation_sil=True)
        assert p._split("Hello, world. Now!") == ["hello", None, "world", None, "now"]
        # a mark inside a token is not a pause, and neither is an apostrophe
        assert p._split("it's 3.5 o'clock") == ["it's", "3", "5", "o'clock"]

    def test_explicit_sil_is_always_honoured(self):
        p = StdPhonemizer("en")
        assert p._split("hello <SIL> world") == ["hello", None, "world"]
        # none at the edges or doubled: the aligner pads its own optional silence there
        assert p._split("<SIL> hello <SIL> <SIL> world <SIL>") == ["hello", None, "world"]

    def test_mandarin_is_segmented_from_the_shipped_dictionary_keys(self):
        """The longest-match segmenter reads `standard_g2p/dicts/`, inside the package."""
        p = StdPhonemizer("zh")
        words = p._split("狂妄的人")
        assert "".join(words) == "狂妄的人"
        assert "狂妄" in words                      # a dictionary word, not four characters

    def test_gold_segment_needs_no_g2p(self):
        p = StdPhonemizer("en")
        d = p.phonemize_gold(gold_segment(HELLO_WORLD))
        assert p._g2p is None                       # the model was never loaded
        assert d["mipa"] == ["h", "ɛ", "l", "o", "w", "ɝ", "l", "d"]
        assert d["word_num"] == [0, 0, 0, 0, 1, 1, 1, 1]
        assert d["words"] == ["hello", "world"]
        keys = ("local_ph", "gold_ph", "gold_phg", "eipa", "mipa", "tone", "stress", "length",
                "word_num")
        assert len({len(d[k]) for k in keys}) == 1

    def test_gold_token_outside_the_group_folds_to_unk(self):
        # A Mandarin retroflex is in gold but not on Japanese's 28-token list.
        outside = next(t for t in LGI.tokens("cjk")[4:] if t not in LGI.tokens("japanese"))
        p = StdPhonemizer("ja")
        d = p.phonemize_gold(gold_segment([("x", [outside])]))
        assert d["mipa"] == ["<unk>"] and d["n_folded"] == 1
        assert d["eipa"] == [outside]               # what it was is still on record


# ---------------------------------------------------------------------------------------------
# checkpoints: family detection and the inventory check
# ---------------------------------------------------------------------------------------------

class TestCheckpoints:

    def test_family_is_read_off_model_config(self):
        assert checkpoint_family({'model_config': {'n_tokens': 208, 'lang_group': 'latin'}}) == 'p4mbfa'
        assert checkpoint_family({'model_config': {'phoneme_classes': 66}}) == 'p3cupe'
        with pytest.raises(ValueError):
            checkpoint_family({'model_config': {'something': 1}})

    def test_extractor_reads_everything_from_the_checkpoint(self, latin_ckpt):
        ex = load_extractor(latin_ckpt)
        assert (ex.family, ex.lang_group) == ('p4mbfa', 'latin')
        assert (ex.classes_dim, ex.groups_dim, ex.tones_dim) == (LGI.n_tokens('latin'), 15, 22)
        assert ex.ms_per_frame == 5.0
        assert ex.decoder_hp['soft_phone_frames'] == 10
        out = ex.encode_clip(torch.randn(1, 16000), return_embeddings=True, return_flux=True)
        frames = out['ph'].shape[1]
        assert out['ph'].shape == (1, frames, ex.classes_dim)
        assert out['group'].shape == (1, frames, 15)
        assert out['tone'].shape == (1, frames, 22)
        assert out['flux'].shape == (1, frames)

    def test_inventory_check_accepts_its_own_group(self, latin_ckpt):
        load_extractor(latin_ckpt).check_inventory(std_inventory('latin'))

    def test_inventory_check_rejects_another_group(self, latin_ckpt):
        with pytest.raises(ValueError, match="cjk"):
            load_extractor(latin_ckpt).check_inventory(std_inventory('cjk'))

    def test_inventory_check_rejects_a_regenerated_token_list(self, tmp_path):
        """Same group, same size, two tokens swapped: only the release's token list catches it."""
        tokens = list(LGI.tokens('latin'))
        tokens[10], tokens[11] = tokens[11], tokens[10]
        path = make_release(str(tmp_path / "stale"), tokens=tokens)
        with pytest.raises(ValueError, match="different 'latin' token list"):
            load_extractor(path).check_inventory(std_inventory('latin'))

    def test_training_checkpoint_still_loads(self, tmp_path):
        """A raw training checkpoint has no token list; the group and the count are checked."""
        path = make_checkpoint(str(tmp_path / "raw.pt"))
        ex = load_extractor(path)
        assert ex.stamp is None and ex.release is None
        ex.check_inventory(std_inventory('latin'))
        with pytest.raises(ValueError, match="cjk"):
            ex.check_inventory(std_inventory('cjk'))

    def test_release_and_training_checkpoint_are_the_same_model(self, latin_ckpt, tmp_path):
        """Both layouts of the same weights give the same posteriors."""
        a = load_extractor(latin_ckpt)
        b = load_extractor(make_checkpoint(str(tmp_path / "raw.pt")))
        assert a.decoder_hp == b.decoder_hp
        assert a.release['training']['experiment'] == 'test'
        wav = torch.randn(1, 16000)
        assert torch.equal(a.encode_clip(wav)['ph'], b.encode_clip(wav)['ph'])

    def test_release_directory_must_be_complete(self, tmp_path):
        path = make_release(str(tmp_path / "half"))
        os.remove(os.path.join(path, 'model.safetensors'))
        with pytest.raises(ValueError, match="model.safetensors"):
            load_extractor(path)

    @pytest.mark.parametrize("override, match", [
        ({'format': 'something'}, "format 'something'"),
        ({'format_version': 2}, "format version 2"),
        ({'format': 'p3cupe'}, "p4mbfa"),
    ])
    def test_release_of_another_format_is_refused(self, tmp_path, override, match):
        path = make_release(str(tmp_path / "other"), **override)
        with pytest.raises(ValueError, match=match):
            load_extractor(path)

    def test_missing_decoder_setting_is_an_error_not_a_default(self, tmp_path):
        path = make_checkpoint(str(tmp_path / "x.pt"))
        ckpt = torch.load(path, weights_only=False)
        del ckpt['hyper_parameters']['soft_phone_frames']
        torch.save(ckpt, path)
        with pytest.raises(ValueError, match="soft_phone_frames"):
            load_extractor(path)


# ---------------------------------------------------------------------------------------------
# the aligner, end to end, on a random-weight checkpoint
# ---------------------------------------------------------------------------------------------

class TestStdAligner:

    def test_checkpoint_decides_the_phoneme_set(self, aligner):
        assert aligner.phoneme_set == "std"
        assert aligner.extractor.family == "p4mbfa"
        assert aligner.inventory.name == "std/latin"
        assert (aligner.silence_class, aligner.pad_class) == (1, 0)
        assert (aligner.phonemes_key, aligner.phoneme_groups_key) == ("local_ph", "gold_phg")

    def test_low_confidence_level_is_the_familys_unless_given(self, aligner, latin_ckpt):
        assert aligner.low_confidence_level == aligner.extractor.low_confidence_level == 0.1
        a = BFA(ckpt_path=latin_ckpt, lang="en", device="cpu",
                                    low_confidence_level=0.3)
        assert a.low_confidence_level == 0.3
        from bournemouth_aligner.p3cupe import CupeExtractor
        assert 0.0 < CupeExtractor.low_confidence_level < 1.0
        assert CupeExtractor.low_confidence_level != aligner.extractor.low_confidence_level

    def test_asking_for_the_other_set_raises(self, latin_ckpt):
        with pytest.raises(ValueError, match="phoneme_set='ph66'"):
            BFA(ckpt_path=latin_ckpt, lang="en", phoneme_set="ph66",
                                    device="cpu")

    def test_language_of_another_group_raises(self, latin_ckpt):
        with pytest.raises(ValueError, match="latin"):
            BFA(ckpt_path=latin_ckpt, lang="ru", device="cpu")

    def test_old_language_spellings_are_accepted(self, latin_ckpt):
        a = BFA(ckpt_path=latin_ckpt, lang="pt-br", device="cpu")
        assert a.lang == "pt-BR"

    def test_gold_segment_is_aligned_without_the_g2p(self, aligner):
        wav = torch.randn(1, 32000)
        result = aligner.process_segments([{"segments": [gold_segment(HELLO_WORLD)]}], [wav],
                                          do_groups=True)[0]
        assert aligner.phonemizer._g2p is None
        assert result["phoneme_set"] == "std/latin" and result["lang"] == "en"

        seg = result["segments"][0]
        phones = seg["phoneme_ts"]
        # forced alignment: every target placed exactly once, in order, inside the audio
        assert [p["target_seq_idx"] for p in phones] == list(range(8))
        assert [p["phoneme_label"] for p in phones] == ["h", "ɛ", "l", "o", "w", "ɝ", "l", "d"]
        assert [p["phoneme_id"] for p in phones] == seg["local_ph"]
        assert all(0.0 <= a["start_ms"] < a["end_ms"] <= 2000.0 for a in phones)
        assert all(a["end_ms"] <= b["start_ms"] + 1e-6 for a, b in zip(phones, phones[1:]))
        # the std layers ride along on every phoneme
        assert all({"tone", "stress", "length"} <= set(p) for p in phones)
        # groups are the phoneme segmentation relabelled, in the 15 gold groups
        assert [g["group_id"] for g in seg["group_ts"]] == seg["gold_phg"]
        assert {g["group_label"] for g in seg["group_ts"]} <= set(aligner.group_id_to_label.values())

        words = seg["words_ts"]
        assert [w["word"] for w in words] == ["hello", "world"]
        assert [len(w["local_ph"]) for w in words] == [4, 4]
        assert words[0]["end_ms"] <= words[1]["start_ms"] + 1e-6

    def test_batch_of_ragged_segments(self, aligner):
        """Padding is `<blank>`, which is never a target, so a short row is not lengthened."""
        short = gold_segment([("hi", ["h", "a"])], end=1.0)
        long = gold_segment(HELLO_WORLD, start=1.0, end=3.0)
        result = aligner.process_segments([{"segments": [short, long]}], [torch.randn(1, 48000)])[0]
        assert [len(s["phoneme_ts"]) for s in result["segments"]] == [2, 8]

    def test_low_level_batch_takes_local_ids(self, aligner):
        ids = [PI.TOKEN_INDEX[p] for p in ("h", "ɛ", "l", "o")]
        local = LGI.to_local(ids, "en")["local_ph"]
        wav = aligner._rms_normalize(torch.randn(16000))
        stamps, _, _ = aligner.extract_timestamps_from_segment_batch(
            wav.unsqueeze(0), [16000], [local], extract_embeddings=False, do_groups=False)
        assert [int(t[0]) for t in stamps[0]["phoneme_timestamps"]] == local

    def test_textgrid(self, aligner):
        result = aligner.process_segments([{"segments": [gold_segment(HELLO_WORLD)]}],
                                          [torch.randn(1, 32000)])[0]
        assert "TextGrid" in aligner.convert_to_textgrid(result)


# ---------------------------------------------------------------------------------------------
# hub: where files are looked for (no network)
# ---------------------------------------------------------------------------------------------

class TestHub:

    def test_names_map_to_this_accounts_repositories(self):
        assert hub.repo_id_of("mbfa-latin") == "Tabahi/mbfa-latin"
        assert hub.repo_id_of("someone/their-model") == "someone/their-model"

    def test_every_preset_names_a_published_model(self):
        assert set(presets.P4MBFA_MODELS.values()) <= set(presets.MODELS)
        for name in (presets.P3CUPE_EN, presets.P3CUPE_EN_BASE, presets.P3CUPE_MLS7):
            assert presets.MODELS[name] == "ph66"
        assert presets.ph66_model("en-gb") == presets.P3CUPE_EN
        assert presets.ph66_model("de") == presets.P3CUPE_MLS7
        assert presets.MODELS[presets.MBFA_LATIN_BENCHMARK] == "std"
        assert presets.get_std_preset("en-us")[0] == "mbfa-latin"

    def test_model_is_found_in_the_model_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BFA_MODEL_DIR", str(tmp_path))
        make_release(str(tmp_path / "mbfa-latin"))
        os.makedirs(tmp_path / "not-a-model")
        assert hub.find_model("mbfa-latin") == str(tmp_path / "mbfa-latin")
        assert hub.find_model("Tabahi/mbfa-latin") == str(tmp_path / "mbfa-latin")
        assert hub.find_model("not-a-model") is None
        assert hub.resolve_model("mbfa-latin") == str(tmp_path / "mbfa-latin")   # no download

    def test_a_cached_model_is_used_without_the_network(self, tmp_path, monkeypatch):
        """Every file in the cache -> that directory, and `snapshot_download` is never called."""
        import huggingface_hub
        monkeypatch.delenv("BFA_MODEL_DIR", raising=False)
        monkeypatch.chdir(tmp_path)                       # no ./models here either
        snapshot = tmp_path / "snapshot"
        make_release(str(snapshot))
        monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                            lambda repo_id, name, revision=None: str(snapshot / name))
        monkeypatch.setattr(huggingface_hub, "snapshot_download",
                            lambda *a, **k: pytest.fail("asked the Hub for a cached model"))
        assert hub.resolve_model("mbfa-latin", verbose=0) == str(snapshot)

    def test_a_missing_model_is_downloaded_once_and_quietly(self, tmp_path, monkeypatch, capsys):
        import huggingface_hub
        monkeypatch.delenv("BFA_MODEL_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        snapshot = tmp_path / "snapshot"
        make_release(str(snapshot))
        calls = []
        monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                            lambda repo_id, name, revision=None: None)
        monkeypatch.setattr(huggingface_hub, "snapshot_download",
                            lambda repo_id, **k: calls.append(repo_id) or str(snapshot))
        assert hub.resolve_model("mbfa-latin", verbose=0) == str(snapshot)
        assert capsys.readouterr().out == ""
        assert hub.resolve_model("mbfa-latin", verbose=1) == str(snapshot)
        assert "Downloading the model Tabahi/mbfa-latin" in capsys.readouterr().out
        assert calls == ["Tabahi/mbfa-latin"] * 2

    def test_an_existing_path_resolves_to_itself(self, latin_ckpt):
        assert hub.resolve_model(latin_ckpt) == latin_ckpt

    def test_model_name_loads_from_the_model_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BFA_MODEL_DIR", str(tmp_path))
        make_release(str(tmp_path / "mbfa-latin"))
        aligner = BFA(model_name="mbfa-latin", device="cpu")
        assert (aligner.phoneme_set, aligner.inventory.lang_group) == ("std", "latin")

    def test_g2p_dir_must_already_hold_the_model(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BFA_G2P_DIR", raising=False)
        with pytest.raises(FileNotFoundError, match="pytorch_model.bin"):
            hub.ensure_g2p_model(str(tmp_path))

    def test_g2p_dir_with_its_own_tokenizer_downloads_nothing(self, tmp_path, monkeypatch):
        for name in hub.G2P_FILES:
            (tmp_path / name).write_text("x")
        os.makedirs(tmp_path / "tokenizer")
        (tmp_path / "tokenizer" / "tokenizer_config.json").write_text("{}")
        assert hub.ensure_g2p_model(str(tmp_path)) == (str(tmp_path),
                                                       str(tmp_path / "tokenizer"))
        monkeypatch.setenv("BFA_G2P_DIR", str(tmp_path))
        assert hub.ensure_g2p_model()[0] == str(tmp_path)

    def test_g2p_tokenizer_next_to_the_model(self, tmp_path):
        for name in (*hub.G2P_FILES, "tokenizer_config.json"):
            (tmp_path / name).write_text("x")
        assert hub.ensure_g2p_model(str(tmp_path)) == (str(tmp_path), str(tmp_path))


if __name__ == "__main__":
    pytest.main([__file__])
