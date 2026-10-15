# File: tests/test_eval_labels.py
"""
Tests for the evaluation parsers' label conversion (`evaluation/dataset/`): TIMIT's and
Buckeye's hand-written phone labels into the std phoneme set's latin tokens.

Needs neither corpus: the tables are checked against the vendored standard_g2p, and the file
readers against a few lines written to a temp directory.

    python -m pytest tests/test_eval_labels.py
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from bournemouth_aligner.standard_g2p import lang_group_inventory as LGI
from evaluation.dataset import buckeye_parser, timit_parser
from evaluation.dataset.std_labels import (BUCKEYE_TO_STD, TIMIT_TO_STD, UNK, StdLabels,
                                           clean_buckeye_label)

# What the std G2P writes for English, measured on both corpora. A hand label has to land on
# one of these, or on one of the allophones the tables name on purpose.
G2P_ENGLISH = set("ɪ t n ə d s a i ɹ ɫ ɛ k æ ʊ ð m e u w z o ɔ f b p ɑ ɝ h ŋ v j ɡ θ ʃ ʒ".split())
EXTRA = {"SIL", "noise", "<unk>", "ɾ", "ʔ"}


@pytest.fixture(scope="module")
def timit():
    return StdLabels(TIMIT_TO_STD, "TIMIT")


@pytest.fixture(scope="module")
def buckeye():
    return StdLabels(BUCKEYE_TO_STD, "Buckeye")


class TestTables:

    def test_every_label_resolves_to_latin_tokens(self, timit, buckeye):
        assert len(TIMIT_TO_STD) == 61                      # TIMIT's whole phone set
        for labels, table in ((timit, TIMIT_TO_STD), (buckeye, BUCKEYE_TO_STD)):
            assert labels.header == {"phoneme_set": "std/latin", "lang": "en",
                                     "lang_group": "latin", "n_tokens": LGI.n_tokens("latin"),
                                     "gold_fingerprint": LGI.gold_fingerprint()}
            for label in table:
                tokens = labels.decode(labels.local_ph(label))
                assert set(tokens) <= G2P_ENGLISH | EXTRA, (label, tokens)

    def test_labels_are_written_as_the_g2p_writes_english(self, timit, buckeye):
        for labels in (timit, buckeye):
            assert labels.decode(labels.local_ph("ay")) == ["a", "ɪ"]      # two tokens
            assert labels.decode(labels.local_ph("ch")) == ["t", "ʃ"]
            assert labels.decode(labels.local_ph("l")) == ["ɫ"]
            assert labels.decode(labels.local_ph("ah")) == ["ə"]
            assert labels.decode(labels.local_ph("dx")) == ["ɾ"]
            assert len(labels.gold_phg("ay")) == 2
        assert timit.decode(timit.local_ph("tcl")) == ["t"]                # closure -> its stop
        assert timit.local_ph("h#") == [timit.sil_id] == timit.local_ph("epi")
        assert buckeye.decode(buckeye.local_ph("own")) == ["o", "ʊ"]       # nasalised -> oral
        assert buckeye.local_ph("IVER") == [buckeye.noise_id]

    def test_buckeye_typos_are_read_not_unknown(self, buckeye):
        assert UNK not in set(BUCKEYE_TO_STD.values())
        assert buckeye.decode(buckeye.local_ph("i")) == ["ɪ"]
        assert buckeye.decode(buckeye.local_ph("j")) == ["d", "ʒ"]

    def test_only_speech_starts_a_boundary(self, timit, buckeye):
        assert timit.is_speech("tcl") and timit.is_speech("q") and timit.is_speech("ay")
        assert not timit.is_speech("epi") and not timit.is_speech("h#")
        assert buckeye.is_speech("tq") and buckeye.is_speech("x")
        for label in ("SIL", "LAUGH", "IVER", "VOCNOISE", "UNKNOWN", "EXCLUDE"):
            assert not buckeye.is_speech(label)

    def test_an_unknown_label_raises(self, timit):
        with pytest.raises(KeyError, match="no entry in the std table"):
            timit.local_ph("zz")

    def test_a_token_outside_the_latin_list_is_refused(self):
        with pytest.raises(ValueError):
            StdLabels({"xx": "not-a-token"}, "test")

    def test_buckeye_label_cleaning(self):
        assert clean_buckeye_label("iy") == ["iy"]
        assert clean_buckeye_label("iy; *") == ["iy"]
        assert clean_buckeye_label("iy ; *") == ["iy"]
        assert clean_buckeye_label("t; *q") == ["t"]
        assert clean_buckeye_label("ih+1") == ["ih"]
        assert clean_buckeye_label("ah+1n") == ["ahn"]
        assert clean_buckeye_label("ah n") == ["ah", "n"]
        assert clean_buckeye_label("<EXCLUDE-name>") == ["EXCLUDE"]
        assert clean_buckeye_label("{B_TRANS}") == ["{B_TRANS}"]


class TestReaders:

    def test_buckeye_times_are_where_a_label_ends(self, tmp_path, buckeye):
        path = tmp_path / "clip.phones"
        path.write_text("signal x\nnfields 2\n#\n"
                        "    7.038  122 {B_TRANS}\n"
                        "    7.326  122 ah\n"
                        "    7.494  122 m; *\n"
                        "   11.342  122 SIL\n"
                        "   11.933  122 ow\n"
                        "   12.000  122\n"                       # an interval nobody labelled
                        "   12.500  122 {E_TRANS}\n"
                        "   12.600  122 junk-after-the-end\n")
        phones, onsets = buckeye_parser.read_phones(str(path), buckeye)
        assert [(p["phone"], p["start"], p["end"]) for p in phones] == [
            ("ah", 7038, 7326), ("m", 7326, 7494), ("SIL", 7494, 11342), ("ow", 11342, 11933),
            ("UNKNOWN", 11933, 12000)]
        assert [buckeye.decode(p["local_ph"]) for p in phones] == [
            ["ə"], ["m"], ["SIL"], ["o", "ʊ"], ["noise"]]
        # where a speech phone starts: not the pause, not the interval nobody labelled
        assert onsets == [7038, 7326, 11342]

    def test_buckeye_segments_break_at_every_tag_but_a_pause(self, tmp_path):
        assert not buckeye_parser.is_break_token("<SIL>")
        assert not buckeye_parser.is_break_token("okay")
        for tag in ("<IVER>", "<LAUGH>", "<LAUGH-yeah>", "<VOCNOISE>", "<CUTOFF-th=the>",
                    "<EXT-I've>", "<NOISE_you", "{E_TRANS}"):
            assert buckeye_parser.is_break_token(tag), tag

        words = [{"timestamp": t, "word": w} for t, w in (
            (1.0, "{B_TRANS}"), (1.5, "yknow"), (2.0, "<SIL>"), (2.4, "hm"), (3.0, "<LAUGH-yeah>"),
            (3.5, "<SIL>"), (4.0, "okay"), (9.0, "<IVER>"), (9.5, "{E_TRANS}"))]
        onsets = [1000, 1250, 2000, 2200, 2400, 2700, 3500, 3700]      # 2000 starts the pause
        srt = tmp_path / "clip.srt.json"
        buckeye_parser.save_srt("clip.wav", str(srt), boundaries=[t for t in onsets if t != 2000],
                                words_data=words)
        import json
        segments = json.loads(srt.read_text())["segments"]
        # the laughed word ends the first segment; the pause after it belongs to neither
        assert [(s["start"], s["end"], s["text"]) for s in segments] == [
            (1.0, 2.4, "you know hum"), (3.5, 4.0, "okay")]
        assert [s["boundaries"] for s in segments] == [[0, 250, 1200], [0, 200]]

    def test_timit_words_carry_their_phones_as_std_tokens(self, tmp_path, timit):
        (tmp_path / "x.WRD").write_text("1600 4800 she\n4800 9600 had\n")
        (tmp_path / "x.PHN").write_text("0 1600 h#\n1600 3200 sh\n3200 4800 iy\n"
                                        "4800 5600 hv\n5600 8000 ay\n8000 8800 dcl\n"
                                        "8800 9600 d\n9600 12000 h#\n")
        words, boundaries = timit_parser.parse_timit_clip(str(tmp_path / "x.WRD"),
                                                          str(tmp_path / "x.PHN"), timit)
        she, had = words
        assert (she["phones"], timit.decode(she["local_ph"])) == (["sh", "iy"], ["ʃ", "i"])
        assert had["phones"] == ["hv", "ay", "dcl", "d"]
        assert timit.decode(had["local_ph"]) == ["h", "a", "ɪ", "d", "d"]
        assert had["ph_count"] == [1, 2, 1, 1]
        assert had["boundaries"] == [350, 500, 550, 600]         # the END of each phone, ms
        assert len(had["gold_phg"]) == len(had["local_ph"])
        # where each in-word phone starts: the closure and its release both, the last end not
        assert boundaries == [100, 200, 300, 350, 500, 550]


if __name__ == "__main__":
    pytest.main([__file__])
