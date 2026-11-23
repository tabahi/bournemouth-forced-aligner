"""
L2-ARCTIC (v5.0) -> the SRT trees and paths lists the evaluation reads.

    python evaluation/dataset/l2arctic_parser.py --base /path/to/l2arctic_release_v5.0

Non-native English: 24 speakers, six first languages (Arabic, Mandarin, Hindi, Korean,
Spanish, Vietnamese), four speakers each. Two datasets come out of it:

  l2arctic           the 3,599 read CMU-ARCTIC sentences that have a manual annotation
                     (about 150 per speaker). One clip = one sentence = one segment, as TIMIT.
                     -> local/data/l2arctic_data/<SPK>/<utt>.srt.json, local/paths/l2arctic_paths_all.json
  l2arctic_suitcase  the 22 retellings of the "suitcase story", spontaneous, about a minute
                     each. Cut into segments at pauses, as Buckeye (`split_narrative`).
                     -> local/data/l2arctic_suitcase_data/<spk>.srt.json, local/paths/l2arctic_suitcase_paths_all.json

The ~23,000 sentences WITHOUT a manual annotation are not used: their only labels are the
corpus's own forced alignment, made with MFA 1.0, which cannot be a reference for scoring an
aligner. Bear in mind that the manual annotations started from that forced alignment too: in
the sentences, 88.8% of the annotated phone onsets are exactly where MFA 1.0 put them (the
annotators moved 11%, by a median of 10 ms). `boundaries_moved` lists the ones that were moved.

## What the aligner gets, and what it is scored against

The TEXT is what the speaker was asked to say: the prompt in `transcript/` (sentences), or the
words tier (suitcase). The audio is what they did say, with the mispronunciations the
annotators marked. That mismatch is the point of the corpus here: the aligner has a correct
transcript of imperfect speech.

A segment's `boundaries` (ms from the segment start) are the onsets of every interval the
speaker produced: a correct phone, a substitution (`CPL,PPL,s`, the phone that was heard
starts there, even when the annotator wrote `err`) and an addition (`sil,PPL,a`, a phone that
is in the audio and not in the text). A deletion (`CPL,sil,d`) produced nothing, so it starts
no boundary; the phone after it does. Silence (`sil`, `sp`, unlabelled) and spoken noise
(`spn`) start none.

Each segment also carries, for analyses beyond the main score (the scorer ignores them):
  boundaries_error  the onsets that start or end an error interval (s / a / d)
  boundaries_moved  the onsets the annotators moved off the MFA 1.0 alignment (sentences only)

    {"labels": {"phoneme_set": "std/latin", ...},
     "text": "...", "speaker": "ABA", "l1": "Arabic", "sex": "M",
     "errors": {"substitution": 3, "addition": 0, "deletion": 1},
     "segments": [{"start": 0.0, "end": 3.2, "text": "...", "boundaries": [30, 170, ...],
                   "boundaries_error": [...], "boundaries_moved": [...]}],
     "phones": [{"start": 30, "end": 170, "canonical": "f", "perceived": "f", "kind": "ok",
                 "local_ph": [..]}, ...]}

`phones` is the annotation tier whole, in ms from the start of the recording; `local_ph` is
the PERCEIVED phone in the std phoneme set (latin local ids, see std_labels.py).
"""

import argparse
import glob
import json
import os
import re
import sys

import soundfile as sf
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.dataset.std_labels import (L2ARCTIC_TO_STD, StdLabels,      # noqa: E402
                                           clean_l2arctic_label)

# https://psi.engr.tamu.edu/l2-arctic-corpus/ (free for non-commercial use, CC BY-NC 4.0)
# base dir: expect <SPK>/{wav,transcript,annotation,textgrid}/ and suitcase_corpus/{wav,transcript,annotation}/
l2arctic_base_dir = '/my_datasets/speech_data/l2arctic_release_v5.0'

SENTENCES = {"name": "l2arctic", "srt_dir": "local/data/l2arctic_data",
             "paths_list": "local/paths/l2arctic_paths_all.json"}
SUITCASE = {"name": "l2arctic_suitcase", "srt_dir": "local/data/l2arctic_suitcase_data",
            "paths_list": "local/paths/l2arctic_suitcase_paths_all.json"}

# The README's speaker table: sex and first language.
SPEAKERS = {
    "ABA": ("M", "Arabic"), "SKA": ("F", "Arabic"), "YBAA": ("M", "Arabic"), "ZHAA": ("F", "Arabic"),
    "BWC": ("M", "Mandarin"), "LXC": ("F", "Mandarin"), "NCC": ("F", "Mandarin"), "TXHC": ("M", "Mandarin"),
    "ASI": ("M", "Hindi"), "RRBI": ("M", "Hindi"), "SVBI": ("F", "Hindi"), "TNI": ("F", "Hindi"),
    "HJK": ("F", "Korean"), "HKK": ("M", "Korean"), "YDCK": ("F", "Korean"), "YKWK": ("M", "Korean"),
    "EBVS": ("M", "Spanish"), "ERMS": ("M", "Spanish"), "MBMPS": ("F", "Spanish"), "NJS": ("F", "Spanish"),
    "HQTV": ("M", "Vietnamese"), "PNV": ("F", "Vietnamese"), "THV": ("F", "Vietnamese"), "TLV": ("M", "Vietnamese"),
}
SPEAKER_IDS = {spk: i for i, spk in enumerate(sorted(SPEAKERS))}
L1_IDS = {l1: i for i, l1 in enumerate(sorted({l1 for _, l1 in SPEAKERS.values()}))}

# The perceived labels that start no boundary: nothing was said, or nothing the aligner places.
NOT_AN_ONSET = {"sil", "spn"}

# Suitcase segments, as Buckeye's: cut at the first pause after TARGET seconds, and at the
# latest pause before MAX (a narrative with no pause for 30 s is cut at a word edge).
TARGET_SECONDS = 10
MAX_SECONDS = 30
# A word-tier gap shorter than this is a word edge, not a pause.
MIN_PAUSE_SECONDS = 0.15


def read_textgrid(path):
    """
    The interval tiers of a long-format Praat TextGrid, as {tier name: [(start, end, text)]}.

    Times in seconds. The corpus's files are all long format, UTF-8 (a few with a BOM); the
    `text` is returned as written, quotes removed, spaces kept.
    """
    raw = open(path, "rb").read()
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"{path}: not UTF-8 or UTF-16")

    tiers, name, current = {}, None, None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("class ="):
            current = None
        elif line.startswith("name =") and current is None:
            name = line.split("=", 1)[1].strip().strip('"').strip()
            tiers[name] = []
        elif line.startswith("intervals [") and name is not None:
            current = {}
        elif current is not None and line.startswith("xmin ="):
            current["s"] = float(line.split("=", 1)[1])
        elif current is not None and line.startswith("xmax ="):
            current["e"] = float(line.split("=", 1)[1])
        elif current is not None and line.startswith("text ="):
            value = line.split("=", 1)[1].strip()
            if value.startswith('"') and value.endswith('"') and len(value) >= 2:
                value = value[1:-1]
            tiers[name].append((current["s"], current["e"], value))
            current = None
    if "phones" not in tiers or "words" not in tiers:
        raise ValueError(f"{path}: no words/phones tiers (found {sorted(tiers)})")
    return tiers


def ms(seconds):
    # round, not int: 0.29 s is 289.99999999999994 ms as a float
    return int(round(seconds * 1000))


def read_phones(tiers, labels):
    """
    The annotation's phones tier as dicts, and the three onset lists of the recording.

    Returns `(phones, onsets, error_onsets, counts)`, times in ms from the start of the file.
    """
    phones, onsets, error_edges = [], [], set()
    counts = {"substitution": 0, "addition": 0, "deletion": 0}
    for start, end, raw in tiers["phones"]:
        if end <= start:
            continue
        canonical, perceived, kind = clean_l2arctic_label(raw)
        labels.local_ph(canonical)                         # every label must be in the table
        start_ms, end_ms = ms(start), ms(end)
        phones.append({"start": start_ms, "end": end_ms, "canonical": canonical,
                       "perceived": perceived, "kind": kind,
                       "local_ph": labels.local_ph(perceived)})
        if perceived not in NOT_AN_ONSET:
            onsets.append(start_ms)
        if kind != "ok":
            counts[kind] += 1
            error_edges.update((start_ms, end_ms))
    onsets = sorted(set(onsets))
    return phones, onsets, sorted(error_edges & set(onsets)), counts


def forced_onsets(textgrid_path):
    """Every phone onset (ms) of the corpus's MFA 1.0 alignment of the same sentence."""
    tiers = read_textgrid(textgrid_path)
    return {ms(s) for s, e, t in tiers["phones"] if e > s and t.strip() not in ("", "sil", "sp")}


def relative(onsets, start_ms, end_ms):
    """The onsets inside [start, end), in ms from `start`."""
    return [o - start_ms for o in onsets if start_ms <= o < end_ms]


# ---------------------------------------------------------------------------------- sentences

def parse_sentences(base_dir, labels, redo):
    paths_list, skipped, total = [], [], {"substitution": 0, "addition": 0, "deletion": 0}
    n_boundaries = n_moved = n_error = 0
    annotations = sorted(glob.glob(os.path.join(base_dir, "*", "annotation", "*.TextGrid")))
    annotations = [p for p in annotations if os.path.basename(os.path.dirname(os.path.dirname(p)))
                   in SPEAKERS]
    for ann_path in tqdm(annotations, desc="l2arctic sentences"):
        spk_dir = os.path.dirname(os.path.dirname(ann_path))
        spk = os.path.basename(spk_dir)
        utt = os.path.basename(ann_path)[:-len(".TextGrid")]
        wav_path = os.path.join(spk_dir, "wav", utt + ".wav")
        txt_path = os.path.join(spk_dir, "transcript", utt + ".txt")
        fa_path = os.path.join(spk_dir, "textgrid", utt + ".TextGrid")
        if not (os.path.exists(wav_path) and os.path.exists(txt_path)):
            skipped.append(f"{spk}/{utt}: no wav or transcript")
            continue
        duration = sf.info(wav_path).duration
        srt_path = os.path.join(SENTENCES["srt_dir"], spk, utt + ".srt.json")

        if redo or not os.path.exists(srt_path):
            with open(txt_path, encoding="utf-8") as f:
                text = " ".join(f.read().split())
            phones, onsets, error_onsets, counts = read_phones(read_textgrid(ann_path), labels)
            end_ms = ms(duration)
            forced = forced_onsets(fa_path) if os.path.exists(fa_path) else None
            segment = {"start": 0.0, "end": duration, "text": text,
                       "boundaries": relative(onsets, 0, end_ms),
                       "boundaries_error": relative(error_onsets, 0, end_ms),
                       "boundaries_moved": (None if forced is None else
                                            relative([o for o in onsets if o not in forced], 0, end_ms))}
            sex, l1 = SPEAKERS[spk]
            srt = {"labels": labels.header, "text": text, "speaker": spk, "l1": l1, "sex": sex,
                   "errors": counts, "segments": [segment], "phones": phones}
            os.makedirs(os.path.dirname(srt_path), exist_ok=True)
            with open(srt_path, "w", encoding="utf-8") as f:
                json.dump(srt, f, ensure_ascii=False)
        else:
            with open(srt_path, encoding="utf-8") as f:
                srt = json.load(f)
            segment, counts = srt["segments"][0], srt["errors"]

        for k in total:
            total[k] += counts[k]
        n_boundaries += len(segment["boundaries"])
        n_error += len(segment["boundaries_error"])
        n_moved += len(segment["boundaries_moved"] or [])
        sex, l1 = SPEAKERS[spk]
        paths_list.append({"audio": "~/" + os.path.relpath(wav_path, base_dir),
                           "srt": srt_path, "set": "all", "lang": "en", "duration": duration,
                           "spkr": SPEAKER_IDS[spk], "clip_id": f"{spk}_{utt}", "sex": sex,
                           "accent": L1_IDS[l1], "l1": l1})

    print(f"l2arctic: {len(paths_list)} sentences, {sum(p['duration'] for p in paths_list) / 3600:.2f} h, "
          f"{n_boundaries} onsets ({n_error} at an error, {n_moved} moved off MFA 1.0); "
          f"errors {total}")
    for line in skipped:
        print("  skipped", line)
    return paths_list


# ---------------------------------------------------------------------------------- suitcase

def is_word(token):
    """A plain word of the words tier (`was***` is `was`); a cut-off (`buil-`) or a symbol is not."""
    return re.fullmatch(r"[a-z']+", token.strip().lower().rstrip("*")) is not None


def split_narrative(words):
    """
    The words tier of a narrative as segments: `[(start, end, [word, ...])]`, in seconds.

    A run of plain words is cut at the first pause (an unlabelled gap of MIN_PAUSE_SECONDS or
    more) once it is TARGET_SECONDS long, at the last pause before MAX_SECONDS, or at the last
    word edge when there is no pause. A token that is not a plain word (a cut-off, a symbol)
    ends the segment, as Buckeye's tags do: its audio has no text to place on it.
    """
    segments, run = [], []          # run: [(start, end, word)]
    pause_after = []                # pause_after[i]: run[i] is followed by a pause

    def flush(upto):
        nonlocal run, pause_after
        if upto > 0:
            segments.append((run[0][0], run[upto - 1][1], [w for _, _, w in run[:upto]]))
        run, pause_after = run[upto:], pause_after[upto:]

    for i, (start, end, token) in enumerate(words):
        token = token.strip()
        if not token:
            if run and end - start >= MIN_PAUSE_SECONDS:
                pause_after[-1] = True
            continue
        if not is_word(token):
            flush(len(run))
            continue
        run.append((start, end, token.lower().rstrip("*")))
        pause_after.append(False)
        length = run[-1][1] - run[0][0]
        if length >= TARGET_SECONDS and pause_after[-2:-1] == [True]:
            flush(len(run) - 1)             # the pause came before this word
        elif length > MAX_SECONDS:
            pauses = [k + 1 for k, p in enumerate(pause_after[:-1]) if p]
            flush(pauses[-1] if pauses else len(run) - 1)
    # the last word may still be followed by a pause: nothing to do, the run just ends
    flush(len(run))
    return segments


def parse_suitcase(base_dir, labels, redo):
    root = os.path.join(base_dir, "suitcase_corpus")
    paths_list = []
    n_boundaries = n_error = n_segments = 0
    total = {"substitution": 0, "addition": 0, "deletion": 0}
    for ann_path in tqdm(sorted(glob.glob(os.path.join(root, "annotation", "*.TextGrid"))),
                         desc="l2arctic suitcase"):
        spk_lower = os.path.basename(ann_path)[:-len(".TextGrid")]
        spk = spk_lower.upper()
        wav_path = os.path.join(root, "wav", spk_lower + ".wav")
        duration = sf.info(wav_path).duration
        srt_path = os.path.join(SUITCASE["srt_dir"], spk_lower + ".srt.json")

        if redo or not os.path.exists(srt_path):
            tiers = read_textgrid(ann_path)
            phones, onsets, error_onsets, counts = read_phones(tiers, labels)
            segments = []
            for start, end, seg_words in split_narrative(tiers["words"]):
                s_ms, e_ms = ms(start), min(ms(end), ms(duration))
                if e_ms <= s_ms:
                    continue
                segments.append({"start": start, "end": e_ms / 1000, "text": " ".join(seg_words),
                                 "boundaries": relative(onsets, s_ms, e_ms),
                                 "boundaries_error": relative(error_onsets, s_ms, e_ms),
                                 "boundaries_moved": None})
            sex, l1 = SPEAKERS[spk]
            srt = {"labels": labels.header, "text": " ".join(s["text"] for s in segments),
                   "speaker": spk, "l1": l1, "sex": sex, "errors": counts,
                   "segments": segments, "phones": phones,
                   "voiced_duration": sum(s["end"] - s["start"] for s in segments)}
            os.makedirs(os.path.dirname(srt_path), exist_ok=True)
            with open(srt_path, "w", encoding="utf-8") as f:
                json.dump(srt, f, ensure_ascii=False)
        else:
            with open(srt_path, encoding="utf-8") as f:
                srt = json.load(f)

        for k in total:
            total[k] += srt["errors"][k]
        n_segments += len(srt["segments"])
        n_boundaries += sum(len(s["boundaries"]) for s in srt["segments"])
        n_error += sum(len(s["boundaries_error"]) for s in srt["segments"])
        sex, l1 = SPEAKERS[spk]
        paths_list.append({"audio": "~/" + os.path.relpath(wav_path, base_dir),
                           "srt": srt_path, "set": "all", "lang": "en", "duration": duration,
                           "spkr": SPEAKER_IDS[spk], "clip_id": f"{spk}_suitcase", "sex": sex,
                           "accent": L1_IDS[l1], "l1": l1})

    print(f"l2arctic_suitcase: {len(paths_list)} narratives, "
          f"{sum(p['duration'] for p in paths_list) / 60:.1f} min, {n_segments} segments, "
          f"{n_boundaries} onsets ({n_error} at an error); errors {total}")
    return paths_list


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default=l2arctic_base_dir, help="the l2arctic_release_v5.0 directory")
    ap.add_argument("--keep", action="store_true", help="reuse SRTs already written")
    args = ap.parse_args()
    base = os.path.abspath(args.base)
    if not os.path.isfile(os.path.join(base, "PROMPTS")):
        sys.exit(f"{base} does not look like L2-ARCTIC v5.0 (no PROMPTS file)")
    os.chdir(REPO_ROOT)                     # the SRT and paths-list paths are repo-relative

    labels = StdLabels(L2ARCTIC_TO_STD, "L2-ARCTIC")
    for dataset, parse in ((SENTENCES, parse_sentences), (SUITCASE, parse_suitcase)):
        paths_list = parse(base, labels, redo=not args.keep)
        os.makedirs(os.path.dirname(dataset["paths_list"]), exist_ok=True)
        with open(dataset["paths_list"], "w") as f:
            json.dump(paths_list, f)
        print("Saved", os.path.abspath(dataset["paths_list"]))
    print(labels.summary())
