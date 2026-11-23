"""
Common Voice selection (the multilingual review set) -> the SRT trees and paths lists the
evaluation reads, plus the set's existing MFA alignments as ready-made experiments.

    python evaluation/dataset/cv_selection_parser.py              # SRTs + paths lists
    python evaluation/dataset/cv_selection_parser.py --import-mfa # + tmp/alignments/cv_*__mfa-cv
    python evaluation/dataset/cv_selection_parser.py --export-bfa DIR   # copy the BFA runs out
    python evaluation/dataset/cv_selection_parser.py --status done      # only clips marked done

(evaluation/cv_eval.py runs all of this, the alignment and the scoring in one go.)

The corpus is `<root>/<lang>/<clip_id>.wav` (16 kHz mono, one sentence each) with the
hand-reviewed alignment in `<root>/alignments/reviewed/<lang>/<clip_id>.vs.json` (BFA web-UI
format). By default every clip whose review status is not `discarded` is used (`--status done`
takes only the `done` ones). One dataset per language (layout.CV_DATASETS), because the BFA
settings `lang` (the phonemizer) and the model's language group differ per language:

    cv_en en  cv_es es  cv_fr fr (latin)   cv_ru ru (cyrillic)   cv_el el  cv_ko ko (other_alphabetic)
    cv_vi vi (vietnamese)   cv_th th (thai_khmer)   cv_ta ta (brahmic)   cv_ar ar  cv_ur ur (abjad)
    cv_ja ja (japanese)   cv_yue yue  cv_zh zh-CN -> BFA `zh` (cjk)

Each SRT holds one segment covering the whole clip:

    {"labels": {"phoneme_set": "std/latin", "lang": "en", "lang_group": "latin", "source": ...},
     "segments": [{"start": 0.0, "end": 6.69, "text": "...",
                   "boundaries": [283, 350, ...],             # phone onsets (what is scored)
                   "word_boundaries": [283, 736, 736, ...],   # word starts and ends
                   "words": [{"w": "...", "s": 0.283, "e": 0.736, "phones": [...],
                              "boundaries": [350, ...]}]}]}   # phone ends, as in timit_parser

`boundaries` = the onset of every phone that the TEXT produced (target_seq_idx >= 0): the
reviewers' inserted SIL / noise / <unk> tokens are not boundaries. Phones are the std tokens of
BFA's G2P, so an affricate or diphthong is two tokens with two onsets (e.g. zh `ʈ ʂʰ`, en
`a ɪ`); an aligner that writes them as one phone (MFA) cannot hit the second onset, which costs
it recall but not precision.

How the reference was made matters for reading the scores (see the dataset's STATE.md /
MEMORY.md): word edges placed blind from the BFA/MFA midpoint and checked by eye, inner phones
re-placed per word, then every boundary snapped to a spectral-change extremum within 20 ms;
the human ear pass edits it further. The reference changes between phases, which is why
cv_eval.py stamps every row with the date and a fingerprint of the reference it was scored on.

MFA (`--import-mfa`): the set's own MFA run (`alignments/mfa/<lang>/<clip>.json`; pretrained
*_mfa models where MFA has one, otherwise a model trained on Common Voice: el/ta `*_cv`,
ar/ur/yue with BFA-G2P pronunciations; the model is in each file's `acoustic_model`). Written
as harness alignment files with a `done` manifest, so `compute_metrics.py` scores them like any
other MFA run. MFA's `spn` (an OOV word aligned as one span) is relabelled `spn_oov`, because the
scorer drops `spn` as non-speech while its onset is a real word onset here.
"""

import argparse
import csv
import glob
import json
import os
import shutil
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.layout import (DATASET_SRT_ROOTS, PATHS_LISTS, experiment_dir,  # noqa: E402
                               alignment_path, manifest_path, list_experiments, CV_DATASETS)

CV_ROOT = "/work/Datasets/speech_data/common_voice_selection"

MFA_MODEL = "mfa-cv"     # the experiment name of the imported MFA runs: cv_<x>__mfa-cv


def _speech(p):
    return (p.get("target_seq_idx", -1) >= 0 and not p.get("inserted")
            and p.get("phoneme_label") != "SIL")


def _ms(x):
    return int(round(float(x)))


# A reference onset is "moved" when it is more than this far from the nearest onset of BOTH the
# latest BFA alignment (alignments/bfa) and MFA (alignments/mfa): the checked reference disagrees
# with both aligners there. Above the snap step's 20 ms limit, so a boundary that only differs by
# snapping or jitter does not count.
MOVED_THR_MS = 20.0
MFA_NONSPEECH = {"sil", "sp", "", "<eps>"}


def _aligner_onsets(root, lang, clip):
    """Speech-phone onsets (ms) of the latest BFA and the MFA alignments, or None if missing."""
    out = []
    b = f"{root}/alignments/bfa/{lang}/{clip}.json"
    m = f"{root}/alignments/mfa/{lang}/{clip}.json"
    if not (os.path.exists(b) and os.path.exists(m)):
        return None
    bfa = json.load(open(b, encoding="utf-8"))
    out.append(sorted(float(p["start_ms"]) for s in bfa["segments"] for p in s["phoneme_ts"]
                      if _speech(p)))
    mfa = json.load(open(m, encoding="utf-8"))
    out.append(sorted(float(p["start_ms"]) for p in mfa["phones"]
                      if p["phone"] not in MFA_NONSPEECH))
    return out


def _moved(onsets, aligners, thr=MOVED_THR_MS):
    """1 per onset that is > thr from every onset of each aligner."""
    if aligners is None:
        return None
    def far(t, ref):
        return not ref or min(abs(t - r) for r in ref) > thr
    return [int(all(far(t, ref) for ref in aligners)) for t in onsets]


def _use(status, which):
    """`which`: 'all' = every clip but the discarded ones | a status name = only that one."""
    status = status or "todo"
    return status != "discarded" if which == "all" else status == which


def build_dataset(dataset, root=CV_ROOT, status="all"):
    lang = CV_DATASETS[dataset]
    meta = {r["clip_id"]: r for r in csv.DictReader(open(f"{root}/metadata.csv", encoding="utf-8"))
            if r["lang"] == lang}
    srt_root = DATASET_SRT_ROOTS[dataset]
    os.makedirs(srt_root, exist_ok=True)
    paths, skipped = [], []
    for vs in sorted(glob.glob(f"{root}/alignments/reviewed/{lang}/*.vs.json")):
        ref = json.load(open(vs, encoding="utf-8"))
        clip = ref["clip_id"]
        if not _use((ref.get("review") or {}).get("status"), status) or clip not in meta:
            skipped.append(clip)
            continue
        r = meta[clip]
        seg = ref["segments"][0]
        phones = sorted((p for p in seg["phoneme_ts"] if _speech(p)),
                        key=lambda p: (p["start_ms"], p["end_ms"]))
        words = []
        for w in seg["words_ts"]:
            inside = [p for p in phones if w["start_ms"] - 0.5 <= p["start_ms"] < w["end_ms"] - 0.5]
            words.append({"w": w["word"], "s": round(w["start_ms"] / 1000, 4),
                          "e": round(w["end_ms"] / 1000, 4),
                          "phones": [p["phoneme_label"] for p in inside],
                          "boundaries": [_ms(p["end_ms"]) for p in inside]})
        duration = float(ref.get("duration") or r["duration"])
        onsets = sorted({_ms(p["start_ms"]) for p in phones})
        srt = {
            "labels": {"phoneme_set": ref.get("phoneme_set"), "lang": r["bfa_lang"],
                       "lang_group": r["lang_group"], "source": os.path.relpath(vs, root),
                       "review_status": (ref.get("review") or {}).get("status"),
                       "review_rev": (ref.get("review") or {}).get("rev")},
            "text": r["sentence"],
            "segments": [{
                "start": 0.0, "end": duration, "text": r["sentence"],
                "boundaries": onsets,
                # 1 = not at either initial alignment (see MOVED_THR_MS); null if one is missing
                "moved": _moved(onsets, _aligner_onsets(root, lang, clip)),
                "moved_thr_ms": MOVED_THR_MS,
                "word_boundaries": sorted({_ms(w["start_ms"]) for w in seg["words_ts"]}
                                          | {_ms(w["end_ms"]) for w in seg["words_ts"]}),
                "words": words,
            }],
        }
        srt_path = os.path.join(srt_root, f"{clip}.srt.json")
        with open(srt_path, "w", encoding="utf-8") as f:
            json.dump(srt, f, ensure_ascii=False)
        paths.append({"audio": "~/" + r["audio"], "srt": os.path.relpath(srt_path, REPO_ROOT),
                      "lang": r["bfa_lang"], "duration": duration, "clip_id": clip,
                      "spkr": r["speaker"], "sex": r["gender"]})
    os.makedirs(os.path.dirname(PATHS_LISTS[dataset]), exist_ok=True)
    with open(PATHS_LISTS[dataset], "w", encoding="utf-8") as f:
        json.dump(paths, f, indent=1, ensure_ascii=False)
    # SRTs of clips that left the set (discarded since the last build) must not linger
    keep = {f"{e['clip_id']}.srt.json" for e in paths}
    for old in os.listdir(srt_root):
        if old.endswith(".srt.json") and old not in keep:
            os.remove(os.path.join(srt_root, old))
    print(f"{dataset}: {len(paths)} clips -> {srt_root}  ({len(skipped)} left out, status != {status})")
    return paths


# Cantonese MFA was trained on BFA-G2P pronunciations from before the voiced onsets, so it writes
# an onset affricate as two phones. Where the reference word has d͡z there, they are merged
# (first start, second end) at import; alignments/mfa itself is not changed.
MFA_MERGE_LANGS = {"yue"}
_AFFRICATES = {("t", "s"): "d͡z", ("t", "ɕ"): "d͡ʑ", ("ʈ", "ʂ"): "d͡ʐ"}
_VOICED = {"p": "b", "t": "d", "k": "ɡ"}


def _merge_word(phones, ref):
    """One MFA word's phones with onset affricates merged, or None if it does not fit `ref`."""
    out, b = [], 0
    a = 0
    while a < len(phones):
        if b >= len(ref):
            return None
        t = phones[a]["phone"]
        pair = (t, phones[a + 1]["phone"]) if a + 1 < len(phones) else None
        if pair in _AFFRICATES and _AFFRICATES[pair] == ref[b]:
            out.append({**phones[a], "phone": ref[b], "end_ms": phones[a + 1]["end_ms"]})
            a += 2
        elif t == ref[b] or _VOICED.get(t) == ref[b]:
            out.append(phones[a])
            a += 1
        else:
            return None
        b += 1
    return out if b == len(ref) else None


def merge_mfa_affricates(m, ref_seg):
    """(phones, merged, words that did not fit): MFA words are matched to the reference in order."""
    tokens = {}
    for w, t in zip(ref_seg["word_num"], ref_seg["ipa"]):
        tokens.setdefault(w, []).append(t)
    ref_words = [(w, tokens.get(i, [])) for i, w in enumerate(ref_seg["words"]) if w != "<SIL>"]
    by_idx = {}
    for p in m["phones"]:
        by_idx.setdefault(p.get("word_idx"), []).append(p)
    mfa_words = [(i, w["word"]) for i, w in enumerate(m["words"]) if w["word"] not in ("<eps>", "")]
    if [w for _, w in mfa_words] != [w for w, _ in ref_words]:
        return m["phones"], 0, len(mfa_words)
    replaced, merged, unfit = {}, 0, 0
    for (i, _), (_, ref) in zip(mfa_words, ref_words):
        ph = by_idx.get(i, [])
        new = _merge_word(ph, ref)
        if new is None:
            unfit += 1
            continue
        merged += len(ph) - len(new)
        replaced[i] = new
    out, seen = [], set()
    for p in m["phones"]:
        i = p.get("word_idx")
        if i not in replaced:
            out.append(p)
        elif i not in seen:
            seen.add(i)
            out.extend(replaced[i])
    return out, merged, unfit


def import_mfa(dataset, paths, root=CV_ROOT):
    lang = CV_DATASETS[dataset]
    exp = f"{dataset}__{MFA_MODEL}"
    run_dir = experiment_dir(exp)
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    n, model = 0, {}
    n_merged = n_unfit = 0
    for entry in paths:
        src = f"{root}/alignments/mfa/{lang}/{entry['clip_id']}.json"
        if not os.path.exists(src):
            continue
        m = json.load(open(src, encoding="utf-8"))
        model = {"acoustic_model": m.get("acoustic_model"), "dictionary": m.get("dictionary"),
                 "g2p_model": m.get("g2p_model"), "aligner": m.get("aligner")}
        source = m["phones"]
        if lang in MFA_MERGE_LANGS:
            ref = json.load(open(f"{root}/alignments/reviewed/{lang}/{entry['clip_id']}.vs.json",
                                 encoding="utf-8"))
            source, k, u = merge_mfa_affricates(m, ref["segments"][0])
            n_merged += k
            n_unfit += u
        phones = [{"phone": "spn_oov" if p["phone"] == "spn" else p["phone"],
                   "start_ms": p["start_ms"], "end_ms": p["end_ms"]} for p in source]
        words = [{"word": w["word"], "start_ms": w["start_ms"], "end_ms": w["end_ms"]}
                 for w in m["words"] if w["word"] not in ("<eps>", "")]
        out = {"segments": [{"start": 0.0, "end": entry["duration"], "text": m.get("text"),
                             "phoneme_ts": phones, "words_ts": words}]}
        with open(alignment_path(exp, dataset, entry["srt"]), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        n += 1
    manifest = {
        "experiment": exp, "method": "MFA", "dataset": dataset, "status": "done",
        "label": f"{MFA_MODEL} (imported from {os.path.relpath(root)}/alignments/mfa/{lang})",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"dataset": dataset, "subset": -1, "split": "all", "model": MFA_MODEL,
                   "method": "MFA", "id": exp, "imported": True},
        "model": model,
        "stats": {"files": n},
    }
    with open(manifest_path(exp), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    extra = f", {n_merged} onset affricates merged, {n_unfit} words not matched" \
        if lang in MFA_MERGE_LANGS else ""
    print(f"{exp}: {n} MFA alignments imported ({model.get('acoustic_model')}){extra}")


def export_bfa(out_root, root=CV_ROOT):
    """Copy every BFA run of the cv_* datasets to OUT/<model>/<lang>/<clip_id>.vs.json."""
    for exp in list_experiments():
        dataset, model = exp.split("__")[:2]
        if dataset not in CV_DATASETS or model == MFA_MODEL:
            continue
        man = json.load(open(manifest_path(exp)))
        if man.get("method") != "BFA" or man.get("status") != "done":
            continue
        dst = os.path.join(out_root, model, CV_DATASETS[dataset])
        os.makedirs(dst, exist_ok=True)
        n = 0
        for f in glob.glob(os.path.join(experiment_dir(exp), "*.json")):
            name = os.path.basename(f)
            if name.startswith("_"):
                continue
            shutil.copy2(f, os.path.join(dst, name[:-len(".json")] + ".vs.json"))
            n += 1
        shutil.copy2(manifest_path(exp), os.path.join(dst, "_manifest.json"))
        print(f"{exp}: {n} files -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--import-mfa", action="store_true")
    ap.add_argument("--export-bfa", metavar="DIR", default=None)
    ap.add_argument("--status", default="all",
                    help="all (default: every clip but the discarded ones) | done | todo | ...")
    ap.add_argument("datasets", nargs="*", default=list(CV_DATASETS))
    a = ap.parse_args()
    if a.export_bfa:
        export_bfa(a.export_bfa)
        sys.exit(0)
    for ds in a.datasets:
        p = build_dataset(ds, status=a.status)
        if a.import_mfa:
            import_mfa(ds, p)
