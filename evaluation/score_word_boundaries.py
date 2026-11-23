"""
Word-edge errors of every experiment on a dataset whose SRTs carry words (cv_* for now).

    python evaluation/score_word_boundaries.py [--only GLOB] [--csv OUT]

compute_metrics.py scores phone onsets. This scores word starts and word ends, 
    paired by WHERE IN THE TEXT they are rather than by time:

  * each word list (reference SRT `words`, the alignment's `words_ts`) is normalised to its
    letters/digits (lower-cased, punctuation and silence words dropped) and concatenated;
  * the two strings are aligned with difflib, so a boundary at character k of the reference is
    paired with the aligner's boundary at the same character - whatever the two word
    segmentations (MFA cuts Mandarin into other words than BFA's segmenter does);
  * a start is paired with a start and an end with an end (they differ where there is a pause).

A reference edge with no counterpart at that character (the aligner joined the two words) is
counted as `unpaired`, not as an error. Reported: paired count, mean / median absolute error,
and the share within 10 / 25 / 50 / 100 ms (of the paired edges), for starts and ends together
and for the clip's first start / last end ("outer", where breath and silence decide) apart.
"""

import argparse
import csv
import difflib
import fnmatch
import json
import os
import sys
import unicodedata

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.layout import (PATHS_LISTS, alignment_path, list_experiments,   # noqa: E402
                               manifest_path, resolve_srt)

SILENCE_WORDS = {"", "sil", "<sil>", "<eps>", "sp", "<unk>", "noise"}
TOLS = (10, 25, 50, 100)


def _norm(word):
    return "".join(c for c in unicodedata.normalize("NFC", str(word)).lower()
                   if unicodedata.category(c)[0] in "LMN")


def _edges(words):
    """[(norm, start_ms, end_ms)] -> (string, {char offset: start}, {char offset: end})."""
    text, starts, ends = "", {}, {}
    for w, s, e in words:
        n = _norm(w)
        if not n or str(w).strip().lower() in SILENCE_WORDS:
            continue
        starts.setdefault(len(text), float(s))
        text += n
        ends[len(text)] = float(e)
    return text, starts, ends


def pair_edges(ref_words, pred_words):
    """Yields (kind, ref_ms, pred_ms or None, outer) for every reference word edge."""
    rt, rs, re_ = _edges(ref_words)
    pt, ps, pe = _edges(pred_words)
    sm = difflib.SequenceMatcher(None, rt, pt, autojunk=False)
    cmap = {}                                   # ref char offset -> pred char offset
    for a, b, size in sm.get_matching_blocks():
        for k in range(size + 1):
            cmap.setdefault(a + k, b + k)
    first, last = min(rs) if rs else None, max(re_) if re_ else None
    for kind, ref, pred in (("start", rs, ps), ("end", re_, pe)):
        for off, t in ref.items():
            q = cmap.get(off)
            outer = off == (first if kind == "start" else last)
            yield kind, t, (pred.get(q) if q is not None else None), outer


def _pred_words(seg):
    return [(w.get("word"), w["start_ms"], w["end_ms"]) for w in seg.get("words_ts", [])]


def score_experiment(experiment, dataset):
    manifest = json.load(open(manifest_path(experiment)))
    rel = manifest["method"] != "BFA"          # MFA & co write segment-relative times
    paths = json.load(open(PATHS_LISTS[dataset]))
    errs = {"all": [], "inner": [], "outer": []}
    unpaired = files = 0
    for entry in paths:
        apath = alignment_path(experiment, dataset, entry["srt"])
        if not os.path.exists(apath):
            continue
        files += 1
        srt = json.load(open(resolve_srt(entry["srt"])))
        pred = json.load(open(apath))
        pred_by_start = {int(round(s["start"] * 1000)): s for s in pred["segments"]}
        for seg in srt["segments"]:
            s0 = int(round(seg["start"] * 1000))
            pseg = pred_by_start.get(s0)
            ref_words = [(w["w"], w["s"] * 1000 - s0, w["e"] * 1000 - s0) for w in seg["words"]]
            if pseg is None:
                unpaired += 2 * len(ref_words)
                continue
            shift = 0 if rel else s0
            pw = [(w, s - shift, e - shift) for w, s, e in _pred_words(pseg)]
            for kind, r, p, outer in pair_edges(ref_words, pw):
                if p is None:
                    unpaired += 1
                    continue
                d = p - r
                errs["all"].append(d)
                errs["outer" if outer else "inner"].append(d)
    out = {"experiment": experiment, "dataset": dataset, "method": manifest["method"],
           "model": manifest["config"]["model"], "files": files, "unpaired": unpaired}
    for part, e in errs.items():
        a = np.abs(np.array(e)) if e else np.array([np.nan])
        out[f"{part}_n"] = len(e)
        out[f"{part}_mean_abs"] = round(float(np.nanmean(a)), 2)
        out[f"{part}_median_abs"] = round(float(np.nanmedian(a)), 2)
        out[f"{part}_signed_median"] = round(float(np.median(e)), 2) if e else float("nan")
        for t in TOLS:
            out[f"{part}_within_{t}ms"] = round(float(np.mean(a <= t)) * 100, 2) if e else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", default="cv_*")
    ap.add_argument("--csv", default="local/results/results_cv_words.csv")
    a = ap.parse_args()
    rows = []
    for exp in list_experiments():
        if not fnmatch.fnmatch(exp, a.only):
            continue
        dataset = exp.split("__")[0]
        if dataset not in PATHS_LISTS:
            continue
        man = json.load(open(manifest_path(exp)))
        if man.get("status") != "done":
            continue
        rows.append(score_experiment(exp, dataset))
    if not rows:
        sys.exit("nothing to score")
    print(f"{'experiment':<58} {'n':>5} {'unp':>4} {'mean':>6} {'med':>5} "
          + " ".join(f"{'<=' + str(t):>6}" for t in TOLS) + f" {'outer_mean':>10}")
    for r in rows:
        print(f"{r['experiment']:<58} {r['all_n']:>5} {r['unpaired']:>4} {r['all_mean_abs']:>6.1f} "
              f"{r['all_median_abs']:>5.1f} "
              + " ".join(f"{r[f'all_within_{t}ms']:>6.1f}" for t in TOLS)
              + f" {r['outer_mean_abs']:>10.1f}")
    out = a.csv if os.path.isabs(a.csv) else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), a.csv)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("->", out)


if __name__ == "__main__":
    main()
