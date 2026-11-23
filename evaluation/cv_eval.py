"""
Evaluate every aligner on the Common Voice selection (14 languages) against the CURRENT
reference, and append dated rows to one results CSV per language.

    python evaluation/cv_eval.py                         # all 14 languages, today's date
    python evaluation/cv_eval.py --phase "after human pass"   # label the rows
    python evaluation/cv_eval.py en ur zh-CN             # some languages only
    python evaluation/cv_eval.py --gpus 0                # one GPU (default: 0,1 in parallel)
    python evaluation/cv_eval.py --status done           # only clips the reviewers marked done

The reference is `<CV_ROOT>/alignments/reviewed/<lang>/*.vs.json`, which changes from phase to
phase (snap, human ear pass, ...). The aligners' outputs do not, so a rerun is cheap:

  1. cv_selection_parser: rebuild the SRTs + paths lists from the reference (every clip that is
     not `discarded`), and import the set's MFA run as cv_<x>__mfa-cv.
  2. process_alignments.py on the `cv_*` experiments of evaluate.yaml, split over the GPUs
     (one shell per GPU, CUDA_VISIBLE_DEVICES). Cached runs are skipped; a cached run that lacks
     some of the current clips is resumed, so only those are aligned.
  3. Score each experiment with compute_metrics (phone onsets) and score_word_boundaries (word
     starts/ends) and APPEND one row per experiment to
         <CV_ROOT>/_results/results_<lang>.csv
     Each row starts with eval_date / phase / ref_fingerprint, so every evaluation ever run
     stays in the same file and rows of different phases sit side by side. Rerunning on the same
     date replaces that date's rows (per experiment) instead of adding duplicates.

ref_fingerprint is a hash of the reference SRTs (clips, texts, boundaries): two rows with the
same fingerprint were scored on the identical reference.
"""

import argparse
import csv
import datetime
import hashlib
import io
import json
import os
import subprocess
import sys
import contextlib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.layout import (CV_DATASETS, alignment_path, manifest_path,  # noqa: E402
                               resolve_srt)
from evaluation.dataset import cv_selection_parser as cvp                    # noqa: E402

LANG_TO_DS = {lang: ds for ds, lang in CV_DATASETS.items()}
LEAD_COLS = ["eval_date", "eval_time", "phase", "lang", "experiment", "method", "aligner",
             "ref_clips", "ref_done", "ref_todo", "ref_fingerprint", "aligned_files"]


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------------------------
# 1. the reference
# ---------------------------------------------------------------------------------------------
def build_reference(datasets, status):
    info = {}
    for ds in datasets:
        paths = cvp.build_dataset(ds, status=status)
        cvp.import_mfa(ds, paths)
        h = hashlib.sha1()
        counts = {}
        for e in sorted(paths, key=lambda e: e["clip_id"]):
            srt = json.load(open(resolve_srt(e["srt"]), encoding="utf-8"))
            st = srt["labels"].get("review_status") or "todo"
            counts[st] = counts.get(st, 0) + 1
            seg = srt["segments"][0]
            h.update(json.dumps([e["clip_id"], seg["text"], seg["boundaries"],
                                 seg["word_boundaries"]], ensure_ascii=False).encode())
        info[ds] = {"ref_clips": len(paths), "ref_done": counts.get("done", 0),
                    "ref_todo": counts.get("todo", 0), "ref_fingerprint": h.hexdigest()[:12],
                    "clips": paths}
    return info


# ---------------------------------------------------------------------------------------------
# 2. the alignments
# ---------------------------------------------------------------------------------------------
def yaml_experiments(datasets):
    import evaluation.process_alignments as pa
    pa.configure(None)
    return [e for e in pa.build_experiments() if e["dataset"] in datasets]


def reopen_incomplete(exps, ref):
    """A finished BFA run that lacks clips of the current reference is set back to `running`,
    so process_alignments resumes it and aligns only the missing ones."""
    todo = []
    for e in exps:
        clips = ref[e["dataset"]]["clips"]
        missing = [c for c in clips
                   if not os.path.exists(alignment_path(e["id"], e["dataset"], c["srt"]))]
        mp = manifest_path(e["id"])
        done = os.path.exists(mp) and json.load(open(mp)).get("status") == "done"
        if done and missing:
            m = json.load(open(mp))
            m["status"] = "running"
            json.dump(m, open(mp, "w"), indent=2, ensure_ascii=False)
            log(f"  {e['id']}: {len(missing)} clip(s) not aligned yet, reopening the run")
        if missing or not done:
            todo.append(e)
    return todo


def align(exps, gpus, logdir):
    if not exps:
        log("All alignments are cached.")
        return
    globs = ",".join(e["id"] for e in exps)
    n = min(len(gpus), len(exps))
    procs = []
    for k in range(n):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpus[k]))
        lf = open(os.path.join(logdir, f"align_gpu{gpus[k]}.log"), "w")
        cmd = [sys.executable, "evaluation/process_alignments.py"] + \
              ([str(n), str(k)] if n > 1 else []) + ["--only", globs]
        log(f"  GPU {gpus[k]}: {' '.join(cmd[1:3])} ... -> {lf.name}")
        procs.append((subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=lf,
                                       stderr=subprocess.STDOUT), lf))
    bad = 0
    for p, lf in procs:
        bad |= p.wait()
        lf.close()
    if bad:
        log("  [WARN] an alignment shell failed; see the logs above. Scoring what is there.")


# ---------------------------------------------------------------------------------------------
# 3. scoring
# ---------------------------------------------------------------------------------------------
def score(manifest, scoring, tmpdir):
    """compute_metrics' CSV row + score_word_boundaries' numbers, as one dict."""
    from evaluation import compute_metrics as cm
    from evaluation import score_word_boundaries as swb
    tmp_csv = os.path.join(tmpdir, manifest["experiment"] + ".csv")
    if os.path.exists(tmp_csv):
        os.remove(tmp_csv)
    os.makedirs(cm.PLOTS_DIR, exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        data = cm.score_one(manifest, scoring, tmp_csv)
    with open(os.path.join(tmpdir, manifest["experiment"] + ".log"), "w") as f:
        f.write(buf.getvalue())
    if data is None or not os.path.exists(tmp_csv):
        return None
    row = next(csv.DictReader(open(tmp_csv)))
    w = swb.score_experiment(manifest["experiment"], manifest["dataset"])
    for k, v in w.items():
        if k not in ("experiment", "dataset", "method", "model"):
            row[f"word_{k}"] = v
    row.update(score_moved(manifest, scoring))
    return row


def score_moved(manifest, scoring):
    """
    Reference phone onsets that disagree with both aligners (SRT `moved` = 1: more than moved_thr_ms from both the
    latest BFA and MFA alignments), scored with the harness's own onset extraction and one-to-one matching over
    ALL onsets of the clip. Only recall and error are defined for a subset of the reference
    (a prediction cannot be assigned to "moved" or not), so no precision / F1.
    """
    import numpy as np
    from evaluation import compute_metrics as cm
    exp, ds = manifest["experiment"], manifest["dataset"]
    rel = manifest["method"] in ("MFA", "MAPS", "CTC", "CHARSIU")
    tols = list(scoring["tolerance_ms"])
    n, hits, errs, thr = 0, {t: 0 for t in tols}, [], ""
    for e in json.load(open(cvp.PATHS_LISTS[ds])):
        srt = json.load(open(resolve_srt(e["srt"]), encoding="utf-8"))
        ap = alignment_path(exp, ds, e["srt"])
        for seg in srt["segments"]:
            mv = seg.get("moved")
            if not mv:
                continue
            thr = seg.get("moved_thr_ms", "")
            idx = {i for i, f in enumerate(mv) if f}
            n += len(idx)
            if not idx or not os.path.exists(ap):
                continue
            pred = json.load(open(ap, encoding="utf-8"))
            pseg = next((s for s in pred["segments"]
                         if int(round(s["start"] * 1000)) == int(round(seg["start"] * 1000))), None)
            if pseg is None:
                continue
            onsets = cm.segment_pred_boundaries(
                pseg, seg["start"] * 1000, timestamps_are_relative=rel,
                offset_start=scoring["offset_start"], phoneme_set=pred.get("phoneme_set"),
                merge_two_token_phones=scoring["merge_two_token_phones"])[0]
            if not onsets:
                continue
            for ki, _, d in cm.levenshtein_align(seg["boundaries"], onsets,
                                                 max_dist=float(scoring["align_max_dist"])):
                if ki in idx:
                    errs.append(abs(d))
                    for t in tols:
                        hits[t] += abs(d) <= t
    out = {"moved_thr_ms": thr, "moved_n": n,
           "moved_abs_err_mean": round(float(np.mean(errs)), 2) if errs else "",
           "moved_abs_err_median": round(float(np.median(errs)), 2) if errs else ""}
    for t in tols:
        out[f"moved_recall_{t}ms"] = round(100 * hits[t] / n, 2) if n else ""
    return out


def append_rows(csv_path, rows, key=("eval_date", "experiment")):
    """Add rows to the language's CSV; a row with the same date + experiment is replaced."""
    old, cols = [], []
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            cols, old = list(r.fieldnames or []), list(r)
    new_keys = {tuple(x[k] for k in key) for x in rows}
    kept = [x for x in old if tuple(x.get(k, "") for k in key) not in new_keys]
    for x in rows:
        for c in x:
            if c not in cols:
                cols.append(c)
    cols = LEAD_COLS + [c for c in cols if c not in LEAD_COLS]
    tmp = csv_path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for x in kept + rows:
            w.writerow({c: x.get(c, "") for c in cols})
    os.replace(tmp, csv_path)
    return len(old) - len(kept)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("langs", nargs="*", default=list(LANG_TO_DS),
                    help=f"languages (default all): {' '.join(LANG_TO_DS)}")
    ap.add_argument("--phase", default="", help="free text stored in every row, e.g. 'human_start'")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--gpus", default="0,1", help="comma-separated GPU ids, one alignment shell each")
    ap.add_argument("--status", default="all", help="all (not discarded) | done | todo")
    ap.add_argument("--no-align", action="store_true", help="score the cached alignments only")
    ap.add_argument("--out", default=os.path.join(cvp.CV_ROOT, "_results"))
    a = ap.parse_args()

    bad = [l for l in a.langs if l not in LANG_TO_DS]
    if bad:
        ap.error(f"unknown language(s) {bad}; known: {list(LANG_TO_DS)}")
    datasets = [LANG_TO_DS[l] for l in a.langs]
    tmpdir = os.path.join(REPO_ROOT, "tmp", "cv_eval", a.date)
    os.makedirs(tmpdir, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)

    log(f"== 1. reference ({cvp.CV_ROOT}/alignments/reviewed, status={a.status})")
    ref = build_reference(datasets, a.status)

    exps = yaml_experiments(set(datasets))
    log(f"== 2. alignments: {len(exps)} BFA experiment(s) in evaluate.yaml")
    todo = reopen_incomplete(exps, ref)
    if a.no_align:
        log(f"  --no-align: {len(todo)} run(s) incomplete, scored as they are")
    else:
        align(todo, [g for g in a.gpus.split(",") if g != ""], tmpdir)

    from evaluation.config import load_config
    scoring = load_config()["scoring"]
    now = datetime.datetime.now().strftime("%H:%M:%S")
    log(f"== 3. scoring -> {a.out}/results_<lang>.csv  (eval_date {a.date})")
    summary = []
    for ds in datasets:
        lang = CV_DATASETS[ds]
        names = [e["id"] for e in exps if e["dataset"] == ds] + [f"{ds}__{cvp.MFA_MODEL}"]
        rows = []
        for name in names:
            mp = manifest_path(name)
            if not os.path.exists(mp):
                log(f"  {name}: no alignments, left out")
                continue
            man = json.load(open(mp))
            if man.get("status") != "done":
                log(f"  {name}: alignment not finished ({man.get('status')}), left out")
                continue
            row = score(man, scoring, tmpdir)
            if row is None:
                log(f"  {name}: nothing to score")
                continue
            n_al = sum(1 for c in ref[ds]["clips"]
                       if os.path.exists(alignment_path(name, ds, c["srt"])))
            row.update({"eval_date": a.date, "eval_time": now, "phase": a.phase, "lang": lang,
                        "aligner": man["config"]["model"], "aligned_files": n_al,
                        **{k: v for k, v in ref[ds].items() if k != "clips"}})
            rows.append(row)
            summary.append(row)
        if rows:
            out = os.path.join(a.out, f"results_{lang}.csv")
            replaced = append_rows(out, rows)
            log(f"  {lang:6} {len(rows)} row(s) -> {out}" + (f" ({replaced} replaced)" if replaced else ""))

    log("")
    log(f"{'lang':6} {'aligner':24} {'clips':>5} {'f1@25':>6} {'abs_err':>7} "
        f"{'word_med':>8} {'word<=25':>8} {'moved_n':>7} {'mv_rec25':>8} {'mv_err':>6}")
    f = lambda v, w, d=1: f"{float(v):{w}.{d}f}" if v != "" else f"{'-':>{w}}"
    for r in summary:
        log(f"{r['lang']:6} {r['aligner']:24} {r['files']:>5} {f(r['f1_25ms'], 6, 2)} "
            f"{f(r['abs_err_mean'], 7, 2)} {f(r['word_all_median_abs'], 8)} "
            f"{f(r['word_all_within_25ms'], 8)} {r['moved_n']:>7} {f(r['moved_recall_25ms'], 8)} "
            f"{f(r['moved_abs_err_mean'], 6)}")
    log(f"\nDetail per experiment: {tmpdir}/<experiment>.log")


if __name__ == "__main__":
    main()
