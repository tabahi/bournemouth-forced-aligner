# File: tests/test_buckeye.py
"""
Quick boundary-accuracy experiments on randomly sampled Buckeye SEGMENTS (not whole clips).

Aligning a whole Buckeye clip takes minutes; a random sample of 10-50 segments takes about a
minute and is enough to compare decoder settings. Scoring mirrors
`UnsupervisedPhones/modules/aligner/alignment_tests.py`:

  - known boundaries: the SRT segment's `boundaries` (segment-relative ms, de-duplicated),
    which include silence/noise transitions and both segment edges;
  - predicted boundaries: ONSETS of speech phonemes only, made segment-relative;
  - one monotonic one-to-one Levenshtein alignment per segment, thresholded at 10/25/50 ms.

On top of that it reports f1@25 by position within the segment (deciles), because the
failure being chased is positional: a bad start that the rest of the segment never recovers
from.

Usage:
    python tests/test_buckeye.py --n 40 --seed 0
    python tests/test_buckeye.py --n 40 --min-dur 8 --cfg '{"enforce_all_targets": false}'
    python tests/test_buckeye.py --clip tmp/buckeye_samples/s0103a    # every segment of a clip
    python tests/test_buckeye.py --n 40 --corrupt extra_ends --cfg '{"enforce_all_targets": false}'

`--corrupt` makes the transcript unreliable while scoring against the untouched ground truth:
  extra_ends   2 words from the neighbouring segments glued onto each end of the text (what a
               segment clipped too tightly looks like: the text says more than the audio)
  missing      2 random interior words deleted from the text (speech with no transcript)

Not collected by pytest (no `test_` functions): it needs the corpus and a model download.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from bisect import bisect_left, bisect_right
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

PATHS_LIST = "/mnt/intelpa-2/rehman/codes/speech_data_manager/paths_cache/paths_list_buckeye_all.json"
LOCAL_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'model3a_bin',
                          'm_ue03c_epoch=81_step=39442_val_timit_lyr_fusion_f1_25ms=64.2429.ckpt')
TOLERANCES = (10, 25, 50)
MAX_DIST = 100.0
NON_SPEECH_LABELS = {
    "", "-", "sil", "sp", "spn", "sils", "silence", "noise", "pau", "pause",
    "<eps>", "<sil>", "<unk>", "{b_trans}", "{e_trans}", "[bracket]", "[noise]",
}


# --- scoring (same definitions as alignment_tests.py) -------------------------------------

def levenshtein_align(known, pred, max_dist=MAX_DIST):
    """Monotonic one-to-one matching maximising matches, then minimising total error.
    O(n*m) DP; segments are small enough that the Fenwick version is not needed."""
    n, m = len(known), len(pred)
    if n == 0 or m == 0:
        return []
    gap = float(max_dist)
    # score[i][j]: best weight using known[:i], pred[:j]
    score = np.zeros((n + 1, m + 1))
    move = np.zeros((n + 1, m + 1), dtype=np.int8)   # 0 diag, 1 up (skip known), 2 left
    for i in range(1, n + 1):
        k = known[i - 1]
        lo = bisect_left(pred, k - max_dist)
        hi = bisect_right(pred, k + max_dist)
        for j in range(1, m + 1):
            best, mv = score[i - 1, j], 1
            if score[i, j - 1] > best:
                best, mv = score[i, j - 1], 2
            if lo <= j - 1 < hi:
                w = score[i - 1, j - 1] + 2.0 * gap - abs(pred[j - 1] - k)
                if w > best:
                    best, mv = w, 0
            score[i, j], move[i, j] = best, mv
    out, i, j = [], n, m
    while i > 0 and j > 0:
        mv = move[i, j]
        if mv == 0:
            out.append((i - 1, j - 1, pred[j - 1] - known[i - 1]))
            i, j = i - 1, j - 1
        elif mv == 1:
            i -= 1
        else:
            j -= 1
    return out[::-1]


def r_value(hit_rate, over_seg):
    r1 = math.sqrt((1.0 - hit_rate) ** 2 + over_seg ** 2)
    r2 = (-over_seg + hit_rate - 1.0) / math.sqrt(2.0)
    return 1.0 - (abs(r1) + abs(r2)) / 2.0


def pred_onsets(seg_out, seg_start_ms, index_to_plabel):
    onsets = set()
    for p in seg_out.get("phoneme_ts", []):
        if p.get("phoneme_id") == 0:
            continue
        if str(p.get("phoneme_label", "")).strip().lower() in NON_SPEECH_LABELS:
            continue
        onsets.add(int(round(float(p["start_ms"]) - seg_start_ms)))
    return sorted(onsets)


class Scorer:
    def __init__(self):
        self.known = 0
        self.pred = 0
        self.hits = {t: 0 for t in TOLERANCES}
        self.abs_err = []
        self.signed = []
        # position deciles, by the KNOWN boundary's relative position in the segment
        self.dec_known = np.zeros(10)
        self.dec_hit25 = np.zeros(10)
        self.per_segment = []

    def add(self, known, pred, dur_ms, tag=""):
        matches = levenshtein_align(known, pred)
        self.known += len(known)
        self.pred += len(pred)
        seg_hits = 0
        matched_known = {}
        for ki, _pj, e in matches:
            a = abs(e)
            self.abs_err.append(a)
            self.signed.append(e)
            for t in TOLERANCES:
                if a <= t:
                    self.hits[t] += 1
            if a <= 25:
                seg_hits += 1
            matched_known[ki] = a
        for ki, k in enumerate(known):
            d = min(9, int(10 * k / max(dur_ms, 1)))
            self.dec_known[d] += 1
            if matched_known.get(ki, 1e9) <= 25:
                self.dec_hit25[d] += 1
        rec = seg_hits / max(1, len(known))
        prec = seg_hits / max(1, len(pred))
        f1 = 2 * rec * prec / (rec + prec) if rec + prec else 0.0
        self.per_segment.append((tag, dur_ms / 1000.0, len(known), f1))
        return f1

    def summary(self):
        out = {"known": self.known, "pred": self.pred}
        os_ = self.pred / max(1, self.known) - 1.0
        for t in TOLERANCES:
            r = self.hits[t] / max(1, self.known)
            p = self.hits[t] / max(1, self.pred)
            out[f"f1_{t}"] = 100 * (2 * r * p / (r + p) if r + p else 0.0)
            out[f"r_{t}"] = r_value(r, os_)
        out["abs_err_median"] = float(np.median(self.abs_err)) if self.abs_err else float('nan')
        out["signed_median"] = float(np.median(self.signed)) if self.signed else float('nan')
        out["unmatched_known_pct"] = 100 * (1 - len(self.abs_err) / max(1, self.known))
        out["deciles_recall25"] = [round(100 * h / k, 1) if k else None
                                   for h, k in zip(self.dec_hit25, self.dec_known)]
        return out


# --- sampling -------------------------------------------------------------------------------

def phn_boundaries(phn_path, seg_start):
    """TIMIT: reference boundaries from a .PHN file (sample indices at 16 kHz), segment-relative ms."""
    b = set()
    for line in open(phn_path):
        a, e, _ = line.split()
        b.update((int(a) / 16.0 - seg_start * 1000, int(e) / 16.0 - seg_start * 1000))
    return sorted(b)


def sample_segments(n, seed, min_dur, max_dur, clip=None, paths_list=PATHS_LIST):
    """Returns {audio_path: [segment, ...]}."""
    if clip is not None:
        srt = json.load(open(clip + ".srt.json"))
        segs = [s for s in srt["segments"] if s.get("boundaries") and min_dur <= s["end"] - s["start"] <= max_dur]
        return {clip + ".wav": segs}
    paths = json.load(open(paths_list))
    rng = random.Random(seed)
    rng.shuffle(paths)
    pool = []
    for x in paths:
        if not (os.path.exists(x["audio"]) and os.path.exists(x["srt"])):
            continue
        for s in json.load(open(x["srt"]))["segments"]:
            if not s.get("boundaries") and x.get("PHN") and os.path.exists(x["PHN"]):
                s["boundaries"] = phn_boundaries(x["PHN"], s["start"])
            if s.get("boundaries") and min_dur <= s["end"] - s["start"] <= max_dur and s["text"].strip():
                pool.append((x["audio"], s))
        if len(pool) > 20 * n:
            break
    picked = rng.sample(pool, min(n, len(pool)))
    by_clip = defaultdict(list)
    for a, s in picked:
        by_clip[a].append(s)
    for a in by_clip:
        by_clip[a].sort(key=lambda s: s["start"])
    return by_clip


def corrupt_text(segs, i, mode, rng):
    words = segs[i]["text"].split()
    if mode == "extra_ends":
        prev = segs[i - 1]["text"].split() if i > 0 else []
        nxt = segs[i + 1]["text"].split() if i + 1 < len(segs) else []
        head = prev[-2:] if prev else ["and", "then"]
        tail = nxt[:2] if nxt else ["you", "know"]
        return " ".join(head + words + tail)
    if mode == "missing" and len(words) > 4:
        drop = set(rng.sample(range(1, len(words) - 1), 2))
        return " ".join(w for k, w in enumerate(words) if k not in drop)
    return segs[i]["text"]


def run(args):
    import torch
    from bournemouth_aligner import PhonemeTimestampAligner

    cfg = dict(preset="en-us", duration_max=30, device="cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(LOCAL_CKPT):
        # The en-us preset's model3a checkpoint is not on the hub yet; use the local copy.
        cfg["cupe_ckpt_path"] = LOCAL_CKPT
    cfg.update(json.loads(args.cfg))
    aligner = PhonemeTimestampAligner(**cfg)
    aligner.warn_level = 0

    by_clip = sample_segments(args.n, args.seed, args.min_dur, args.max_dur, args.clip, args.paths)
    scorer = Scorer()
    t0 = time.time()
    n_segs = 0
    for audio_path, segs in by_clip.items():
        wav = aligner.load_audio(audio_path)
        rng = random.Random(args.seed)
        if args.corrupt:
            # neighbours come from the full SRT, not the sample, so the glued words are real
            full = sorted(json.load(open(audio_path.replace(".wav", ".srt.json")))["segments"]
                          if os.path.exists(audio_path.replace(".wav", ".srt.json")) else
                          json.load(open(next(x["srt"] for x in json.load(open(args.paths))
                                              if x["audio"] == audio_path)))["segments"],
                          key=lambda s: s["start"])
            index = {round(s["start"], 3): k for k, s in enumerate(full)}
            texts = [corrupt_text(full, index[round(s["start"], 3)], args.corrupt, rng) for s in segs]
        else:
            texts = [s["text"] for s in segs]
        clean = [{"start": s["start"], "end": s["end"], "text": t} for s, t in zip(segs, texts)]
        res = aligner.process_segments([{"segments": clean}], [wav], batch_size=8)[0]
        out_by_start = {int(round(o["start"] * 1000)): o for o in res["segments"]}
        for s in segs:
            st = int(round(s["start"] * 1000))
            dur_ms = (s["end"] - s["start"]) * 1000
            known = sorted({int(round(b)) for b in s["boundaries"]})
            o = out_by_start.get(st)
            pred = pred_onsets(o, st, aligner.phonemizer.index_to_plabel) if o else []
            f1 = scorer.add(known, pred, dur_ms, tag=f"{os.path.basename(audio_path)}@{s['start']:.2f}")
            n_segs += 1
            if args.verbose:
                print(f"  {os.path.basename(audio_path)} {s['start']:8.2f}-{s['end']:8.2f}s "
                      f"known={len(known):3d} pred={len(pred):3d} f1@25={100*f1:5.1f}")
    elapsed = time.time() - t0
    summ = scorer.summary()
    summ.update(segments=n_segs, seconds=round(elapsed, 1), cfg=cfg)
    print(json.dumps(summ))
    if args.worst:
        for tag, dur, nk, f1 in sorted(scorer.per_segment, key=lambda r: r[3])[:args.worst]:
            print(f"  worst: {tag} dur={dur:.1f}s known={nk} f1@25={100*f1:.1f}")
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-dur", type=float, default=0.5)
    ap.add_argument("--max-dur", type=float, default=30.0)
    ap.add_argument("--paths", default=PATHS_LIST, help="paths list JSON (e.g. a TIMIT one; "
                    "boundaries then come from each clip's .PHN when the SRT has none)")
    ap.add_argument("--clip", default=None, help="path prefix, e.g. tmp/buckeye_samples/s0103a")
    ap.add_argument("--cfg", default="{}", help="JSON kwargs for PhonemeTimestampAligner")
    ap.add_argument("--corrupt", choices=["extra_ends", "missing"], default=None,
                    help="make the transcript unreliable; scoring still uses the true boundaries")
    ap.add_argument("--worst", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    run(ap.parse_args())
