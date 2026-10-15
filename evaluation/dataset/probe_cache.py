"""
Build a boundary-probe cache for the p4mbfa training repository, from TIMIT or Buckeye.

    python evaluation/dataset/probe_cache.py timit
    python evaluation/dataset/probe_cache.py buckeye --n 400
    python evaluation/dataset/probe_cache.py timit --like /path/to/timit_probe_400_latin.json

## What a probe cache is

During training, `bfa_models/p4mbfa/timit_probe.py` force-aligns a few hundred held-out clips
every epoch and scores the onsets against hand-labelled boundaries. To do that it needs each
clip's phoneme sequence IN THE MODEL'S OWN LABEL SPACE - a language group's local tokens - and
it must not phonemize at training time (slow, and it would put the G2P inside the training
loop). So the sequences are produced once, with the same G2P that labelled the training data,
and frozen in a file:

    {"dataset": "buckeye", "lang": "en", "lang_group": "latin", "n_tokens": 208,
     "gold_fingerprint": "...", "source": "...",
     "clips": [{"clip_id", "audio", "start", "end", "duration", "text",
                "local_ph", "gold_phg", "n_folded", "boundaries_ms"}]}

This is the format `p4mbfa/timit_probe_prep.py` writes, with two additions: `dataset`, and
`start` / `end` that are no longer always the whole file.

## Why it is built here

The ground truth - text and hand-labelled boundaries - is what `timit_parser.py` and
`buckeye_parser.py` already wrote into this repository's SRT trees for the evaluation
pipeline, and the phonemizer is the standard_g2p vendored here, the one the released
checkpoints are checked against. One builder over both gives the training-time probe and the
offline evaluation the same clips, the same boundaries and the same phonemes.

## TIMIT and Buckeye are not the same kind of entry

A TIMIT clip is one sentence in one file: `start` is 0 and `end` is the file's length. A
Buckeye "clip" is a SEGMENT of a ~7 minute interview, so `start` / `end` are offsets into
`audio` and the probe has to cut the segment out before aligning it. `boundaries_ms` is
relative to `start` in both.

Buckeye is spontaneous speech: the transcript has what was said, the G2P has the dictionary
pronunciation, and the two disagree more than on read speech (reductions, deletions). That is
the point of scoring on it, but it is also why segments are filtered - a segment whose
phoneme rate is implausible is a transcript that does not describe its audio, and it would
measure the corpus rather than the model.

English only: both corpora are, so the language group is the one English belongs to.
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.layout import PATHS_LISTS, resolve_srt                         # noqa: E402

LANG = "en"
SEED = 42
DEFAULT_N = 400

# The corpus roots the parsers replaced with `~` in the paths lists. A probe cache is read by
# another repository, so it stores real, absolute audio paths.
DATASET_AUDIO_ROOTS = {
    "timit": "/my_datasets/speech_data_small/timit/raw/TIMIT",
    "buckeye": "/my_datasets/speech_data_small/buckeye_corpus",
}

# Buckeye segment filters. The duration ceiling is p4mbfa's `duration_max` (the probe pads to
# it); the rate window is the collator's `tokens_per_sec_min/max`, so the probe is not scored
# on a kind of segment training itself throws away.
BUCKEYE_MIN_DUR, BUCKEYE_MAX_DUR = 2.0, 20.0
TOKENS_PER_SEC_MIN, TOKENS_PER_SEC_MAX = 2.0, 20.0
MIN_BOUNDARIES = 5


def default_out(dataset, n, lang_group):
    return os.path.join(REPO_ROOT, "tmp", "data", f"{dataset}_probe_{n}_{lang_group}.json")


def _audio_path(dataset, stored):
    root = DATASET_AUDIO_ROOTS.get(dataset)
    return root + stored[1:] if (root and stored.startswith("~")) else stored


# --- candidates ------------------------------------------------------------------------------

def timit_candidates(paths_list, test_only=True, seed=SEED):
    """
    One candidate per TIMIT clip, in a seeded order.

    `test_only` keeps TIMIT's own test split: the probe selects checkpoints, so it should not
    be read off training-adjacent material.
    """
    entries = [e for e in paths_list if not test_only or "/TEST/" in e["audio"]]
    entries.sort(key=lambda e: e["srt"])              # deterministic order before sampling
    random.Random(seed).shuffle(entries)
    for entry in entries:
        srt_path = resolve_srt(entry["srt"])
        if not os.path.exists(srt_path):
            continue
        with open(srt_path) as f:
            segments = json.load(f).get("segments") or []
        if len(segments) != 1:
            continue
        seg = segments[0]
        yield {"clip_id": str(entry.get("clip_id", os.path.basename(srt_path))),
               "audio": _audio_path("timit", entry["audio"]),
               "start": float(seg["start"]), "end": float(seg["end"]),
               "text": (seg.get("text") or "").strip(),
               "boundaries_ms": sorted({int(round(b)) for b in seg.get("boundaries", [])})}


def buckeye_candidates(paths_list, seed=SEED, min_dur=BUCKEYE_MIN_DUR, max_dur=BUCKEYE_MAX_DUR):
    """
    Buckeye segments, interleaved across speakers in a seeded order.

    Round-robin over speakers rather than a flat shuffle: Buckeye has 40 speakers with very
    different amounts of usable speech, and a flat sample of a few hundred segments can leave
    some of them out entirely.
    """
    rng = random.Random(seed)
    by_speaker = defaultdict(list)
    for entry in sorted(paths_list, key=lambda e: e["srt"]):
        srt_path = resolve_srt(entry["srt"])
        if not os.path.exists(srt_path):
            continue
        stem = os.path.basename(srt_path).split(".")[0]
        with open(srt_path) as f:
            segments = json.load(f).get("segments") or []
        for seg in segments:
            duration = float(seg["end"]) - float(seg["start"])
            boundaries = sorted({int(round(b)) for b in seg.get("boundaries", [])})
            text = (seg.get("text") or "").strip()
            if not (min_dur <= duration <= max_dur) or len(boundaries) < MIN_BOUNDARIES or not text:
                continue
            by_speaker[entry.get("spkr", stem[:3])].append({
                "clip_id": f"{stem}@{float(seg['start']):.3f}",
                "audio": _audio_path("buckeye", entry["audio"]),
                "start": float(seg["start"]), "end": float(seg["end"]),
                "text": text, "boundaries_ms": boundaries})
    queues = []
    for speaker in sorted(by_speaker, key=str):
        segments = by_speaker[speaker]
        rng.shuffle(segments)
        queues.append(segments)
    while any(queues):
        for queue in queues:
            if queue:
                yield queue.pop()


def like_candidates(cache_path):
    """The clips of an existing cache, as they are: same ids, audio, text and boundaries."""
    with open(cache_path) as f:
        cache = json.load(f)
    for clip in (cache["clips"] if isinstance(cache, dict) else cache):
        yield {k: clip[k] for k in ("clip_id", "audio", "start", "end", "text", "boundaries_ms")}


# --- the builder -----------------------------------------------------------------------------

def build_cache(dataset, n_clips=DEFAULT_N, out_path=None, like=None, device=None, seed=SEED,
                rate_filter=None):
    """
    Phonemize candidates until `n_clips` are kept, and write the cache.

    Args:
        dataset: 'timit' or 'buckeye'.
        like: path of an existing cache whose clips to reuse exactly (ids, audio, boundaries),
            re-phonemizing only. Use it to move a clip set to new token lists without
            changing what is scored.
        rate_filter: drop segments outside the collator's tokens-per-second window. Default:
            on for Buckeye, off for TIMIT (read speech, and it would change the clip set the
            existing TIMIT probe uses).
    Returns:
        The cache dict that was written.
    """
    import torch
    from bournemouth_aligner import presets
    from bournemouth_aligner.phonemizers.std import StdPhonemizer

    if rate_filter is None:
        rate_filter = (dataset == "buckeye") and like is None
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    lang, _ = presets.normalize_std_lang(LANG)
    phonemizer = StdPhonemizer(lang, device=device)
    inventory = phonemizer.inventory
    lang_group = phonemizer.lang_group

    if like:
        candidates, source = like_candidates(like), f"clips of {like}"
    else:
        with open(PATHS_LISTS[dataset]) as f:
            paths_list = json.load(f)
        candidates = (timit_candidates(paths_list, seed=seed) if dataset == "timit"
                      else buckeye_candidates(paths_list, seed=seed))
        source = f"{os.path.relpath(PATHS_LISTS[dataset], REPO_ROOT)}, seed {seed}"

    clips, skipped = [], defaultdict(int)
    for cand in candidates:
        if len(clips) >= n_clips and not like:
            break
        if not cand["boundaries_ms"] or not cand["text"]:
            skipped["no boundaries or text"] += 1
            continue
        if not os.path.exists(cand["audio"]):
            skipped["audio missing"] += 1
            continue
        d = phonemizer.phonemize_sentence(cand["text"])
        duration = cand["end"] - cand["start"]
        if not d["local_ph"]:
            skipped["no phonemes"] += 1
            continue
        # A segment the G2P could not fully label would be scored on the labelling, not on
        # the alignment.
        if d["gold_unmapped"] and not like:
            skipped["unmapped phonemes"] += 1
            continue
        if rate_filter and not (TOKENS_PER_SEC_MIN <= len(d["local_ph"]) / duration
                                <= TOKENS_PER_SEC_MAX):
            skipped["phoneme rate outside the training window"] += 1
            continue
        clips.append({"clip_id": cand["clip_id"], "audio": cand["audio"],
                      "start": cand["start"], "end": cand["end"], "duration": duration,
                      "text": cand["text"], "local_ph": d["local_ph"],
                      "gold_phg": d["gold_phg"], "n_folded": d["n_folded"],
                      "boundaries_ms": cand["boundaries_ms"]})

    out_path = out_path or default_out(dataset, len(clips), lang_group)
    cache = {"dataset": dataset, "lang": lang, "lang_group": lang_group,
             "n_tokens": inventory.n_classes, "gold_fingerprint": inventory.fingerprint,
             "source": source, "clips": clips}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cache, f, ensure_ascii=False)

    n_bnd = sum(len(c["boundaries_ms"]) for c in clips)
    n_ph = sum(len(c["local_ph"]) for c in clips)
    dur = sum(c["duration"] for c in clips)
    print(f"Wrote {len(clips)} {dataset} clips to {out_path}")
    print(f"  {lang_group} / {inventory.n_classes} tokens, gold fingerprint {inventory.fingerprint}")
    print(f"  {n_bnd} ground-truth boundaries, {n_ph} target tokens (ratio "
          f"{n_ph / max(n_bnd, 1):.3f}), {dur / 60:.1f} min of audio, "
          f"{sum(c['n_folded'] for c in clips)} tokens folded to <unk>")
    if dataset == "buckeye":
        print(f"  {len({c['clip_id'].split('@')[0][:3] for c in clips})} speakers; segment "
              f"length {min(c['duration'] for c in clips):.1f}-"
              f"{max(c['duration'] for c in clips):.1f} s")
    if skipped:
        print("  skipped: " + ", ".join(f"{n} {why}" for why, n in sorted(skipped.items())))
    if clips:
        print(f"  e.g. {clips[0]['text'][:60]!r} -> "
              f"{[inventory.id_to_label[i] for i in clips[0]['local_ph'][:14]]}")
    return cache


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset", choices=["timit", "buckeye"])
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="clips to keep")
    ap.add_argument("--out", default=None, help="default: tmp/data/<dataset>_probe_<n>_<group>.json")
    ap.add_argument("--like", default=None, help="reuse the clips of this existing cache")
    ap.add_argument("--device", default=None, help="device for the G2P model")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    build_cache(args.dataset, n_clips=args.n, out_path=args.out, like=args.like,
                device=args.device, seed=args.seed)
