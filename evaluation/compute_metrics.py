"""
Phoneme alignment evaluation for MFA and BFA methods.

Scoring, in one paragraph: ground-truth boundaries live in the SRT JSON as
segment-relative milliseconds (every transition stored twice, as one phone's end
and the next one's start).  Predicted boundaries are brought into that same frame
— MFA timestamps already are segment-relative, BFA timestamps are clip-absolute
and get the segment start subtracted — then de-duplicated.  Only phoneme onsets
are scored by default; phoneme ends are not independent segmentation points.  The
two boundary lists are aligned once per segment with a Levenshtein DP (monotonic,
one-to-one, see levenshtein_align) and that single alignment is thresholded at
each tolerance to give recall (known -> predicted), precision (predicted ->
known), F1 and the R-value.
"""

import csv
import json
import math
import os
import re
import shutil
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from tqdm import tqdm


# Repo root on the path, so the script runs from any working directory. `parents[1]`, not
# `parents[2]`: this file sits at <repo>/evaluation/, so parents[0] is `evaluation` and parents[1] is
# the repo - parents[2] is the directory ABOVE the repo, which put every sibling checkout on
# the path and `bournemouth_aligner` on none of them.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Where the alignments, manifests and results live - the SAME definitions the sweep writes
# under. See evaluation/layout.py.
from evaluation.layout import (MANIFEST_DIR, PATHS_LISTS, PLOTS_DIR, RESULTS_DIR,
                               alignment_path, resolve_srt)

from bournemouth_aligner.ipamappers.ph66_mapper import (
    phoneme_groups_mapper,
    phoneme_groups_index,
    phoneme_mapped_index,
)




# ─── Metric functions ────────────────────────────────────────────────────────
#
# Everything below scores a *single* monotonic one-to-one alignment between the
# known (ground-truth) boundary times and the predicted boundary times, computed
# with a Levenshtein / edit-distance DP.  That alignment is computed once per
# segment and then thresholded at each tolerance, so recall and precision always
# refer to the same set of matched pairs and no boundary is ever counted twice.


class _PrefixMax:
    """Fenwick tree over predicted-boundary indices holding (max score, argmax)."""

    __slots__ = ("n", "val", "arg")

    def __init__(self, n):
        self.n   = n
        self.val = [0.0] * (n + 1)   # 0.0 == "empty chain", all real scores are > 0
        self.arg = [-1]  * (n + 1)

    def update(self, i, value, arg):
        """Insert `value` at 1-based position i."""
        while i <= self.n:
            if value > self.val[i]:
                self.val[i] = value
                self.arg[i] = arg
            i += i & -i

    def query(self, i):
        """Max over 1-based positions [1..i]; returns (value, arg)."""
        best, best_arg = 0.0, -1
        while i > 0:
            if self.val[i] > best:
                best, best_arg = self.val[i], self.arg[i]
            i -= i & -i
        return best, best_arg


def levenshtein_align(known_boundaries, pred_boundaries, max_dist=100.0, gap_cost=None):
    """
    Levenshtein-style monotonic one-to-one alignment of two sorted boundary lists.

    Both lists are treated as symbol sequences in time order.  A predicted boundary
    may be *substituted* for a known boundary at cost |pred - known| (only allowed
    when that distance is <= max_dist), a known boundary may be *deleted* (a miss)
    and a predicted boundary may be *inserted* (a false alarm), each at cost
    gap_cost.  The alignment minimising the total edit cost is returned.

    Because the total cost is
        sum_matched |d| + gap_cost * (unmatched_known + unmatched_pred)
      = gap_cost * (n_known + n_pred) - sum_matched (2*gap_cost - |d|),
    minimising the cost is the same as picking the non-crossing set of pairs that
    maximises sum(2*gap_cost - |d|).  With gap_cost == max_dist every admissible
    pair carries a weight in [max_dist, 2*max_dist], so the alignment maximises the
    number of matched pairs first and minimises the total timing error second.
    That reduction is solved here in O(P log n_pred) over the P admissible pairs
    (a Fenwick prefix-max over predicted indices), instead of a full O(n*m) matrix.

    Args:
        known_boundaries: Sorted ground-truth boundary times (ms)
        pred_boundaries:  Sorted predicted boundary times (ms)
        max_dist:         Largest |pred - known| that may be matched at all (ms).
                          Must be >= the largest tolerance being scored.
        gap_cost:         Cost of a miss / false alarm (defaults to max_dist)

    Returns:
        List of (known_idx, pred_idx, signed_error) in increasing index order,
        where signed_error = pred - known (positive => prediction is late).
    """
    n, m = len(known_boundaries), len(pred_boundaries)
    if n == 0 or m == 0:
        return []

    gap = float(max_dist) if gap_cost is None else float(gap_cost)

    tree        = _PrefixMax(m)
    pair_score  = []
    pair_parent = []
    pair_known  = []
    pair_pred   = []
    best_score, best_pair = 0.0, -1

    for ki, k in enumerate(known_boundaries):
        lo = bisect_left(pred_boundaries,  k - max_dist)
        hi = bisect_right(pred_boundaries, k + max_dist)
        row_start = len(pair_score)

        for pj in range(lo, hi):
            weight = 2.0 * gap - abs(pred_boundaries[pj] - k)
            # best chain ending strictly before this pair (pred index < pj, known index < ki,
            # the latter guaranteed because this row is inserted only after it is finished)
            prev_score, prev_pair = tree.query(pj)
            score = prev_score + weight

            pair_score.append(score)
            pair_parent.append(prev_pair)
            pair_known.append(ki)
            pair_pred.append(pj)

            if score > best_score:
                best_score, best_pair = score, len(pair_score) - 1

        for idx in range(row_start, len(pair_score)):
            tree.update(pair_pred[idx] + 1, pair_score[idx], idx)

    matches = []
    idx = best_pair
    while idx >= 0:
        ki, pj = pair_known[idx], pair_pred[idx]
        matches.append((ki, pj, pred_boundaries[pj] - known_boundaries[ki]))
        idx = pair_parent[idx]
    matches.reverse()
    return matches


def nearest_distances(known_boundaries, pred_boundaries):
    """
    Nearest-neighbour distance from every known boundary to the closest predicted
    boundary (ms).  Purely descriptive — used for the error histogram, never for
    recall/precision (a nearest-neighbour match is many-to-one and would inflate
    both).  Returns a numpy array.
    """
    if not known_boundaries or not pred_boundaries:
        return np.empty(0, dtype=np.float64)

    known = np.asarray(known_boundaries, dtype=np.float64)
    pred  = np.asarray(pred_boundaries,  dtype=np.float64)
    pos   = np.searchsorted(pred, known)
    left  = np.abs(known - pred[np.clip(pos - 1, 0, len(pred) - 1)])
    right = np.abs(known - pred[np.clip(pos,     0, len(pred) - 1)])
    return np.minimum(left, right)


def best_shift(signed_errors, tolerance_ms, max_shift_ms):
    """
    Shift (ms) that would put the most matched pairs inside +-tolerance_ms.

    The mean/median signed error badly under-estimates a systematic lag, because the
    aligner re-matches around it and the surviving errors stay roughly symmetric.
    Scanning the error histogram for the densest tolerance-wide window measures the
    lag the metrics actually care about.  Returns 0.0 when there is nothing to scan.
    """
    if len(signed_errors) == 0 or max_shift_ms <= 0:
        return 0.0

    limit = int(max_shift_ms)
    edges = np.arange(-limit - 0.5, limit + 1.5, 1.0)
    hist, _ = np.histogram(signed_errors, bins=edges)
    window = np.ones(2 * int(tolerance_ms) + 1)
    counts = np.convolve(hist, window, mode="same")
    return float(np.arange(-limit, limit + 1)[int(np.argmax(counts))])


def r_value(hit_rate, over_segmentation):
    """
    Räsänen's R-value for segmentation quality (1.0 = perfect, penalises the
    over-segmentation that makes plain recall look good).

    Args:
        hit_rate: Recall as a fraction in [0, 1]
        over_segmentation: n_pred / n_known - 1
    """
    r1 = math.sqrt((1.0 - hit_rate) ** 2 + over_segmentation ** 2)
    r2 = (-over_segmentation + hit_rate - 1.0) / math.sqrt(2.0)
    return 1.0 - (abs(r1) + abs(r2)) / 2.0





# ─── Evaluation ──────────────────────────────────────────────────────────────

# Labels that are not phonemes and must never contribute a boundary.
NON_SPEECH_LABELS = {
    "", "-", "sil", "sp", "spn", "sils", "silence", "noise", "pau", "pause",
    "<eps>", "<sil>", "<unk>", "{b_trans}", "{e_trans}", "[bracket]", "[noise]",
}


# Phoneme-group bookkeeping.
# phoneme_groups_mapper: ph66 phoneme_id -> group id;  phoneme_groups_index: name -> group id.
PHONEME_GROUP_NAMES = {gid: name for name, gid in phoneme_groups_index.items()}
# 'SIL' is dropped by _is_speech_phone(), so it can never collect an error.
REPORTED_PHONEME_GROUPS = [name for name in sorted(phoneme_groups_index, key=lambda n: phoneme_groups_index[n])
                           if name != "SIL"]

# BFA labels its phonemes with a ph66 'phoneme_id'; MFA (english_us_arpa) writes
# ARPABET symbols with a stress digit instead.  Mapping ARPABET onto the ph66
# inventory lets both methods reach the same 16 groups by the same rule.
# Stress is stripped first, which is lossless here: AH0 (schwa) and AH1/AH2 (ʌ)
# both land in central_vowels.
ARPABET_TO_PH66 = {
    # Vowels
    "AA": "a:",  "AE": "æ",   "AH": "ʌ",   "AO": "ɔ",   "AX": "ə",
    "AW": "aʊ",  "AY": "aɪ",  "EH": "ɛ",   "ER": "ɚ",   "EY": "eɪ",
    "IH": "ɪ",   "IY": "i:",  "OW": "oʊ",  "OY": "ɔɪ",  "UH": "ʊ",   "UW": "u:",
    # Consonants
    "B": "b",    "CH": "tʃ",  "D": "d",    "DH": "ð",   "DX": "ɾ",
    "F": "f",    "G": "g",    "HH": "h",   "JH": "dʒ",  "K": "k",
    "L": "l",    "M": "m",    "N": "n",    "NG": "ŋ",   "P": "p",
    "R": "ɹ",    "S": "s",    "SH": "ʃ",   "T": "t",    "TH": "θ",
    "V": "v",    "W": "w",    "WH": "w",   "Y": "j",    "Z": "z",   "ZH": "ʒ",
    # Syllabic consonants of the wider ARPABET set (MFA's US dictionary omits them)
    "EL": "l",   "EM": "m",   "EN": "n",   "NX": "n",
}

_STRESS_DIGITS = re.compile(r"\d+$")


def _phone_label(phone):
    """Label of a phoneme_ts entry ('phoneme_label' for BFA, 'phone' for MFA)."""
    for key in ("phoneme_label", "phone", "ipa_label"):
        if phone.get(key):
            return str(phone[key])
    return ""


def _phone_group(phone):
    """
    Phoneme-group id of a phoneme_ts entry, or None when its phoneme cannot be
    identified.  BFA entries carry a ph66 'phoneme_id' directly; MFA entries are
    resolved through ARPABET (stress stripped), and an IPA label is looked up in the
    ph66 inventory as a last resort.
    """
    phoneme_id = phone.get("phoneme_id")
    if phoneme_id is not None:
        return phoneme_groups_mapper.get(int(phoneme_id))

    label = _phone_label(phone).strip()
    if not label:
        return None

    ph66 = ARPABET_TO_PH66.get(_STRESS_DIGITS.sub("", label.upper()))
    if ph66 is None and label in phoneme_mapped_index:
        ph66 = label
    if ph66 is None:
        return None
    return phoneme_groups_mapper.get(phoneme_mapped_index[ph66])


def _is_speech_phone(phone):
    """False for padding / silence / noise entries that must not produce boundaries."""
    if phone.get("phoneme_id") == 0:
        return False
    return _phone_label(phone).strip().lower() not in NON_SPEECH_LABELS


def detect_timestamp_frame(vs_seg, seg_start_ms, seg_duration_ms, slack_ms=100.0):
    """
    Work out whether a segment's predicted timestamps are segment-relative or
    clip-absolute, by checking which interpretation places the phonemes inside the
    segment.  MFA writes segment-relative times (it aligns the extracted segment
    audio); BFA writes clip-absolute times.

    Returns 'relative', 'absolute', or 'unknown'.  'unknown' covers both the empty
    segment and the genuinely ambiguous one (a segment that starts early enough
    that both readings land inside it).
    """
    starts = [float(p["start_ms"]) for p in vs_seg.get("phoneme_ts", []) if _is_speech_phone(p)]
    if not starts:
        return "unknown"

    lo, hi = -slack_ms, seg_duration_ms + slack_ms
    as_relative = sum(1 for s in starts             if lo <= s <= hi)
    as_absolute = sum(1 for s in starts if lo <= s - seg_start_ms <= hi)

    if as_relative == as_absolute:
        return "unknown"
    return "relative" if as_relative > as_absolute else "absolute"


def segment_pred_boundaries(
    vs_seg,
    seg_start_ms,
    timestamps_are_relative,
    offset_start=0.0,
    offset_end=0.0,
    drop_estimated=False,
    unmapped_labels=None,
):
    """
    Collect the predicted boundaries of one segment, expressed in the ground-truth
    time frame (milliseconds relative to the segment start).

    Silence/padding phonemes are dropped, clip-absolute timestamps are shifted back
    to segment-relative, offsets are applied, and duplicate boundary times are
    collapsed (BFA emits runs of 'is_estimated' phonemes that all share one
    timestamp — without the de-duplication they would each count as a prediction).

    Returns:
        (onsets, ends, gaps, n_phones, boundary_groups) — onsets/ends are sorted,
        de-duplicated, integer millisecond boundary times; gaps are the silences
        between consecutive predicted phonemes (ms); boundary_groups maps a boundary
        time to the phoneme group of the phoneme it belongs to.  When phonemes share
        a timestamp the earliest wins, and a time that is one phoneme's end and the
        next one's onset is attributed to the onset, since onsets are what the
        metrics score.

    If a counter is passed as unmapped_labels, the labels of phonemes whose group
    could not be resolved are tallied into it.
    """
    phones = sorted(
        (p for p in vs_seg.get("phoneme_ts", []) if _is_speech_phone(p)),
        key=lambda p: (float(p["start_ms"]), float(p["end_ms"])),
    )
    if drop_estimated:
        phones = [p for p in phones if not p.get("is_estimated")]

    shift = 0.0 if timestamps_are_relative else float(seg_start_ms)

    onsets, ends, gaps = [], [], []
    onset_groups, end_groups = {}, {}
    last_end = None
    for phone in phones:
        start_ms = float(phone["start_ms"]) - shift
        end_ms   = float(phone["end_ms"])   - shift
        if last_end is not None and start_ms > last_end:
            gaps.append(start_ms - last_end)
        last_end = end_ms

        onset = int(round(start_ms - offset_start))
        end   = int(round(end_ms   + offset_end))
        onsets.append(onset)
        ends.append(end)

        group = _phone_group(phone)
        if group is not None:
            onset_groups.setdefault(onset, group)
            end_groups.setdefault(end,     group)
        elif unmapped_labels is not None:
            unmapped_labels[_phone_label(phone)] += 1

    onsets = sorted(set(onsets))
    ends   = sorted(set(ends))
    boundary_groups = {**end_groups, **onset_groups}   # onsets win the overlaps
    return onsets, ends, gaps, len(phones), boundary_groups


def check_method_alignment(
    paths_list_path,
    output_json_path,
    dataset,
    vs_ext=".bfa1.json",
    limit_clips=-1,
    relative_boundaries=True,
    tolerance_ms_list=None,
    offset_start=0,
    offset_end=0,
    use_end_boundaries=False,
    drop_estimated=False,
    measure_group_errors=False,
    align_max_dist=100.0,
    save_distances=True,
    gross_error_ms=None,
):
    """
    Score a predicted phoneme alignment against the ground-truth SRT boundaries.

    Everything is compared in the ground-truth time frame: milliseconds relative to
    the start of the segment.  Ground-truth 'boundaries' are stored that way, MFA
    timestamps are already that way, and clip-absolute BFA timestamps are shifted
    back by the segment start before anything is compared.

    For every segment the known and predicted boundary lists are aligned once with
    a Levenshtein (monotonic, one-to-one) DP — see levenshtein_align — and that
    single alignment is then thresholded at each tolerance:

        hits(t)      = matched pairs whose |pred - known| <= t
        recall(t)    = hits(t) / n_known      (known -> predicted)
        precision(t) = hits(t) / n_predicted  (predicted -> known)
        F1(t)        = harmonic mean of the two

    Because the matching is one-to-one and order-preserving, one predicted boundary
    can never satisfy two ground-truth boundaries (or vice versa), and the recall
    and precision numerators come from the same set of matched pairs.

    Args:
        paths_list_path: JSON file listing {srt, vs} path pairs
        output_json_path: Path to write the results JSON
        vs_ext: File extension identifying the alignment method (e.g. ".mfa1.json", ".bf0.json")
        limit_clips: Maximum number of files to process (-1 = all)
        relative_boundaries: Time frame of the predicted timestamps.
            True   — segment-relative (MFA);
            False  — clip-absolute (BFA), shifted back using the segment start;
            "auto" — decided per file, by majority vote of detect_timestamp_frame()
                     over its segments.
            When True/False is given the frame is still checked and a warning is
            printed if the data disagrees.
        tolerance_ms_list: Tolerance thresholds in ms (default [10, 25, 50])
        offset_start: ms subtracted from every predicted onset (systematic-lateness
            correction; use the reported suggested_offset_start)
        offset_end: ms added to every predicted end boundary (only used when
            use_end_boundaries is True)
        use_end_boundaries: If False (default) only phoneme onsets are scored.  BFA
            end boundaries are not real segmentation points — they nearly double the
            predicted boundary count and wreck precision — so they are ignored.
        drop_estimated: Drop phonemes flagged 'is_estimated' by BFA (degenerate
            fallback timings).  They are de-duplicated in any case.
        measure_group_errors: Break the mean timing error down by phoneme group
            (the 16 groups of ph66_mapper.phoneme_groups_index).  Works for BFA (via
            'phoneme_id') and for MFA (via ARPABET_TO_PH66), so the two are directly
            comparable.  Each matched pair is attributed to the group of the
            *predicted* phoneme, so the means only cover boundaries the aligner
            matched (within align_max_dist).
        align_max_dist: Largest |pred - known| the aligner may match (ms).  Must be
            >= max(tolerance_ms_list); also caps the error statistics.
        save_distances: Store the per-boundary nearest-neighbour distances in the
            JSON (needed by plot_alignment_histogram; a few MB on large corpora).
        gross_error_ms: Radius outside which a boundary counts as a GROSS error -
            a known boundary with no prediction anywhere near it, or a prediction
            with no ground truth anywhere near it.  Defaults to align_max_dist.
            These are reported as a parallel `ex_*` family of metrics that use the
            same hit counts over restricted denominators; see below.

    Returns:
        Dict of computed alignment metrics

    Two families of scores are returned:

        recall/precision/f1_{t}ms      - every boundary counts.  A boundary 4 s from
                                         anything costs exactly what a 99 ms error
                                         costs.  These are the comparable numbers:
                                         MFA's published 78.51 on buckeye is one.
        ex_recall/ex_precision/ex_f1   - gross errors dropped from BOTH denominators.
                                         Measures placement GIVEN detection, and is
                                         deliberately blind to how much was found:
                                         an aligner that places 10% of boundaries
                                         perfectly and misses the rest scores the same
                                         as one that places all of them.  Read it
                                         beside ex_excluded_known_pct, never alone,
                                         and never against a published F1.
    """
    if tolerance_ms_list is None:
        tolerance_ms_list = [10, 25, 50]
    tolerance_ms_list = sorted(tolerance_ms_list)

    if align_max_dist < max(tolerance_ms_list):
        align_max_dist = float(max(tolerance_ms_list))
        print(f"[WARN] align_max_dist raised to {align_max_dist} ms to cover the largest tolerance")

    gross_error_ms = float(align_max_dist if gross_error_ms is None else gross_error_ms)


    if relative_boundaries not in (True, False, "auto"):
        raise ValueError("relative_boundaries must be True, False or 'auto'")

    boundary_mode = "onsets+ends" if use_end_boundaries else "onsets"
    print(f"[INFO] scoring {vs_ext}: boundary_mode={boundary_mode}, relative_boundaries={relative_boundaries}, "
          f"offset_start={offset_start}, offset_end={offset_end}, align_max_dist={align_max_dist}, "
          f"drop_estimated={drop_estimated}")

    with open(paths_list_path, "r") as f:
        paths_list = json.load(f)
    if limit_clips is not None and limit_clips > 0:
        paths_list = paths_list[:limit_clips]

    total_known = 0
    total_pred  = 0
    # Same boundaries, minus the gross errors (no counterpart within gross_error_ms).
    ex_total_known = 0
    ex_total_pred  = 0
    hits        = {t: 0 for t in tolerance_ms_list}
    # onset-only counters (identical to the main ones unless use_end_boundaries)
    total_onset_pred = 0
    onset_hits       = {t: 0 for t in tolerance_ms_list}

    matched_abs_errors    = []
    matched_signed_errors = []
    end_signed_errors     = []
    nn_distances          = []
    all_gaps              = []

    group_error_sum   = defaultdict(float)   # group id -> sum of |error| over matched pairs
    group_error_count = defaultdict(int)
    group_pred_count  = defaultdict(int)     # group id -> predicted boundaries offered
    ungrouped_pred    = 0                    # predicted boundaries with no resolvable group
    unmapped_labels   = defaultdict(int)     # phoneme label -> times it could not be mapped

    processed_files        = 0
    valid_i                = []
    segments_total_count   = 0
    segments_missing_count = 0
    segments_scored_count  = 0
    segments_empty_pred    = 0
    segments_empty_known   = 0
    missing_SRT_files      = 0
    missing_VS_files       = 0
    frame_mismatch_files   = 0
    debug_printed          = False

    for i in tqdm(range(len(paths_list)), desc="Processing alignment files"):
        srt_path = resolve_srt(paths_list[i]['srt'])
        # Derived from the SRT path, not from a 'vs' key: the paths lists do not carry one
        # (the parsers write audio/srt/WRD/PHN and nothing else), so the old
        # `paths_list[i]['vs']` raised KeyError on the first file. `alignment_path` comes from
        # `evaluation/layout.py`, the same definition the sweep writes under - a second copy here
        # would be free to drift, and the symptom of drift is "every extension reports as
        # never aligned".
        vs_path  = alignment_path(dataset, srt_path, vs_ext)

        if not os.path.exists(srt_path):
            missing_SRT_files += 1
            continue
        if not os.path.exists(vs_path):
            missing_VS_files += 1
            continue

        with open(srt_path, "r") as f:
            srt_data = json.load(f)
        with open(vs_path, "r") as f:
            vs_data = json.load(f)

        if not vs_data.get("segments"):
            print(f"VS file has no segments: {vs_path}, skipping.")
            missing_VS_files += 1
            continue

        # Match segments by start time (rounded to ms) so that segments the aligner
        # dropped do not shift the indices of the ones it kept.
        vs_seg_by_start = {int(round(seg["start"] * 1000)): seg for seg in vs_data["segments"]}

        segment_pairs = []
        for srt_seg in srt_data["segments"]:
            segments_total_count += 1
            seg_start_ms = int(round(srt_seg["start"] * 1000))
            vs_seg = vs_seg_by_start.get(seg_start_ms)
            if vs_seg is None:
                segments_missing_count += 1
                continue
            segment_pairs.append((srt_seg, vs_seg, seg_start_ms,
                                  max(0.0, (srt_seg["end"] - srt_seg["start"]) * 1000.0)))

        # Decide the time frame once per file, by majority vote over the segments
        # where the two readings can actually be told apart.
        votes = defaultdict(int)
        for _, vs_seg, seg_start_ms, seg_duration_ms in segment_pairs:
            votes[detect_timestamp_frame(vs_seg, seg_start_ms, seg_duration_ms)] += 1
        if votes["relative"] == votes["absolute"]:
            file_frame = "unknown"
        else:
            file_frame = "relative" if votes["relative"] > votes["absolute"] else "absolute"

        if relative_boundaries == "auto":
            file_is_relative = (file_frame != "absolute")
        else:
            file_is_relative = relative_boundaries
            expected = "relative" if relative_boundaries else "absolute"
            if file_frame not in ("unknown", expected):
                frame_mismatch_files += 1
                print(f"[WARN] {os.path.basename(vs_path)}: timestamps look {file_frame} "
                      f"({votes['relative']} vs {votes['absolute']} segments) but "
                      f"relative_boundaries={relative_boundaries} reads them as {expected}")

        for srt_seg, vs_seg, seg_start_ms, seg_duration_ms in segment_pairs:
            # Ground truth: segment-relative boundary times, each transition stored
            # twice (as an end and as the next start), so de-duplicate.
            known_boundaries = sorted({int(round(b)) for b in srt_seg.get("boundaries", [])})
            if not known_boundaries:
                segments_empty_known += 1
                continue

            onsets, ends, gaps, _, boundary_groups = segment_pred_boundaries(
                vs_seg,
                seg_start_ms,
                timestamps_are_relative=file_is_relative,
                offset_start=offset_start,
                offset_end=offset_end,
                drop_estimated=drop_estimated,
                unmapped_labels=unmapped_labels if measure_group_errors else None,
            )
            all_gaps.extend(gaps)

            pred_boundaries = sorted(set(onsets + ends)) if use_end_boundaries else onsets
            if not pred_boundaries:
                segments_empty_pred += 1
                continue

            if not debug_printed:
                debug_printed = True
                print(f"[DEBUG] first scored segment ({os.path.basename(vs_path)} @ {seg_start_ms} ms, "
                      f"frame={file_frame}, read as {'relative' if file_is_relative else 'absolute'})")
                print(f"[DEBUG]   known: {known_boundaries[:8]}")
                print(f"[DEBUG]   pred:  {pred_boundaries[:8]}")

            matches = levenshtein_align(known_boundaries, pred_boundaries, max_dist=align_max_dist)

            total_known += len(known_boundaries)
            total_pred  += len(pred_boundaries)
            segments_scored_count += 1

            # Gross errors, counted in both directions because they feed different
            # denominators: a known boundary with nothing predicted within
            # gross_error_ms can never be recalled, and a prediction with no ground
            # truth that close can never be correct.  Nearest-neighbour, not the DP
            # matching - a boundary can have a close counterpart and still go unmatched
            # because the one-to-one constraint gave that counterpart to a better pair,
            # and that is a placement failure, not a gross one.
            nn_known = nearest_distances(known_boundaries, pred_boundaries)
            nn_pred  = nearest_distances(pred_boundaries, known_boundaries)
            ex_total_known += int((nn_known <= gross_error_ms).sum())
            ex_total_pred  += int((nn_pred  <= gross_error_ms).sum())

            for _, pj, signed_error in matches:
                abs_error = abs(signed_error)
                matched_abs_errors.append(abs_error)
                matched_signed_errors.append(signed_error)
                for t in tolerance_ms_list:
                    if abs_error <= t:
                        hits[t] += 1
                if measure_group_errors:
                    group = boundary_groups.get(pred_boundaries[pj])
                    if group is not None:
                        group_error_sum[group]   += abs_error
                        group_error_count[group] += 1

            if measure_group_errors:
                for boundary in pred_boundaries:
                    group = boundary_groups.get(boundary)
                    if group is None:
                        ungrouped_pred += 1
                    else:
                        group_pred_count[group] += 1

            if use_end_boundaries:
                # Onset-only variant, scored against the same ground truth.
                total_onset_pred += len(onsets)
                for _, _, signed_error in levenshtein_align(
                    known_boundaries, onsets, max_dist=align_max_dist
                ):
                    for t in tolerance_ms_list:
                        if abs(signed_error) <= t:
                            onset_hits[t] += 1
                for _, _, signed_error in levenshtein_align(
                    known_boundaries, ends, max_dist=align_max_dist
                ):
                    end_signed_errors.append(signed_error)

            if save_distances:
                nn_distances.append(np.minimum(nn_known, 500).astype(np.int32))
            # break # only score the first segment of each file, to avoid double-counting - debugging
        processed_files += 1
        valid_i.append(i)

    if processed_files == 0:
        raise ValueError(
            f"No valid files found (missing SRT: {missing_SRT_files}, missing VS: {missing_VS_files} "
            f"of {len(paths_list)}). Check paths_list, vs_ext and file existence."
        )
    if total_known == 0:
        raise ValueError("No boundaries were scored — every segment was empty or unmatched.")

    if not use_end_boundaries:
        total_onset_pred = total_pred
        onset_hits = dict(hits)

    matched_total  = len(matched_abs_errors)
    abs_errors     = np.asarray(matched_abs_errors,    dtype=np.float64)
    signed_errors  = np.asarray(matched_signed_errors, dtype=np.float64)
    all_distances  = (np.concatenate(nn_distances).tolist() if nn_distances else [])

    over_segmentation = (total_pred / total_known) - 1.0

    # Systematic lag of the predictions, measured at the tightest tolerance.  The
    # suggestion is cumulative — it is the value to pass as offset_start next time,
    # not an extra correction on top of the offsets this run already applied.
    shift_limit = max(0.0, align_max_dist - min(tolerance_ms_list))
    suggested_offset_start = offset_start + best_shift(signed_errors, min(tolerance_ms_list), shift_limit)
    suggested_offset_end   = offset_end   - best_shift(
        np.asarray(end_signed_errors, dtype=np.float64), min(tolerance_ms_list), shift_limit)

    alignment_data = {
        "vs_ext":                    vs_ext,
        "boundary_mode":             boundary_mode,
        "relative_boundaries":       relative_boundaries,
        "align_max_dist":            align_max_dist,
        "offset_start":              offset_start,
        "offset_end":                offset_end,
        "drop_estimated":            drop_estimated,

        "processed_files":           processed_files,
        "missing_srt_files":         missing_SRT_files,
        "missing_vs_files":          missing_VS_files,
        "frame_mismatch_files":      frame_mismatch_files,
        "segments_total":            segments_total_count,
        "segments_missing":          segments_missing_count,
        "segments_scored":           segments_scored_count,
        "segments_empty_pred":       segments_empty_pred,
        "segments_empty_known":      segments_empty_known,
        "valid_indices":             valid_i,

        "total_known_boundaries":    total_known,
        "total_pred_boundaries":     total_pred,
        "total_known_onset_boundaries": total_known,
        "total_onset_pred_boundaries":  total_onset_pred,
        "detection_ratio":           total_pred / total_known,
        "over_segmentation":         over_segmentation,
        "tolerance_ms_list":         tolerance_ms_list,

        # Gross errors excluded (see the docstring before using these).
        "gross_error_ms":            gross_error_ms,
        "ex_total_known_boundaries": ex_total_known,
        "ex_total_pred_boundaries":  ex_total_pred,
        "ex_excluded_known":         total_known - ex_total_known,
        "ex_excluded_pred":          total_pred  - ex_total_pred,
        "ex_excluded_known_pct":     (total_known - ex_total_known) / total_known * 100,
        "ex_excluded_pred_pct":      (total_pred  - ex_total_pred)  / total_pred  * 100,
        "ex_detection_ratio":        (ex_total_pred / ex_total_known) if ex_total_known else 0.0,

        # Statistics over the aligned pairs only (i.e. within align_max_dist).
        "matched_total":             matched_total,
        "matched_pct_of_known":      matched_total / total_known * 100,
        "matched_pct_of_pred":       matched_total / total_pred * 100,
        "missed":                    total_known - matched_total,
        "fakes":                     total_pred - matched_total,
        "missed_pct":                (total_known - matched_total) / total_known * 100,
        "fakes_pct":                 (total_pred - matched_total) / total_pred * 100,
        "abs_error_mean":            float(abs_errors.mean())    if matched_total else 0.0,
        "abs_error_median":          float(np.median(abs_errors)) if matched_total else 0.0,
        "abs_error_std":             float(abs_errors.std())     if matched_total else 0.0,
        "signed_error_mean":         float(signed_errors.mean()) if matched_total else 0.0,
        "signed_error_median":       float(np.median(signed_errors)) if matched_total else 0.0,

        "gap_total":   len(all_gaps),
        "gap_mean":    float(np.mean(all_gaps))   if all_gaps else 0.0,
        "gap_median":  float(np.median(all_gaps)) if all_gaps else 0.0,
        "gap_std":     float(np.std(all_gaps))    if all_gaps else 0.0,

        "suggested_offset_start": round(suggested_offset_start, 1),
        "suggested_offset_end":   round(suggested_offset_end,   1),

        "all_distances":          all_distances,
    }

    # Mean |timing error| per phoneme group, over the matched pairs attributed to
    # the group of the predicted phoneme.
    alignment_data["measure_group_errors"] = measure_group_errors
    if measure_group_errors:
        group_errors = {}
        for name in REPORTED_PHONEME_GROUPS:
            gid   = phoneme_groups_index[name]
            count = group_error_count[gid]
            group_errors[name] = {
                "error_mean": (group_error_sum[gid] / count) if count else None,
                "matched":    count,
                "predicted":  group_pred_count[gid],
            }
            alignment_data[f"group_err_{name}"] = (group_error_sum[gid] / count) if count else None
        alignment_data["group_errors"]   = group_errors
        alignment_data["ungrouped_pred"] = ungrouped_pred
        alignment_data["unmapped_labels"] = dict(sorted(unmapped_labels.items(),
                                                        key=lambda kv: -kv[1])[:20])
        if ungrouped_pred:
            worst = ", ".join(f"{label}×{n}" for label, n in
                              sorted(unmapped_labels.items(), key=lambda kv: -kv[1])[:8])
            print(f"[WARN] {ungrouped_pred} predicted boundaries could not be assigned a phoneme "
                  f"group{' — no group at all was resolved' if not any(group_pred_count.values()) else ''}. "
                  f"Unmapped labels: {worst or 'none (entries carry no label)'}")

    for t in tolerance_ms_list:
        recall    = hits[t] / total_known * 100
        precision = hits[t] / total_pred  * 100
        alignment_data[f"hits_{t}ms"]      = hits[t]
        alignment_data[f"recall_{t}ms"]    = recall
        alignment_data[f"precision_{t}ms"] = precision
        alignment_data[f"f1_{t}ms"]        = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
        alignment_data[f"r_value_{t}ms"]   = r_value(hits[t] / total_known, over_segmentation)
        # kept for backwards compatibility with older result JSONs / CSVs
        alignment_data[f"detected_{t}ms"]  = hits[t]
        alignment_data[f"predicted_{t}ms"] = hits[t]

        # Same hits, denominators restricted to boundaries that have a counterpart
        # within gross_error_ms.  hits[t] can never exceed either restricted total:
        # a pair matched within t <= gross_error_ms leaves both its endpoints inside.
        ex_recall    = hits[t] / ex_total_known * 100 if ex_total_known else 0.0
        ex_precision = hits[t] / ex_total_pred  * 100 if ex_total_pred  else 0.0
        alignment_data[f"ex_recall_{t}ms"]    = ex_recall
        alignment_data[f"ex_precision_{t}ms"] = ex_precision
        alignment_data[f"ex_f1_{t}ms"]        = (2 * ex_recall * ex_precision / (ex_recall + ex_precision)) if (ex_recall + ex_precision) > 0 else 0.0

        onset_recall    = onset_hits[t] / total_known      * 100
        onset_precision = onset_hits[t] / total_onset_pred * 100 if total_onset_pred else 0.0
        alignment_data[f"onset_hits_{t}ms"]      = onset_hits[t]
        alignment_data[f"onset_recall_{t}ms"]    = onset_recall
        alignment_data[f"onset_precision_{t}ms"] = onset_precision
        alignment_data[f"onset_f1_{t}ms"]        = (2 * onset_recall * onset_precision / (onset_recall + onset_precision)) if (onset_recall + onset_precision) > 0 else 0.0
        alignment_data[f"onset_detected_{t}ms"]  = onset_hits[t]
        alignment_data[f"onset_predicted_{t}ms"] = onset_hits[t]

    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(alignment_data, f)

    print(
        f"Alignment data saved to {output_json_path}\n"
        f"  files: {processed_files} scored, {missing_SRT_files} missing SRT, {missing_VS_files} missing VS "
        f"of {len(paths_list)}\n"
        f"  segments: {segments_scored_count} scored, {segments_missing_count} not in the VS file, "
        f"{segments_empty_pred} without predictions, {segments_empty_known} without ground truth "
        f"of {segments_total_count}"
    )
    return alignment_data


# ─── Experiment manifests ────────────────────────────────────────────────────
#
# `process_alignments.py` writes one manifest per `vs_ext` at alignment time, recording the
# checkpoint, snap source/radius, subset and the rest. Reading it back here is what lets a CSV
# row describe itself instead of relying on someone remembering what `.bf12` was.

# MANIFEST_DIR comes from scoring.layout - see the import at the top.


def load_experiment_manifest(vs_ext, manifest_dir=None):
    """Manifest for one `vs_ext`, or None (MFA rows and pre-manifest runs have none)."""
    manifest_dir = manifest_dir or MANIFEST_DIR   # resolved at CALL time, so the module-level
    vs_tag = str(vs_ext).replace(".json", "").lstrip(".")   # MANIFEST_DIR stays authoritative
    path = os.path.join(manifest_dir, f"{vs_tag}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        print(f"[WARN] could not read manifest {path}: {exc}")
        return None


def discover_vs_extensions(manifest_dir=None, vs_range=None):
    """
    Every `.bfN.json` that has a manifest, in N order.

    `vs_range` optionally restricts it (e.g. `range(0, 14)`); without it, everything found is
    scored. Extensions with a manifest but no alignment files on disk are still returned - the
    caller reports them as skipped, which is more useful than silently omitting a run you
    thought you had.
    """
    manifest_dir = manifest_dir or MANIFEST_DIR
    if not os.path.isdir(manifest_dir):
        return []

    found = []
    for name in os.listdir(manifest_dir):
        m = re.fullmatch(r"bf(\d+)\.json", name)
        if not m:
            continue
        n = int(m.group(1))
        if vs_range is not None and n not in vs_range:
            continue
        found.append((n, f".bf{n}.json"))
    return [ext for _, ext in sorted(found)]


# ─── Output / reporting ──────────────────────────────────────────────────────

def append_results_to_csv(csv_path, method, dataset, vs_ext, alignment_data):
    """
    Append one row of alignment results to a CSV file.

    Creates the file with a header row if it does not exist.
    Prints a warning if the existing header does not match the expected columns
    (which happens when tolerance_ms_list changes between runs).

    Columns: method, dataset, vs_ext, boundary_mode, files, boundary counts,
             detection_ratio, recall/precision/F1/R-value @ each tolerance,
             matched-pair error statistics, and the mean error of each phoneme
             group (blank for methods without phoneme ids, e.g. MFA).
    """
    tolerance_ms_list = alignment_data.get("tolerance_ms_list", [20, 40, 60])

    # `notes` and the `cfg_*` columns identify a row: which checkpoint, which decoder
    # settings, how many files. They used to be a hand-typed "-", which is how a 100-file row
    # and a full-corpus row ended up side by side looking comparable. They are read from the
    # manifest `process_alignments.py` wrote AT ALIGNMENT TIME - the only place that actually
    # knows - and fall back to "" for rows produced before manifests existed.
    #
    # `decoder_effective` is preferred over `experiment` for the decoder axes, because those
    # accept None meaning "the checkpoint's trained value": the request is not the run, and
    # without the resolved value a later checkpoint silently reinterprets historical rows.
    manifest = load_experiment_manifest(vs_ext)
    cfg = (manifest or {}).get("experiment", {}) or {}
    eff = (manifest or {}).get("decoder_effective", {}) or {}
    manifest_stats = (manifest or {}).get("stats", {}) or {}
    notes = (manifest or {}).get("label", "-") or "-"

    # Column order is the reading order: identify the run first, then headline scores, then
    # supporting diagnostics. Group errors stay last because they are a detailed breakdown,
    # not the primary comparison columns.
    id_cols    = ["method", "cfg_ckpt", "cfg_model", "dataset", "vs_ext", "notes"]
    cfg_cols   = ["cfg_subset", "cfg_bonus_weight",
                  "cfg_min_phone_frames", "cfg_soft_phone_frames", "cfg_sub_frame_onsets",
                  "cfg_run_at"]
    # `files` sits with the scores, not with the config: two rows are comparable only at the
    # same file count, and a 100-file row reads several f1 points high against a full corpus.
    scale_cols = ["files", "total_known", "total_pred", "detection_ratio",
                  "mean_confidence", "bad_confidence_segments",
                  "bad_confidence_segment_fraction", "segment_duration_avg_s",
                  "segment_duration_min_s", "segment_duration_max_s"]
    f1_cols    = [f"f1_{t}ms" for t in tolerance_ms_list]
    r_value_cols = [f"r_value_{t}ms" for t in tolerance_ms_list]
    other_score_cols = ([f"recall_{t}ms" for t in tolerance_ms_list]
                        + [f"precision_{t}ms" for t in tolerance_ms_list])
    err_cols   = ["abs_err_median", "abs_err_mean", "signed_err_median",
                  "missed_pct", "fakes_pct"]
    # Gross errors excluded. Kept in their own block, AFTER the comparable columns, so a
    # reader cannot mistake ex_f1 for f1 when scanning a row - it is placement given
    # detection, and blind to whatever share of the ground truth it dropped.
    ex_cols    = ([f"ex_{m}_{t}ms" for m in ("f1", "recall", "precision")
                   for t in tolerance_ms_list]
                  + ["ex_excluded_known_pct", "ex_excluded_pred_pct", "gross_error_ms"])
    diag_cols  = ["boundary_mode", "offset_start", "suggested_offset_start"]
    # Always emitted so the header stays stable across methods.
    group_cols = [f"err_{name}" for name in REPORTED_PHONEME_GROUPS]
    all_cols   = (id_cols + cfg_cols + f1_cols + r_value_cols + other_score_cols
                  + scale_cols + err_cols + ex_cols + diag_cols + group_cols)

    ckpt = ((manifest or {}).get("model", {}) or {}).get("cupe_ckpt_path")
    row = {
        "method":           method,
        "notes":            notes,
        "cfg_model":        cfg.get("model", ""),
        "cfg_ckpt":         os.path.basename(ckpt) if ckpt else
                            ((manifest or {}).get("model", {}) or {}).get("model_name", ""),
        "cfg_subset":       cfg.get("subset", ""),
        "cfg_bonus_weight":      eff.get("boundary_bonus_weight",
                                         cfg.get("boundary_bonus_weight", "")),
        "cfg_min_phone_frames":  eff.get("min_phone_frames", cfg.get("min_phone_frames", "")),
        "cfg_soft_phone_frames": eff.get("soft_phone_frames", cfg.get("soft_phone_frames", "")),
        "cfg_sub_frame_onsets":  eff.get("sub_frame_onsets", cfg.get("sub_frame_onsets", "")),
        "cfg_run_at":       (manifest or {}).get("timestamp", ""),
        "dataset":          dataset,
        "vs_ext":           vs_ext,
        "boundary_mode":    alignment_data.get("boundary_mode", "onsets"),
        "files":            alignment_data["processed_files"],
        "total_known":      alignment_data["total_known_boundaries"],
        "total_pred":       alignment_data["total_pred_boundaries"],
        "detection_ratio":  round(alignment_data["detection_ratio"], 4),
        "mean_confidence": alignment_data.get("mean_confidence",
                                              manifest_stats.get("mean_confidence", "")),
        "bad_confidence_segments": alignment_data.get(
            "bad_confidence_segments", manifest_stats.get("bad_confidence_segments", "")),
        "bad_confidence_segment_fraction": alignment_data.get(
            "bad_confidence_segment_fraction",
            manifest_stats.get("bad_confidence_segment_fraction", "")),
        "segment_duration_avg_s": alignment_data.get(
            "segment_duration_avg_s", manifest_stats.get("segment_duration_avg_s", "")),
        "segment_duration_min_s": alignment_data.get(
            "segment_duration_min_s", manifest_stats.get("segment_duration_min_s", "")),
        "segment_duration_max_s": alignment_data.get(
            "segment_duration_max_s", manifest_stats.get("segment_duration_max_s", "")),
        "abs_err_mean":     round(alignment_data.get("abs_error_mean",      0), 2),
        "abs_err_median":   round(alignment_data.get("abs_error_median",    0), 2),
        "signed_err_median": round(alignment_data.get("signed_error_median", 0), 2),
        "missed_pct":       round(alignment_data.get("missed_pct", 0), 2),
        "fakes_pct":        round(alignment_data.get("fakes_pct",  0), 2),
        "offset_start":     alignment_data.get("offset_start", 0),
        "suggested_offset_start": alignment_data.get("suggested_offset_start", 0),
    }
    for t in tolerance_ms_list:
        row[f"recall_{t}ms"]    = round(alignment_data.get(f"recall_{t}ms",    0), 2)
        row[f"precision_{t}ms"] = round(alignment_data.get(f"precision_{t}ms", 0), 2)
        row[f"f1_{t}ms"]        = round(alignment_data.get(f"f1_{t}ms",       0), 2)
        row[f"r_value_{t}ms"]   = round(alignment_data.get(f"r_value_{t}ms",  0), 4)
        for m in ("recall", "precision", "f1"):
            row[f"ex_{m}_{t}ms"] = round(alignment_data[f"ex_{m}_{t}ms"], 2) if f"ex_{m}_{t}ms" in alignment_data else ""

    for col in ("ex_excluded_known_pct", "ex_excluded_pred_pct", "gross_error_ms"):
        row[col] = round(alignment_data[col], 2) if col in alignment_data else ""

    group_errors = alignment_data.get("group_errors", {})
    for name in REPORTED_PHONEME_GROUPS:
        mean = group_errors.get(name, {}).get("error_mean")
        row[f"err_{name}"] = round(mean, 2) if mean is not None else ""

    if os.path.exists(csv_path):
        _migrate_csv_header(csv_path, all_cols)
    else:
        with open(csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=all_cols).writeheader()

    with open(csv_path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=all_cols).writerow(row)

    print(f"Results appended to {csv_path}  [{notes}]")


def _migrate_csv_header(csv_path, all_cols):
    """
    Bring an existing results CSV up to the current column set, preserving every old row.

    Appending a row with MORE fields than the file's header does not fail - `csv.DictWriter`
    happily writes them and every value after the first new column lands under the wrong
    heading, which is worse than an error because the file still parses. So when the header has
    changed, the file is rewritten with the union of columns (old rows get blanks for the new
    ones) after a `.bak` copy.
    """
    with open(csv_path, 'r', newline='') as f:
        rows = list(csv.DictReader(f))
        f.seek(0)
        existing_header = next(csv.reader(f), [])

    if existing_header == all_cols:
        return

    # keep any column the old file had that the new set drops, so nothing is ever lost
    merged = list(all_cols) + [c for c in existing_header if c not in all_cols]

    backup = csv_path + ".bak"
    shutil.copy2(csv_path, backup)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=merged)
        writer.writeheader()
        for old in rows:
            writer.writerow({c: old.get(c, "") for c in merged})

    added = [c for c in all_cols if c not in existing_header]
    print(f"[INFO] CSV header changed; rewrote {csv_path} with {len(merged)} columns "
          f"({len(rows)} existing rows preserved, backup at {backup}).\n"
          f"       New columns: {added}")
    all_cols[:] = merged


def show_alignment_stats(output_json_path, method, dataset, vs_ext, csv_path=None):
    """
    Print alignment metrics from a results JSON produced by check_method_alignment.
    If csv_path is provided, also appends the results to a CSV via append_results_to_csv.
    """
    with open(output_json_path, 'r') as f:
        alignment_data = json.load(f)

    processed_files      = alignment_data["processed_files"]
    total_known          = alignment_data["total_known_boundaries"]
    total_pred           = alignment_data["total_pred_boundaries"]
    total_onset_pred     = alignment_data.get("total_onset_pred_boundaries", total_pred)
    detection_ratio      = alignment_data["detection_ratio"]
    tolerance_ms_list    = alignment_data.get("tolerance_ms_list", [20, 40, 60])
    boundary_mode        = alignment_data.get("boundary_mode", "onsets")

    print("\n" + "=" * 72)
    print(f"Method: {method}, Dataset: {dataset}, Method version: {vs_ext}")
    print(f"\n=== ALIGNMENT METRICS SUMMARY FOR {processed_files} FILES ===")
    print(f"Scored boundaries:            {boundary_mode}"
          f"{'  (phoneme ends ignored)' if boundary_mode == 'onsets' else ''}")
    print(f"Segments scored:              {alignment_data.get('segments_scored', '?')}"
          f" / {alignment_data.get('segments_total', '?')}")
    print(f"Total known boundaries:       {total_known}")
    print(f"Total predicted boundaries:   {total_pred}")
    print(f"Detection ratio (pred/known): {detection_ratio:.3f}   "
          f"(over-segmentation {alignment_data.get('over_segmentation', detection_ratio - 1) * 100:+.1f}%)")

    print(f"\n=== LEVENSHTEIN-ALIGNED SCORES (one-to-one, monotonic) ===")
    print(f"  {'tol':>5}  {'recall':>8}  {'precision':>10}  {'F1':>7}  {'R-val':>7}   hits")
    for t in tolerance_ms_list:
        hits = alignment_data.get(f"hits_{t}ms", alignment_data.get(f"detected_{t}ms", 0))
        print(f"  {t:>3}ms  {alignment_data[f'recall_{t}ms']:>7.2f}%  "
              f"{alignment_data[f'precision_{t}ms']:>9.2f}%  "
              f"{alignment_data.get(f'f1_{t}ms', 0):>6.2f}%  "
              f"{alignment_data.get(f'r_value_{t}ms', 0):>7.3f}   "
              f"{hits}/{total_known} known, {hits}/{total_pred} pred")

    if f"ex_f1_{tolerance_ms_list[0]}ms" in alignment_data:
        gross = alignment_data.get("gross_error_ms", "?")
        print(f"\n=== GROSS ERRORS EXCLUDED (no counterpart within {gross} ms) ===")
        print(f"  Dropped: {alignment_data['ex_excluded_known']} known "
              f"({alignment_data['ex_excluded_known_pct']:.2f}%), "
              f"{alignment_data['ex_excluded_pred']} predicted "
              f"({alignment_data['ex_excluded_pred_pct']:.2f}%)")
        print(f"  Scored on {alignment_data['ex_total_known_boundaries']} known / "
              f"{alignment_data['ex_total_pred_boundaries']} predicted")
        print(f"  {'tol':>5}  {'recall':>8}  {'precision':>10}  {'F1':>7}")
        for t in tolerance_ms_list:
            print(f"  {t:>3}ms  {alignment_data[f'ex_recall_{t}ms']:>7.2f}%  "
                  f"{alignment_data[f'ex_precision_{t}ms']:>9.2f}%  "
                  f"{alignment_data[f'ex_f1_{t}ms']:>6.2f}%")
        print(f"  ^ placement given detection - blind to the {alignment_data['ex_excluded_known_pct']:.1f}% "
              f"of ground truth dropped. NOT comparable to a published F1.")

    if boundary_mode != "onsets" and all(f"onset_recall_{t}ms" in alignment_data for t in tolerance_ms_list):
        print(f"\n=== ONSET-ONLY SCORES (predicted ends excluded) ===")
        for t in tolerance_ms_list:
            print(f"  {t:>3}ms  recall {alignment_data[f'onset_recall_{t}ms']:>7.2f}%  "
                  f"precision {alignment_data[f'onset_precision_{t}ms']:>7.2f}%  "
                  f"({alignment_data.get(f'onset_hits_{t}ms', 0)}/{total_known} known, "
                  f"{alignment_data.get(f'onset_hits_{t}ms', 0)}/{total_onset_pred} pred)")

    if "matched_total" in alignment_data:
        print(f"\n=== MATCHED-PAIR ERROR (pairs within {alignment_data.get('align_max_dist', '?')} ms) ===")
        print(f"  Matched: {alignment_data['matched_total']} "
              f"({alignment_data['matched_pct_of_known']:.1f}% of known, "
              f"{alignment_data['matched_pct_of_pred']:.1f}% of predicted)")
        print(f"  |error|: mean {alignment_data['abs_error_mean']:.1f} ms, "
              f"median {alignment_data['abs_error_median']:.1f} ms, "
              f"std {alignment_data['abs_error_std']:.1f} ms")
        print(f"  signed:  mean {alignment_data['signed_error_mean']:+.1f} ms, "
              f"median {alignment_data['signed_error_median']:+.1f} ms  (+ = prediction is late)")
        print(f"  Missed (unmatched known): {alignment_data['missed_pct']:.2f}%  ({alignment_data['missed']})")
        print(f"  Fakes  (unmatched pred):  {alignment_data['fakes_pct']:.2f}%  ({alignment_data['fakes']})")

    group_errors = alignment_data.get("group_errors") or {}
    rows = sorted(((name, g) for name, g in group_errors.items() if g["matched"]),
                  key=lambda item: item[1]["error_mean"])
    if rows:
        print(f"\n=== MEAN ERROR BY PHONEME GROUP (matched onsets, by predicted phoneme) ===")
        for name, g in rows:
            share = g["matched"] / g["predicted"] * 100 if g["predicted"] else 0.0
            print(f"  {name:<22} {g['error_mean']:>6.1f} ms   "
                  f"{g['matched']:>8} matched / {g['predicted']:>8} predicted ({share:.1f}%)")
        empty = [name for name, g in group_errors.items() if not g["matched"]]
        if empty:
            print(f"  (no matched boundaries: {', '.join(empty)})")

    print(f"\n=== GAP ANALYSIS (silence between consecutive predicted phonemes) ===")
    print(f"  Gaps: {alignment_data['gap_total']}")
    print(f"  Mean: {alignment_data['gap_mean']:.1f} ms  "
          f"Median: {alignment_data.get('gap_median', alignment_data.get('gap_mediamn', 0)):.1f} ms  "
          f"Std: {alignment_data['gap_std']:.1f} ms")

    if "suggested_offset_start" in alignment_data:
        print(f"\n=== SUGGESTED OFFSETS (re-run with these to remove the systematic bias) ===")
        print(f"  offset_start = {alignment_data['suggested_offset_start']:.1f} ms  "
              f"(currently {alignment_data.get('offset_start', 0)}; residual onset bias "
              f"{alignment_data.get('signed_error_median', 0):+.1f} ms)")
        if alignment_data.get("boundary_mode") != "onsets":
            print(f"  offset_end   = {alignment_data['suggested_offset_end']:.1f} ms")

    if alignment_data.get("frame_mismatch_files"):
        print(f"\n[WARN] {alignment_data['frame_mismatch_files']} file(s) had timestamps in a different "
              f"time frame than relative_boundaries={alignment_data.get('relative_boundaries')} assumes.")

    print("\n" + "=" * 72)

    if csv_path is not None:
        append_results_to_csv(csv_path, method, dataset, vs_ext, alignment_data)


def plot_alignment_histogram(
    output_json_path,
    max_distance=500,
    thresholds_ms_list=[10, 25, 50],
    title_suffix="",
    plot_output_path="alignment_distance_histogram.png",
):
    """
    Plot a histogram of boundary distances (known→nearest predicted) from a results JSON.

    Args:
        output_json_path: Results JSON produced by check_method_alignment
        max_distance: Distances above this value (ms) are excluded
        title_suffix: Prepended to the plot title
        plot_output_path: Where to save the PNG
    """
    with open(output_json_path, 'r') as f:
        alignment_data = json.load(f)

    processed_files = alignment_data["processed_files"]
    all_distances   = alignment_data["all_distances"]

    if not all_distances:
        print("No distances found in results JSON.")
        return

    filtered = [d for d in all_distances if 0 < d <= max_distance]
    overall_mean   = np.mean(filtered)
    overall_std    = np.std(filtered)
    overall_median = np.median(filtered)

    bins = np.linspace(1, max_distance, 500)

    plt.figure(figsize=(6, 4))
    plt.hist(filtered, bins=bins.tolist(), alpha=0.7, color='gray', edgecolor='black', linewidth=0.2)
    plt.xlabel('Distance (ms)')
    plt.ylabel('Frequency')
    plt.title(
        f'{title_suffix} (Alignment error)\n'
        f'({len(all_distances)} boundaries from {processed_files} files, '
        f'μ/σ={round(overall_mean)}/{round(overall_std)} ms)'
    )
    plt.xscale('linear')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)

    def format_y(x, pos):
        if x >= 1e6:   return f'{x / 1e6:.1f}M'
        elif x >= 1e3: return f'{x / 1e3:.0f}K'
        return f'{x:.0f}'

    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y))
    thresh_colors = ['purple', 'orange', 'green', 'blue', 'yellow', 'brown']
    for x in range(len(thresholds_ms_list)):
        if x < len(thresh_colors):
            plt.axvline(thresholds_ms_list[x], color=thresh_colors[x], linestyle='--', alpha=0.7,
                        label=f'{thresholds_ms_list[x]}ms threshold')
        else:            plt.axvline(thresholds_ms_list[x], color='gray', linestyle='--', alpha=0.7,
                        label=f'{thresholds_ms_list[x]}ms threshold')
            
    plt.axvline(float(overall_median), color='red', linestyle='-', alpha=0.8,
                label=f'Median ({overall_median:.1f}ms)')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as '{plot_output_path}'")


# ─── Entry point ─────────────────────────────────────────────────────────────

def score_one(vs_ext, method, dataset, paths_list_path, csv_path, *, limit_clips=-1,
              tolerance_ms_list=(10, 25, 50), offset_start=0, offset_end=0,
              use_end_boundaries=False, measure_group_errors=True, make_plot=True):
    """
    Score one `vs_ext` end to end: metrics -> JSON -> histogram -> printed summary -> CSV row.

    Returns the alignment_data dict, or None when there was nothing to score (no alignment
    files for this extension yet).
    """
    predicted_timestamps_are_relative = True if method == "MFA" else False
    output_json_path = os.path.join(PLOTS_DIR, f"{dataset}_{limit_clips}_align{vs_ext}")

    try:
        check_method_alignment(
            paths_list_path, output_json_path, dataset,
            vs_ext=vs_ext,
            limit_clips=limit_clips,
            relative_boundaries=predicted_timestamps_are_relative,
            tolerance_ms_list=list(tolerance_ms_list),
            offset_start=offset_start,
            offset_end=offset_end,
            use_end_boundaries=use_end_boundaries,
            measure_group_errors=measure_group_errors,
            align_max_dist=100.0,
        )
    except ValueError as exc:
        # raised when every file is missing - i.e. this extension was never aligned
        print(f"[SKIP] {vs_ext}: {exc}")
        return None

    if make_plot:
        plot_alignment_histogram(
            output_json_path, max_distance=500, thresholds_ms_list=list(tolerance_ms_list),
            title_suffix=f"{method.upper()} - {dataset.capitalize()} {vs_ext}",
            plot_output_path=output_json_path + ".png",
        )

    show_alignment_stats(output_json_path, method, dataset, vs_ext, csv_path=csv_path)
    with open(output_json_path) as f:
        return json.load(f)


if __name__ == "__main__":
    method    = "BFA"      # BFA or MFA
    dataset   = "buckeye"    # "timit" | "buckeye" - must match EXPERIMENT_GRID's dataset
    limit_clips = -1       # -1 = all clips

    # ---- batch mode ------------------------------------------------------------------
    # Score every `.bfN.json` that `process_alignments.py` has written a manifest for, in N
    # order, appending one self-describing CSV row each. Set SCORE_ALL = False to fall back to
    # the single `vs_ext` below.
    #
    # `limit_clips` is left at -1 even for subset runs: the scorer walks the whole paths list
    # and simply finds no alignment file for the files that were not aligned, so a subset run
    # scores exactly its own files. The subset size is recorded in `cfg_subset`/`notes`, which
    # is what stops a 100-file row being read against a 6300-file one.
    SCORE_ALL = True
    VS_RANGE  = range(3, 4)                # e.g. range(1, 3) to score just those; None = every manifest

    # Time frame of the predicted phoneme timestamps:
    #   MFA aligns the extracted segment audio -> segment-relative  -> True
    #   BFA timestamps are clip-absolute                            -> False
    #   "auto" lets each segment be detected from the data.
    predicted_timestamps_are_relative = True if method == "MFA" else False
    # Per-phoneme-group mean error: BFA resolves groups from 'phoneme_id',
    # MFA from its ARPABET labels, so both are comparable.
    measure_group_errors = True

    tolerance_ms_list = [10, 25, 50]  # Adjust tolerance thresholds here (e.g. [25, 50, 75])

    # BFA phoneme end boundaries are not segmentation points and only add false
    # alarms, so onsets alone are scored. Set True to score ends as well.
    use_end_boundaries = False

    vs_ext = ".bf1.json" if method == "BFA" else ".mfa.json"

    # The SAME paths list `process_alignments.py` aligns from. They have to match: the
    # alignments are keyed by each list's `srt` entry, so scoring a different list finds no
    # files at all and reports the extension as never aligned.
    paths_list_path  = PATHS_LISTS[dataset]
    output_json_path = os.path.join(PLOTS_DIR, f"{dataset}_{limit_clips}", f"align{vs_ext}")
    csv_path         = os.path.join(RESULTS_DIR, f"results_{dataset}.csv")


    #align_with_mfa(paths_list_path,vs_ext=".mfa1.json")

    # offset_start is subtracted from every predicted onset, offset_end is added to
    # every predicted end boundary. Use the suggested_offset_start printed in the
    # summary (a positive value means the predictions are systematically late).
    offset_start = 0
    offset_end   = 0

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if SCORE_ALL and method == "BFA":
        extensions = discover_vs_extensions(vs_range=VS_RANGE)
        if not extensions:
            print(f"No manifests found in {MANIFEST_DIR}. Run process_alignments.py first, "
                  f"or set SCORE_ALL = False to score a single vs_ext.")
        print(f"\nScoring {len(extensions)} extension(s): {', '.join(extensions)}\n")

        summary = []
        for ext in extensions:
            manifest = load_experiment_manifest(ext)
            label = (manifest or {}).get("label", ext)
            print("\n" + "#" * 78)
            print(f"# {ext}   {label}")
            print("#" * 78)

            data = score_one(ext, method, dataset, paths_list_path, csv_path,
                             limit_clips=limit_clips, tolerance_ms_list=tolerance_ms_list,
                             offset_start=offset_start, offset_end=offset_end,
                             use_end_boundaries=use_end_boundaries,
                             measure_group_errors=measure_group_errors)
            if data is None:
                summary.append((ext, label, None, None, None))
            else:
                summary.append((ext, label, data["processed_files"],
                                data.get("f1_25ms", data.get(f"f1_{tolerance_ms_list[1]}ms")),
                                data.get("abs_error_mean")))

        print("\n" + "=" * 88)
        print(f"{'vs_ext':<12} {'label':<34} {'files':>7} {'f1@25':>8} {'abs_err':>9}")
        print("-" * 88)
        for ext, label, files, f1, err in summary:
            if files is None:
                print(f"{ext:<12} {label:<34} {'-':>7} {'skipped (no alignments)':>19}")
            else:
                print(f"{ext:<12} {label:<34} {files:>7} {f1:>8.2f} {err:>9.2f}")
        print("=" * 88)
        print("Rows are only comparable at the SAME `files` count - a 100-file row reads "
              "several f1 points high.")

    else:
        score_one(vs_ext, method, dataset, paths_list_path, csv_path,
                  limit_clips=limit_clips, tolerance_ms_list=tolerance_ms_list,
                  offset_start=offset_start, offset_end=offset_end,
                  use_end_boundaries=use_end_boundaries,
                  measure_group_errors=measure_group_errors, make_plot=True)


#  python evaluation/compute_metrics.py        # run from the repository root
