'''
Labelling what the text does not cover: silence or noise.

The decoder places the phones of the text and, with the 2.1 defaults, may leave optional SILs
visited at the clip's edges and at word boundaries. Those come back as GAPS between phones -
`core._framestamps_from_alignment` drops every position that is not in the caller's sequence -
and this module fills each gap with `SIL` and `noise` entries, so a segment's `phoneme_ts` covers
it from start to end.

Why from the audio and not from the model: every released model has a `noise` class, but none
was trained on noise (breath, lip and tongue clicks, background) and none ever predicts it - its
posterior stays under 0.004 on every frame of the hand-labelled Common Voice en clips. What the
model calls such a stretch is mostly SIL, sometimes a phone. Energy above the recording's own
floor is what separates breath from silence there: noise sits ~35 dB below the speech level,
silence ~50-55 dB.

The rules (all times in ms, energy in dB over 10 ms windows at a 1 ms hop):
  - floor = the segment's `floor_pct` percentile; a frame is loud when its energy, averaged
    over `smooth_ms`, is more than `noise_db` above the floor;
  - loud runs shorter than `min_noise_ms` are silence, unless they are a CLICK: a short burst
    (raw energy `click_db` over the floor) of at most `click_max_ms`, with at least
    `click_quiet_before_ms` of quiet before it and `click_quiet_after_ms` after it;
  - quiet stretches shorter than `bridge_ms` between two noise runs join them;
  - an internal gap shorter than `min_gap_ms` is given to the neighbouring phone, and one
    shorter than `stop_gap_ms` next to a stop is that stop's closure (the right-hand stop if
    there is one, else the left): the silence before a burst is part of the stop;
  - no entry is shorter than `min_token_ms`.
'''

import numpy as np

STOP_CHARS = set("pbtdkgɡqcɟʔʈɖɓɗ")

DEFAULTS = dict(
    floor_pct=10.0, noise_db=12.0, smooth_ms=30, min_noise_ms=40, click_db=20.0,
    click_max_ms=100, click_quiet_before_ms=50, click_quiet_after_ms=20, bridge_ms=40,
    min_gap_ms=40, stop_gap_ms=120, min_token_ms=10,
)


def energy_db_per_ms(wav, sample_rate=16000, win_ms=10):
    '''dB per 1 ms hop of a mono waveform; entry k is the window centred on k + 0.5 ms.'''
    x = np.asarray(wav, dtype=np.float64).reshape(-1)
    hop = sample_rate // 1000
    win = sample_rate * win_ms // 1000
    n_ms = int(len(x) // hop)
    c = np.concatenate([[0.0], np.cumsum(x * x)])
    centre = np.arange(n_ms) * hop + hop // 2
    a = np.clip(centre - win // 2, 0, len(x))
    b = np.clip(centre + win // 2, 0, len(x))
    e = (c[b] - c[a]) / np.maximum(b - a, 1)
    return 10.0 * np.log10(e + 1e-10)


def is_stop(label):
    import unicodedata
    base = unicodedata.normalize("NFD", label or "")
    return bool(base) and base[0] in STOP_CHARS


def _runs(mask):
    out, start = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append([start, i])
            start = None
    return out


def classify_span(energy, floor, p):
    '''
    Noise runs inside one non-speech span: a list of [start, end) offsets in ms.

    `energy` is the span's per-ms dB; `floor` the segment's floor.
    '''
    n = len(energy)
    if n == 0:
        return []
    k = max(1, int(p['smooth_ms']))
    pad = np.pad(energy, (k // 2, k - 1 - k // 2), mode='edge')
    smooth = np.convolve(pad, np.ones(k) / k, mode='valid')
    loud = smooth > floor + p['noise_db']
    noise = [r for r in _runs(loud) if r[1] - r[0] >= p['min_noise_ms']]

    # Clicks: short, loud, isolated. Found on the raw energy, which a click's few ms survive.
    quiet = energy < floor + p['noise_db']
    for a, b in _runs(energy > floor + p['click_db']):
        if b - a > p['click_max_ms']:
            continue
        before = 0
        i = a - 1
        while i >= 0 and quiet[i]:
            before += 1
            i -= 1
        after = 0
        i = b
        while i < n and quiet[i]:
            after += 1
            i += 1
        # The span's edge counts as quiet only if the span reaches the segment's edge, which
        # the caller signals by padding; inside the clip a phone sits there, so be strict.
        if before >= p['click_quiet_before_ms'] and after >= p['click_quiet_after_ms']:
            noise.append([max(0, a - 2), min(n, b + 2)])

    if not noise:
        return []
    noise.sort()
    merged = [noise[0]]
    for a, b in noise[1:]:
        if a - merged[-1][1] < p['bridge_ms']:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def click_frames(rel_db, n_valid, ms_per_frame=5.0, params=None, max_level_db=10.0):
    '''
    Frames of isolated clicks, on the model's frame grid: [T] bool.

    `rel_db` is each frame's level in dB relative to the clip's speech level
    (`core.frame_energy_db`). A click is a burst more than `click_db` over the clip's floor that
    lasts at most `click_max_ms`, has at least `click_quiet_before_ms` of quiet before it and
    `click_quiet_after_ms` after it (the clip's edges count as quiet), and peaks at least
    `max_level_db` below the speech level - a short word said on its own is as loud as speech;
    lip and tongue clicks are not.
    '''
    p = dict(DEFAULTS)
    p.update(params or {})
    x = np.asarray(rel_db, dtype=np.float64)
    out = np.zeros(len(x), dtype=bool)
    n = int(min(n_valid, len(x)))
    if n < 3:
        return out
    v = x[:n]
    floor = float(np.percentile(v, p['floor_pct']))
    quiet = v < floor + p['noise_db']
    f = float(ms_per_frame)
    for a, b in _runs(v > floor + p['click_db']):
        if (b - a) * f > p['click_max_ms'] or v[a:b].max() > -float(max_level_db):
            continue
        i = a - 1
        while i >= 0 and quiet[i]:
            i -= 1
        before = (a - 1 - i) * f if i >= 0 else float('inf')
        i = b
        while i < n and quiet[i]:
            i += 1
        after = (i - b) * f if i < n else float('inf')
        if before >= p['click_quiet_before_ms'] and after >= p['click_quiet_after_ms']:
            out[max(0, a - 1):min(n, b + 1)] = True
    return out


def _entry(label, class_id, start_ms, end_ms):
    return {"phoneme_id": class_id, "phoneme_label": label, "ipa_label": label,
            "start_ms": float(start_ms), "end_ms": float(end_ms), "confidence": None,
            "target_seq_idx": -1, "index": -1, "inserted": True, "auto": "nonspeech"}


def fill_nonspeech(phoneme_ts, seg_start_ms, seg_end_ms, energy, sil_id, noise_id,
                   params=None, layers=()):
    '''
    `phoneme_ts` with every gap filled by SIL / noise entries, in time order.

    Args:
        phoneme_ts: the segment's entries (the text's phones), any order.
        seg_start_ms, seg_end_ms: the segment's bounds; `energy[0]` is the ms at seg_start_ms.
        energy: per-ms dB over the segment (`energy_db_per_ms`).
        sil_id, noise_id: the class ids to stamp on the new entries (noise_id may be None).
        layers: extra per-entry keys to set to 0 on new entries (`tone`, `stress`, `length`).
    '''
    p = dict(DEFAULTS)
    p.update(params or {})
    phones = sorted((dict(x) for x in phoneme_ts), key=lambda x: x["start_ms"])
    if not phones:
        return phones
    n_ms = len(energy)
    floor = float(np.percentile(energy, p['floor_pct'])) if n_ms else 0.0

    def ms(t):
        return int(round(t - seg_start_ms))

    # --- internal gaps that are too short to be a pause, or that are a stop's closure
    for i in range(len(phones) - 1):
        left, right = phones[i], phones[i + 1]
        gap = right["start_ms"] - left["end_ms"]
        if gap <= 0:
            continue
        lab_l, lab_r = left.get("phoneme_label", ""), right.get("phoneme_label", "")
        if lab_l == "SIL" or lab_r == "SIL":
            limit = p['min_gap_ms']
        elif is_stop(lab_r) or is_stop(lab_l):
            limit = p['stop_gap_ms']
        else:
            limit = p['min_gap_ms']
        if gap >= limit:
            continue
        seg = energy[max(0, ms(left["end_ms"])):max(0, ms(right["start_ms"]))]
        if limit == p['stop_gap_ms'] and classify_span(seg, floor, p):
            continue                                     # noise in it: not a closure
        if is_stop(lab_r) and lab_r != "SIL":
            right["start_ms"] = left["end_ms"]
        else:
            left["end_ms"] = right["start_ms"]

    # --- a gap next to a SIL of the text (punctuation) extends that SIL instead of adding one
    for i in range(len(phones) - 1):
        left, right = phones[i], phones[i + 1]
        if right["start_ms"] - left["end_ms"] <= 0:
            continue
        if left.get("phoneme_label") == "SIL":
            left["end_ms"] = right["start_ms"]
        elif right.get("phoneme_label") == "SIL":
            right["start_ms"] = left["end_ms"]

    # --- every remaining gap, the segment's head and tail included
    spans = []
    if phones[0]["start_ms"] - seg_start_ms > 0:
        spans.append((seg_start_ms, phones[0]["start_ms"]))
    for i in range(len(phones) - 1):
        if phones[i + 1]["start_ms"] - phones[i]["end_ms"] > 0:
            spans.append((phones[i]["end_ms"], phones[i + 1]["start_ms"]))
    if seg_end_ms - phones[-1]["end_ms"] > 0:
        spans.append((phones[-1]["end_ms"], seg_end_ms))

    new = []
    for a, b in spans:
        ia, ib = max(0, ms(a)), min(n_ms, max(0, ms(b)))
        runs = classify_span(energy[ia:ib], floor, p) if (noise_id is not None and ib > ia) \
            else []
        # Build the alternating SIL / noise pieces in absolute ms.
        pieces, cur = [], a
        for ra, rb in runs:
            s, e = a + ra, min(b, a + rb)
            if s > cur:
                pieces.append(["SIL", cur, s])
            pieces.append(["noise", max(cur, s), e])
            cur = e
        if b > cur:
            pieces.append(["SIL", cur, b])
        # Fold pieces shorter than min_token_ms into a neighbour.
        out = []
        for piece in pieces:
            if piece[2] - piece[1] <= 0:
                continue
            if out and (piece[2] - piece[1] < p['min_token_ms'] or out[-1][0] == piece[0]):
                out[-1][2] = piece[2]
            else:
                out.append(piece)
        if len(out) > 1 and out[0][2] - out[0][1] < p['min_token_ms']:
            out[1][1] = out[0][1]
            out.pop(0)
        for label, s, e in out:
            entry = _entry(label, sil_id if label == "SIL" else noise_id, s, e)
            for layer in layers:
                entry[layer] = 0
            new.append(entry)

    return sorted(phones + new, key=lambda x: (x["start_ms"], x["end_ms"]))
