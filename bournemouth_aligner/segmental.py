'''
Segmental forced-alignment decoder: every frame belongs to exactly one phone, a pause, or
(in the non-forced mode) a filler, and phone boundaries are placed by the posteriors AND an
acoustic-change signal in the same Viterbi pass.

## Why a second decoder

The CTC decoder in `forced_alignment.py` inherits CTC's peaky posteriors: a phone is a one-
or two-frame spike surrounded by blank, so the spike's position - not the phone's onset - is
what the lattice places. Onset snapping (`snap_onsets_to_embeddings`) then moves each onset
to the strongest spectral-flux frame within a few frames, but it does so greedily, one onset
at a time, after the sequence has already been committed.

This decoder changes three things, each measured on 100 Buckeye segments (model3a ue03c,
onset f1 at 25 ms, same scorer as `tests/test_buckeye.py`):

    CTC lattice + greedy snap (radius 2)                                58.9
    phone HMM, no blank, raw posteriors                                 48.1
    + blank mass shared into every phone: log(P(c) + 0.1 P(blank))     ~48
    + flux-contrast bonus on every phone-onset transition              64.7
    + 2-frame minimum duration, 3rd frame skippable at cost 1          68.4

On all 200 dev segments: 59.0 -> 68.2, and on `s0103a` (30 segments): 57.9 -> 65.5.
The signed onset error median goes from +5 ms (late, the CTC spike) to 0.

 - **The bonus is the whole idea.** Adding `w * contrast(flux)[t]` to every transition INTO
   a phone's first state lets the Viterbi trade posterior evidence against acoustic change
   globally. The same flux used greedily after decoding is worth +2.3; used inside it,
   +16. `contrast` is flux minus its 7-frame moving average, over the segment's flux std;
   plain z-scores, log flux, ranks and posterior JSD all scored lower.
 - **Minimum duration** stops the decoder from emitting a phone as a 1-frame sliver next
   to a boundary it could not resolve. A hard 3-frame minimum scores the same (68.3) but
   cannot fit fast speech; 4 frames drops to 64.
 - Without the flux signal this topology scores 55.5, BELOW the CTC lattice (56.7), so
   the caller falls back to CTC when no boundary tap is available.

## Long segments

The decoder has no band constraint and no stride to reduce, and it does not "fall off" on
long segments: recall@25 by decile of position within the segment is 66-74 % across all
ten deciles on segments up to 30 s decoded in one pass. On Buckeye `s0102a`, a single 225 s
segment, whole-segment decoding scores 67.3 against 54.7 when the same decoder is walked
through the chunker (which cuts where CTC crams, a signal this decoder does not produce) and
45.3 for the chunked CTC decoder.

## `enforce=False`

Adds, for text that does not match the audio:
  - word skips: jump over 1-3 whole words at `word_skip_cost` per word, including at the
    start and end of the segment (clipped audio);
  - a filler emission on every pause state, `log(max non-blank P + 0.1 P(blank)) -
    filler_cost`, so speech with no transcript absorbs into the gap instead of dragging the
    neighbouring phones over it.

Measured on 100 segments with 3 foreign words glued onto the head AND tail of the text:
forced 65.3, non-forced 66.4; with 3 words deleted from the text: 65.3 / 65.4; clean text:
68.4 / 68.3. Skipped phones have no framestamp - the caller reports them as missing.
'''

import numpy as np

_NEG = -1e30

# Measured defaults (see module docstring).
DEFAULT_BLANK_WEIGHT = 0.1
DEFAULT_PAUSE_BIAS = -1.0
DEFAULT_BOUNDARY_WEIGHT = 1.0
DEFAULT_CONTRAST_FRAMES = 7
DEFAULT_MIN_FRAMES = 2
DEFAULT_SOFT_FRAMES = 1
DEFAULT_SOFT_COST = 1.0
DEFAULT_WORD_SKIP_COST = 6.0
DEFAULT_FILLER_COST = 6.0
MAX_WORDS_SKIPPED = 3

# State kinds
_PHONE, _SOFT, _PAUSE, _TARGET_SIL = 0, 1, 2, 3


def boundary_bonus(change, num_frames, contrast_frames=DEFAULT_CONTRAST_FRAMES, per_frame=False):
    '''
    Turn a change signal into a per-frame onset bonus, `bonus[t]` = evidence that a phone
    starts AT frame t.

    Args:
        change: 1-D array. With `per_frame=False` it is a pair signal, `change[t]` comparing
            frames t and t+1 (spectral flux), so the onset is t+1. With `per_frame=True` it is
            already indexed by frame (a boundary-head logit).
        num_frames: T.
        contrast_frames: moving-average window subtracted before scaling. 7 frames measured
            best; 5 and 11 within 0.5.
    '''
    b = np.zeros(num_frames)
    f = np.asarray(change, dtype=np.float64).reshape(-1)
    if f.size < 2 or num_frames < 2:
        return b
    k = int(max(1, min(contrast_frames, f.size)))
    pad = np.pad(f, (k // 2, k - 1 - k // 2), mode='edge')
    g = (f - np.convolve(pad, np.ones(k) / k, mode='valid')) / (f.std() + 1e-6)
    if per_frame:
        n = min(g.size, num_frames)
        b[:n] = g[:n]
    else:
        n = min(g.size, num_frames - 1)
        b[1:n + 1] = g[:n]
    return b


class _Graph:
    __slots__ = ('cls', 'tidx', 'onset', 'kind', 'edges', 'starts', 'ends')


def _build_graph(ids, word_idx, sil_id, min_frames, soft_frames, soft_cost, enforce,
                 word_skip_cost):
    cls, tidx, onset, kind = [], [], [], []
    edges = []                                     # (src, dst, cost)
    word_first, word_last = {}, {}

    def add(c, ti, is_onset, k):
        cls.append(c); tidx.append(ti); onset.append(is_onset); kind.append(k)
        return len(cls) - 1

    add(-1, -1, False, _PAUSE)
    for i, p in enumerate(ids):
        if i > 0 and word_idx[i] != word_idx[i - 1] and kind[-1] != _PAUSE:
            add(-1, -1, False, _PAUSE)
        first = len(cls)
        if p == sil_id:
            # A SIL from punctuation: a real state when the speaker paused, skippable for free
            # when they did not.
            add(p, i, True, _TARGET_SIL)
        else:
            for q in range(min_frames + soft_frames):
                add(p, i, q == 0, _PHONE if q < min_frames else _SOFT)
        word_first.setdefault(word_idx[i], first)
        word_last[word_idx[i]] = len(cls) - 1
    add(-1, -1, False, _PAUSE)
    S = len(cls)
    kind_a = np.array(kind)

    # Cost of NOT visiting a state; inf for states every path must visit.
    skip_cost = np.full(S, np.inf)
    skip_cost[kind_a == _PAUSE] = 0.0
    skip_cost[kind_a == _TARGET_SIL] = 0.0
    skip_cost[kind_a == _SOFT] = soft_cost

    # Base jumps, over a virtual START (-1) and END (S): the plain chain, plus whole-word
    # skips (and starting/ending inside the text) when not enforcing.
    START, END = -1, S
    base = {s: [(s + 1, 0.0)] for s in range(-1, S)}
    if not enforce:
        words = sorted(word_first)

        def cost(a, b):     # skipping words a..b-1
            # Per word, not per phone: a per-phone cost (so short function words are cheap
            # to skip) measured no better - +0.2 with foreign words at both ends, -0.1 clean.
            return word_skip_cost * (b - a)

        for a in range(len(words)):
            # The state just before word `a`: the pause separating it from word a-1 when
            # there is one. Starting or ending the path THERE (rather than on the word itself)
            # is what lets clipped audio open or close on silence.
            src = word_first[words[a]] - 1
            for b in range(a + 1, min(len(words), a + 1 + MAX_WORDS_SKIPPED)):
                base[src].append((word_first[words[b]], cost(a, b)))
            if 0 < a <= MAX_WORDS_SKIPPED:
                entry = src if kind[src] == _PAUSE else word_first[words[a]]
                base[START].append((entry, cost(0, a)))
            tail = len(words) - a
            if 0 < a and tail <= MAX_WORDS_SKIPPED:
                # skip words a..end, leaving from the state before word a
                base[src].append((END, cost(a, len(words))))

    # Close the jumps over optional states: from `src`, any state reachable through a chain of
    # skipped states is ONE transition, costing the base jumps plus each skipped state's cost.
    # Without this, a soft sub-state followed by a pause is two transitions, i.e. two frames,
    # and a segment whose phones exactly fill the audio has no complete path at all.
    def closure(src):
        best = {}
        frontier = [(d, c) for d, c in base[src]]
        while frontier:
            d, c = frontier.pop()
            if c >= best.get(d, np.inf):
                continue
            best[d] = c
            if d < S and np.isfinite(skip_cost[d]):
                for d2, c2 in base[d]:
                    frontier.append((d2, c + skip_cost[d] + c2))
        return best

    starts, ends = {}, {}
    for d, c in closure(START).items():
        if d < S:
            starts[d] = c
    for src in range(S):
        for d, c in closure(src).items():
            if d == END:
                ends[src] = c
            elif d != src + 1 or c != 0.0:
                edges.append((src, d, c))

    g = _Graph()
    g.cls = np.array(cls); g.tidx = np.array(tidx); g.onset = np.array(onset)
    g.kind = kind_a; g.edges = edges; g.starts = starts; g.ends = ends
    return g


def _edge_layers(edges, S):
    '''Split extra edges into layers with unique destinations, so each layer is one
    vectorised max-update.'''
    layers = []
    for src, dst, c in sorted(set(edges), key=lambda e: (e[1], e[2])):
        if not (0 <= src < S and 0 <= dst < S):
            continue
        for lay in layers:
            if dst not in lay[3]:
                lay[0].append(src); lay[1].append(dst); lay[2].append(c); lay[3].add(dst)
                break
        else:
            layers.append(([src], [dst], [c], {dst}))
    return [(np.array(a), np.array(b), np.array(c, dtype=np.float64)) for a, b, c, _ in layers]


class _Emissions:
    """
    Per-frame emissions, computed on demand. The full frames x states matrix is 8 bytes a cell
    and was the decoder's largest allocation (~1 GB for a 225 s segment); this keeps only the
    [T, C] class table, with C = 67.
    """

    def __init__(self, log_probs, g, blank_id, sil_id, blank_weight, pause_bias, filler_cost):
        P = np.exp(log_probs)
        pb = P[:, blank_id]
        self.mix = np.log(P + blank_weight * pb[:, None] + 1e-12)          # [T, C]
        pause = self.mix[:, sil_id] + pause_bias
        if filler_cost is not None:
            speech = P.copy()
            speech[:, blank_id] = 0.0
            speech[:, sil_id] = 0.0
            pause = np.maximum(pause, np.log(speech.max(1) + blank_weight * pb + 1e-12) - filler_cost)
        self.pause = pause                                                  # [T]
        self.cls = np.clip(g.cls, 0, None)
        self.is_pause = g.kind == _PAUSE
        self.shape = (log_probs.shape[0], len(g.cls))

    def __getitem__(self, t):
        e = self.mix[t, self.cls]
        e[self.is_pause] = self.pause[t]
        return e


def _emissions(log_probs, g, blank_id, sil_id, blank_weight, pause_bias, filler_cost):
    return _Emissions(log_probs, g, blank_id, sil_id, blank_weight, pause_bias, filler_cost)


def _viterbi(em, g, bonus, weight):
    T, S = em.shape
    onset_mask = g.onset.astype(np.float64)
    layers = _edge_layers(g.edges, S)
    if len(layers) > 250:
        raise ValueError(f"too many overlapping skip edges ({len(layers)} layers)")
    # Backpointers are 1-byte MOVE codes, not source states: 0 stay, 1 advance, 2+k the k-th
    # edge layer (whose destinations are unique, so `layer_src[k][dst]` recovers the source).
    layer_src = []
    for src, dst, _ in layers:
        lut = np.full(S, -1, dtype=np.int64)
        lut[dst] = src
        layer_src.append(lut)

    def step(dp, t):
        bon = (weight * bonus[t]) * onset_mask
        best = dp.copy()
        code = np.zeros(S, dtype=np.uint8)
        adv = dp[:-1] + bon[1:]
        m = adv > best[1:]
        best[1:][m] = adv[m]
        code[1:][m] = 1
        for k, (src, dst, cost) in enumerate(layers):
            cand = dp[src] - cost + bon[dst]
            m = cand > best[dst]
            best[dst[m]] = cand[m]
            code[dst[m]] = 2 + k
        return best + em[t], code

    def back(s, c):
        return s if c == 0 else (s - 1 if c == 1 else int(layer_src[c - 2][s]))

    dp = np.full(S, _NEG)
    em0 = em[0]
    for s, c in g.starts.items():
        if s < S:
            dp[s] = em0[s] - c
    # One byte per frame x state: ~30 MB for 120 s of conversational speech.
    bp = np.zeros((T, S), dtype=np.uint8)
    for t in range(1, T):
        dp, bp[t] = step(dp, t)

    fin = np.full(S, _NEG)
    for s, c in g.ends.items():
        if s < S:
            fin[s] = dp[s] - c
    s = int(np.argmax(fin))
    if fin[s] <= _NEG / 2:
        return None
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s = back(s, int(bp[t, s]))
    return path


def decode_segmental(log_probs, target_ids, blank_id, silence_id, change=None,
                     word_idx=None, enforce=True, change_per_frame=False,
                     blank_weight=DEFAULT_BLANK_WEIGHT, pause_bias=DEFAULT_PAUSE_BIAS,
                     boundary_weight=DEFAULT_BOUNDARY_WEIGHT,
                     contrast_frames=DEFAULT_CONTRAST_FRAMES,
                     min_frames=DEFAULT_MIN_FRAMES, soft_frames=DEFAULT_SOFT_FRAMES,
                     soft_cost=DEFAULT_SOFT_COST, word_skip_cost=DEFAULT_WORD_SKIP_COST,
                     filler_cost=DEFAULT_FILLER_COST):
    '''
    Align `target_ids` to `log_probs`.

    Args:
        log_probs: [T, C] log posteriors (numpy or torch).
        target_ids: target class ids, in order.
        blank_id, silence_id: class ids.
        change: acoustic change signal (see `boundary_bonus`), or None for no bonus.
        word_idx: word index per target, same length. None treats every phone as a word,
            which puts an optional pause between every pair of phones (-0.05 f1 on Buckeye)
            and makes `enforce=False` skip phones rather than words.
        enforce: every target is aligned (forced alignment). False allows word skips and
            filler, see module docstring.

    Returns:
        List of `(class_id, start_frame, end_frame, target_idx)` in time order, one per
        target that the path visited (every non-SIL target when `enforce=True`).

    Raises:
        ValueError: when the targets cannot fit the frames even at one frame each.
    '''
    if hasattr(log_probs, 'detach'):
        log_probs = log_probs.detach().float().cpu().numpy()
    lp = np.asarray(log_probs, dtype=np.float64)
    ids = [int(x) for x in target_ids]
    T = lp.shape[0]
    n = len(ids)
    if n == 0 or T == 0:
        return []
    word_idx = list(range(n)) if word_idx is None else [int(w) for w in word_idx]
    if len(word_idx) != n:
        raise ValueError(f"word_idx has {len(word_idx)} entries for {n} targets")

    n_speech = sum(1 for p in ids if p != silence_id)
    mf, sf = int(max(1, min_frames)), int(max(0, soft_frames))
    while mf > 1 and mf * n_speech > T:
        mf -= 1
        sf = 0
    if mf * n_speech > T:
        raise ValueError(f"Audio too short to align: {n_speech} phonemes cannot be fit into "
                         f"{T} frames (need at least 1 frame per phoneme).")

    g = _build_graph(ids, word_idx, silence_id, mf, sf, soft_cost, enforce, word_skip_cost)
    em = _emissions(lp, g, blank_id, silence_id, blank_weight, pause_bias,
                    None if enforce else filler_cost)
    bonus = (boundary_bonus(change, T, contrast_frames, per_frame=change_per_frame)
             if change is not None else np.zeros(T))
    path = _viterbi(em, g, bonus, boundary_weight)
    if path is None:
        raise ValueError(f"No complete alignment path for {n} targets in {T} frames.")

    # Sub-states inherit their phone's target index; pauses have none.
    state_t = g.tidx.copy()
    cur = -1
    for s in range(len(state_t)):
        if g.kind[s] in (_PHONE, _SOFT, _TARGET_SIL):
            if g.onset[s]:
                cur = g.tidx[s]
            state_t[s] = cur
        else:
            state_t[s] = -1
    ti = state_t[path]
    out = []
    start = 0
    for t in range(1, T + 1):
        if t == T or ti[t] != ti[start]:
            if ti[start] >= 0:
                out.append((ids[ti[start]], start, t, int(ti[start])))
            start = t
    return out
