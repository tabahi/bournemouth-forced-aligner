r'''
The forced aligner: the same segmental Viterbi the models were trained against.

Shared by both model families. p4mbfa's training decoder is this one with the label spaces
changed (its SIL is id 1, not 0 - which is why `sil_class` is an argument everywhere below and
never assumed), so everything said about p3cupe here holds for p4mbfa's `ph_head` too.

## Why the decoder lives with the model

p3cupe is trained so that its own posteriors carry the boundary. Each frame belongs to exactly
one phone of the given sequence, and that phone's posterior is a hat over its segment: 1.0
across the interior, falling linearly to exactly 0.5 at each boundary, where the neighbour
rises through 0.5. So two phones CROSS at their shared boundary and nowhere else, and the
boundary is a continuous quantity that `refine_onsets` reads off by linear interpolation - which
is why a 5 ms frame grid is not the resolution limit.

The targets that shape was trained towards came from this DP, run on the model's own posteriors
(hard-EM). Decoding with anything else would be decoding with a different model than the one
that was trained, so `viterbi_align` here is a port of `bfa_models/p3cupe/loss.py`'s, not a second
implementation.

## The topology

Each phone is a chain of `min_frames + soft_frames` sub-states; only the last has a self-loop.
The first `min_frames` must be visited. The `soft_frames` after them may be skipped at
`soft_cost` nats each, so a phone shorter than `min_frames + soft_frames` pays for every frame
it is short by - a SOFT duration prior, which a constant per-phone cost can never be (every
path enters the same number of phones and pays it equally).

An optional SIL is padded onto each end of the sequence and may be skipped for free, because
the phonemizer rarely spells out leading silence and without a blank class nothing else absorbs
it. With `sil_skippable` every SIL in the sequence may be skipped too - the phonemizer writes
one per comma whether or not the speaker paused - and a visited skippable SIL needs only one
frame.

`expand_targets` (2.1) can also insert an optional SIL after any position - BFA asks for one at
every word boundary - with a soft minimum duration of its own (`entry_frames`: the chain is
entered that many states before its end, so each frame short of it costs `soft_cost`). Such a
SIL is not in the caller's sequence (`target_idx` -1); a visited one leaves a gap between the
phones on either side, which `nonspeech.fill_nonspeech` later labels SIL or noise.

`shape_emissions` (2.1) turns the posteriors into the per-frame scores: the prior, an energy and a
voicing term that keep vowels and sonorants off frames far quieter than the clip's speech or
without periodicity (breath, lip smacks), and a filler that lets SIL take a quiet frame whatever
the model calls it. The model's `noise` class is never used: it was not trained on noise.

## Scoring

`class_prior_norm` scores a frame on `log p(c|x) - log p_bar(c)`, the hybrid-HMM scaled
likelihood, rather than on the raw posterior. Without it the path is bought by whatever class
the model likes everywhere; see `class_prior_normalise`.

`onset_bonus[t] * bonus_weight` nats are added to every transition that STARTS a phone at
frame t, so posterior evidence and acoustic change are traded off inside one global search
instead of snapping onsets onto flux peaks afterwards. The bonus is off by default here - see
`boundary_bonus` for why.
'''

import contextlib

import torch
import torch.nn.functional as F

NEG = -1e9
SIL_CLASS = 0


@contextlib.contextmanager
def _single_threaded():
    '''
    Pin torch's intra-op pool to one thread for the duration of the DP.

    The recursion is a python loop over frames on tensors of a few thousand elements
    ([B, S, min+soft]), which is far below the size that pays for a thread fan-out - so the
    threads only contend. MEASURED on a 32 s clip with 358 phones on a loaded 32-core box:
    1.70 s at one thread, against 1.84 s steady-state and 34-117 s on the FIRST call at 32
    threads, where the pool spin-up lands on top of the contention. The pathological first
    call is what this exists to remove; a caller's first alignment is not a warm-up run.
    '''
    try:
        previous = torch.get_num_threads()
    except Exception:
        yield
        return
    if previous <= 1:
        yield
        return
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


# ----------------------------------------------------------------------------- targets

def prepare_targets(targets, target_lens, sil_class=SIL_CLASS, optional_sil=True):
    '''
    Pad an optional SIL onto both ends of every target sequence.

    Returns:
        ext_targets [B, S+2], ext_lens [B], opt_head [B] bool, opt_tail [B] bool
        (which of the two pads exists, and may therefore be skipped). `opt_head` doubles as
        the offset from an extended position back to the caller's target index.
    '''
    b, s = targets.shape
    device = targets.device

    # Padding beyond `target_lens` is never read past the length masks, but the emission
    # matrix is gathered for EVERY column, so a negative id there is an out-of-bounds index
    # into the class dimension - on CUDA a device-side assert several kernels later, with a
    # stack trace pointing somewhere else. Clamp once, here.
    targets = targets.clamp(min=0)

    if not optional_sil:
        zeros = torch.zeros(b, dtype=torch.bool, device=device)
        return targets, target_lens, zeros, zeros

    idx = torch.arange(s, device=device)[None, :]
    last = (target_lens - 1).clamp(min=0)[:, None]
    # Only pad where the sequence does not already say SIL, otherwise a clip with a labelled
    # leading silence would get two SIL states and the DP would spend min_frames on each.
    opt_head = targets.gather(1, torch.zeros_like(last)).squeeze(1) != sil_class
    opt_tail = targets.gather(1, last).squeeze(1) != sil_class

    ext = torch.full((b, s + 2), sil_class, dtype=targets.dtype, device=device)
    pos = idx + opt_head.long()[:, None]                   # shift by 1 where a head pad exists
    ext.scatter_(1, pos, targets)
    # scatter wrote the padding slots too; re-stamp the tail pad after the real sequence.
    ext_lens = target_lens + opt_head.long() + opt_tail.long()
    tail_pos = (target_lens + opt_head.long())[:, None]
    ext.scatter_(1, tail_pos, torch.full_like(tail_pos, sil_class))
    return ext, ext_lens, opt_head, opt_tail


def expand_targets(targets, target_lens, sil_class=SIL_CLASS, optional_sil=True,
                   sil_after=None):
    '''
    `prepare_targets` plus an optional SIL after every position flagged in `sil_after` -
    normally every word boundary, so a pause the text does not mark has somewhere to go other
    than into the phone next to it.

    No SIL is inserted next to one the sequence already has (a punctuation SIL), which would
    give the path two SIL states in a row to split one pause between.

    Returns a dict:
        ext [B, S'], ext_lens [B]
        src [B, S'] long: the caller's index of each position, -1 for an inserted SIL
        inserted [B, S'] bool: the positions that are not in the caller's sequence (the two
            edge pads and the word-boundary SILs); all of them may be skipped
        word_sil [B, S'] bool: the word-boundary SILs among them
        opt_head, opt_tail [B] bool: as `prepare_targets`
    '''
    b = targets.shape[0]
    device = targets.device
    tg = targets.clamp(min=0).cpu()
    lens = target_lens.cpu()
    sa = None if sil_after is None else sil_after.cpu().bool()
    rows = []
    for i in range(b):
        n = int(lens[i])
        t = tg[i, :n].tolist()
        seq, src, word = [], [], []
        head = optional_sil and (n == 0 or t[0] != sil_class)
        tail = optional_sil and (n == 0 or t[-1] != sil_class)
        if head:
            seq.append(sil_class), src.append(-1), word.append(False)
        for k in range(n):
            seq.append(t[k]), src.append(k), word.append(False)
            if (sa is not None and k + 1 < n and bool(sa[i, k]) and t[k] != sil_class
                    and t[k + 1] != sil_class):
                seq.append(sil_class), src.append(-1), word.append(True)
        if tail:
            seq.append(sil_class), src.append(-1), word.append(False)
        rows.append((seq, src, word, head, tail))

    width = max(1, max(len(r[0]) for r in rows))
    ext = torch.full((b, width), sil_class, dtype=targets.dtype)
    src = torch.full((b, width), -1, dtype=torch.long)
    word = torch.zeros((b, width), dtype=torch.bool)
    ext_lens = torch.zeros(b, dtype=torch.long)
    opt_head = torch.zeros(b, dtype=torch.bool)
    opt_tail = torch.zeros(b, dtype=torch.bool)
    for i, (s_, r_, w_, h_, t_) in enumerate(rows):
        m = len(s_)
        if m:
            ext[i, :m] = torch.tensor(s_, dtype=targets.dtype)
            src[i, :m] = torch.tensor(r_, dtype=torch.long)
            word[i, :m] = torch.tensor(w_, dtype=torch.bool)
        ext_lens[i] = m
        opt_head[i], opt_tail[i] = h_, t_
    valid = torch.arange(width)[None, :] < ext_lens[:, None]
    inserted = (src < 0) & valid
    return dict(ext=ext.to(device), ext_lens=ext_lens.to(target_lens.device),
                src=src.to(device), inserted=inserted.to(device), word_sil=word.to(device),
                opt_head=opt_head.to(device), opt_tail=opt_tail.to(device))


# ----------------------------------------------------------------------------- the DP

@torch.no_grad()
def class_prior_normalise(log_probs, input_lens, smoothing=0.1, scale=1.0):
    '''
    Posteriors -> scaled likelihoods for the aligner: `log p(c|x) - log p_bar(c)`.

    Viterbi on raw posteriors rewards every frame given to whichever class the model favours
    EVERYWHERE. Dividing by the class prior is what a hybrid HMM does with a network's
    posteriors for exactly this reason: a class predicted uniformly everywhere scores ~0 on
    every frame and so buys the path nothing, and only where a class is MORE likely than usual
    does it pull a boundary.

    `p_bar` is taken over this batch's valid frames, mixed with `smoothing` of a uniform so a
    class the model never predicts is boosted by at most log(C / smoothing).

    `scale` is the hybrid-HMM prior scale (`log p(c|x) - scale * log p_bar(c)`). At 1 a class
    the clip rarely uses is boosted so much that, in silence, where every phone's posterior is
    tiny, it scores nearly as well as SIL - so a phone can be stretched over a long pause or a
    breath for almost nothing.
    '''
    b, t, c = log_probs.shape
    valid = (torch.arange(t, device=log_probs.device)[None, :] < input_lens[:, None])
    valid = valid.to(log_probs.dtype)[..., None]
    p_bar = (log_probs.exp() * valid).sum((0, 1)) / valid.sum().clamp(min=1.0)
    p_bar = (1.0 - smoothing) * p_bar + smoothing / c
    return log_probs - float(scale) * p_bar.log()[None, None, :]


@torch.no_grad()
def boundary_bonus(flux, input_lens, contrast_frames=17, cap=None):
    '''
    Acoustic change -> per-frame onset bonus: `bonus[b, t]` is evidence that a phone STARTS at
    frame t.

    Args:
        flux: [B, T] change ENTERING each frame (entry t compares frames t-1 and t, entry 0 is
            meaningless), e.g. the log-mel L2 flux from `CupeExtractor.encode_clip`.
        contrast_frames: moving-average window subtracted before scaling. LOCAL contrast, not
            the raw flux, is what carries a boundary - a loud stretch of speech has high flux
            everywhere - so the bonus is in units of the clip's own flux standard deviation.
        cap: upper limit on that z-score, None for none. Without one, the clip's single
            biggest change - very often not speech at all but a lip click or the start of a
            breath before the first word - is worth ~100 nats at weight 15, more than the
            posteriors charge for stretching the first phone back across hundreds of ms of
            silence to collect it.
    Returns:
        [B, T] float, zero at frame 0 and past each clip's end.

    ## What it costs and buys

    It snaps onsets onto abrupt acoustic change, which means stop BURSTS. On TIMIT's standard
    boundary scorer, which labels a stop's closure and its burst as two separate boundaries,
    that is worth +2 f1@25 on the CUPE-3 models and up to +6.5 on the mbfa ones. Scored per
    phone ONSET - which is what this library emits - it costs about 5, because it drags stop
    onsets to the burst and pulls vowel onsets early (CUPE-3-en-base, TIMIT: 68.6% of onsets
    within 25 ms at weight 30, 74.6% at weight 0). The released models are published with 30,
    which is what `core.py` uses unless told otherwise; its docstring has the table.
    '''
    b, t = flux.shape
    out = torch.zeros(b, t, device=flux.device)
    for i in range(b):
        n = int(input_lens[i])
        f = flux[i, 1:n].float()
        if f.numel() < 2:
            continue
        k = int(max(1, min(contrast_frames, f.numel())))
        pad = F.pad(f[None, None], (k // 2, k - 1 - k // 2), mode='replicate')
        avg = F.avg_pool1d(pad, k, stride=1)[0, 0]
        z = (f - avg) / (f.std() + 1e-6)
        out[i, 1:n] = z if cap is None else z.clamp(max=float(cap))
    return out


def shape_emissions(log_probs, input_lens, class_prior_norm=True, prior_scale=1.0,
                    sil_class=SIL_CLASS, energy=None, energy_mask=None, energy_weight=0.0,
                    energy_db=25.0, filler_penalty=None, filler_db=25.0, clicks=None,
                    click_penalty=10.0, voicing=None, voicing_weight=0.0,
                    voicing_threshold=0.4):
    '''
    The per-frame, per-class scores the DP adds up: the posteriors after the prior, the
    energy term and the filler. [B, T, C] float.

    energy: [B, T] frame level in dB relative to the clip's speech level (its p95), or None.
        With `energy_weight`, class c loses `energy_weight` nats per 10 dB the frame lies
        below `-energy_mask[c]` dB (`energy_mask`: [C], a depth per class, inf = never; e.g.
        25 for vowels and sonorants, 40 for fricatives, inf for stops, whose closure IS
        silence). `energy_db` is the depth used for a plain 0/1 mask (1 = `energy_db`).
        The model sees 120 ms windows of RMS-normalised audio and has no notion of how loud a
        frame is against the rest of the clip, so it happily reads a breath 35 dB under the
        speech as an /i/ or an /m/; a vowel that quiet is very rare (6% of vowel frames below
        -25 dB on Common Voice en, against 73% of breath/noise frames).
    voicing: [B, T] periodicity per frame in [0, 1] (`core.frame_voicing`), or None. With
        `voicing_weight`, the classes `energy_mask` gives the sonorant depth (`energy_db`) lose
        `voicing_weight` nats per 0.1 the frame's periodicity lies below `voicing_threshold`.
        Vowels and sonorants are voiced; lip smacks, clicks and breath that are loud enough to
        pass the energy test are not (Common Voice en: periodicity < 0.4 on 70% of such
        noise frames, 21% of vowel frames).
    filler_penalty: None, or a number of nats. SIL then scores
        `max(SIL, best class - filler_penalty)` on frames quieter than `-filler_db` (needs
        `energy`). The model has a `noise` class but was never trained on noise and never
        predicts it, so non-speech that is not silence - breath, lip and tongue clicks - comes
        out as SIL most of the time and as some phone (s, h, t, i ...) the rest. On a quiet
        frame this lets the non-speech states take it whatever the model calls it, at a price,
        instead of a phone of the text stretching over it.
    '''
    lp = log_probs.float()
    if class_prior_norm:
        lp = class_prior_normalise(lp, input_lens, scale=prior_scale)
    else:
        lp = lp.clone()
    t = lp.shape[1]
    if energy is not None:
        en = energy[:, :t].float().to(lp.device)
        if energy_mask is not None and float(energy_weight) != 0.0:
            m = torch.as_tensor(energy_mask, dtype=lp.dtype, device=lp.device)
            if bool(((m == 0) | (m == 1)).all()):        # a 0/1 mask: one depth for all
                m = torch.where(m > 0, torch.full_like(m, float(energy_db)),
                                torch.full_like(m, float('inf')))
            depth = -en[..., None] - m[None, None, :]
            pen = (depth / 10.0).clamp(min=0.0).nan_to_num(0.0) * float(energy_weight)
            pen[..., sil_class] = 0.0
            lp = lp - pen
    if voicing is not None and float(voicing_weight) != 0.0 and energy_mask is not None:
        m = torch.as_tensor(energy_mask, dtype=lp.dtype, device=lp.device)
        son = (m == float(energy_db)) | (m == 1)
        vo = voicing[:, :t].float().to(lp.device)
        pen = ((float(voicing_threshold) - vo) / 0.1).clamp(min=0.0) * float(voicing_weight)
        lp = lp - pen[..., None] * son[None, None, :].to(lp.dtype)
        if filler_penalty is not None:
            best = lp.max(-1).values - float(filler_penalty)
            quiet = en < -float(filler_db)
            sil = lp[..., sil_class]
            lp[..., sil_class] = torch.where(quiet, torch.maximum(sil, best), sil)
    if clicks is not None:
        # An isolated click (`nonspeech.click_frames`): no phone of the text may sit on it,
        # and the non-speech states take it whatever the model calls it (often a stop).
        ck = clicks[:, :t].to(lp.device).bool()
        best = lp.max(-1).values
        sil = torch.where(ck, torch.maximum(lp[..., sil_class], best), lp[..., sil_class])
        lp = lp - ck[..., None].to(lp.dtype) * float(click_penalty)
        lp[..., sil_class] = sil
    return lp


def _segmental_topology(log_probs, ext_targets, ext_lens, input_lens, min_frames, opt_head,
                        opt_tail, class_prior_norm, soft_frames, soft_cost, sil_skippable,
                        sil_class, onset_bonus, bonus_weight, inserted=None,
                        entry_frames=None, sil_bonus=True, emissions=None, entry_cost=None):
    '''
    The per-phone state chains, which phones may be skipped, the exit penalties, and the
    emission and bonus terms - everything `viterbi_align` needs that does not depend on t.
    '''
    b, t_padded, _ = log_probs.shape
    s = ext_targets.shape[1]
    d, k = int(min_frames), int(soft_frames)
    if d < 1 or k < 0:
        raise ValueError(f"need min_frames >= 1 and soft_frames >= 0, got {d}, {k}")
    n_sub = d + k
    device = log_probs.device
    NEGT = torch.tensor(NEG, device=device)

    # The recursions run once per FRAME, so they must not run over the batch's padding.
    t_max = max(1, min(t_padded, int(input_lens.max())))
    if emissions is not None:
        lp = emissions[:, :t_max].float()
    else:
        lp = log_probs[:, :t_max].float()
        if class_prior_norm:
            lp = class_prior_normalise(lp, input_lens)

    rows = torch.arange(b, device=device)
    pos = torch.arange(s, device=device)
    valid_phone = pos[None, :] < ext_lens[:, None]                          # [B, S]
    last = (ext_lens - 1).clamp(min=0)
    skippable = torch.zeros(b, s, dtype=torch.bool, device=device)
    if opt_head is not None:
        skippable[:, 0] |= opt_head
    if opt_tail is not None:
        skippable[rows, last] |= opt_tail
    if inserted is not None:
        skippable |= inserted.to(device)
    if sil_skippable:
        skippable |= ext_targets == sil_class
    skippable &= valid_phone
    # Where a visited phone enters its chain. A skippable phone enters `entry_frames` states
    # before the end (default 1: straight into its last, self-looping state, so one frame is
    # enough); everything else enters at state 0. Entering m states before the end makes a
    # visit shorter than m frames cost `soft_cost` per missing frame - a soft floor.
    if sil_skippable or inserted is not None:
        soft_skip = skippable if sil_skippable else (skippable & inserted.to(device))
    else:
        soft_skip = torch.zeros_like(skippable)
    if entry_frames is None:
        ef = torch.ones(b, s, dtype=torch.long, device=device)
    else:
        ef = entry_frames.to(device).long().clamp(min=1, max=n_sub)
    entry_j = torch.where(soft_skip, n_sub - ef, torch.zeros_like(ef))
    if n_sub == 1:
        entry_j.zero_()
    feasible = input_lens >= ((valid_phone & ~skippable).sum(1) * d).clamp(min=1)

    # Leaving phone i from sub-state j costs the soft states it skips; j < d-1 cannot leave.
    j_idx = torch.arange(n_sub, device=device)
    exit_pen = torch.where(j_idx >= d - 1, -(n_sub - 1 - j_idx).float() * float(soft_cost),
                           NEGT)                                             # [D]

    tgt = ext_targets.clamp(min=0)
    phone_ok = torch.where(valid_phone, 0.0, NEGT)                          # [B, S]

    def emission(t):
        return (lp[:, t].gather(1, tgt) + phone_ok)[:, :, None]              # [B, S, 1]

    use_bonus = onset_bonus is not None and float(bonus_weight) != 0.0
    bonus = onset_bonus[:, :t_max].float() if use_bonus else None
    # Which positions collect the onset bonus when entered. The bonus is evidence that a
    # PHONE starts; a pause starts where speech stops, and a SIL that could collect it would
    # be a way to buy bonus with a one-frame visit at every word boundary.
    bonus_on = torch.ones(b, s, device=device)
    if not sil_bonus:
        bonus_on = torch.where(ext_targets == sil_class, 0.0, 1.0).to(device)
    cost = (torch.zeros(b, s, device=device) if entry_cost is None
            else entry_cost.to(device).float())
    return dict(t_max=t_max, n_sub=n_sub, rows=rows, last=last, skippable=skippable,
                entry_j=entry_j, feasible=feasible, exit_pen=exit_pen,
                emission=emission, use_bonus=use_bonus, bonus=bonus,
                bonus_w=float(bonus_weight), bonus_on=bonus_on, entry_cost=cost)


@torch.no_grad()
def viterbi_align(log_probs, ext_targets, ext_lens, input_lens, min_frames=6,
                  opt_head=None, opt_tail=None, class_prior_norm=True, soft_frames=4,
                  soft_cost=1.0, sil_skippable=True, sil_class=SIL_CLASS, onset_bonus=None,
                  bonus_weight=0.0, inserted=None, entry_frames=None, sil_bonus=True,
                  emissions=None, entry_cost=None):
    '''
    Monotonic forced alignment of `ext_targets` onto the frames. See the module docstring for
    the topology and the scoring.

    Args:
        log_probs: [B, T, C] log posteriors (NOT logits)
        ext_targets: [B, S] class ids, from `prepare_targets` / `expand_targets`
        ext_lens: [B] valid entries of ext_targets
        input_lens: [B] valid frames
        inserted: [B, S] bool, positions that are not in the caller's sequence (skippable)
        entry_frames: [B, S] soft minimum duration, in frames, of each skippable position
            (default 1)
        sil_bonus: False = entering a SIL collects no onset bonus
        entry_cost: [B, S] nats charged for visiting each position (e.g. a word-boundary SIL)
        emissions: [B, T, C] scores to use instead of the (prior-normalised) log_probs,
            from `shape_emissions`
    Returns:
        phone_idx [B, T] long: index into ext_targets for each frame (garbage past input_lens)
        feasible [B] bool: False where the clip cannot hold the mandatory frames, in which case
            that clip's row of phone_idx is meaningless and the caller must drop it.

    Cost: one python loop over T each way. Emissions are gathered per frame ([B, S]) rather
    than materialised per state, and backpointers are two bytes per (frame, phone), so memory
    does not grow with the number of sub-states.
    '''
    b, t_padded, _ = log_probs.shape
    s = ext_targets.shape[1]
    device = log_probs.device
    NEGT = torch.tensor(NEG, device=device)
    top = _segmental_topology(log_probs, ext_targets, ext_lens, input_lens, min_frames,
                              opt_head, opt_tail, class_prior_norm, soft_frames, soft_cost,
                              sil_skippable, sil_class, onset_bonus, bonus_weight,
                              inserted=inserted, entry_frames=entry_frames,
                              sil_bonus=sil_bonus, emissions=emissions,
                              entry_cost=entry_cost)
    t_max, n_sub, rows, last = top['t_max'], top['n_sub'], top['rows'], top['last']
    skippable, entry_j, feasible = top['skippable'], top['entry_j'], top['feasible']
    exit_pen, emission = top['exit_pen'], top['emission']
    use_bonus, bonus, bonus_w = top['use_bonus'], top['bonus'], top['bonus_w']
    bonus_on, entry_cost = top['bonus_on'], top['entry_cost']

    # One-hot of each position's entry sub-state.
    is_entry = entry_j[:, :, None] == torch.arange(n_sub, device=device)[None, None, :]
    entry_last = entry_j == n_sub - 1                                         # [B, S]

    # --- t = 0: start in phone 0, or in phone 1 past a skippable phone 0
    alpha = torch.full((b, s, n_sub), NEG, device=device)
    alpha[rows, 0, entry_j[:, 0]] = 0.0
    if s > 1:
        start1 = torch.where(skippable[:, 0], 0.0, NEGT)
        alpha[rows, 1, entry_j[:, 1]] = torch.maximum(alpha[rows, 1, entry_j[:, 1]], start1)
    alpha = alpha + emission(0)

    # Backpointers. last_code, for the last sub-state: 0 stay, 1 from the sub-state before,
    # 2 entered. enter_src, for any entry: skip * n_sub + the exit sub-state of the source.
    # mid_entered: the entry sub-state of a position entered somewhere other than state 0 or
    # the last one was reached by ENTERING it (else from the sub-state before).
    last_code = torch.zeros(t_max, b, s, dtype=torch.uint8, device=device)
    enter_src = torch.zeros(t_max, b, s, dtype=torch.uint8, device=device)
    mid_entered = torch.zeros(t_max, b, s, dtype=torch.uint8, device=device)
    active = torch.arange(t_max, device=device)[None, :] < input_lens[:, None]    # [B, T]
    skip_mid = skippable[:, 1:-1] if s > 2 else None
    ent = torch.full((b, s), NEG, device=device)
    src = torch.zeros(b, s, dtype=torch.long, device=device)

    for t in range(1, t_max):
        exit_v, exit_j = (alpha + exit_pen).max(-1)                          # [B, S]
        ent.fill_(NEG)
        src.zero_()
        ent[:, 1:] = exit_v[:, :-1]
        src[:, 1:] = exit_j[:, :-1]
        if skip_mid is not None:
            cand = torch.where(skip_mid, exit_v[:, :-2], NEGT)
            take = cand > ent[:, 2:]
            ent[:, 2:] = torch.where(take, cand, ent[:, 2:])
            src[:, 2:] = torch.where(take, exit_j[:, :-2] + n_sub, src[:, 2:])
        if use_bonus:
            ent = ent + bonus[:, t:t + 1] * bonus_w * bonus_on
        ent = ent - entry_cost

        ent_full = torch.where(is_entry, ent[:, :, None], NEGT)              # [B, S, D]
        stay = alpha[:, :, -1]
        best, code = stay, torch.zeros(b, s, dtype=torch.uint8, device=device)
        nxt = torch.empty_like(alpha)
        if n_sub > 1:
            adv = torch.full_like(alpha[:, :, :-1], NEG)
            adv[:, :, 1:] = alpha[:, :, :-2]
            took = ent_full[:, :, :-1] > adv
            nxt[:, :, :-1] = torch.where(took, ent_full[:, :, :-1], adv)
            mid_entered[t] = (took & is_entry[:, :, :-1]).any(-1).to(torch.uint8)
            within = alpha[:, :, -2]
            m = within > best
            best = torch.where(m, within, best)
            code = torch.where(m, 1, code).to(torch.uint8)
        ent_last = torch.where(entry_last, ent, NEGT)
        m = ent_last > best
        best = torch.where(m, ent_last, best)
        code = torch.where(m, 2, code).to(torch.uint8)
        nxt[:, :, -1] = best
        nxt = nxt + emission(t)

        on = active[:, t]
        alpha = torch.where(on[:, None, None], nxt, alpha)
        last_code[t] = torch.where(on[:, None], code, 0).to(torch.uint8)
        enter_src[t] = src.to(torch.uint8)

    # --- end: leave the last phone, or the one before a skippable last phone
    exit_v, exit_j = (alpha + exit_pen).max(-1)
    p = last.clone()
    if s > 1:
        alt = (last - 1).clamp(min=0)
        use_alt = skippable[rows, last] & (last > 0) & (exit_v[rows, alt] > exit_v[rows, last])
        p = torch.where(use_alt, alt, p)
    j = exit_j[rows, p]

    phone_idx = torch.zeros(b, t_max, dtype=torch.long, device=device)
    for t in range(t_max - 1, -1, -1):
        phone_idx[:, t] = p
        if t == 0:
            break
        lc = last_code[t, rows, p].long()
        es = enter_src[t, rows, p].long()
        me = mid_entered[t, rows, p].long()
        ej = entry_j[rows, p]
        is_last = j == n_sub - 1
        at_entry = (~is_last) & (j == ej)
        entered = (is_last & (lc == 2)) | (at_entry & ((j == 0) | (me == 1)))
        back_one = (is_last & (lc == 1)) | (~is_last & ~entered & (j > 0))
        skip = es // n_sub
        p_new = torch.where(entered, p - 1 - skip, p)
        j_new = torch.where(entered, es % n_sub, torch.where(back_one, j - 1, j))
        on = t < input_lens              # past a clip's end its state is frozen
        p = torch.where(on, p_new, p)
        j = torch.where(on, j_new, j)

    if t_padded > t_max:                 # restore the caller's width; these frames are padding
        phone_idx = torch.cat(
            [phone_idx, phone_idx[:, -1:].expand(b, t_padded - t_max)], dim=1)
    return phone_idx, feasible


def path_to_onsets(phone_idx, input_lens, n_positions):
    '''
    Frame index at which each target position starts.

    Returns [B, n_positions + 1]; entry `i` is the first frame of phone `i` and the last entry
    is the clip length, so `onsets[i + 1] - onsets[i]` is phone `i`'s duration. A phone the
    path skipped (an unused optional SIL) gets a zero-length segment rather than a wrong one.
    '''
    b, t_max = phone_idx.shape
    device = phone_idx.device
    frame_ok = torch.arange(t_max, device=device)[None, :] < input_lens[:, None]
    # Frames past the clip are parked above every position so they are never counted.
    pi = torch.where(frame_ok, phone_idx, torch.full_like(phone_idx, n_positions + 1))
    cols = torch.arange(n_positions + 1, device=device)[None, None, :]
    return (pi[:, :, None] < cols).sum(dim=1)


# ----------------------------------------------------------------------------- readout

@torch.no_grad()
def refine_onsets(log_probs, seq, onsets):
    '''
    Sub-frame boundary times from the crossing of two posteriors.

    The training target makes phone `i` and phone `i+1` cross exactly at their shared boundary,
    so the boundary is where `logp(i+1) - logp(i)` changes sign, and linear interpolation of
    that difference between the two frames either side of the crossing puts it at a fraction of
    a frame. At 5 ms frames this is the difference between a 2.5 ms quantisation floor and
    none - it is the reason the loss was shaped this way rather than with hard labels.

    Args:
        log_probs: [T, C] for one clip
        seq: [S] the (extended) target sequence
        onsets: [S + 1] integer onsets from `path_to_onsets`
    Returns:
        [S] float frame positions. Position 0 stays at its integer onset (nothing to cross).
    '''
    out = onsets[:-1].float().clone()
    t_max = log_probs.shape[0]
    prev = -1                                          # last phone the path visited
    for i in range(int(seq.shape[0])):
        if onsets[i + 1] == onsets[i]:
            continue                                   # skipped: no frames, no boundary
        left_i, prev = prev, i
        b = int(onsets[i])
        if left_i < 0 or b <= 0 or b >= t_max:
            continue                                   # first visited phone, or a clip edge
        # The phone across the boundary is the previous VISITED one, not seq[i - 1]: that may
        # be a skipped SIL, whose posterior has nothing to do with this crossing.
        left, right = int(seq[left_i]), int(seq[i])
        if left == right:
            continue
        g0 = (log_probs[b - 1, right] - log_probs[b - 1, left]).item()
        g1 = (log_probs[b, right] - log_probs[b, left]).item()
        if g0 < 0 < g1:                                 # the expected bracket
            out[i] = (b - 1) + g0 / (g0 - g1)
        else:
            out[i] = b - 0.5                            # no crossing here: mid-way is the
        # honest answer, and it matches the integer reading to within half a frame.
    return out


@torch.no_grad()
def confidence_dip(log_probs):
    '''
    The boundary signal read WITHOUT the phoneme sequence: `1 - (top1 - top2)` per frame.

    The trained shape makes confidence dip to zero at a boundary and nowhere else, so peaks of
    this signal are boundary candidates for audio that has no transcript. Not used by the
    forced aligner; exposed because it is the same model's unsupervised segmenter.
    '''
    top2 = log_probs.exp().topk(2, dim=-1).values
    return 1.0 - (top2[..., 0] - top2[..., 1])


# ----------------------------------------------------------------------------- batch API

def _feasible_min_frames(n_mandatory, n_frames, min_frames, soft_frames):
    '''
    The largest duration floor this clip can actually afford, and whether the soft states
    survive it.

    A clip whose text outruns its audio would otherwise come back infeasible and lose every
    timestamp. Shortening the floor is strictly better than that: the phones are still placed
    in order, just with less duration prior holding them apart.
    '''
    if n_mandatory <= 0:
        return max(1, int(min_frames)), int(soft_frames)
    afford = int(n_frames) // int(n_mandatory)
    if afford >= int(min_frames):
        return int(min_frames), int(soft_frames)
    return max(1, afford), 0


@torch.no_grad()
def align_batch(log_probs, seqs, seq_lens, spectral_lens, min_frames=6, soft_frames=4,
                soft_cost=1.0, sil_skippable=True, optional_sil=True, class_prior_norm=True,
                sil_class=SIL_CLASS, flux=None, bonus_weight=0.0, contrast_frames=17,
                sub_frame=True, bonus_cap=None, prior_scale=1.0, sil_after=None,
                word_sil_frames=4, word_sil_cost=0.0, sil_bonus=True, energy=None, energy_mask=None,
                energy_weight=0.0, energy_db=25.0, filler_penalty=None, filler_db=25.0,
                clicks=None, click_penalty=10.0, voicing=None, voicing_weight=0.0,
                voicing_threshold=0.4):
    '''
    Force-align a whole batch.

    This is the deliverable - the MFA-equivalent call - and it is the same DP the model was
    trained against, so what comes out here is what the posteriors were shaped to say.

    Args:
        log_probs: [B, T, C] log posteriors
        seqs: [B, S] target phoneme ids, seq_lens: [B]
        spectral_lens: [B] valid frames per clip
        flux: [B, T] acoustic change entering each frame; required when `bonus_weight` is set
        sub_frame: interpolate each onset at the posterior crossing (`refine_onsets`). False
            returns the integer frame grid.
        bonus_cap: upper limit on the onset bonus's z-score (`boundary_bonus`).
        prior_scale: the prior scale of `class_prior_normalise`.
        sil_after: [B, S] bool, or None. An optional SIL is inserted after every flagged
            position (`expand_targets`); `word_sil_frames` is its soft minimum duration and
            `word_sil_cost` the nats a visit costs.
        sil_bonus: False = entering a SIL collects no onset bonus.
        energy, energy_mask, energy_weight, energy_db, filler_penalty, filler_db, clicks,
            click_penalty: see `shape_emissions`. `energy` is [B, T], dB relative to each
            clip's speech level; `clicks` [B, T] bool.
    Returns:
        A list of B dicts, each with
            `phones`     the extended sequence (the optional SILs included),
            `target_idx` the caller's index for each position, -1 for an inserted SIL,
            `onsets`     float frame per position (integer-valued when `sub_frame` is False),
            `used`       False for a position the path skipped,
            `n_frames`   the clip's valid frame count.
        An empty dict marks a clip whose sequence cannot be fitted into its frames at all.

    The DP runs on whatever device `log_probs` is on. It is a python loop over frames, so tiny
    per-frame tensors make it launch-bound on a GPU: measured on a 32 s clip with 358 phones,
    1.8 s on CPU against 5.0 s on CUDA, bit-identical either way. `core.py` therefore hands it
    CPU tensors.
    '''
    b = log_probs.shape[0]
    device = log_probs.device
    seqs = seqs.to(device)
    seq_lens = seq_lens.to(device)
    spectral_lens = spectral_lens.to(device)
    if flux is not None:
        flux = flux.to(device)
    tg = expand_targets(seqs, seq_lens, sil_class=sil_class, optional_sil=optional_sil,
                        sil_after=sil_after)
    ext, ext_lens, src = tg['ext'], tg['ext_lens'], tg['src']
    opt_head, opt_tail, inserted = tg['opt_head'], tg['opt_tail'], tg['inserted']
    entry_frames = torch.where(tg['word_sil'], int(word_sil_frames), 1)
    entry_cost = torch.where(tg['word_sil'], float(word_sil_cost), 0.0)

    onset_bonus = None
    if bonus_weight:
        if flux is None:
            raise ValueError("bonus_weight is set but no flux was passed to align_batch()")
        onset_bonus = boundary_bonus(flux, spectral_lens, contrast_frames, cap=bonus_cap)

    # The prior is the BATCH's, as it always was; the energy and filler terms are per clip.
    emissions = shape_emissions(
        log_probs, spectral_lens, class_prior_norm=class_prior_norm, prior_scale=prior_scale,
        sil_class=sil_class, energy=None if energy is None else energy.to(device),
        energy_mask=energy_mask, energy_weight=energy_weight, energy_db=energy_db,
        filler_penalty=filler_penalty, filler_db=filler_db,
        clicks=None if clicks is None else clicks.to(device), click_penalty=click_penalty,
        voicing=None if voicing is None else voicing.to(device),
        voicing_weight=voicing_weight, voicing_threshold=voicing_threshold)

    # How many positions the path is obliged to visit, per clip - a skippable SIL is not one.
    pos = torch.arange(ext.shape[1], device=device)[None, :]
    valid = pos < ext_lens[:, None]
    skippable = inserted.clone()
    if sil_skippable:
        skippable |= ext == sil_class
    n_mandatory = (valid & ~skippable).sum(1).tolist()

    # Clips that cannot afford the configured duration floor get a shorter one rather than no
    # alignment. Grouping by the resolved floor keeps this to one DP run in the normal case.
    groups = {}
    for i in range(b):
        key = _feasible_min_frames(n_mandatory[i], int(spectral_lens[i]),
                                   min_frames, soft_frames)
        groups.setdefault(key, []).append(i)

    out = [{} for _ in range(b)]
    lp_cpu = log_probs.detach().float().cpu() if sub_frame else None
    ext_cpu, ext_lens_cpu, src_cpu = ext.cpu(), ext_lens.cpu(), src.cpu()
    spec_lens_cpu = spectral_lens.cpu()

    for (mf, sf), ids in groups.items():
        sel = torch.tensor(ids, device=device)
        with _single_threaded() if device.type == 'cpu' else contextlib.nullcontext():
            phone_idx, feasible = viterbi_align(
                log_probs.index_select(0, sel), ext.index_select(0, sel),
                ext_lens.index_select(0, sel), spectral_lens.index_select(0, sel),
                min_frames=mf, opt_head=opt_head.index_select(0, sel),
                opt_tail=opt_tail.index_select(0, sel), class_prior_norm=class_prior_norm,
                soft_frames=sf, soft_cost=soft_cost, sil_skippable=sil_skippable,
                sil_class=sil_class,
                onset_bonus=None if onset_bonus is None else onset_bonus.index_select(0, sel),
                bonus_weight=bonus_weight, inserted=inserted.index_select(0, sel),
                entry_frames=entry_frames.index_select(0, sel), sil_bonus=sil_bonus,
                emissions=emissions.index_select(0, sel),
                entry_cost=entry_cost.index_select(0, sel))
        onsets = path_to_onsets(phone_idx, spectral_lens.index_select(0, sel),
                                ext.shape[1]).cpu()

        for k, i in enumerate(ids):
            n_ext, n_frames = int(ext_lens_cpu[i]), int(spec_lens_cpu[i])
            if not bool(feasible[k]) or n_ext == 0:
                continue
            on = onsets[k, :n_ext + 1]
            if sub_frame:
                # A handful of scalar reads per boundary, so it goes to the CPU once per clip
                # rather than syncing with the GPU a few thousand times.
                place = [float(v) for v in
                         refine_onsets(lp_cpu[i, :n_frames], ext_cpu[i, :n_ext], on)]
            else:
                place = [float(v) for v in on[:n_ext]]
            out[i] = {
                'phones': ext_cpu[i, :n_ext].tolist(),
                'target_idx': src_cpu[i, :n_ext].tolist(),
                'onsets': place,
                'used': [int(on[p + 1]) != int(on[p]) for p in range(n_ext)],
                'n_frames': n_frames,
            }
    return out
