// The alignment being edited, and every edit operation on it.
//
// A clip's alignment is a `.vs.json` object (`clip.vs`), edited in place. Each phoneme gets one
// private field, `_w` (its word index), which is dropped on export: BFA stores the word of each
// TARGET phoneme in `word_num`, and after an insert or delete the targets are simply the
// phonemes as they now stand, so `_w` is what carries a phoneme's word across edits.
//
// Entries BFA marks `inserted` (the SIL / noise it labels where the text has no phone: before the
// first word, after the last, at pauses between words) belong to NO word: their `_w` is null,
// they keep `target_seq_idx` -1, and they are left out of the target lists and `words_ts`.
//
// After every edit `rebuildSegment` regenerates what BFA derives from `phoneme_ts`: the id,
// group, IPA, word and tone/stress/length lists, `words_ts`, and each phone's confidence (the
// peak posterior of its id over its own frames, as `utils._calculate_confidences` computes it).

import { peakProbs } from "./api.js";

export const MIN_PHONE_MS = 5;      // one model frame
const SHARED_EPS_MS = 0.5;          // edges closer than this are one boundary

// ---------------------------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------------------------

/** Give every phoneme its `_w`, sort by time. Safe to call more than once. */
export function prepareVs(vs) {
  for (const seg of vs.segments || []) {
    const ph = seg.phoneme_ts || (seg.phoneme_ts = []);
    ph.sort((a, b) => a.start_ms - b.start_ms);
    const wn = seg.word_num || [];
    let last = 0;
    for (const p of ph) {
      if (p._w === undefined) {
        const ti = p.target_seq_idx;
        p._w = p.inserted ? null : (ti >= 0 && ti < wn.length) ? wn[ti] : last;
      }
      if (p._w !== null) last = p._w;
    }
    if (!Array.isArray(seg.words)) seg.words = [];
  }
  return vs;
}

export function isEdited(vs) {
  return !!(vs && (vs.segments || []).some((s) => s.edited));
}

// ---------------------------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------------------------

/** Frame span [f0, f1) of a phone in a segment's posteriors, as BFA computes it. */
export function frameSpan(an, startMs, endMs) {
  const mpf = an.ms_per_frame;
  const off = an.start_sec * 1000;
  const T = an.phoneme.shape[0];
  const f0 = Math.max(0, Math.min(Math.floor((startMs - off) / mpf), T - 1));
  const f1 = Math.max(f0 + 1, Math.min(Math.ceil((endMs - off) / mpf), T));
  return [f0, f1];
}

/** Peak probability of every phoneme class over a time span, or null without posteriors. */
export function spanPhonemeProbs(clip, si, startMs, endMs) {
  const an = clip.analysis && clip.analysis.segments[si];
  if (!an) return null;
  const [f0, f1] = frameSpan(an, startMs, endMs);
  return peakProbs(an.phoneme, f0, f1);
}

export function spanToneProbs(clip, si, startMs, endMs) {
  const an = clip.analysis && clip.analysis.segments[si];
  if (!an || !an.tone) return null;
  const [f0, f1] = frameSpan(an, startMs, endMs);
  return peakProbs(an.tone, f0, f1);
}

/** Boundaries of one segment: {t, left, right} where left/right are phoneme indices or -1. */
export function segmentEdges(seg) {
  const ph = seg.phoneme_ts;
  const edges = [];
  for (let j = 0; j < ph.length; j++) {
    const prev = ph[j - 1];
    if (prev && Math.abs(prev.end_ms - ph[j].start_ms) < SHARED_EPS_MS) {
      edges.push({ t: ph[j].start_ms, left: j - 1, right: j });
    } else {
      if (prev) edges.push({ t: prev.end_ms, left: j - 1, right: -1 });
      edges.push({ t: ph[j].start_ms, left: -1, right: j });
    }
  }
  if (ph.length) edges.push({ t: ph[ph.length - 1].end_ms, left: ph.length - 1, right: -1 });
  return edges;
}

/** Where a boundary may move to: [lo, hi] in ms. */
export function edgeLimits(seg, edge) {
  const ph = seg.phoneme_ts;
  const segStart = seg.start * 1000, segEnd = seg.end * 1000;
  let lo, hi;
  if (edge.left >= 0) lo = ph[edge.left].start_ms + MIN_PHONE_MS;
  else lo = edge.right > 0 ? ph[edge.right - 1].end_ms : segStart;
  if (edge.right >= 0) hi = ph[edge.right].end_ms - MIN_PHONE_MS;
  else hi = edge.left < ph.length - 1 ? ph[edge.left + 1].start_ms : segEnd;
  return [Math.max(lo, segStart), Math.min(hi, segEnd)];
}

/** The gap around time t in a segment, if t is not inside a phoneme: {t0, t1, after}. */
export function gapAt(seg, t) {
  const ph = seg.phoneme_ts;
  const segStart = seg.start * 1000, segEnd = seg.end * 1000;
  if (t < segStart || t > segEnd) return null;
  let after = -1;
  for (let j = 0; j < ph.length; j++) {
    if (t >= ph[j].start_ms && t < ph[j].end_ms) return null;
    if (ph[j].end_ms <= t) after = j;
  }
  const t0 = after >= 0 ? ph[after].end_ms : segStart;
  const t1 = after + 1 < ph.length ? ph[after + 1].start_ms : segEnd;
  if (t1 - t0 < MIN_PHONE_MS) return null;
  return { t0, t1, after };
}

/** Phonemes of word w in segment si, in time order. */
export function wordPhones(seg, w) {
  return seg.phoneme_ts.filter((p) => p._w === w);
}

// ---------------------------------------------------------------------------------------------
// Rebuilding the derived fields
// ---------------------------------------------------------------------------------------------

export function rebuildSegment(clip, si) {
  const inv = clip.inventory;
  const seg = clip.vs.segments[si];
  const ph = seg.phoneme_ts;
  ph.sort((a, b) => a.start_ms - b.start_ms);
  const an = clip.analysis && clip.analysis.segments[si];
  const pk = inv.phonemes_key, gk = inv.groups_key;

  let ti = 0;
  ph.forEach((p, i) => {
    if (p._w === null || p._w === undefined) {
      // An inserted SIL / noise: not one of the text's phonemes.
      p._w = null;
      p.inserted = true;
      p.index = -1;
      p.target_seq_idx = -1;
      p.phoneme_label = inv.tokens[p.phoneme_id] ?? p.phoneme_label;
      // Peak SIL posterior, for noise too: no model predicts its `noise` class.
      if (an) {
        const [f0, f1] = frameSpan(an, p.start_ms, p.end_ms);
        p.confidence = peakProbs(an.phoneme, f0, f1)[inv.sil_id] ?? 0;
      } else p.confidence = p.confidence ?? 0;
      for (const k of ["tone", "stress", "length"]) if (p[k] === undefined) p[k] = 0;
      return;
    }
    delete p.inserted;
    p.index = i;
    p.target_seq_idx = ti++;
    p.phoneme_label = inv.tokens[p.phoneme_id] ?? `UNK_${p.phoneme_id}`;
    if (an) {
      const [f0, f1] = frameSpan(an, p.start_ms, p.end_ms);
      p.confidence = peakProbs(an.phoneme, f0, f1)[p.phoneme_id] ?? 0;
    }
    for (const k of ["tone", "stress", "length"]) if (p[k] === undefined) p[k] = 0;
  });

  const lex = ph.filter((p) => p._w !== null);
  seg[pk] = lex.map((p) => p.phoneme_id);
  seg[gk] = lex.map((p) => inv.group_of[p.phoneme_id] ?? 0);
  seg.ipa = lex.map((p) => p.ipa_label);
  seg.word_num = lex.map((p) => p._w);
  for (const k of ["tone", "stress", "length"]) seg[k] = lex.map((p) => p[k]);

  const byWord = new Map();
  for (const p of lex) {
    if (!byWord.has(p._w)) byWord.set(p._w, []);
    byWord.get(p._w).push(p);
  }
  seg.words_ts = [...byWord.keys()].sort((a, b) => a - b).map((w) => {
    const phones = byWord.get(w);
    return {
      word: seg.words[w] ?? `UNK_WORD_${w}`,
      start_ms: phones[0].start_ms,
      end_ms: phones[phones.length - 1].end_ms,
      confidence: phones.reduce((s, p) => s + p.confidence, 0) / phones.length,
      [pk]: phones.map((p) => p.phoneme_id),
      ipa: phones.map((p) => p.ipa_label),
    };
  });

  const ca = seg.coverage_analysis || {};
  seg.coverage_analysis = {
    ...ca, target_count: lex.length, aligned_count: lex.length, missing_count: 0,
    extra_count: 0, coverage_ratio: 1.0, missing_phonemes: [], extra_phonemes: [],
    bad_alignment: false,
  };
  seg.edited = true;
}

/** Recompute every confidence from the posteriors (just loaded), leaving edit marks alone. */
export function refreshConfidence(clip) {
  if (!clip.vs || !clip.analysis || !clip.inventory) return;
  clip.vs.segments.forEach((seg, si) => {
    const was = seg.edited;
    rebuildSegment(clip, si);
    seg.edited = was;
  });
}

// ---------------------------------------------------------------------------------------------
// Edits. Each returns nothing and leaves the segment rebuilt. Undo snapshots are the caller's.
// ---------------------------------------------------------------------------------------------

function mark(p) { p.edited = true; }

export function moveEdge(clip, si, edge, t) {
  const seg = clip.vs.segments[si];
  const [lo, hi] = edgeLimits(seg, edge);
  t = Math.min(hi, Math.max(lo, t));
  const ph = seg.phoneme_ts;
  if (edge.left >= 0) { ph[edge.left].end_ms = t; mark(ph[edge.left]); }
  if (edge.right >= 0) { ph[edge.right].start_ms = t; mark(ph[edge.right]); }
  edge.t = t;
  rebuildSegment(clip, si);
}

/** Set a phone's start/end from the inspector, within its neighbours and the segment. */
export function setTimes(clip, si, j, startMs, endMs) {
  const seg = clip.vs.segments[si];
  const ph = seg.phoneme_ts;
  const lo = j > 0 ? ph[j - 1].end_ms : seg.start * 1000;
  const hi = j < ph.length - 1 ? ph[j + 1].start_ms : seg.end * 1000;
  // A shared edge with a neighbour is kept shared: move the neighbour's edge too.
  const sharedL = j > 0 && Math.abs(ph[j - 1].end_ms - ph[j].start_ms) < SHARED_EPS_MS;
  const sharedR = j < ph.length - 1 && Math.abs(ph[j + 1].start_ms - ph[j].end_ms) < SHARED_EPS_MS;
  let s = Math.max(sharedL ? ph[j - 1].start_ms + MIN_PHONE_MS : lo, startMs);
  let e = Math.min(sharedR ? ph[j + 1].end_ms - MIN_PHONE_MS : hi, endMs);
  if (e - s < MIN_PHONE_MS) return false;
  ph[j].start_ms = s; ph[j].end_ms = e; mark(ph[j]);
  if (sharedL) { ph[j - 1].end_ms = s; mark(ph[j - 1]); }
  if (sharedR) { ph[j + 1].start_ms = e; mark(ph[j + 1]); }
  rebuildSegment(clip, si);
  return true;
}

export function relabel(clip, si, j, id) {
  const p = clip.vs.segments[si].phoneme_ts[j];
  p.phoneme_id = id;
  p.ipa_label = clip.inventory.tokens[id];
  mark(p);
  rebuildSegment(clip, si);
}

export function setTone(clip, si, j, tone) {
  const p = clip.vs.segments[si].phoneme_ts[j];
  p.tone = tone; mark(p);
  rebuildSegment(clip, si);
}

export function setWord(clip, si, j, w) {
  const p = clip.vs.segments[si].phoneme_ts[j];
  p._w = w; mark(p);
  rebuildSegment(clip, si);
}

/** Split phone j at its middle; the new right half is a copy. Returns the new index. */
export function split(clip, si, j) {
  const ph = clip.vs.segments[si].phoneme_ts;
  const p = ph[j];
  if (p.end_ms - p.start_ms < 2 * MIN_PHONE_MS) return -1;
  const mid = (p.start_ms + p.end_ms) / 2;
  const q = { ...p, start_ms: mid, edited: true };
  p.end_ms = mid; mark(p);
  ph.splice(j + 1, 0, q);
  rebuildSegment(clip, si);
  return j + 1;
}

/** Spread the phonemes of word w evenly between its first start and last end. */
export function spreadWord(clip, si, w) {
  const seg = clip.vs.segments[si];
  const ph = wordPhones(seg, w);
  if (!ph.length) return false;
  const s = ph[0].start_ms, e = ph[ph.length - 1].end_ms;
  if (e - s < ph.length * MIN_PHONE_MS) return false;
  const d = (e - s) / ph.length;
  if (ph.every((p, k) => Math.abs(p.start_ms - (s + k * d)) < 0.01)) return false;   // already even
  ph.forEach((p, k) => {
    p.start_ms = s + k * d;
    p.end_ms = k === ph.length - 1 ? e : s + (k + 1) * d;
    mark(p);
  });
  rebuildSegment(clip, si);
  return true;
}

// ---------------------------------------------------------------------------------------------
// Aligning one word again, on its own
// ---------------------------------------------------------------------------------------------

/** Indices (into phoneme_ts) of word w's first and last phone, or null. */
function wordRange(seg, w) {
  const ph = seg.phoneme_ts;
  let a = -1, b = -1;
  ph.forEach((p, j) => { if (p._w === w) { if (a < 0) a = j; b = j; } });
  return a < 0 ? null : [a, b];
}

/**
 * The window [lo, hi] (ms) to align word w in: its span widened by padMs, or `want` when given.
 * It never reaches past the far edge of a neighbouring phone, so at most the phone just before
 * and the phone just after the word are trimmed.
 */
export function wordAlignWindow(seg, w, padMs, want = null) {
  const r = wordRange(seg, w);
  if (!r) return null;
  const ph = seg.phoneme_ts, prev = ph[r[0] - 1], next = ph[r[1] + 1];
  let lo = want ? want[0] : ph[r[0]].start_ms - padMs;
  let hi = want ? want[1] : ph[r[1]].end_ms + padMs;
  lo = Math.max(lo, seg.start * 1000, prev ? prev.start_ms + MIN_PHONE_MS : -Infinity);
  hi = Math.min(hi, seg.end * 1000, next ? next.end_ms - MIN_PHONE_MS : Infinity);
  return hi - lo >= (r[1] - r[0] + 1) * MIN_PHONE_MS ? [lo, hi] : null;
}

/**
 * Put the result of an `align_word` job in place: the word's phones take the new times, and
 * the neighbouring phones are trimmed where they would overlap. Labels are untouched. Returns
 * an error message, or "" when done.
 */
export function applyWordAlignment(clip, si, w, labels, result) {
  const seg = clip.vs.segments[si];
  const ph = wordPhones(seg, w);
  if (ph.length !== labels.length || ph.some((p, k) => p.phoneme_label !== labels[k])) {
    return "The word's phonemes changed while it was being aligned. Try again.";
  }
  const byTarget = new Map(result.phonemes.map((r) => [r.target, r]));
  if (ph.some((_, k) => !byTarget.has(k))) return "The aligner did not place every phoneme of the word.";
  ph.forEach((p, k) => {
    const r = byTarget.get(k);
    p.start_ms = r.start_ms; p.end_ms = r.end_ms;
    p.realigned = true; mark(p);
  });
  const all = seg.phoneme_ts;
  const prev = all[all.indexOf(ph[0]) - 1], next = all[all.indexOf(ph[ph.length - 1]) + 1];
  const s = ph[0].start_ms, e = ph[ph.length - 1].end_ms;
  if (prev && prev.end_ms > s) { prev.end_ms = Math.max(prev.start_ms + MIN_PHONE_MS, s); mark(prev); }
  if (next && next.start_ms < e) { next.start_ms = Math.min(next.end_ms - MIN_PHONE_MS, e); mark(next); }
  rebuildSegment(clip, si);
  return "";
}

export function deletePhone(clip, si, j) {
  clip.vs.segments[si].phoneme_ts.splice(j, 1);
  rebuildSegment(clip, si);
}

/** Insert a phone filling a gap. Its label is the most probable token there. Returns its index. */
export function fillGap(clip, si, gap) {
  const inv = clip.inventory;
  const seg = clip.vs.segments[si];
  const ph = seg.phoneme_ts;
  const probs = spanPhonemeProbs(clip, si, gap.t0, gap.t1);
  let id = inv.sil_id;
  if (probs) {
    let best = -1;
    probs.forEach((v, c) => { if (c !== inv.pad_id && v > best) { best = v; id = c; } });
  }
  const neighbour = ph[gap.after] || ph[gap.after + 1];
  const p = {
    phoneme_id: id, phoneme_label: inv.tokens[id], ipa_label: inv.tokens[id],
    start_ms: gap.t0, end_ms: gap.t1, confidence: 0, target_seq_idx: 0, index: 0,
    tone: 0, stress: 0, length: 0, edited: true, _w: neighbour ? neighbour._w : 0,
  };
  if (p._w === null) p.inserted = true;     // next to an inserted SIL / noise: one too
  ph.splice(gap.after + 1, 0, p);
  rebuildSegment(clip, si);
  return gap.after + 1;
}

// ---------------------------------------------------------------------------------------------
// Uniform seed: a starting point that no aligner's phoneme boundaries survive in
// ---------------------------------------------------------------------------------------------
//
// Hand correction anchors on whatever it starts from (on L2-ARCTIC 89% of the "manual" onsets
// are exactly the MFA alignment the annotators began with). A uniform seed keeps the word
// boundaries of the alignment, each moved a random 20-40 ms earlier or later, and spreads every
// word's phonemes evenly across it. The seed is recorded in the file (`vs.seed`, and per segment
// `seed_words_ts` / `seed_phoneme_ts`), so how far annotators moved each boundary can be
// measured later. The offsets themselves are not stored: the file does not reveal the
// alignment it was seeded from.

export const UNIFORM_JITTER_MS = [20, 40];

/** mulberry32: small, fast, and the same on every browser for a given seed. */
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randomSeed() {
  const a = new Uint32Array(1);
  crypto.getRandomValues(a);
  return a[0];
}

export function isUniformSeeded(vs) {
  return !!(vs && vs.seed && vs.seed.method === "uniform");
}

/** The words of a segment in time order: {w, s, e, phones}. */
function wordUnits(seg) {
  const byWord = new Map();
  for (const p of seg.phoneme_ts) {
    if (p._w === null) continue;
    if (!byWord.has(p._w)) byWord.set(p._w, []);
    byWord.get(p._w).push(p);
  }
  return [...byWord.entries()].map(([w, phones]) => ({
    w, phones, s: Math.min(...phones.map((p) => p.start_ms)), e: Math.max(...phones.map((p) => p.end_ms)),
  })).sort((a, b) => a.s - b.s);
}

/** Seed one segment. Returns how many word boundaries could not take a full 20-40 ms offset. */
function seedSegment(clip, si, rand) {
  const seg = clip.vs.segments[si];
  const segStart = seg.start * 1000, segEnd = seg.end * 1000;
  const [jMin, jMax] = UNIFORM_JITTER_MS;
  const units = wordUnits(seg);
  if (!units.length) return 0;

  // Boundary points in time order. Adjacent words share one point, so they stay adjacent.
  const pts = [];                        // {t, span}: span = phones of the word ending here, or 0
  units.forEach((u, k) => {
    const prev = units[k - 1];
    if (!(prev && Math.abs(prev.e - u.s) < SHARED_EPS_MS)) pts.push({ t: u.s, span: 0 });
    u.p0 = pts.length - 1;
    pts.push({ t: u.e, span: u.phones.length });
    u.p1 = pts.length - 1;
  });

  let clamped = 0;
  let prevNew = segStart;
  pts.forEach((pt, i) => {
    const lo = Math.max(segStart, prevNew + pt.span * MIN_PHONE_MS);
    const next = pts[i + 1];
    const hi = Math.min(segEnd, next ? next.t - next.span * MIN_PHONE_MS : segEnd);
    const mag = jMin + (jMax - jMin) * rand();
    const sign = rand() < 0.5 ? -1 : 1;
    let t = pt.t + sign * mag;
    if (t < lo || t > hi) t = pt.t - sign * mag;           // the other direction
    if (t < lo || t > hi) { t = Math.min(Math.max(pt.t + sign * mag, lo), Math.max(lo, hi)); clamped++; }
    pt.t = t;
    prevNew = t;
  });

  for (const u of units) {
    const s = pts[u.p0].t, e = pts[u.p1].t, n = u.phones.length;
    const d = (e - s) / n;
    u.phones.sort((a, b) => a.start_ms - b.start_ms);
    u.phones.forEach((p, k) => {
      p.start_ms = s + k * d;
      p.end_ms = k === n - 1 ? e : s + (k + 1) * d;
      p.confidence = 0;            // recomputed below when the posteriors are loaded
      delete p.edited;
      delete p.realigned;
    });
  }
  rebuildSegment(clip, si);
  seg.edited = false;
  seg.seed_words_ts = seg.words_ts.map((w) => ({ word: w.word, start_ms: w.start_ms, end_ms: w.end_ms }));
  seg.seed_phoneme_ts = seg.phoneme_ts.map((p) => ({
    phoneme_label: p.phoneme_label, start_ms: p.start_ms, end_ms: p.end_ms,
  }));
  return clamped;
}

/**
 * Replace every phoneme time of the clip with a uniform seed. Labels, words and tone are kept.
 * Returns {seed, clamped}: clamped = boundaries that had no room for a full offset.
 */
export function applyUniformSeed(clip, seed = randomSeed()) {
  const rand = rng(seed);
  let clamped = 0;
  clip.vs.segments.forEach((_, si) => { clamped += seedSegment(clip, si, rand); });
  clip.vs.seed = {
    method: "uniform",
    word_jitter_ms: [...UNIFORM_JITTER_MS],
    rng: "mulberry32",
    rng_seed: seed,
    created: new Date().toISOString(),
    note: "Word boundaries of the aligner moved a random 20-40 ms earlier or later; each word's " +
      "phonemes spread evenly. Seeded times per segment: seed_words_ts, seed_phoneme_ts.",
  };
  return { seed, clamped };
}

// ---------------------------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------------------------

export function exportJSON(vs) {
  return JSON.stringify(vs, (k, v) => (k.startsWith("_") ? undefined : v), 1);
}

function tgEscape(s) { return String(s).replace(/"/g, '""'); }

/** Praat TextGrid with a words and a phonemes tier covering every segment. */
export function toTextGrid(vs, durationSec) {
  const words = [], phones = [];
  for (const seg of vs.segments || []) {
    for (const w of seg.words_ts || []) words.push([w.start_ms / 1000, w.end_ms / 1000, w.word]);
    for (const p of seg.phoneme_ts || []) {
      phones.push([p.start_ms / 1000, p.end_ms / 1000, p.ipa_label || p.phoneme_label]);
    }
  }
  const end = Math.max(durationSec || 0, ...words.map((x) => x[1]), ...phones.map((x) => x[1]), 0.001);

  function tier(name, items) {
    items.sort((a, b) => a[0] - b[0]);
    const iv = [];
    let cur = 0;
    for (let [s, e, label] of items) {
      s = Math.max(s, cur);
      if (e <= s) continue;
      if (s > cur + 1e-6) iv.push([cur, s, ""]);
      iv.push([s, e, label]);
      cur = e;
    }
    if (cur < end - 1e-6) iv.push([cur, end, ""]);
    const out = [
      '        class = "IntervalTier"', `        name = "${name}"`,
      "        xmin = 0", `        xmax = ${end}`, `        intervals: size = ${iv.length}`,
    ];
    iv.forEach(([s, e, l], i) => {
      out.push(`        intervals [${i + 1}]:`, `            xmin = ${s}`, `            xmax = ${e}`,
        `            text = "${tgEscape(l)}"`);
    });
    return out;
  }

  return [
    'File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0", `xmax = ${end}`,
    "tiers? <exists>", "size = 2", "item []:",
    "    item [1]:", ...tier("words", words),
    "    item [2]:", ...tier("phonemes", phones), "",
  ].join("\n");
}
