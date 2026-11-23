// The editor canvas: spectrogram or waveform, segment / phoneme / word tiers, a time ruler,
// boundary dragging, selection, zoom and scroll. Plus the overview strip under it.

import { segmentEdges, edgeLimits, gapAt, wordPhones } from "./model.js";
import { peaks as computePeaks } from "./audio.js";

const RULER_H = 18, SEG_H = 18, PH_H = 42, WORD_H = 28, GAP = 2;
const EDGE_GRAB_PX = 5;
const TILE_W = 4096;

// Magma-like colour map, 256 entries.
const LUT = (() => {
  const stops = [[0, 0, 0, 4], [0.25, 59, 15, 112], [0.5, 140, 41, 129], [0.75, 222, 73, 104],
    [0.9, 252, 165, 103], [1, 252, 253, 191]];
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const x = i / 255;
    let k = 0;
    while (k < stops.length - 2 && x > stops[k + 1][0]) k++;
    const [x0, ...a] = stops[k], [x1, ...b] = stops[k + 1];
    const f = (x - x0) / (x1 - x0);
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = a[c] + (b[c] - a[c]) * f;
  }
  return lut;
})();

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** The text, shortened with an ellipsis to fit maxW pixels, or "" if even a letter won't fit. */
function fitText(g, text, maxW) {
  if (g.measureText(text).width <= maxW) return text;
  for (let n = text.length - 1; n >= 1; n--) {
    const t = text.slice(0, n) + "…";
    if (g.measureText(t).width <= maxW) return t;
  }
  return "";
}

function confColor(c) {
  // 0 -> red, ~0.1 (std models' low-confidence level) -> amber, >= 0.5 -> green
  const x = Math.min(1, Math.sqrt(Math.max(0, c)) / Math.sqrt(0.5));
  const hue = 0 + 120 * x;
  return `hsl(${hue}, 70%, 48%)`;
}

export class Editor {
  constructor(canvas, overview, callbacks) {
    this.cv = canvas;
    this.ov = overview;
    this.cb = callbacks;          // {onSelect, onEditStart, onEdit, onEditEnd, onSeek, onHover}
    this.clip = null;
    this.mode = "mel";
    this.view = [0, 1];           // visible seconds
    this.sel = null;
    this.drag = null;
    this.hoverEdge = null;
    this.playhead = null;
    this._tiles = null; this._tilesFor = null;
    this._peaks = null; this._peaksFor = null;
    this._raf = 0;

    new ResizeObserver(() => this.draw()).observe(canvas);
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => this.draw());
    canvas.addEventListener("pointerdown", (e) => this._down(e));
    canvas.addEventListener("pointermove", (e) => this._move(e));
    canvas.addEventListener("pointerup", (e) => this._up(e));
    canvas.addEventListener("pointercancel", (e) => this._up(e));
    canvas.addEventListener("pointerleave", () => this.cb.onHover && this.cb.onHover(null));
    canvas.addEventListener("wheel", (e) => this._wheel(e), { passive: false });
    canvas.addEventListener("dblclick", (e) => this._dbl(e));
    overview.addEventListener("pointerdown", (e) => this._ovDown(e));
    overview.addEventListener("pointermove", (e) => this._ovMove(e));
    overview.addEventListener("pointerup", () => { this._ovDrag = false; });
  }

  // ------------------------------------------------------------------ state

  get duration() {
    const c = this.clip;
    if (!c) return 1;
    if (c.audioBuffer) return c.audioBuffer.duration;
    return (c.vs && c.vs.duration) || 1;
  }

  setClip(clip, keepView = false) {
    this.clip = clip;
    if (!keepView) this.view = [0, this.duration];
    this.sel = null;
    this.draw();
  }

  setMode(m) { this.mode = m; this.draw(); }

  setSelection(sel, scroll = true) {
    this.sel = sel;
    if (scroll && sel) {
      const [t0, t1] = this.selSpan(sel) || [];
      if (t0 !== undefined && (t1 < this.view[0] || t0 > this.view[1])) {
        const w = this.view[1] - this.view[0];
        this.setView(t0 - w * 0.3, t0 - w * 0.3 + w);
      }
    }
    this.draw();
    this.cb.onSelect(sel);
  }

  /** Seconds covered by a selection, or null. */
  selSpan(sel = this.sel) {
    if (!sel || !this.clip) return null;
    const segs = this.clip.vs ? this.clip.vs.segments : [];
    if (sel.type === "range") return [sel.t0, sel.t1];
    if (sel.type === "segment") return [segs[sel.si].start, segs[sel.si].end];
    if (sel.type === "gap") return [sel.gap.t0 / 1000, sel.gap.t1 / 1000];
    if (sel.type === "phoneme") {
      const p = segs[sel.si] && segs[sel.si].phoneme_ts[sel.j];
      return p ? [p.start_ms / 1000, p.end_ms / 1000] : null;
    }
    if (sel.type === "word") {
      const ph = wordPhones(segs[sel.si], sel.w);
      return ph.length ? [ph[0].start_ms / 1000, ph[ph.length - 1].end_ms / 1000] : null;
    }
    return null;
  }

  setView(t0, t1) {
    const d = this.duration;
    let w = Math.max(0.02, Math.min(d, t1 - t0));
    t0 = Math.max(0, Math.min(d - w, t0));
    this.view = [t0, t0 + w];
    this.draw();
  }

  zoom(factor, center) {
    const [a, b] = this.view;
    const c = center ?? (a + b) / 2;
    this.setView(c - (c - a) * factor, c + (b - c) * factor);
  }

  fit() { this.setView(0, this.duration); }

  setPlayhead(t) {
    this.playhead = t;
    if (t !== null && (t > this.view[1] || t < this.view[0])) {
      const w = this.view[1] - this.view[0];
      this.setView(t - w * 0.05, t - w * 0.05 + w);
      return;
    }
    this.draw();
  }

  // ------------------------------------------------------------------ geometry

  rows() {
    const h = this.cv.clientHeight;
    const main = h - RULER_H - SEG_H - PH_H - WORD_H - 4 * GAP;
    let y = 0;
    const r = {};
    r.ruler = [y, RULER_H]; y += RULER_H + GAP;
    r.main = [y, main]; y += main + GAP;
    r.seg = [y, SEG_H]; y += SEG_H + GAP;
    r.ph = [y, PH_H]; y += PH_H + GAP;
    r.word = [y, WORD_H];
    return r;
  }

  x(t) { return (t - this.view[0]) / (this.view[1] - this.view[0]) * this.cv.clientWidth; }
  t(x) { return this.view[0] + x / this.cv.clientWidth * (this.view[1] - this.view[0]); }

  rowAt(y) {
    const r = this.rows();
    for (const k of ["ruler", "main", "seg", "ph", "word"]) {
      if (y >= r[k][0] && y < r[k][0] + r[k][1] + GAP) return k;
    }
    return null;
  }

  /** Phones of a segment are wide enough on screen for boundary lines through the spectrogram. */
  _linesThroughMain(seg, edges) {
    return (this.x(seg.end) - this.x(seg.start)) / Math.max(1, edges.length) >= 6;
  }

  /**
   * Nearest draggable boundary to x, within EDGE_GRAB_PX: {si, edge} or null. On the
   * spectrogram row only where its lines are drawn, so a drag there selects a span otherwise.
   */
  edgeAt(x, row = null) {
    const vs = this.clip && this.clip.vs;
    if (!vs) return null;
    let best = null, bestD = EDGE_GRAB_PX + 1;
    vs.segments.forEach((seg, si) => {
      if (seg.end < this.view[0] || seg.start > this.view[1]) return;
      const edges = segmentEdges(seg);
      if (row === "main" && !this._linesThroughMain(seg, edges)) return;
      for (const edge of edges) {
        const d = Math.abs(this.x(edge.t / 1000) - x);
        if (d < bestD) { bestD = d; best = { si, edge }; }
      }
    });
    return best;
  }

  hit(x, y) {
    const row = this.rowAt(y);
    const t = this.t(x);
    const vs = this.clip && this.clip.vs;
    if (!vs) return { row, t };
    const tm = t * 1000;
    for (let si = 0; si < vs.segments.length; si++) {
      const seg = vs.segments[si];
      if (t < seg.start || t > seg.end) continue;
      if (row === "seg") return { row, t, sel: { type: "segment", si } };
      if (row === "ph") {
        const j = seg.phoneme_ts.findIndex((p) => tm >= p.start_ms && tm < p.end_ms);
        if (j >= 0) return { row, t, sel: { type: "phoneme", si, j } };
        const gap = gapAt(seg, tm);
        if (gap) return { row, t, sel: { type: "gap", si, gap } };
      }
      if (row === "word") {
        const p = seg.phoneme_ts.find((q) => tm >= q.start_ms && tm < q.end_ms);
        if (p && p._w !== null) return { row, t, sel: { type: "word", si, w: p._w } };
      }
    }
    return { row, t };
  }

  // ------------------------------------------------------------------ input

  _pos(e) {
    const r = this.cv.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  _down(e) {
    if (!this.clip || e.button !== 0) return;
    const [x, y] = this._pos(e);
    const row = this.rowAt(y);
    this.cv.setPointerCapture(e.pointerId);
    if (row !== "ruler" && row !== "seg") {
      const near = this.edgeAt(x, row);
      if (near) {
        this.drag = { kind: "edge", ...near, moved: false };
        return;
      }
    }
    if (row === "main" || row === "ruler") {
      this.drag = { kind: "range", t0: this.t(x), x0: x, moved: false };
      return;
    }
    const h = this.hit(x, y);
    this.setSelection(h.sel || null, false);
  }

  _move(e) {
    if (!this.clip) return;
    const [x, y] = this._pos(e);
    const d = this.drag;
    if (d && d.kind === "edge") {
      if (!d.moved) { d.moved = true; this.cb.onEditStart(); }
      this.cb.onEdit(d.si, d.edge, this.t(x) * 1000);
      return;
    }
    if (d && d.kind === "range") {
      if (Math.abs(x - d.x0) > 3) d.moved = true;
      if (d.moved) {
        const t = this.t(x);
        this.sel = { type: "range", t0: Math.min(d.t0, t), t1: Math.max(d.t0, t) };
        this.draw();
      }
      return;
    }
    const row = this.rowAt(y);
    const near = (row !== "ruler" && row !== "seg") ? this.edgeAt(x, row) : null;
    this.cv.style.cursor = near ? "col-resize" : (row === "main" ? "crosshair" : "default");
    if ((near && near.edge.t) !== (this.hoverEdge && this.hoverEdge.edge.t)) {
      this.hoverEdge = near;
      this.draw();
    }
    if (this.cb.onHover) {
      const h = near ? { row, t: this.t(x), edge: near } : this.hit(x, y);
      this.cb.onHover(h, e.clientX, e.clientY);
    }
  }

  _up(e) {
    const d = this.drag;
    this.drag = null;
    if (!d) return;
    if (d.kind === "edge" && d.moved) this.cb.onEditEnd(d.si);
    if (d.kind === "edge" && !d.moved) {
      // A click on a boundary selects the phone to its right (or left at the end).
      const j = d.edge.right >= 0 ? d.edge.right : d.edge.left;
      this.setSelection({ type: "phoneme", si: d.si, j }, false);
    }
    if (d.kind === "range") {
      if (d.moved) this.cb.onSelect(this.sel);
      else {
        this.cb.onSeek(d.t0);
        const [x, y] = this._pos(e);
        const h = this.hit(x, y);
        this.setSelection(h.sel || null, false);
      }
    }
  }

  _dbl(e) {
    const [x, y] = this._pos(e);
    const h = this.hit(x, y);
    if (h.sel) {
      const [a, b] = this.selSpan(h.sel);
      const pad = Math.max(0.05, (b - a) * 0.5);
      this.setView(a - pad, b + pad);
    }
  }

  _wheel(e) {
    if (!this.clip) return;
    e.preventDefault();
    const [x] = this._pos(e);
    if (e.ctrlKey || e.metaKey) {
      this.zoom(Math.exp(e.deltaY * 0.002), this.t(x));
    } else {
      const delta = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
      const w = this.view[1] - this.view[0];
      this.setView(this.view[0] + delta / this.cv.clientWidth * w, this.view[1] + delta / this.cv.clientWidth * w);
    }
  }

  _ovDown(e) {
    if (!this.clip) return;
    this._ovDrag = true;
    this.ov.setPointerCapture(e.pointerId);
    this._ovMove(e);
  }

  _ovMove(e) {
    if (!this._ovDrag) return;
    const r = this.ov.getBoundingClientRect();
    const t = (e.clientX - r.left) / r.width * this.duration;
    const w = this.view[1] - this.view[0];
    this.setView(t - w / 2, t + w / 2);
  }

  // ------------------------------------------------------------------ drawing

  _ensureTiles() {
    const an = this.clip && this.clip.analysis;
    const mel = an && an.mel;
    if (!mel) return null;
    if (this._tilesFor === mel) return this._tiles;
    const [T, B] = mel.shape;
    const tiles = [];
    for (let c0 = 0; c0 < T; c0 += TILE_W) {
      const w = Math.min(TILE_W, T - c0);
      const cv = document.createElement("canvas");
      cv.width = w; cv.height = B;
      const ctx = cv.getContext("2d");
      const img = ctx.createImageData(w, B);
      for (let c = 0; c < w; c++) {
        for (let b = 0; b < B; b++) {
          const v = mel.data[(c0 + c) * B + b];
          const o = ((B - 1 - b) * w + c) * 4;
          img.data[o] = LUT[v * 3]; img.data[o + 1] = LUT[v * 3 + 1];
          img.data[o + 2] = LUT[v * 3 + 2]; img.data[o + 3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
      tiles.push({ c0, w, cv });
    }
    this._tiles = tiles; this._tilesFor = mel;
    return tiles;
  }

  _ensurePeaks() {
    const buf = this.clip && this.clip.audioBuffer;
    if (!buf) return null;
    if (this._peaksFor !== buf) { this._peaks = computePeaks(buf, 256); this._peaksFor = buf; }
    return this._peaks;
  }

  draw() {
    cancelAnimationFrame(this._raf);
    this._raf = requestAnimationFrame(() => { this._draw(); this._drawOverview(); });
  }

  _draw() {
    const cv = this.cv;
    const dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth, H = cv.clientHeight;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
      cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    }
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const col = {
      bg: cssVar("--card"), text: cssVar("--text"), muted: cssVar("--muted"),
      border: cssVar("--border"), accent: cssVar("--accent"), sel: cssVar("--sel"),
      a: cssVar("--tier-a"), b: cssVar("--tier-b"), edited: cssVar("--edited"),
    };
    g.fillStyle = col.bg;
    g.fillRect(0, 0, W, H);
    if (!this.clip) return;
    const R = this.rows();
    const [v0, v1] = this.view;

    this._drawRuler(g, R.ruler, W, col);
    if (this.mode === "mel")
    {
      if( this._ensureTiles()) this._drawMel(g, R.main, W);
      else 
      {
        const cv = document.createElement("canvas");
        const ctx = cv.getContext("2d");
        cv.width = W; cv.height = R.main[1];
        ctx.fillStyle = col.muted; ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.font = "13px system-ui, sans-serif";
        ctx.fillText("No spectrogram yet. Press 'Align' to generate one along with alignments.", W / 2, R.main[0] + R.main[1] / 2);
        g.drawImage(cv, 0, R.main[0]);
      }
    }
    else this._drawWave(g, R.main, W, col);

    const vs = this.clip.vs;
    g.font = "13px system-ui, sans-serif";
    g.textBaseline = "middle";
    g.textAlign = "center";

    if (vs) {
      vs.segments.forEach((seg, si) => {
        if (seg.end < v0 || seg.start > v1) return;
        this._drawSegment(g, R, seg, si, col);
      });
    }

    // selection
    const span = this.selSpan();
    if (span) {
      const x0 = this.x(span[0]), x1 = this.x(span[1]);
      g.fillStyle = col.sel;
      g.fillRect(x0, R.main[0], Math.max(1, x1 - x0), R.word[0] + R.word[1] - R.main[0]);
      g.strokeStyle = col.accent;
      g.lineWidth = 1;
      g.strokeRect(x0 + 0.5, R.main[0] + 0.5, Math.max(1, x1 - x0) - 1, R.word[0] + R.word[1] - R.main[0] - 1);
    }

    // playhead
    if (this.playhead !== null) {
      const x = this.x(this.playhead);
      g.strokeStyle = "#e53935";
      g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke();
    }
  }

  _drawRuler(g, [y, h], W, col) {
    const [v0, v1] = this.view;
    const pxPerS = W / (v1 - v0);
    const steps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600];
    const step = steps.find((s) => s * pxPerS >= 70) || 600;
    g.fillStyle = col.muted; g.strokeStyle = col.border;
    g.font = "11px system-ui, sans-serif"; g.textBaseline = "middle"; g.textAlign = "left";
    g.beginPath();
    for (let t = Math.ceil(v0 / step) * step; t <= v1; t += step) {
      const x = Math.round(this.x(t)) + 0.5;
      g.moveTo(x, y + h - 6); g.lineTo(x, y + h);
      const label = step < 1 ? t.toFixed(step < 0.01 ? 3 : step < 0.1 ? 2 : 1) : String(Math.round(t));
      g.fillText(label + "s", x + 3, y + h / 2 - 1);
    }
    g.stroke();
  }

  _drawMel(g, [y, h], W) {
    const mel = this.clip.analysis.mel;
    const hop = mel.hop_ms / 1000;
    const [v0, v1] = this.view;
    g.imageSmoothingEnabled = (v1 - v0) / hop > W;   // smooth only when squeezing columns
    for (const tile of this._tiles) {
      const ta = tile.c0 * hop, tb = (tile.c0 + tile.w) * hop;
      if (tb < v0 || ta > v1) continue;
      const s0 = Math.max(ta, v0), s1 = Math.min(tb, v1);
      const sx = (s0 - ta) / hop, sw = (s1 - s0) / hop;
      g.drawImage(tile.cv, sx, 0, sw, tile.cv.height, this.x(s0), y, this.x(s1) - this.x(s0), h);
    }
    g.imageSmoothingEnabled = true;
  }

  _drawWave(g, [y, h], W, col) {
    const buf = this.clip.audioBuffer;
    if (!buf) {
      g.fillStyle = col.muted; g.textAlign = "center";
      g.fillText("No audio", W / 2, y + h / 2);
      return;
    }
    const sr = buf.sampleRate;
    const [v0, v1] = this.view;
    const mid = y + h / 2, amp = h / 2 - 2;
    const sPerPx = (v1 - v0) * sr / W;
    g.strokeStyle = col.border;
    g.beginPath(); g.moveTo(0, mid + 0.5); g.lineTo(W, mid + 0.5); g.stroke();
    g.fillStyle = col.accent; g.strokeStyle = col.accent;
    if (sPerPx > 256) {
      const P = this._ensurePeaks();
      for (let px = 0; px < W; px++) {
        const b0 = Math.floor((v0 * sr + px * sPerPx) / P.bucket);
        const b1 = Math.max(b0 + 1, Math.floor((v0 * sr + (px + 1) * sPerPx) / P.bucket));
        let lo = 1, hi = -1;
        for (let b = b0; b < b1 && b < P.mins.length; b++) {
          if (P.mins[b] < lo) lo = P.mins[b];
          if (P.maxs[b] > hi) hi = P.maxs[b];
        }
        if (hi < lo) continue;
        g.fillRect(px, mid - hi * amp, 1, Math.max(1, (hi - lo) * amp));
      }
    } else {
      const ch = buf.getChannelData(0);
      g.lineWidth = 1;
      g.beginPath();
      const i0 = Math.max(0, Math.floor(v0 * sr)), i1 = Math.min(ch.length, Math.ceil(v1 * sr) + 1);
      for (let i = i0; i < i1; i++) {
        const x = this.x(i / sr), yy = mid - ch[i] * amp;
        if (i === i0) g.moveTo(x, yy); else g.lineTo(x, yy);
      }
      g.stroke();
    }
  }

  _drawSegment(g, R, seg, si, col) {
    const W = this.cv.clientWidth;
    const sx0 = this.x(seg.start), sx1 = this.x(seg.end);
    // segment tier
    g.fillStyle = col.b;
    g.fillRect(sx0, R.seg[0], sx1 - sx0, R.seg[1]);
    g.fillStyle = col.muted;
    g.font = "11px system-ui, sans-serif";
    g.textAlign = "left";
    g.save();
    g.beginPath(); g.rect(Math.max(0, sx0), R.seg[0], Math.min(W, sx1) - Math.max(0, sx0), R.seg[1]); g.clip();
    g.fillText(`${si + 1}. ${seg.text || ""}`, Math.max(0, sx0) + 4, R.seg[0] + R.seg[1] / 2);
    g.restore();
    g.textAlign = "center";

    const [v0, v1] = this.view;
    const ph = seg.phoneme_ts;
    const selPh = this.sel && this.sel.type === "phoneme" && this.sel.si === si ? this.sel.j : -1;
    const selW = this.sel && this.sel.type === "word" && this.sel.si === si ? this.sel.w : null;

    // phoneme tier
    for (let j = 0; j < ph.length; j++) {
      const p = ph[j];
      if (p.end_ms / 1000 < v0 || p.start_ms / 1000 > v1) continue;
      const x0 = this.x(p.start_ms / 1000), x1 = this.x(p.end_ms / 1000);
      const [y, h] = R.ph;
      g.fillStyle = j === selPh ? col.sel : (j % 2 ? col.a : col.b);
      g.fillRect(x0, y, x1 - x0, h);
      // confidence bar (none on an inserted SIL / noise: it is not one of the text's phones)
      if (!p.inserted) {
        g.fillStyle = confColor(p.confidence || 0);
        g.fillRect(x0 + 1, y + h - 4, Math.max(1, x1 - x0 - 2), 3);
      }
      if (p.edited) {
        g.fillStyle = col.edited;
        g.beginPath(); g.arc(x1 - 5, y + 5, 2.5, 0, 7); g.fill();
      }
      const label = p.phoneme_label;
      g.font = "15px system-ui, sans-serif";
      if (g.measureText(label).width < x1 - x0 - 2) {
        g.fillStyle = col.text;
        g.fillText(label, (x0 + x1) / 2, y + h / 2 - 3);
      }
    }

    // word tier
    for (const w of seg.words_ts || []) {
      if (w.end_ms / 1000 < v0 || w.start_ms / 1000 > v1) continue;
      const x0 = this.x(w.start_ms / 1000), x1 = this.x(w.end_ms / 1000);
      const [y, h] = R.word;
      const first = ph.find((p) => Math.abs(p.start_ms - w.start_ms) < 0.01);
      const isSel = selW !== null && first && first._w === selW;
      g.fillStyle = isSel ? col.sel : col.a;
      g.fillRect(x0, y, x1 - x0, h);
      g.strokeStyle = col.border;
      g.strokeRect(x0 + 0.5, y + 0.5, x1 - x0 - 1, h - 1);
      g.font = "13px system-ui, sans-serif";
      const label = fitText(g, w.word, x1 - x0 - 4);
      if (label) {
        g.fillStyle = col.text;
        g.fillText(label, (x0 + x1) / 2, y + h / 2);
      }
    }

    // boundaries, drawn through the spectrogram and the phoneme tier
    const onMel = this.mode === "mel" && this.clip.analysis;
    const edges = segmentEdges(seg);
    // Zoomed far out, lines through the spectrogram would hide it: draw them only when the
    // phones are at least a few pixels wide on average.
    const throughMain = this._linesThroughMain(seg, edges);
    for (const edge of edges) {
      const x = Math.round(this.x(edge.t / 1000)) + 0.5;
      if (x < -2 || x > W + 2) continue;
      const hot = this.hoverEdge && this.hoverEdge.si === si && Math.abs(this.hoverEdge.edge.t - edge.t) < 0.01;
      g.strokeStyle = hot ? col.accent : (onMel ? "rgba(255,255,255,0.55)" : col.muted);
      g.lineWidth = hot ? 2 : 1;
      g.beginPath();
      if (throughMain || hot) { g.moveTo(x, R.main[0]); g.lineTo(x, R.main[0] + R.main[1]); }
      g.moveTo(x, R.ph[0]); g.lineTo(x, R.ph[0] + R.ph[1]);
      g.stroke();
    }
  }

  _drawOverview() {
    const cv = this.ov;
    const dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth, H = cv.clientHeight;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
      cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    }
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = cssVar("--tier-a");
    g.fillRect(0, 0, W, H);
    if (!this.clip) return;
    const P = this._ensurePeaks();
    const d = this.duration;
    if (P) {
      g.fillStyle = cssVar("--muted");
      const bPerPx = P.mins.length / W;
      for (let px = 0; px < W; px++) {
        let lo = 1, hi = -1;
        for (let b = Math.floor(px * bPerPx); b < Math.floor((px + 1) * bPerPx) + 1 && b < P.mins.length; b++) {
          if (P.mins[b] < lo) lo = P.mins[b];
          if (P.maxs[b] > hi) hi = P.maxs[b];
        }
        if (hi >= lo) g.fillRect(px, H / 2 - hi * H / 2, 1, Math.max(1, (hi - lo) * H / 2));
      }
    }
    const x0 = this.view[0] / d * W, x1 = this.view[1] / d * W;
    g.fillStyle = cssVar("--sel");
    g.fillRect(x0, 0, Math.max(2, x1 - x0), H);
    g.strokeStyle = cssVar("--accent");
    g.strokeRect(x0 + 0.5, 0.5, Math.max(2, x1 - x0) - 1, H - 1);
    if (this.playhead !== null) {
      g.fillStyle = "#e53935";
      g.fillRect(this.playhead / d * W, 0, 1.5, H);
    }
  }
}
