// SPDX-License-Identifier: GPL-3.0-only
// MIT OCR step as the app runs it with PANEL_OCR=paddle (paddle_rec._paddle_infer):
// CommonOCR._generate_text_direction → Quadrilateral.get_transformed_region (48 px strips) →
// restack_glyphs for vertical columns → PP-OCRv6 recognizer + CTC decode → MIN_CONF / empty filters.

import { Graph, mostCommon } from "./pyset.js";
import { canMerge, pyRound } from "./quad.js";
import { pyStrip } from "./text.js";

export const REC = { textH: 48, buckets: [512, 1024, 2048], batch: 8, minConf: 0.5, glyphMerge: 1.15 };

const f32 = Math.fround;
const norm32 = (x, y) => f32(Math.sqrt(f32(f32(x * x) + f32(y * y))));
const NORM = Float32Array.from({ length: 256 }, (_, v) => f32(f32(f32(v / 255) - 0.5) / 0.5));

/** _generate_text_direction: [quad, direction] in MIT's order. */
export function textDirections(quads) {
  const G = new Graph();
  quads.forEach((_, i) => G.addNode(i));
  for (let u = 0; u < quads.length; u++) {
    for (let v = u + 1; v < quads.length; v++) if (canMerge(quads[u], quads[v], { aspectRatioTol: 1 })) G.addEdge(u, v);
  }
  const out = [];
  for (const comp of G.connectedComponents()) {
    let nodes = [...comp];
    const dir = mostCommon(nodes.map((i) => quads[i].direction))[0][0];
    const key = dir === "h" ? (i) => quads[i].aabb.y + Math.floor(quads[i].aabb.h / 2) : (i) => -(quads[i].aabb.x + quads[i].aabb.w);
    nodes = nodes.map((i) => ({ i, k: key(i) })).sort((a, b) => a.k - b.k).map((o) => o.i);
    for (const i of nodes) out.push([quads[i], dir]);
  }
  return out;
}

const matToImg = (m) => ({ data: new Uint8Array(m.data), width: m.cols, height: m.rows });

/**
 * Quadrilateral.get_transformed_region(img, direction, 48). Vertical strips are returned upright
 * (MIT rotates them 90° CCW and restack_glyphs rotates them straight back).
 */
export function transformedRegion(cv, img, q, direction, textH = REC.textH) {
  const [l1a, l1b, l2a, l2b] = q.structure;
  const ratio = f32(norm32(l1b[0] - l1a[0], l1b[1] - l1a[1]) / norm32(l2b[0] - l2a[0], l2b[1] - l2a[1]));
  const clip = (v, hi) => Math.min(Math.max(v, 0), hi);
  const xs = q.pts.map((p) => p[0]);
  const ys = q.pts.map((p) => p[1]);
  const x1 = clip(Math.min(...xs), img.cols);
  const y1 = clip(Math.min(...ys), img.rows);
  const x2 = clip(Math.max(...xs), img.cols);
  const y2 = clip(Math.max(...ys), img.rows);
  q.assignedDirection = direction;
  if (x2 <= x1 || y2 <= y1) return null;
  let w, h;
  if (direction === "h") {
    h = Math.max(textH, 2);
    w = Math.max(pyRound(textH / ratio), 2);
  } else {
    w = Math.max(textH, 2);
    h = Math.max(pyRound(textH * ratio), 2);
  }
  const src = cv.matFromArray(4, 1, cv.CV_32FC2, q.pts.flatMap(([x, y]) => [x - x1, y - y1]));
  const dst = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, w - 1, 0, w - 1, h - 1, 0, h - 1]);
  const M = cv.findHomography(src, dst, cv.RANSAC, 5.0);
  const crop = img.roi(new cv.Rect(x1, y1, x2 - x1, y2 - y1));
  const out = new cv.Mat();
  cv.warpPerspective(crop, out, M, new cv.Size(w, h), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());
  const res = matToImg(out);
  [src, dst, M, crop, out].forEach((m) => m.delete());
  return res;
}

function resizeImg(cv, im, w, h) {
  const m = cv.matFromArray(im.height, im.width, cv.CV_8UC3, im.data);
  const r = new cv.Mat();
  cv.resize(m, r, new cv.Size(w, h), 0, 0, cv.INTER_LINEAR);
  const out = matToImg(r);
  m.delete();
  r.delete();
  return out;
}

function hstack(cells, height) {
  const width = cells.reduce((s, c) => s + c.width, 0);
  const data = new Uint8Array(width * height * 3);
  let x0 = 0;
  for (const c of cells) {
    for (let y = 0; y < height; y++) data.set(c.data.subarray(y * c.width * 3, (y + 1) * c.width * 3), (y * width + x0) * 3);
    x0 += c.width;
  }
  return { data, width, height };
}

const rowsOf = (im, s, e) => ({ data: im.data.slice(s * im.width * 3, e * im.width * 3), width: im.width, height: e - s });

/** paddle_rec.restack: fixed square cells (fallback when no ink is found). */
function restackFixed(cv, column) {
  const cell = column.width;
  const n = Math.max(1, pyRound(column.height / cell));
  const col = resizeImg(cv, column, cell, n * cell);
  return hstack(Array.from({ length: n }, (_, i) => rowsOf(col, i * cell, (i + 1) * cell)), cell);
}

const median = (vals) => {
  const s = Float64Array.from(vals).sort();
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/** paddle_rec.restack_glyphs on the upright column. */
export function restackGlyphs(cv, column) {
  const width = column.width;
  const H = column.height;
  const m = cv.matFromArray(H, width, cv.CV_8UC3, column.data);
  const gray = new cv.Mat();
  cv.cvtColor(m, gray, cv.COLOR_RGB2GRAY);
  const bin = new cv.Mat();
  cv.threshold(gray, bin, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
  const b = bin.data;
  let zeros = 0;
  for (let i = 0; i < b.length; i++) zeros += b[i] === 0;
  const inkVal = zeros / b.length < 0.5 ? 0 : 255;
  const ink = new Uint8Array(b.length);
  for (let i = 0; i < b.length; i++) ink[i] = b[i] === inkVal ? 1 : 0;
  [m, gray, bin].forEach((x) => x.delete());
  for (let y = 0; y < H; y++) {
    for (const x of [0, 1, width - 2, width - 1]) if (x >= 0 && x < width) ink[y * width + x] = 0;
  }
  const colAny = new Uint8Array(width);
  const profile = new Int32Array(H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < width; x++) {
      if (ink[y * width + x]) {
        colAny[x] = 1;
        profile[y]++;
      }
    }
  }
  const cols = [];
  colAny.forEach((v, x) => v && cols.push(x));
  const minRow = Math.max(2, Math.trunc(width * 0.05));
  const rows = Array.from(profile, (p) => p >= minRow);
  if (!cols.length || !rows.some(Boolean)) return restackFixed(cv, column);
  const glyph = cols[cols.length - 1] - cols[0] + 1;

  const runs = [];
  let start = null;
  rows.forEach((on, y) => {
    if (on && start === null) start = y;
    else if (!on && start !== null) {
      runs.push([start, y]);
      start = null;
    }
  });
  if (start !== null) runs.push([start, rows.length]);
  const merged = [runs[0].slice()];
  for (const [s, e] of runs.slice(1)) {
    if (e - merged[merged.length - 1][0] <= glyph * REC.glyphMerge) merged[merged.length - 1][1] = e;
    else merged.push([s, e]);
  }

  const groups = [];
  for (const [s, e] of merged) {
    if (e - s <= glyph * REC.glyphMerge) {
      groups.push([s, e]);
      continue;
    }
    const n = Math.max(2, pyRound((e - s) / glyph));
    const window = Math.max(1, Math.trunc(glyph * 0.3));
    const cuts = [];
    for (let k = 1; k < n; k++) {
      const expected = s + pyRound((k * (e - s)) / n);
      const lo = Math.max(s + 1, expected - window);
      const hi = Math.min(e - 1, expected + window);
      if (hi >= lo) {
        let best = lo;
        for (let y = lo + 1; y <= hi; y++) if (profile[y] < profile[best]) best = y;
        cuts.push(best);
      } else cuts.push(expected);
    }
    const bounds = [s, ...cuts, e];
    for (let i = 0; i + 1 < bounds.length; i++) if (bounds[i + 1] > bounds[i]) groups.push([bounds[i], bounds[i + 1]]);
  }

  const bgVals = [[], [], []];
  for (let p = 0; p < ink.length; p++) {
    if (!ink[p]) for (let c = 0; c < 3; c++) bgVals[c].push(column.data[p * 3 + c]);
  }
  const bg = bgVals[0].length ? bgVals.map((v) => Math.trunc(median(v))) : [255, 255, 255];

  const cells = groups.map(([s, e]) => {
    let piece = rowsOf(column, s, e);
    if (piece.height > width) piece = resizeImg(cv, piece, Math.max(1, pyRound((width * width) / piece.height)), width);
    const ph = piece.height;
    const pw = piece.width;
    let top = Math.floor((width - ph) / 2);
    let c0 = -1;
    let c1 = -1;
    for (let y = s; y < e; y++) {
      for (let x = 0; x < width; x++) {
        if (ink[y * width + x]) {
          if (c0 < 0 || x < c0) c0 = x;
          if (x > c1) c1 = x;
        }
      }
    }
    const narrow = c0 >= 0 && c1 - c0 + 1 < glyph * 0.85;
    if (narrow && ph < glyph * 0.65) top = Math.max(0, Math.min(width - ph, Math.floor((width + glyph) / 2) - ph));
    const left = Math.floor((width - pw) / 2);
    const data = new Uint8Array(width * width * 3);
    for (let p = 0; p < width * width; p++) data.set(bg, p * 3);
    for (let y = 0; y < ph; y++) data.set(piece.data.subarray(y * pw * 3, (y + 1) * pw * 3), ((y + top) * width + left) * 3);
    return { data, width, height: width };
  });
  return hstack(cells, width);
}

/** PaddleRecognizer._decode for one row, from the model's per-step argmax and max probability. */
function ctcDecode(bestIds, bestP, off, T, charset) {
  let text = "";
  let sum = 0;
  let n = 0;
  let prev = -1;
  for (let t = 0; t < T; t++) {
    const best = bestIds[off + t];
    const bv = bestP[off + t];
    if (best !== 0 && best !== prev) {
      text += best < charset.length ? charset[best] : " ";
      sum += bv;
      n++;
    }
    prev = best;
  }
  return [text, n ? f32(sum / n) : 0];
}

/** Padded batch for n strips: a power of two up to REC.batch, so few distinct shapes need WebGPU shaders. */
export function batchSize(n) {
  let b = 1;
  while (b < n && b < REC.batch) b *= 2;
  return b;
}

/** PaddleRecognizer.read: strips of any size → [text, conf]. runModel(Float32Array, dims) → { best, bestP, dims: [N, T] }. */
export async function recognize(cv, crops, charset, runModel) {
  const { textH, buckets, batch } = REC;
  const byBucket = new Map();
  crops.forEach((c, i) => {
    const w = Math.ceil((textH * c.width) / c.height);
    const bw = buckets.find((b) => w <= b) ?? buckets[buckets.length - 1];
    if (!byBucket.has(bw)) byBucket.set(bw, []);
    byBucket.get(bw).push([i, Math.min(w, bw)]);
  });
  const results = new Array(crops.length);
  for (const [bw, items] of byBucket) {
    for (let s = 0; s < items.length; s += batch) {
      const chunk = items.slice(s, s + batch);
      const n = batchSize(chunk.length);
      const x = new Float32Array(n * 3 * textH * bw);
      chunk.forEach(([i, w], j) => {
        const im = resizeImg(cv, crops[i], w, textH);
        const base = j * 3 * textH * bw;
        for (let y = 0; y < textH; y++) {
          for (let xx = 0; xx < w; xx++) {
            const p = (y * w + xx) * 3;
            for (let c = 0; c < 3; c++) x[base + c * textH * bw + y * bw + xx] = NORM[im.data[p + c]];
          }
        }
      });
      const { best, bestP, dims } = await runModel(x, [n, 3, textH, bw]);
      const T = dims[1];
      chunk.forEach(([i], j) => (results[i] = ctcDecode(best, bestP, j * T, T, charset)));
    }
  }
  return results;
}

/** _paddle_infer + MangaTranslator._run_ocr's empty-text filter. Returns the kept quads (text, prob set). */
export async function ocr(cv, img, textlines, charset, runModel) {
  const t0 = performance.now();
  const lines = textDirections(textlines);
  const crops = lines.map(([q, dir]) => {
    const strip = transformedRegion(cv, img, q, dir);
    if (!strip) return null;
    return dir === "v" ? restackGlyphs(cv, strip) : strip;
  });
  const t1 = performance.now();
  const valid = crops.map((c, i) => [c, i]).filter(([c]) => c);
  const reads = await recognize(cv, valid.map(([c]) => c), charset, runModel);
  const byLine = new Map(valid.map(([, i], k) => [i, reads[k]]));
  const out = [];
  lines.forEach(([q], i) => {
    const r = byLine.get(i);
    if (!r) return;
    const [text, conf] = r;
    if (!text || conf < REC.minConf) return;
    q.text = text;
    q.prob = conf;
    if (pyStrip(q.text)) out.push(q);
  });
  return { textlines: out, crops, ms: { crops: t1 - t0, rec: performance.now() - t1 } };
}
