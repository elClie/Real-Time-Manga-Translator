// SPDX-License-Identifier: GPL-3.0-only
// MIT textline_merge (merge_bboxes_text_region / split_text_region / dispatch) and the rest of
// MangaTranslator._run_textline_merge: bracket fix, is_valuable_text + English filters, simple sort.
// Plus app/geom.py region_to_dict.

import { Graph, PySet, mostCommon } from "./pyset.js";
import { canMerge, polygonDistance } from "./quad.js";
import { isValuableText, langOf, pyStrip } from "./text.js";

const f32 = Math.fround;

/** numpy pairwise sum (n ≤ 128 path), optionally in float32. */
function npSum(vals, single) {
  const add = single ? (a, b) => f32(a + b) : (a, b) => a + b;
  if (vals.length < 8) return vals.reduce(add, single ? f32(0) : 0);
  const r = vals.slice(0, 8);
  let i = 8;
  for (; i + 8 <= vals.length; i += 8) for (let k = 0; k < 8; k++) r[k] = add(r[k], vals[i + k]);
  let res = add(add(add(r[0], r[1]), add(r[2], r[3])), add(add(r[4], r[5]), add(r[6], r[7])));
  for (; i < vals.length; i++) res = add(res, vals[i]);
  return res;
}
const npMean = (vals, single = false) => (single ? f32(npSum(vals, true) / vals.length) : npSum(vals, false) / vals.length);
function npStd(vals) {
  const m = npMean(vals);
  return Math.sqrt(npMean(vals.map((v) => (v - m) * (v - m))));
}

function splitTextRegion(bboxes, indices, gamma = 0.5, sigma = 2) {
  const idx = [...indices];
  if (idx.length === 1) return [new PySet(idx)];
  if (idx.length === 2) {
    const a = bboxes[idx[0]];
    const b = bboxes[idx[1]];
    const fs = Math.max(a.fontSize, b.fontSize);
    if (a.distance(b) < (1 + gamma) * fs && Math.abs(a.angle - b.angle) < 0.2 * Math.PI) return [new PySet(idx)];
    return [new PySet([idx[0]]), new PySet([idx[1]])];
  }
  const G = new Graph();
  idx.forEach((i) => G.addNode(i));
  for (let i = 0; i < idx.length; i++) {
    for (let j = i + 1; j < idx.length; j++) G.addEdge(idx[i], idx[j], { weight: bboxes[idx[i]].distance(bboxes[idx[j]]) });
  }
  const edges = G.kruskal().sort((a, b) => b[2].weight - a[2].weight);
  const dists = edges.map((e) => e[2].weight);
  const fontsize = npMean(idx.map((i) => bboxes[i].fontSize), true);
  const std = npStd(dists);
  const mean = npMean(dists);
  const stdThreshold = Math.max(0.3 * fontsize + 5, 5);
  const b1 = bboxes[edges[0][0]];
  const b2 = bboxes[edges[0][1]];
  const maxPolyDistance = polygonDistance(b1.pts, b2.pts);
  const maxCentroidAlignment = Math.min(Math.abs(b1.centroid[0] - b2.centroid[0]), Math.abs(b1.centroid[1] - b2.centroid[1]));
  if (
    (dists[0] <= mean + std * sigma || dists[0] <= fontsize * (1 + gamma)) &&
    (std < stdThreshold || (maxPolyDistance === 0 && maxCentroidAlignment < 5))
  ) {
    return [new PySet(idx)];
  }
  const G2 = new Graph();
  idx.forEach((i) => G2.addNode(i));
  for (const e of edges.slice(1)) G2.addEdge(e[0], e[1]);
  const out = [];
  for (const comp of G2.connectedComponents()) out.push(...splitTextRegion(bboxes, comp));
  return out;
}

const isCjk = (ch) => ch >= "\u3000" && ch <= "\u9fff";

/** A TextBlock with the fields the app uses. */
export function makeBlock(lines, texts, fontSize, angle) {
  let text = texts[0];
  if (text && texts.length > 1) {
    for (const t of texts.slice(1)) {
      const last = [...text].pop();
      text += isCjk(last) || (t && isCjk([...t][0])) ? t : " " + t;
    }
  }
  const xs = lines.flatMap((l) => l.map((p) => p[0]));
  const ys = lines.flatMap((l) => l.map((p) => p[1]));
  const xyxy = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  return { text, texts, lines, fontSize, angle, xyxy, center: [(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2] };
}

/** textline_merge.dispatch */
export function mergeTextlines(bboxes) {
  const G = new Graph();
  bboxes.forEach((_, i) => G.addNode(i));
  for (let u = 0; u < bboxes.length; u++) {
    for (let v = u + 1; v < bboxes.length; v++) {
      if (canMerge(bboxes[u], bboxes[v], { aspectRatioTol: 1.3, fontSizeRatioTol: 2, charGapTol: 1, charGapTol2: 3 })) G.addEdge(u, v);
    }
  }
  const regionSets = [];
  for (const comp of G.connectedComponents()) regionSets.push(...splitTextRegion(bboxes, comp));

  return regionSets.map((set) => {
    let nodes = [...set];
    const boxes = nodes.map((i) => bboxes[i]);
    const top2 = mostCommon(boxes.map((b) => b.direction)).slice(0, 2);
    let dir;
    if (top2.length === 1) dir = top2[0][0];
    else if (top2[0][1] === top2[1][1]) {
      let maxAr = -100;
      for (const b of boxes) {
        if (b.aspectRatio > maxAr) {
          maxAr = b.aspectRatio;
          dir = b.direction;
        }
        if (1 / b.aspectRatio > maxAr) {
          maxAr = 1 / b.aspectRatio;
          dir = b.direction;
        }
      }
    } else dir = top2[0][0];
    const key = dir === "h" ? (i) => bboxes[i].centroid[1] : (i) => -bboxes[i].centroid[0];
    nodes = nodes.map((i) => ({ i, k: key(i) })).sort((a, b) => a.k - b.k).map((o) => o.i);
    const lines = nodes.map((i) => bboxes[i]);
    const fontSize = Math.trunc(Math.min(...lines.map((l) => l.fontSize)));
    let angle = (npMean(lines.map((l) => l.angle)) * 180) / Math.PI - 90;
    if (Math.abs(angle) < 3) angle = 0;
    return makeBlock(lines.map((l) => l.pts.map((p) => [...p])), lines.map((l) => l.text), fontSize, angle);
  });
}

const BRACKETS = {
  "(": ")", "（": "）", "[": "]", "【": "】", "{": "}", "〔": "〕", "〈": "〉", "「": "」",
  '"': '"', "＂": "＂", "'": "'", "“": "”", "《": "》", "『": "』", "〝": "〞", "﹁": "﹂", "﹃": "﹄",
  "⸂": "⸃", "⸄": "⸅", "⸉": "⸊", "⸌": "⸍", "⸜": "⸝", "⸠": "⸡", "‹": "›", "«": "»", "＜": "＞", "<": ">",
};
const LEFT = new Set(Object.keys(BRACKETS));
const RIGHT = new Set(Object.values(BRACKETS));

/** _run_textline_merge's unpaired / mismatched bracket fix. */
export function fixBrackets(text) {
  const chars = [...text];
  if (!chars.some((c) => LEFT.has(c) || RIGHT.has(c))) return text;
  let stack = [];
  const skip = new Set();
  chars.forEach((c, i) => {
    if (LEFT.has(c)) stack.push(i);
    else if (RIGHT.has(c)) {
      if (stack.length) stack.pop();
      else skip.add(i);
    }
  });
  stack.forEach((i) => skip.add(i));
  stack = [];
  const out = [];
  chars.forEach((c, i) => {
    if (skip.has(i)) return;
    if (LEFT.has(c)) {
      stack.push(c);
      out.push(c);
    } else if (RIGHT.has(c)) {
      if (stack.length) {
        const expected = BRACKETS[stack.pop()];
        out.push(c !== expected ? expected : c);
      }
    } else out.push(c);
  });
  return out.join("");
}

/** sort.py _simple_sort */
export function simpleSort(regions, rtl = true) {
  const out = [];
  const byY = regions.map((r, i) => ({ r, i })).sort((a, b) => a.r.center[1] - b.r.center[1]).map((o) => o.r);
  for (const region of byY) {
    let placed = false;
    for (let i = 0; i < out.length; i++) {
      const s = out[i];
      if (region.center[1] > s.xyxy[3]) continue;
      if (region.center[1] < s.xyxy[1] || (rtl && region.center[0] > s.center[0]) || (!rtl && region.center[0] < s.center[0])) {
        out.splice(i, 0, region);
        placed = true;
        break;
      }
    }
    if (!placed) out.push(region);
  }
  return out;
}

/** Merge → bracket fix → filters → simple sort. Returns { merged, kept }. */
export function textlineMerge(textlines) {
  const merged = mergeTextlines(textlines);
  const snapshot = merged.map((r) => ({ ...r }));
  const kept = [];
  for (const r of merged) {
    r.text = pyStrip(fixBrackets(pyStrip(r.text)));
    if (!isValuableText(r.text)) continue;
    r.lang = langOf(r.text);
    if (r.lang === "en") continue;
    kept.push(r);
  }
  return { merged: snapshot, kept: simpleSort(kept) };
}

/** app/geom.py region_to_dict */
export function regionToDict(r) {
  let sourceVertical = false;
  if (r.lines.length) {
    let best = 0;
    let bestArea = -Infinity;
    const dims = r.lines.map((l) => {
      const xs = l.map((p) => p[0]);
      const ys = l.map((p) => p[1]);
      return [Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)];
    });
    dims.forEach(([w, h], i) => {
      if (w * h > bestArea) {
        bestArea = w * h;
        best = i;
      }
    });
    sourceVertical = dims[best][1] > dims[best][0];
  }
  return { text: r.text, translation: null, bbox: [...r.xyxy], font_size: r.fontSize, source_vertical: sourceVertical, lines: r.lines };
}
