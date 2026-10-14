// app/bubbles.py: speech-bubble geometry by flood fill (region.bubble = { poly, inner, fill, id }).

import { pyRound } from "./quad.js";
import { pyStrip } from "./text.js";

export const BUBBLE = { colorTol: 40, seal: 1, windows: [1.0, 3.0], maxAreaRatio: 12.0, minSolidity: 0.75, minCovered: 0.9, rectGrid: 32, sameIou: 0.7 };

const median = (vals) => {
  const s = Float64Array.from(vals).sort();
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/** _largest_rect: largest axis-aligned all-true rectangle (histogram method), [x1, y1, x2, y2]. */
function largestRect(mask, w, h) {
  const heights = new Int32Array(w);
  let best = 0;
  let rect = [0, 0, 0, 0];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) heights[x] = mask[y * w + x] ? heights[x] + 1 : 0;
    const stack = [];
    for (let x = 0; x <= w; x++) {
      const cur = x < w ? heights[x] : 0;
      let start = x;
      while (stack.length && stack[stack.length - 1][1] >= cur) {
        const [s, hh] = stack.pop();
        start = s;
        if (hh * (x - s) > best) {
          best = hh * (x - s);
          rect = [s, y - hh + 1, x, y + 1];
        }
      }
      stack.push([start, cur]);
    }
  }
  return rect;
}

function contourPoints(c) {
  const d = c.data32S;
  const pts = [];
  for (let i = 0; i < d.length; i += 2) pts.push([d[i], d[i + 1]]);
  return pts;
}

function findInWindow(cv, img, gray, region, grow, trace) {
  const H = gray.rows;
  const W = gray.cols;
  const [x1, y1, x2, y2] = region.bbox;
  const bw = x2 - x1;
  const bh = y2 - y1;
  const wx1 = Math.max(0, Math.trunc(x1 - grow * bw));
  const wy1 = Math.max(0, Math.trunc(y1 - grow * bh));
  const wx2 = Math.min(W, Math.trunc(x2 + grow * bw));
  const wy2 = Math.min(H, Math.trunc(y2 + grow * bh));
  const ww = wx2 - wx1;
  const wh = wy2 - wy1;
  const mats = [];
  const keep = (m) => (mats.push(m), m);
  const done = (v, why) => {
    mats.forEach((m) => m.delete());
    trace?.push({ grow, why: why || (v ? "found" : "none") });
    return v;
  };

  const winRoi = gray.roi(new cv.Rect(wx1, wy1, ww, wh));
  const win = keep(new cv.Mat());
  winRoi.copyTo(win); // roi().clone() keeps the parent's row stride in opencv.js; copyTo packs it
  winRoi.delete();
  const text = keep(cv.Mat.zeros(wh, ww, cv.CV_8UC1));
  for (const q of region.lines) {
    const poly = cv.matFromArray(q.length, 1, cv.CV_32SC2, q.flatMap(([x, y]) => [x - wx1, y - wy1]));
    const polys = new cv.MatVector();
    polys.push_back(poly);
    cv.fillPoly(text, polys, new cv.Scalar(255));
    poly.delete();
    polys.delete();
  }
  // Mat.data builds a new view per access and views go stale when the wasm heap grows:
  // take views right before each loop, with no OpenCV allocation inside it.
  const npx = ww * wh;
  let td = text.data;
  let wd = win.data;
  const under = [];
  for (let i = 0; i < npx; i++) if (td[i]) under.push(wd[i]);
  if (!under.length) return done(null, "no text pixels");
  const bg = Math.trunc(median(under));
  const lo = keep(new cv.Mat(wh, ww, cv.CV_8UC1, new cv.Scalar(Math.max(0, bg - BUBBLE.colorTol))));
  const hi = keep(new cv.Mat(wh, ww, cv.CV_8UC1, new cv.Scalar(Math.min(255, bg + BUBBLE.colorTol))));
  const similar = keep(new cv.Mat());
  cv.inRange(win, lo, hi, similar);
  let sd = similar.data;
  for (let i = 0; i < npx; i++) sd[i] = sd[i] ? 1 : 0;
  const k = 2 * BUBBLE.seal + 1;
  const kernel = keep(cv.Mat.ones(k, k, cv.CV_8U));
  const anchor = new cv.Point(-1, -1);
  const sealed = keep(new cv.Mat());
  cv.erode(similar, sealed, kernel, anchor, 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());

  const labels = keep(new cv.Mat());
  cv.connectedComponents(sealed, labels, 4, cv.CV_32S);
  const eq = keep(new cv.Mat(wh, ww, cv.CV_8UC1));
  const L = labels.data32S;
  td = text.data;
  const sld = sealed.data;
  const counts = new Map();
  for (let i = 0; i < npx; i++) {
    if (td[i] && sld[i] && L[i] > 0) counts.set(L[i], (counts.get(L[i]) || 0) + 1);
  }
  if (trace) {
    sd = similar.data;
    let sim = 0;
    let sea = 0;
    for (let i = 0; i < npx; i++) (sim += sd[i]), (sea += sld[i]);
    wd = win.data;
    let winSum = 0;
    for (let i = 0; i < npx; i++) winSum += wd[i];
    trace.push({ grow, win: [wx1, wy1, wx2, wy2], under: under.length, winSum, bg, similar: sim, sealed: sea, seeds: [...counts] });
  }
  if (!counts.size) return done(null, "no seeds");
  let id = -1;
  let idCount = -1;
  for (const [lab, n] of counts) if (n > idCount || (n === idCount && lab < id)) (id = lab), (idCount = n);
  const eqd = eq.data;
  for (let i = 0; i < npx; i++) eqd[i] = L[i] === id ? 1 : 0;
  const comp = keep(new cv.Mat());
  cv.dilate(eq, comp, kernel, anchor, 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());

  const contours = new cv.MatVector();
  const hierarchy = keep(new cv.Mat());
  cv.findContours(comp, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  let ci = 0;
  let bestArea = -1;
  for (let i = 0; i < contours.size(); i++) {
    const c = contours.get(i);
    const a = cv.contourArea(c);
    if (a > bestArea) (bestArea = a), (ci = i);
    c.delete();
  }
  const contour = keep(contours.get(ci));
  contours.delete();
  const filled = keep(cv.Mat.zeros(wh, ww, cv.CV_8UC1));
  const one = new cv.MatVector();
  one.push_back(contour);
  cv.drawContours(filled, one, -1, new cv.Scalar(1), cv.FILLED);
  one.delete();
  let area = 0;
  let fd = filled.data;
  for (let i = 0; i < npx; i++) area += fd[i];
  const br = cv.boundingRect(contour);
  const { x: cx, y: cy, width: cw, height: ch } = br;
  const leaked = (cx <= 0 && 0 < wx1) || (cy <= 0 && 0 < wy1) || (cx + cw >= ww && wx2 < W) || (cy + ch >= wh && wy2 < H);
  if (leaked) return done("leaked", `leaked ${JSON.stringify([cx, cy, cw, ch, ww, wh, wx1, wy1, wx2, wy2])}`);
  if (area / (bw * bh) > BUBBLE.maxAreaRatio) return done(null, `area ${area / (bw * bh)}`);
  const hull = keep(new cv.Mat());
  cv.convexHull(contour, hull, false, true);
  if (area / Math.max(1, cv.contourArea(hull)) < BUBBLE.minSolidity) return done(null, `solidity ${area / Math.max(1, cv.contourArea(hull))}`);
  const ry1 = Math.max(0, y1 - wy1);
  const ry2 = Math.min(wh, y2 - wy1);
  const rx1 = Math.max(0, x1 - wx1);
  const rx2 = Math.min(ww, x2 - wx1);
  let cov = 0;
  fd = filled.data;
  for (let y = ry1; y < ry2; y++) for (let x = rx1; x < rx2; x++) cov += fd[y * ww + x];
  const n = Math.max(0, ry2 - ry1) * Math.max(0, rx2 - rx1);
  if (n && cov / n < BUBBLE.minCovered) return done(null, `covered ${cov / n}`);

  const scale = Math.max(1, Math.max(cw, ch) / BUBBLE.rectGrid);
  const sub = filled.roi(new cv.Rect(cx, cy, cw, ch));
  const small = keep(new cv.Mat());
  cv.resize(sub, small, new cv.Size(Math.max(1, pyRound(cw / scale)), Math.max(1, pyRound(ch / scale))), 0, 0, cv.INTER_AREA);
  sub.delete();
  const k3 = keep(cv.Mat.ones(3, 3, cv.CV_8U));
  const eroded = keep(new cv.Mat());
  cv.erode(small, eroded, k3, anchor, 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
  const [sx1, sy1, sx2, sy2] = largestRect(new Uint8Array(eroded.data), eroded.cols, eroded.rows);
  const inner = [wx1 + cx + Math.trunc(sx1 * scale), wy1 + cy + Math.trunc(sy1 * scale), wx1 + cx + Math.trunc(sx2 * scale), wy1 + cy + Math.trunc(sy2 * scale)];

  const bgPx = [[], [], []];
  fd = filled.data;
  sd = similar.data;
  for (let y = 0; y < wh; y++) {
    for (let x = 0; x < ww; x++) {
      const i = y * ww + x;
      if (fd[i] && sd[i]) {
        const p = ((wy1 + y) * img.width + wx1 + x) * 3;
        for (let c = 0; c < 3; c++) bgPx[c].push(img.data[p + c]);
      }
    }
  }
  const fill = bgPx[0].length ? bgPx.map((v) => Math.trunc(median(v))) : [255, 255, 255];
  const eps = Math.max(1.0, 0.004 * cv.arcLength(contour, true));
  const approx = keep(new cv.Mat());
  cv.approxPolyDP(contour, approx, eps, true);
  const poly = contourPoints(approx).map(([x, y]) => [x + wx1, y + wy1]);
  return done(withId({ poly, inner, fill }));
}

export function withId(b) {
  const xs = b.poly.map((p) => p[0]);
  const ys = b.poly.map((p) => p[1]);
  return { ...b, id: `b${Math.min(...xs)}_${Math.min(...ys)}_${Math.max(...xs)}_${Math.max(...ys)}` };
}

const bboxOf = (poly) => [Math.min(...poly.map((p) => p[0])), Math.min(...poly.map((p) => p[1])), Math.max(...poly.map((p) => p[0])), Math.max(...poly.map((p) => p[1]))];

function iouSmall(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const small = Math.min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]));
  return (ix * iy) / Math.max(1, small);
}

function containing(cv, found, region) {
  const [x1, y1, x2, y2] = region.bbox;
  const centre = new cv.Point((x1 + x2) / 2, (y1 + y2) / 2);
  for (const b of found) {
    const m = cv.matFromArray(b.poly.length, 1, cv.CV_32SC2, b.poly.flat());
    const inside = cv.pointPolygonTest(m, centre, false) >= 0;
    m.delete();
    if (inside) return b;
  }
  return null;
}

function find(cv, img, gray, region, trace) {
  const [x1, y1, x2, y2] = region.bbox;
  if (x2 <= x1 || y2 <= y1 || !region.lines?.length) return null;
  for (const grow of BUBBLE.windows) {
    const r = findInWindow(cv, img, gray, region, grow, trace);
    if (r !== "leaked") return r;
  }
  return null;
}

/** bubbles.assign: img = { data: RGB Uint8Array, width, height }; sets region.bubble. */
export function assignBubbles(cv, img, regions, trace = false) {
  const rgb = cv.matFromArray(img.height, img.width, cv.CV_8UC3, img.data);
  const gray = new cv.Mat();
  cv.cvtColor(rgb, gray, cv.COLOR_RGB2GRAY);
  rgb.delete();
  const hasText = (r) => !!pyStrip(r.text || "");
  const found = [];
  for (const r of regions) {
    if (!hasText(r)) continue;
    const inside = containing(cv, found, r);
    if (inside) {
      r.bubble = inside;
      continue;
    }
    if (trace) r.bubbleTrace = [];
    let b = find(cv, img, gray, r, r.bubbleTrace);
    if (!b) continue;
    const same = found.find((o) => iouSmall(bboxOf(o.poly), bboxOf(b.poly)) >= BUBBLE.sameIou);
    if (same) b = same;
    else found.push(b);
    r.bubble = b;
  }
  for (const r of regions) {
    if (!r.bubble && hasText(r)) {
      const inside = containing(cv, found, r);
      if (inside) r.bubble = inside;
    }
  }
  gray.delete();
  return regions.filter((r) => r.bubble).length;
}
