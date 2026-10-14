// MIT default detector as the app runs it (det_speedups._fast_infer, no bilateral filter):
// resize + reflect margin → DBNet probability map (models/det*.onnx) → SegDetectorRepresenter
// boxes_from_bitmap → page coordinates → CommonDetector filters / small-image border.

import { Quad, pyRound } from "./quad.js";

export const DET = { size: 2048, textThresh: 0.5, boxThresh: 0.7, unclip: 2.3, margin: 32, maxCandidates: 1000, minSide: 400 };

const f32 = Math.fround;
const up256 = (v) => Math.ceil(v / 256) * 256;

/** det_speedups._resize_with_margin (reflect fill). */
function resizeWithMargin(cv, src, h, w) {
  const ratio = DET.size / Math.max(h, w);
  const th = pyRound(h * ratio);
  const tw = pyRound(w * ratio);
  const resized = new cv.Mat();
  cv.resize(src, resized, new cv.Size(tw, th), 0, 0, cv.INTER_LINEAR);
  const top = Math.max(32, Math.floor(DET.margin / 32) * 32);
  const left = up256(tw) - tw >= 2 * top ? top : 0;
  const ch = up256(th + 2 * top);
  const cw = up256(tw + 2 * left);
  const canvas = new cv.Mat();
  cv.copyMakeBorder(resized, canvas, top, ch - th - top, left, cw - tw - left, cv.BORDER_REFLECT_101);
  resized.delete();
  return { canvas, ratio, top, left };
}

/** cv::RotatedRect::points in float32, for builds without cv.boxPoints. */
function rectPoints(r) {
  const angle = (r.angle * Math.PI) / 180;
  const b = f32(f32(Math.cos(angle)) * 0.5);
  const a = f32(f32(Math.sin(angle)) * 0.5);
  const { x: cx, y: cy } = r.center;
  const { width: w, height: h } = r.size;
  const p0 = [f32(f32(cx - f32(a * h)) - f32(b * w)), f32(f32(cy + f32(b * h)) - f32(a * w))];
  const p1 = [f32(f32(cx + f32(a * h)) - f32(b * w)), f32(f32(cy - f32(b * h)) - f32(a * w))];
  return [p0, p1, [f32(f32(2 * cx) - p0[0]), f32(f32(2 * cy) - p0[1])], [f32(f32(2 * cx) - p1[0]), f32(f32(2 * cy) - p1[1])]];
}

function boxPoints(cv, rect) {
  if (!cv.boxPoints) return rectPoints(rect);
  const r = cv.boxPoints(rect);
  if (Array.isArray(r)) return r.map((p) => [p.x, p.y]);
  const n = r.size();
  const pts = [];
  for (let i = 0; i < n; i++) {
    const p = r.get(i);
    pts.push([p.x, p.y]);
  }
  r.delete?.();
  return pts;
}

/** SegDetectorRepresenter.get_mini_boxes on a CV_32SC2 point Mat. */
function miniBox(cv, mat) {
  const rect = cv.minAreaRect(mat);
  const p = boxPoints(cv, rect)
    .map((q, i) => ({ q, i }))
    .sort((a, b) => a.q[0] - b.q[0] || a.i - b.i)
    .map((o) => o.q);
  const [i1, i4] = p[1][1] > p[0][1] ? [0, 1] : [1, 0];
  const [i2, i3] = p[3][1] > p[2][1] ? [2, 3] : [3, 2];
  return { box: [p[i1], p[i2], p[i3], p[i4]], sside: Math.min(rect.size.width, rect.size.height) };
}

/** box_score_fast: mean probability inside the contour polygon. */
function boxScore(cv, prob, W, H, pts) {
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (let i = 0; i < pts.length; i += 2) {
    xmin = Math.min(xmin, pts[i]);
    xmax = Math.max(xmax, pts[i]);
    ymin = Math.min(ymin, pts[i + 1]);
    ymax = Math.max(ymax, pts[i + 1]);
  }
  xmin = Math.min(Math.max(Math.floor(xmin), 0), W - 1);
  xmax = Math.min(Math.max(Math.ceil(xmax), 0), W - 1);
  ymin = Math.min(Math.max(Math.floor(ymin), 0), H - 1);
  ymax = Math.min(Math.max(Math.ceil(ymax), 0), H - 1);
  const mw = xmax - xmin + 1;
  const mh = ymax - ymin + 1;
  const shifted = new Int32Array(pts.length);
  for (let i = 0; i < pts.length; i += 2) {
    shifted[i] = pts[i] - xmin;
    shifted[i + 1] = pts[i + 1] - ymin;
  }
  const mask = cv.Mat.zeros(mh, mw, cv.CV_8UC1);
  const poly = cv.matFromArray(pts.length / 2, 1, cv.CV_32SC2, shifted);
  const polys = new cv.MatVector();
  polys.push_back(poly);
  cv.fillPoly(mask, polys, new cv.Scalar(1));
  const m = mask.data;
  let sum = 0;
  let n = 0;
  for (let y = 0; y < mh; y++) {
    const row = (y + ymin) * W + xmin;
    for (let x = 0; x < mw; x++) {
      if (m[y * mw + x]) {
        sum += prob[row + x];
        n++;
      }
    }
  }
  mask.delete();
  poly.delete();
  polys.delete();
  return n ? sum / n : 0;
}

/** SegDetectorRepresenter.unclip: pyclipper round offset by area * ratio / perimeter. */
function unclip(Clipper, box) {
  let area = 0;
  let length = 0;
  for (let i = 0; i < 4; i++) {
    const a = box[i];
    const b = box[(i + 1) % 4];
    area += a[0] * b[1] - b[0] * a[1];
    length += Math.hypot(b[0] - a[0], b[1] - a[1]);
  }
  const distance = (Math.abs(area) / 2) * DET.unclip / length;
  const co = new Clipper.ClipperOffset(2, 0.25);
  co.AddPath(box.map(([x, y]) => ({ X: Math.trunc(x), Y: Math.trunc(y) })), Clipper.JoinType.jtRound, Clipper.EndType.etClosedPolygon);
  const sol = new Clipper.Paths();
  co.Execute(sol, distance);
  return sol.flat().map((p) => [p.X, p.Y]);
}

/** boxes_from_bitmap (dest size = map size), then the _fast_infer row filter. */
function boxesFromBitmap(cv, Clipper, prob, W, H) {
  const bitmap = new cv.Mat(H, W, cv.CV_8UC1);
  const bits = bitmap.data;
  for (let i = 0; i < bits.length; i++) bits[i] = prob[i] > DET.textThresh ? 255 : 0;
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(bitmap, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
  const boxes = [];
  const n = Math.min(contours.size(), DET.maxCandidates);
  for (let i = 0; i < n; i++) {
    const contour = contours.get(i);
    const pts = Int32Array.from(contour.data32S);
    const { box: mini, sside } = miniBox(cv, contour);
    let keep = sside >= 3 && boxScore(cv, prob, W, H, pts) >= DET.boxThresh;
    contour.delete();
    if (!keep) continue;
    const expanded = unclip(Clipper, mini);
    if (!expanded.length) continue;
    const em = cv.matFromArray(expanded.length, 1, cv.CV_32SC2, expanded.flat());
    const { box, sside: s2 } = miniBox(cv, em);
    em.delete();
    if (s2 < 5) continue;
    const scaled = box.map(([x, y]) => [
      Math.min(Math.max(pyRound(f32(f32(x / W) * W)), 0), W),
      Math.min(Math.max(pyRound(f32(f32(y / H) * H)), 0), H),
    ]);
    const sums = scaled.map(([x, y]) => x + y);
    const start = sums.indexOf(Math.min(...sums));
    const rolled = [0, 1, 2, 3].map((k) => scaled[(k + start) % 4]);
    if (rolled.some(([x, y]) => x + y !== 0)) boxes.push(rolled);
  }
  bitmap.delete();
  contours.delete();
  hierarchy.delete();
  return boxes;
}

/** Small-image border + resize with margin. Returns the model input and what post() needs. */
export function prepare(cv, rgb) {
  const W0 = rgb.width;
  const H0 = rgb.height;
  let img = cv.matFromArray(H0, W0, cv.CV_8UC3, rgb.data);
  const border = Math.min(W0, H0) < DET.minSide;
  if (border) {
    const S = Math.max(W0, H0, DET.minSide);
    const sq = new cv.Mat(S, S, cv.CV_8UC3, new cv.Scalar(0, 0, 0));
    const roi = sq.roi(new cv.Rect(0, 0, W0, H0));
    img.copyTo(roi);
    roi.delete();
    img.delete();
    img = sq;
  }
  const H = img.rows;
  const W = img.cols;
  const long = Math.max(H, W);
  if (long / DET.size > 2.5 && long / Math.min(H, W) > 3) console.warn("detect: MIT would rearrange this tall image; single pass used");

  const { canvas, ratio, top, left } = resizeWithMargin(cv, img, H, W);
  img.delete();
  const ch = canvas.rows;
  const cw = canvas.cols;
  const src = canvas.data;
  const plane = ch * cw;
  const x = new Float32Array(3 * plane);
  for (let p = 0, q = 0; p < plane; p++, q += 3) {
    x[p] = src[q];
    x[plane + p] = src[q + 1];
    x[2 * plane + p] = src[q + 2];
  }
  const canvasRGB = new Uint8Array(src);
  canvas.delete();
  return { x, dims: [1, 3, ch, cw], canvasRGB, meta: { W0, H0, W, H, border, ratio, top, left, cw, ch } };
}

/** Probability map → textline quads in page coordinates (MIT order). */
export function post(cv, Clipper, prob, meta) {
  const { W0, H0, W, H, border, ratio, top, left, cw, ch } = meta;
  const boxes = boxesFromBitmap(cv, Clipper, prob, cw, ch);
  const inv = 1 / ratio;
  let quads = boxes
    .map((box) =>
      box.map(([bx, by]) => [
        Math.min(Math.max(Math.trunc((bx - left) * inv), 0), W - 1),
        Math.min(Math.max(Math.trunc((by - top) * inv), 0), H - 1),
      ]),
    )
    .map((pts) => new Quad(pts))
    .filter((q) => q.area > 16)
    .filter((q) => q.area > 1);
  if (border) {
    quads = quads
      .filter((q) => !(q.xyxy[0] >= W0 && q.xyxy[1] >= H0))
      .map((q) => new Quad(q.pts.map(([px, py]) => [Math.min(Math.max(px, 0), W0), Math.min(Math.max(py, 0), H0)])));
  }
  return { textlines: quads, detLines: quads.map((q) => q.pts.map((p) => [...p])) };
}

/**
 * rgb: { data: Uint8Array (RGB), width, height }. runModel(Float32Array NCHW, [1,3,h,w]) → Float32Array prob map.
 * Returns { textlines: Quad[], detLines: [[x,y]*4][], ms: {...} }.
 */
export async function detect(cv, Clipper, rgb, runModel) {
  const t0 = performance.now();
  const { x, dims, meta } = prepare(cv, rgb);
  const t1 = performance.now();
  const prob = await runModel(x, dims);
  const t2 = performance.now();
  const out = post(cv, Clipper, prob, meta);
  return { ...out, prob, meta, ms: { prep: t1 - t0, model: t2 - t1, post: performance.now() - t2 } };
}
