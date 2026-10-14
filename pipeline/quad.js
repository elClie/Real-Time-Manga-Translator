// MIT utils/generic.py Quadrilateral + quadrilateral_can_merge_region, and the shapely bits they use.
// MIT keeps corner points as ints and structure norms as float32 (float32 / float32 stays float32,
// float32 with a Python scalar becomes float64); Math.fround reproduces the float32 steps.

const f32 = Math.fround;
const norm32 = (x, y) => f32(Math.sqrt(f32(f32(x * x) + f32(y * y))));

/** Python round(): half to even. */
export function pyRound(v) {
  const r = Math.round(v);
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** Stable argsort (numpy's small-array sort is an insertion sort, so stable). */
const argsort = (vals) => vals.map((v, i) => i).sort((a, b) => vals[a] - vals[b]);

function sortPnts(pts) {
  const vecs = [];
  for (const a of pts) for (const b of pts) vecs.push([a[0] - b[0], a[1] - b[1]]);
  const norms = vecs.map(([x, y]) => Math.hypot(x, y));
  const order = argsort(norms);
  const l0 = [...vecs[order[8]]];
  const l1 = vecs[order[10]];
  if (l0[0] * l1[0] + l0[1] * l1[1] < 0) {
    l0[0] = -l0[0];
    l0[1] = -l0[1];
  }
  const sx = Math.abs((l0[0] + l1[0]) / 2);
  const sy = Math.abs((l0[1] + l1[1]) / 2);
  const vertical = sx <= sy;
  if (vertical) {
    let p = argsort(pts.map((q) => q[1])).map((i) => pts[i]);
    const top = argsort([p[0][0], p[1][0]]).map((i) => p[i]);
    const bottom = argsort([p[2][0], p[3][0]]).reverse().map((i) => p[i + 2]);
    return { pts: [...top, ...bottom], vertical };
  }
  const p = argsort(pts.map((q) => q[0])).map((i) => pts[i]);
  const left = [p[0], p[1]].sort((a, b) => a[1] - b[1]);
  const right = [p[2], p[3]].sort((a, b) => a[1] - b[1]);
  return { pts: [left[0], right[0], right[1], left[1]], vertical };
}

export class Quad {
  constructor(pts, text = "", prob = 0) {
    const s = sortPnts(pts.map(([x, y]) => [Math.trunc(x), Math.trunc(y)]));
    this.pts = s.pts;
    this.direction = s.vertical ? "v" : "h";
    this.text = text;
    this.prob = prob;
    this.assignedDirection = null;
    const [a, b, c, d] = this.pts;
    const mid = (p, q) => [Math.trunc((p[0] + q[0]) / 2), Math.trunc((p[1] + q[1]) / 2)];
    this.structure = [mid(a, b), mid(c, d), mid(b, c), mid(d, a)];
    const [l1a, l1b, l2a, l2b] = this.structure;
    this.v1 = [l1b[0] - l1a[0], l1b[1] - l1a[1]];
    this.v2 = [l2b[0] - l2a[0], l2b[1] - l2a[1]];
    this.n1 = norm32(...this.v1);
    this.n2 = norm32(...this.v2);
    this.aspectRatio = f32(this.n2 / this.n1);
    this.fontSize = Math.min(this.n1, this.n2);
    const xs = this.pts.map((p) => p[0]);
    const ys = this.pts.map((p) => p[1]);
    const x = Math.min(...xs);
    const y = Math.min(...ys);
    this.aabb = { x, y, w: Math.max(...xs) - x, h: Math.max(...ys) - y };
    this.xyxy = [x, y, x + this.aabb.w, y + this.aabb.h];
    this.centroid = [xs.reduce((s, v) => s + v, 0) / 4, ys.reduce((s, v) => s + v, 0) / 4];
    const u1 = [f32(this.v1[0] / this.n1), f32(this.v1[1] / this.n1)];
    const u2 = [f32(this.v2[0] / this.n2), f32(this.v2[1] / this.n2)];
    this.approxAxisAligned = Math.abs(u1[1]) < 0.05 || Math.abs(u1[0]) < 0.05 || Math.abs(u2[1]) < 0.05 || Math.abs(u2[0]) < 0.05;
    this.angle = (Math.acos(u1[0]) + Math.PI) % Math.PI;
    this.polygon = convexHull(this.pts);
    this.area = polygonArea(this.polygon);
  }

  /** Quadrilateral.distance (distance_impl), rho = 0.5. Uses this line's assigned direction. */
  distance(other, rho = 0.5) {
    const fs = Math.max(this.fontSize, other.fontSize);
    const P = this.pts;
    const Q = other.pts;
    const d = (p, q) => Math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2);
    if (this.assignedDirection === "h") {
      let pattern = "h_left";
      const dist1 = hullArea([P[0], P[3], Q[0], Q[3]]) / fs;
      const dist2 = hullArea([P[2], P[1], Q[2], Q[1]]) / fs;
      const dist3 = hullArea([this.structure[0], this.structure[1], other.structure[0], other.structure[1]]) / fs;
      if (dist1 < fs * rho) pattern = "h_left";
      if (dist2 < fs * rho && dist2 < dist1) pattern = "h_right";
      if (dist3 < fs * rho && dist3 < dist1 && dist3 < dist2) pattern = "h_middle";
      if (pattern === "h_left") return d(P[0], Q[0]);
      if (pattern === "h_right") return d(P[1], Q[1]);
      return d(this.structure[0], other.structure[0]);
    }
    let pattern = "v_top";
    const dist1 = hullArea([P[0], P[1], Q[0], Q[1]]) / fs;
    const dist2 = hullArea([P[2], P[3], Q[2], Q[3]]) / fs;
    if (dist1 < fs * rho) pattern = "v_top";
    if (dist2 < fs * rho && dist2 < dist1) pattern = "v_bottom";
    return pattern === "v_top" ? d(P[0], Q[0]) : d(P[2], Q[2]);
  }

  polyDistance(other) {
    return polygonDistance(this.polygon, other.polygon);
  }
}

// ---------- shapely stand-ins ----------

function cross(o, a, b) {
  return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
}

/** Convex hull (monotone chain); degenerate input gives 1–2 points, like shapely's Point/LineString. */
export function convexHull(points) {
  const p = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const uniq = p.filter((q, i) => i === 0 || q[0] !== p[i - 1][0] || q[1] !== p[i - 1][1]);
  if (uniq.length < 3) return uniq;
  const lower = [];
  for (const q of uniq) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
    lower.push(q);
  }
  const upper = [];
  for (let i = uniq.length - 1; i >= 0; i--) {
    const q = uniq[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
    upper.push(q);
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

export function polygonArea(poly) {
  if (poly.length < 3) return 0;
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % poly.length];
    s += a[0] * b[1] - b[0] * a[1];
  }
  return Math.abs(s) / 2;
}

const hullArea = (pts) => polygonArea(convexHull(pts));

function pointSegDist(p, a, b) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len2 = dx * dx + dy * dy;
  let t = len2 ? ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2 : 0;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
}

function segsIntersect(a, b, c, d) {
  const d1 = cross(c, d, a);
  const d2 = cross(c, d, b);
  const d3 = cross(a, b, c);
  const d4 = cross(a, b, d);
  if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) return true;
  const on = (p, q, r) => Math.min(p[0], q[0]) <= r[0] && r[0] <= Math.max(p[0], q[0]) && Math.min(p[1], q[1]) <= r[1] && r[1] <= Math.max(p[1], q[1]);
  return (d1 === 0 && on(c, d, a)) || (d2 === 0 && on(c, d, b)) || (d3 === 0 && on(a, b, c)) || (d4 === 0 && on(a, b, d));
}

function insidePolygon(p, poly) {
  if (poly.length < 3) return false;
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > p[1] !== yj > p[1] && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

const edgesOf = (poly) => (poly.length === 1 ? [[poly[0], poly[0]]] : poly.length === 2 ? [[poly[0], poly[1]]] : poly.map((p, i) => [p, poly[(i + 1) % poly.length]]));

/** shapely geometry.distance for (possibly degenerate) polygons: 0 when they touch or overlap. */
export function polygonDistance(A, B) {
  const ea = edgesOf(A);
  const eb = edgesOf(B);
  for (const [a, b] of ea) for (const [c, d] of eb) if (segsIntersect(a, b, c, d)) return 0;
  if (insidePolygon(A[0], B) || insidePolygon(B[0], A)) return 0;
  let best = Infinity;
  for (const p of A) for (const [c, d] of eb) best = Math.min(best, pointSegDist(p, c, d));
  for (const p of B) for (const [a, b] of ea) best = Math.min(best, pointSegDist(p, a, b));
  return best;
}

// ---------- merge test ----------

/** quadrilateral_can_merge_region behind the app's aabb prefilter (ocr_speedups), same answers. */
export function canMerge(a, b, { ratio = 1.9, gap = 2, charGapTol = 0.6, charGapTol2 = 1.5, fontSizeRatioTol = 1.5, aspectRatioTol = 2 } = {}) {
  const b1 = a.aabb;
  const b2 = b.aabb;
  const charSize = Math.min(a.fontSize, b.fontSize);
  const gx = Math.max(b1.x - (b2.x + b2.w), b2.x - (b1.x + b1.w), 0);
  const gy = Math.max(b1.y - (b2.y + b2.h), b2.y - (b1.y + b1.h), 0);
  const limit = gap * charSize;
  if (gx * gx + gy * gy > limit * limit) return false;

  const { x: x1, y: y1, w: w1, h: h1 } = b1;
  const { x: x2, y: y2, w: w2, h: h2 } = b2;
  const dist = polygonDistance(a.pts, b.pts);
  if (dist > gap * charSize) return false;
  if (f32(Math.max(a.fontSize, b.fontSize) / charSize) > fontSizeRatioTol) return false;
  if (a.aspectRatio > aspectRatioTol && b.aspectRatio < 1 / aspectRatioTol) return false;
  if (b.aspectRatio > aspectRatioTol && a.aspectRatio < 1 / aspectRatioTol) return false;
  if (a.approxAxisAligned && b.approxAxisAligned) {
    if (dist < charSize * charGapTol) {
      if (Math.abs(x1 + Math.floor(w1 / 2) - (x2 + Math.floor(w2 / 2))) < charGapTol2) return true;
      if (w1 > h1 * ratio && h2 > w2 * ratio) return false;
      if (w2 > h2 * ratio && h1 > w1 * ratio) return false;
      if (w1 > h1 * ratio || w2 > h2 * ratio) {
        return Math.abs(x1 - x2) < charSize * charGapTol2 || Math.abs(x1 + w1 - (x2 + w2)) < charSize * charGapTol2;
      }
      if (h1 > w1 * ratio || h2 > w2 * ratio) {
        return Math.abs(y1 - y2) < charSize * charGapTol2 || Math.abs(y1 + h1 - (y2 + h2)) < charSize * charGapTol2;
      }
      return false;
    }
    return false;
  }
  if (Math.abs(a.angle - b.angle) < (15 * Math.PI) / 180) {
    const fs = Math.min(a.fontSize, b.fontSize);
    if (a.polyDistance(b) > fs * charGapTol2) return false;
    if (f32(Math.abs(f32(a.fontSize - b.fontSize)) / fs) > 0.25) return false;
    return true;
  }
  return false;
}
