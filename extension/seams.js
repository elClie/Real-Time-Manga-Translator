// Bubbles cut by the boundary between two stacked strips (webtoon slices); port of app/seams.py.
//
// Each strip is analyzed alone, so a bubble crossing the boundary comes out as two broken halves.
// When the extension says two analyzed strips are stacked and a detection touches their shared
// edge, the rows around the edge are stitched and analyzed again. Bubbles crossing the boundary
// replace the fragments in both strips and are kept by the strip holding most of them; their
// coordinates may then run past that strip's edge.

import { withId } from "./pipeline/bubbles.js";
import { pyStrip } from "./pipeline/text.js";

const EDGE = 8; // px: a detection this close to the shared edge may continue in the neighbour
const BAND = 700; // rows taken from each strip
const MAX_SEAM_DIFF = 25.0; // mean abs pixel difference across the edge; real neighbours measure 0–22
const FRAGMENT_OVERLAP = 0.3; // share of a region's bbox inside a stitched bubble for it to be replaced

const touchesBottom = (page) => page.det_lines.some((q) => Math.max(...q.map((p) => p[1])) >= page.height - 1 - EDGE);
const touchesTop = (page) => page.det_lines.some((q) => Math.min(...q.map((p) => p[1])) <= EDGE);

/** img: { data: RGB Uint8Array, width, height } */
function continuous(upper, lower, rows = 2) {
  if (upper.width !== lower.width) return false;
  const n = upper.width * 3;
  const ur = Math.min(rows, upper.height);
  const lr = Math.min(rows, lower.height);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    let u = 0;
    for (let r = upper.height - ur; r < upper.height; r++) u += upper.data[r * n + i];
    let l = 0;
    for (let r = 0; r < lr; r++) l += lower.data[r * n + i];
    sum += Math.abs(u / ur - l / lr);
  }
  return sum / n <= MAX_SEAM_DIFF;
}

function overlapFrac(box, other) {
  const x1 = Math.max(box[0], other[0]);
  const y1 = Math.max(box[1], other[1]);
  const x2 = Math.min(box[2], other[2]);
  const y2 = Math.min(box[3], other[3]);
  const area = Math.max(1, (box[2] - box[0]) * (box[3] - box[1]));
  return (Math.max(0, x2 - x1) * Math.max(0, y2 - y1)) / area;
}

function shift(region, dy) {
  const [x1, y1, x2, y2] = region.bbox;
  const out = {
    ...region,
    bbox: [x1, y1 + dy, x2, y2 + dy],
    lines: (region.lines || []).map((q) => q.map(([x, y]) => [x, y + dy])),
    seam: true,
  };
  if (region.bubble) {
    const b = region.bubble;
    const [bx1, by1, bx2, by2] = b.inner;
    out.bubble = withId({ poly: b.poly.map(([x, y]) => [x, y + dy]), inner: [bx1, by1 + dy, bx2, by2 + dy], fill: b.fill });
  }
  return out;
}

/** Re-reads bubbles crossing the upper/lower boundary; edits both pages in place. True if anything changed. */
export async function resolveSeam(analyze, upperPage, upperImg, lowerPage, lowerImg) {
  if (!(touchesBottom(upperPage) || touchesTop(lowerPage))) return false;
  if (!continuous(upperImg, lowerImg)) return false;

  const bu = Math.min(BAND, upperImg.height);
  const bl = Math.min(BAND, lowerImg.height);
  const row = upperImg.width * 3;
  const data = new Uint8Array((bu + bl) * row);
  data.set(upperImg.data.subarray((upperImg.height - bu) * row, upperImg.height * row), 0);
  data.set(lowerImg.data.subarray(0, bl * row), bu * row);
  const seamPage = await analyze({ data, width: upperImg.width, height: bu + bl });
  const crossing = seamPage.regions.filter((r) => pyStrip(r.text || "") && r.bbox[1] < bu - 2 && r.bbox[3] > bu + 2);
  if (!crossing.length) return false;

  const toUpper = upperImg.height - bu;
  const toLower = -bu;
  for (const r of crossing) {
    const inUpper = bu - r.bbox[1] >= r.bbox[3] - bu;
    const [owner, dy] = inUpper ? [upperPage, toUpper] : [lowerPage, toLower];
    const stitched = shift(r, dy);
    for (const [page, off] of [[upperPage, toUpper], [lowerPage, toLower]]) {
      const box = [r.bbox[0], r.bbox[1] + off, r.bbox[2], r.bbox[3] + off];
      const kept = [];
      for (const f of page.regions) {
        if (overlapFrac(f.bbox, box) < FRAGMENT_OVERLAP) kept.push(f);
        else if (page === owner && f.text === stitched.text && f.translation != null) {
          stitched.translation = f.translation; // same bubble stitched again (re-analysis)
        }
      }
      page.regions = kept;
    }
    owner.regions.push(stitched);
  }
  return true;
}
