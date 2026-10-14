// Manga panel detector for calibration: leoxs22/manga-panel-detector-yolo26n (YOLO26-nano trained on
// Manga109-s; classes 0 panel, 1 text), exported to ONNX at 640x640 → [1, 6, 8400] (cx, cy, w, h, panel,
// text). It tells a site's manga pages apart from its covers, banners, photos and thumbnails.
const SIZE = 640;
const CONF = 0.5;
const IOU = 0.7;
const PANEL = 4; // output channel of class 0

export class PanelDetector {
  /** @param {string} url  ONNX model URL */
  static async create(ort, url, ep = "webgpu") {
    const t0 = performance.now();
    const session = await ort.InferenceSession.create(url, { executionProviders: [ep], graphOptimizationLevel: "all" });
    const d = new PanelDetector(ort, session);
    d.loadMs = performance.now() - t0;
    return d;
  }

  constructor(ort, session) {
    this.ort = ort;
    this.session = session;
  }

  /** Panels as [x1, y1, x2, y2, score] in 0..1 of the image, best first. `source`: ImageBitmap / canvas. */
  async detect(source) {
    const { width, height } = source;
    const s = Math.min(SIZE / width, SIZE / height);
    const w = Math.round(width * s);
    const h = Math.round(height * s);
    const left = Math.floor((SIZE - w) / 2);
    const top = Math.floor((SIZE - h) / 2);
    const canvas = new OffscreenCanvas(SIZE, SIZE);
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.fillStyle = "rgb(114,114,114)"; // ultralytics letterbox padding
    ctx.fillRect(0, 0, SIZE, SIZE);
    ctx.drawImage(source, left, top, w, h);
    const px = ctx.getImageData(0, 0, SIZE, SIZE).data;
    const n = SIZE * SIZE;
    const x = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      x[i] = px[i * 4] / 255;
      x[n + i] = px[i * 4 + 1] / 255;
      x[2 * n + i] = px[i * 4 + 2] / 255;
    }
    const out = await this.session.run({ [this.session.inputNames[0]]: new this.ort.Tensor("float32", x, [1, 3, SIZE, SIZE]) });
    const y = out[this.session.outputNames[0]];
    const anchors = y.dims[2];
    const d = y.data;
    const found = [];
    for (let i = 0; i < anchors; i++) {
      const score = d[PANEL * anchors + i];
      if (score < CONF) continue;
      const cx = d[i];
      const cy = d[anchors + i];
      const bw = d[2 * anchors + i];
      const bh = d[3 * anchors + i];
      found.push([clamp((cx - bw / 2 - left) / w), clamp((cy - bh / 2 - top) / h), clamp((cx + bw / 2 - left) / w), clamp((cy + bh / 2 - top) / h), score]);
    }
    found.sort((a, b) => b[4] - a[4]);
    const keep = [];
    for (const b of found) if (keep.every((k) => iou(k, b) < IOU)) keep.push(b);
    return keep;
  }
}

const clamp = (v) => Math.min(1, Math.max(0, v));

function iou(a, b) {
  const iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = iw * ih;
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter);
}

/** A manga page rather than a cover/banner/photo/thumbnail: portrait, with a panel inside it instead of
 * the whole picture framed as one (dev/tools/panel_negatives.py: 80/82 pages, 2/196 other images). */
export function looksLikePage(boxes, width, height) {
  return height >= 1.2 * width && boxes.some((b) => (b[2] - b[0]) * (b[3] - b[1]) < 0.85);
}
