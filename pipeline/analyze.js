// SPDX-License-Identifier: GPL-3.0-only
// Browser version of the app's POST /analyze: detect → OCR → merge → manga-ocr reread → bubbles.
// Same output shape as Analyzer.analyze (width, height, regions, det_lines, stage_ms, total_ms).

import { assignBubbles } from "./bubbles.js";
import { detect } from "./detect.js";
import { regionToDict, textlineMerge } from "./merge.js";
import { MOCR, MangaOcr, reread } from "./mocr.js";
import { ocr, REC } from "./ocr.js";
import { loadTextTables } from "./text.js";

let libs = null;

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error(`failed to load ${src}`));
    document.head.appendChild(s);
  });
}

/** opencv.js (4.11, same as the app's cv2) and clipper-lib (Clipper 6.4.2, same as pyclipper): classic scripts. */
export function loadLibs(base = "/node_modules/") {
  libs ??= (async () => {
    await Promise.all([loadScript(base + "@techstark/opencv-js/dist/opencv.js"), loadScript(base + "clipper-lib/clipper.js")]);
    let cv = globalThis.cv;
    if (cv instanceof Promise) cv = await cv;
    else if (!cv.Mat) await new Promise((r) => (cv.onRuntimeInitialized = r));
    return { cv, Clipper: globalThis.ClipperLib };
  })();
  return libs;
}

/** fp16 weight file → the fp32 bytes the graph's external data expects. */
async function widenHalf(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to load ${url}: HTTP ${res.status}`);
  return new Uint8Array(new Float32Array(new Float16Array(await res.arrayBuffer())).buffer);
}

/** RGBA ImageData (or RGB) → RGB, alpha dropped like PIL convert("RGB"). With cv, opencv's cvtColor
 * does it (the JS loop runs once per page, never gets optimized, and costs ~35 ms on a 690×2000 strip). */
export function toRGB({ data, width, height }, cv = null) {
  if (data.length === width * height * 3) return { data, width, height };
  if (cv) {
    const rgba = cv.matFromArray(height, width, cv.CV_8UC4, data);
    const rgb = new cv.Mat();
    cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB);
    const out = { data: rgb.data.slice(), width, height };
    rgba.delete();
    rgb.delete();
    return out;
  }
  const rgb = new Uint8Array(width * height * 3);
  for (let p = 0, q = 0; q < rgb.length; p += 4, q += 3) {
    rgb[q] = data[p];
    rgb[q + 1] = data[p + 1];
    rgb[q + 2] = data[p + 2];
  }
  return { data: rgb, width, height };
}

export class Pipeline {
  static async create(ort, opts = {}) {
    const o = {
      ep: "webgpu",
      models: "/models/",
      libs: "/node_modules/",
      // fp32 graph whose external weights ship as fp16 and are widened on load: fp16 compute on
      // WebGPU wrecks the probability map on some pages (the final ConvTranspose)
      det: "det16.onnx",
      detWeights: "det16.bin",
      rec: "PP-OCRv6_small_rec_top1.onnx", // outputs per-step argmax + max, not the full softmax
      dict: "ppocrv6_dict.txt",
      mocr: "fp16",
      useMocr: true,
      beams: MOCR.beams,
      ...opts,
    };
    const t0 = performance.now();
    const sess = (file, externalData) => ort.InferenceSession.create(o.models + file, { executionProviders: [o.ep], graphOptimizationLevel: "all", externalData });
    const suffix = o.mocr === "fp16" ? "_fp16" : "";
    const detData = o.detWeights ? [{ path: o.detWeights, data: await widenHalf(o.models + o.detWeights) }] : undefined;
    const aux = Promise.all([
      loadLibs(o.libs),
      fetch(o.models + o.dict).then((r) => r.text()),
      o.useMocr ? fetch(o.models + "mocr_vocab.json").then((r) => r.json()) : null,
      loadTextTables(o.models),
    ]);
    // the WebGPU EP refuses concurrent session creation
    const det = await sess(o.det, detData);
    const rec = await sess(o.rec);
    const enc = o.useMocr ? await sess(`mocr_encoder${suffix}.onnx`) : null;
    const dec = o.useMocr ? await sess(`mocr_decoder${suffix}.onnx`) : null;
    const [{ cv, Clipper }, dict, vocab] = await aux;
    const p = new Pipeline();
    Object.assign(p, { ort, opts: o, cv, Clipper, det, rec });
    p.charset = dict.replace(/\n+$/, "").split("\n");
    p.reader = o.useMocr ? new MangaOcr(ort, o.ep, enc, dec, vocab) : null;
    p.loadMs = performance.now() - t0;
    return p;
  }

  runDet = async (x, dims) => {
    const { ort, det } = this;
    const out = await det.run({ [det.inputNames[0]]: new ort.Tensor("float32", x, dims) });
    return out[det.outputNames[0]].data;
  };

  runRec = async (x, dims) => {
    const { ort, rec } = this;
    const out = await rec.run({ [rec.inputNames[0]]: new ort.Tensor("float32", x, dims) });
    return { best: out.best.data, bestP: out.best_p.data, dims: out.best.dims };
  };

  /** Runs every model once on blank input so the first real page doesn't pay for WebGPU shader
   * compilation. Detector shapes are canvas [h, w]; 2304×768 is what ~700 px wide strips give. */
  async warmup(detShapes = [[2304, 768]]) {
    const t0 = performance.now();
    for (const [h, w] of detShapes) await this.runDet(new Float32Array(3 * h * w), [1, 3, h, w]);
    const { textH, batch, buckets } = REC;
    for (let n = 1; n <= batch; n *= 2) await this.runRec(new Float32Array(n * 3 * textH * buckets[0]), [n, 3, textH, buckets[0]]);
    if (this.reader) await this.reader.warmup(this.opts.beams);
    this.warmMs = performance.now() - t0;
    return this.warmMs;
  }

  /** image: ImageData-like ({ data: RGBA or RGB, width, height }); beams overrides the manga-ocr beam count. */
  async analyze(image, { beams = this.opts.beams } = {}) {
    const { cv, Clipper } = this;
    const t0 = performance.now();
    const stage = {};
    const mark = (() => {
      let last = t0;
      return (name) => {
        const now = performance.now();
        stage[name] = Math.round((now - last) * 10) / 10;
        last = now;
      };
    })();
    const rgb = toRGB(image);
    const out = { width: rgb.width, height: rgb.height, regions: [], det_lines: [], stage_ms: stage, total_ms: 0, stages: {} };

    const d = await detect(cv, Clipper, rgb, this.runDet);
    mark("detection");
    out.det_lines = d.detLines;
    out.stages.det = d.ms;
    if (d.textlines.length) {
      const img = cv.matFromArray(rgb.height, rgb.width, cv.CV_8UC3, rgb.data);
      const o = await ocr(cv, img, d.textlines, this.charset, this.runRec);
      img.delete();
      mark("ocr");
      out.stages.ocr = o.textlines.map((q) => ({ pts: q.pts, text: q.text, prob: q.prob, assigned: q.assignedDirection }));
      if (o.textlines.length) {
        const m = textlineMerge(o.textlines);
        mark("textline_merge");
        out.stages.merged = m.merged;
        out.stages.kept = m.kept.map((r) => ({ ...r }));
        if (this.reader && m.kept.length) {
          out.stages.reads = await reread(this.reader, rgb, m.kept, beams);
          mark("manga_ocr");
        }
        out.regions = m.kept.map(regionToDict);
      }
    }
    assignBubbles(cv, rgb, out.regions);
    mark("bubbles");
    out.total_ms = Math.round((performance.now() - t0) * 10) / 10;
    return out;
  }
}
