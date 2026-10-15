// SPDX-License-Identifier: GPL-3.0-only
// The in-browser replacement for the local app's POST /analyze: runs the pipeline (webport/pipeline)
// and keeps the app's page cache, re-analysis rules and seam pass. Talks only to the service worker.
import * as ort from "./vendor/onnxruntime-web/dist/ort.webgpu.min.mjs";
import { Pipeline, toRGB } from "./pipeline/analyze.js";
import { PanelDetector, looksLikePage } from "./pipeline/panels.js";
import { pyStrip } from "./pipeline/text.js";
import { resolveSeam } from "./seams.js";

ort.env.wasm.wasmPaths = chrome.runtime.getURL("vendor/onnxruntime-web/dist/");
ort.env.wasm.numThreads = 1; // extension pages aren't cross-origin isolated, so no wasm threads

// All sessions share one asyncify wasm module and one WebGPU device, neither of which survives being
// re-entered mid-call (a session created while another loads or runs breaks with "setPipeline ...
// not of type GPUComputePipeline"). The panel detector (calibration) can start while the pipeline
// loads, so session creation and runs go one at a time.
let ortQueue = Promise.resolve();
const oneAtATime = (fn) => {
  const call = ortQueue.then(fn);
  ortQueue = call.catch(() => {});
  return call;
};
const sessionCreate = ort.InferenceSession.create.bind(ort.InferenceSession);
const sessionRun = ort.InferenceSession.prototype.run;
ort.InferenceSession.create = (...args) => oneAtATime(() => sessionCreate(...args));
ort.InferenceSession.prototype.run = function (...args) {
  return oneAtATime(() => sessionRun.apply(this, args));
};

const CACHE_SIZE = 500;
const IMAGE_CACHE_SIZE = 16;
const IDLE_MS = 10 * 60_000; // models hold ~0.5 GB of GPU memory; hand it back when unused

let pipe = null;
let loading = null;
let loadError = null;
let ep = null;
let busy = 0;
const pages = new Map(); // id → analyzed page (translations filled in later)
const images = new Map(); // id → RGB pixels of recent pages, for stitching seams
const seamsDone = new Set(); // "upper|lower" ids already stitched

async function pickEp() {
  const adapter = await navigator.gpu?.requestAdapter().catch(() => null);
  return adapter ? "webgpu" : "wasm";
}

function load() {
  loading ??= (async () => {
    ep = await pickEp();
    const p = await Pipeline.create(ort, { ep, models: chrome.runtime.getURL("models/"), libs: chrome.runtime.getURL("vendor/") });
    await p.warmup();
    console.log(`engine ready on ${ep} in ${Math.round(p.loadMs)} ms + ${Math.round(p.warmMs)} ms warm-up`);
    pipe = p;
    loadError = null;
    return p;
  })().catch((err) => {
    console.error("engine load failed", err?.stack || err);
    loadError = String(err?.message || err);
    loading = null;
    throw err;
  });
  return loading;
}

let chain = Promise.resolve();
let beams; // manga-ocr beam count from the latest request (settings.fastReread → 1); seam passes reuse it
function runPipeline(rgb) {
  const run = chain.then(async () => {
    const p = await load();
    busy++;
    try {
      const page = await p.analyze(rgb, { beams });
      delete page.stages;
      return page;
    } finally {
      busy--;
    }
  });
  chain = run.catch(() => {});
  return run;
}

function remember(map, key, value, limit) {
  map.delete(key);
  map.set(key, value);
  while (map.size > limit) map.delete(map.keys().next().value);
}

const isTranslated = (page) => page.regions.every((r) => !pyStrip(r.text || "") || r.translation != null);
const payload = (id, page) => ({ id, ...page, translated: isTranslated(page) });

async function decode(dataUrl) {
  const blob = await (await fetch(dataUrl)).blob();
  const digest = await crypto.subtle.digest("SHA-1", await blob.arrayBuffer());
  const id = [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const canvas = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  return { id, rgb: toRGB(ctx.getImageData(0, 0, canvas.width, canvas.height), pipe?.cv) };
}

/** Runs the seam pass against the given neighbours; returns neighbour pages that changed. */
async function stitchNeighbours(id, above, below) {
  const changed = {};
  for (const [upper, lower] of [[above, id], [id, below]]) {
    const key = `${upper}|${lower}`;
    if (!upper || !lower || seamsDone.has(key)) continue;
    if (!pages.has(upper) || !pages.has(lower) || !images.has(upper) || !images.has(lower)) continue;
    seamsDone.add(key);
    if (await resolveSeam(runPipeline, pages.get(upper), images.get(upper), pages.get(lower), images.get(lower))) {
      const other = lower === id ? upper : lower;
      changed[other] = payload(other, pages.get(other));
      console.log(`seam ${upper.slice(0, 8)}/${lower.slice(0, 8)}: stitched bubble across strips`);
    }
  }
  return changed;
}

async function analyze({ image, label, nocache, above, below, fastReread }) {
  beams = fastReread ? 1 : undefined;
  const t0 = performance.now();
  const { id, rgb } = await decode(image);
  const decodeMs = Math.round(performance.now() - t0);
  if (images.has(id)) remember(images, id, images.get(id), IMAGE_CACHE_SIZE);
  else remember(images, id, rgb, IMAGE_CACHE_SIZE);
  if (pages.has(id) && !nocache) {
    const page = pages.get(id);
    remember(pages, id, page, CACHE_SIZE);
    const updates = await stitchNeighbours(id, above, below);
    return { ...payload(id, page), cached: true, seam_updates: updates, decode_ms: decodeMs };
  }

  for (const key of [...seamsDone]) if (key.split("|").includes(id)) seamsDone.delete(key);
  const page = await runPipeline(rgb);
  if (nocache && pages.has(id)) {
    const previous = pages.get(id).regions;
    if (JSON.stringify(previous.map((r) => r.text)) === JSON.stringify(page.regions.map((r) => r.text))) {
      page.regions = previous; // keep translations already paid for
    }
  }
  remember(pages, id, page, CACHE_SIZE);
  const bubbles = page.regions.filter((r) => pyStrip(r.text || "")).length;
  console.log(`analyze ${label || id.slice(0, 8)}: ${page.width}x${page.height} bubbles=${bubbles} ${Math.round(page.total_ms)} ms`);
  const updates = await stitchNeighbours(id, above, below);
  return { ...payload(id, page), cached: false, seam_updates: updates, decode_ms: decodeMs };
}

let panelDetector = null;

/** Calibration: panels in each sample image (data URLs) and whether it looks like a manga page. */
async function detectPanels({ images: samples }) {
  panelDetector ??= (async () => {
    ep ??= await pickEp();
    const d = await PanelDetector.create(ort, chrome.runtime.getURL("models/panels.onnx"), ep);
    console.log(`panel detector ready on ${ep} in ${Math.round(d.loadMs)} ms`);
    return d;
  })().catch((err) => {
    panelDetector = null;
    throw err;
  });
  const detector = await panelDetector;
  const t0 = performance.now();
  const out = [];
  for (const dataUrl of samples || []) {
    try {
      const bmp = await createImageBitmap(await (await fetch(dataUrl)).blob());
      const boxes = await detector.detect(bmp);
      out.push({ width: bmp.width, height: bmp.height, panels: boxes.length, page: looksLikePage(boxes, bmp.width, bmp.height) });
      bmp.close();
    } catch (err) {
      out.push({ error: String(err?.message || err) });
    }
  }
  console.log(`panels: ${out.length} samples in ${Math.round(performance.now() - t0)} ms`);
  return out;
}

/** Translations from the service worker, written back by text (a seam pass may have changed the regions). */
function storeTranslations({ items }) {
  for (const { id, texts, translations } of items || []) {
    const page = pages.get(id);
    if (!page) continue;
    const got = new Map();
    texts.forEach((t, i) => {
      if (t && translations[i] != null) got.set(t, translations[i]);
    });
    for (const r of page.regions) if (r.translation == null && got.has(r.text)) r.translation = got.get(r.text);
  }
  return true;
}

function status() {
  return {
    state: pipe ? "ready" : loadError ? "error" : loading ? "loading" : "idle",
    ep,
    loadMs: pipe ? Math.round(pipe.loadMs) : null,
    warmMs: pipe ? Math.round(pipe.warmMs) : null,
    error: loadError,
    busy: busy > 0,
    pages: pages.size,
  };
}

const handlers = {
  status: () => status(),
  warmup: () => {
    load().catch(() => {});
    return status();
  },
  analyze,
  panels: detectPanels,
  translations: storeTranslations,
};

let idleTimer = null;
function resetIdle() {
  clearTimeout(idleTimer);
  idleTimer = setTimeout(() => {
    chrome.runtime.sendMessage({ type: "ENGINE_IDLE" }).catch(() => {});
    resetIdle();
  }, IDLE_MS);
}
resetIdle();

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.target !== "offscreen" || !handlers[msg.type]) return false;
  resetIdle();
  Promise.resolve()
    .then(() => handlers[msg.type](msg))
    .then(
      (data) => sendResponse({ ok: true, data }),
      (err) => sendResponse({ ok: false, error: String(err?.message || err) })
    );
  return true;
});
