// SPDX-License-Identifier: GPL-3.0-only
import { Gemini } from "./gemini.js";

const ENGINE_TIMEOUT_MS = 150_000;
const URGENT_PANELS = 5;
const TARGET_BUBBLES = 15;

const gemini = new Gemini();

/**
 * Per-tab translation session.
 * Panels flow: ocrQueue → engine analyze (offscreen document) → (first URGENT_PANELS: own urgent
 *              Gemini call) → pending buffer → batched Gemini call → overlay.
 * A batch is whole panels stacked until it reaches TARGET_BUBBLES (a panel that alone has
 * that many goes by itself). It goes out once the buffer can fill one or nothing is left to
 * OCR; non-urgent Gemini calls start ≥ 6 s apart.
 */
const sessions = new Map();
let keepAlive = null;

function panelKey(p) {
  const id = p.kind === "canvas" ? `${p.ref}|${p.fp || ""}` : `${p.kind}|${p.url || ""}`;
  return `${p.frameId}|${id}`;
}

function getSession(tabId) {
  let s = sessions.get(tabId);
  if (!s) {
    s = {
      tabId,
      known: new Map(), // key → { panel, status, bubbles, analysis }
      ocrQueue: [],
      pending: [],
      analyzing: false,
      dispatching: false,
      urgentLeft: URGENT_PANELS,
      inflight: 0,
      auto: null, // { frameId, signature }
      dryRun: false, // OCR only, timed; nothing is sent to Gemini
      startedAt: Date.now(),
      timings: [], // dry run: { grabMs, grabWaitMs, analyzeMs, serverMs, stageMs, bubbles, finishedAt }
    };
    sessions.set(tabId, s);
    updateKeepAlive();
  }
  return s;
}

function endSession(tabId) {
  const s = sessions.get(tabId);
  if (!s) return;
  sessions.delete(tabId);
  updateKeepAlive();
  if (s.auto) {
    return runInFrame(tabId, s.auto.frameId, () => window.__panelProbe?.stopAuto(), []).catch(() => {});
  }
}

function isLive(s) {
  return sessions.get(s.tabId) === s;
}

/** Manual (non-auto) sessions end once all their work is finished. */
function endIfIdle(s) {
  const busy = s.analyzing || s.dispatching || s.inflight || s.ocrQueue.length || s.pending.length;
  if (!s.auto && !busy && isLive(s)) endSession(s.tabId);
}

// MV3 workers idle out after ~30s without extension API calls; sessions can run for minutes.
function updateKeepAlive() {
  if (sessions.size && !keepAlive) {
    keepAlive = setInterval(() => chrome.runtime.getPlatformInfo(), 20_000);
  } else if (!sessions.size && keepAlive) {
    clearInterval(keepAlive);
    keepAlive = null;
  }
}

// ---------- messaging ----------

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  switch (message?.type) {
    case "FETCH_AS_DATA_URL":
      (message.tabId != null ? frameUrl(message.tabId, message.frameId) : Promise.resolve(null))
        .then((referrer) => fetchAsDataUrl(message.url, referrer))
        .then((dataUrl) => sendResponse({ ok: true, dataUrl }))
        .catch((err) => sendResponse({ ok: false, error: String(err?.message || err) }));
      return true;

    case "CAPTURE_PANEL":
      capturePanel(message.tabId, message.panel)
        .then((dataUrl) => sendResponse({ ok: Boolean(dataUrl), dataUrl }))
        .catch((err) => sendResponse({ ok: false, error: String(err?.message || err) }));
      return true;

    case "TRANSLATE_PANELS":
      addPanels(getSession(message.tabId), message.panels || []);
      sendResponse({ ok: true, queued: (message.panels || []).length });
      return false;

    case "START_AUTO":
      startAuto(message.tabId, message.frameId, message.signature, message.panels || [], Boolean(message.dryRun))
        .then(() => sendResponse({ ok: true }))
        .catch((err) => sendResponse({ ok: false, error: String(err?.message || err) }));
      return true;

    case "STOP_AUTO":
      endSession(message.tabId);
      sendResponse({ ok: true });
      return false;

    case "GET_STATUS":
      sendResponse(statusOf(message.tabId));
      return false;

    case "CHECK_KEY":
      checkKey()
        .then((res) => sendResponse(res))
        .catch((err) => sendResponse({ ok: false, error: String(err?.message || err) }));
      return true;

    case "HEALTH":
      health()
        .then((res) => sendResponse(res))
        .catch((err) => sendResponse({ engine: { state: "error", error: String(err?.message || err) } }));
      return true;

    case "ENGINE_WARMUP":
      engine("warmup")
        .then((res) => sendResponse(res))
        .catch((err) => sendResponse({ state: "error", error: String(err?.message || err) }));
      return true;

    case "TOGGLE_OVERLAY":
      if (sender.tab?.id != null) {
        chrome.scripting
          .executeScript({ target: { tabId: sender.tab.id, allFrames: true }, func: () => window.__panelProbe?.toggleOverlay() })
          .catch(() => {});
      }
      return false;

    case "ENGINE_IDLE":
      if (!sessions.size) closeEngine();
      return false;

    case "DETECT_PANELS":
      engine("panels", { images: message.images || [] }, 60_000)
        .then((data) => sendResponse({ ok: true, data }))
        .catch((err) => sendResponse({ ok: false, error: String(err?.message || err) }));
      return true;

    case "PICK_GROUP":
      pickGroup(message.body)
        .then((data) => sendResponse({ ok: true, data }))
        .catch((err) => sendResponse({ ok: false, error: String(err?.message || err) }));
      return true;

    case "AUTO_PANELS": {
      const s = sender.tab && sessions.get(sender.tab.id);
      if (s?.auto && s.auto.frameId === sender.frameId) {
        addPanels(s, (message.panels || []).map((p) => ({ ...p, frameId: sender.frameId })));
      }
      return false;
    }
  }
  return false;
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
  if ((changeInfo.url || changeInfo.status === "loading") && sessions.has(tabId)) checkDocument(tabId);
});

const DOC_CHECK_MS = 1500;

/** Readers rewrite the URL as you read (Jump TOON's #slice, history.pushState): an auto session survives
 * that as long as its page is still the same document; a real navigation ends it. */
function checkDocument(tabId) {
  const s = sessions.get(tabId);
  if (!s.auto) return endSession(tabId);
  setTimeout(async () => {
    if (sessions.get(tabId) !== s) return;
    const alive = await runInFrame(tabId, s.auto.frameId, () => Boolean(window.__panelProbe?.autoActive())).catch(() => false);
    if (!alive) endSession(tabId);
  }, DOC_CHECK_MS);
}
chrome.tabs.onRemoved.addListener((tabId) => endSession(tabId));

function statusOf(tabId) {
  const s = sessions.get(tabId);
  if (!s) return { active: false };
  const counts = {};
  for (const e of s.known.values()) counts[e.status] = (counts[e.status] || 0) + 1;
  return {
    active: true,
    auto: Boolean(s.auto),
    dryRun: s.dryRun,
    panels: s.known.size,
    counts,
    errors: [...s.known.values()].filter((e) => e.error).map((e) => `${e.panel.ref}: ${e.error}`),
    pendingBubbles: s.pending.reduce((n, e) => n + e.bubbles, 0),
    timing: s.dryRun ? timingSummary(s) : null,
  };
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0;
}

function timingSummary(s) {
  const t = s.timings;
  if (!t.length) return { panels: 0 };
  const stages = {};
  for (const name of Object.keys(t[t.length - 1].stageMs || {})) {
    stages[name] = Math.round(median(t.map((x) => x.stageMs?.[name] || 0)));
  }
  return {
    panels: t.length,
    bubbles: t.reduce((n, x) => n + x.bubbles, 0),
    medianPanelMs: Math.round(median(t.map((x) => x.grabWaitMs + x.analyzeMs))),
    medianGrabMs: Math.round(median(t.map((x) => x.grabMs))),
    medianGrabWaitMs: Math.round(median(t.map((x) => x.grabWaitMs))),
    medianServerMs: Math.round(median(t.map((x) => x.serverMs))),
    stages,
    wallMs: Math.max(...t.map((x) => x.finishedAt)) - s.startedAt,
  };
}

async function startAuto(tabId, frameId, signature, panels, dryRun) {
  const existing = sessions.get(tabId);
  if (existing && existing.dryRun !== dryRun) await endSession(tabId);
  const s = getSession(tabId);
  s.dryRun = dryRun;
  s.auto = { frameId, signature };
  addPanels(s, panels);
  const knownKeys = panels.map((p) => panelKey(p).slice(`${frameId}|`.length));
  await runInFrame(tabId, frameId, (sig, keys) => window.__panelProbe?.startAuto(sig, keys), [signature, knownKeys]);
}

// ---------- helpers ----------

let refererRule = 0;

/** With `referrer` (the page the image sits on), sends that page's origin as Referer the way the page
 * itself would: hotlink-protected hosts (pixiv's i.pximg.net) answer 403 without it, and fetch() can't
 * set the header. The session rule only covers this host, for requests outside tabs, during this fetch. */
async function fetchAsDataUrl(url, referrer) {
  const ref = /^https?:/i.test(referrer || "") ? `${new URL(referrer).origin}/` : null;
  const id = ref ? (refererRule = (refererRule % 1000) + 1) : 0;
  if (ref) {
    await chrome.declarativeNetRequest.updateSessionRules({
      removeRuleIds: [id],
      addRules: [
        {
          id,
          action: { type: "modifyHeaders", requestHeaders: [{ header: "referer", operation: "set", value: ref }] },
          condition: { requestDomains: [new URL(url).hostname], tabIds: [chrome.tabs.TAB_ID_NONE], resourceTypes: ["xmlhttprequest", "other"] },
        },
      ],
    });
  }
  try {
    const res = await fetch(url, { credentials: "omit", redirect: "follow" });
    if (!res.ok) throw new Error(`fetch ${res.status}`);
    return blobToDataUrl(await res.blob());
  } finally {
    if (ref) await chrome.declarativeNetRequest.updateSessionRules({ removeRuleIds: [id] }).catch(() => {});
  }
}

const frameUrl = (tabId, frameId) => runInFrame(tabId, frameId ?? 0, () => location.href).catch(() => null);

function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("FileReader failed"));
    reader.onload = () => resolve(reader.result);
    reader.readAsDataURL(blob);
  });
}

async function runInFrame(tabId, frameId, func, args) {
  const [res] = await chrome.scripting.executeScript({
    target: { tabId, frameIds: [frameId] },
    func,
    args,
  });
  return res?.result;
}

function setState(tabId, panel, update) {
  const item = { ref: panel.ref, url: panel.url, kind: panel.kind, fp: panel.fp, ...update };
  return runInFrame(tabId, panel.frameId, (i) => window.__panelProbe?.setPanelState(i), [item]).catch(
    (err) => console.warn("setState failed", err)
  );
}

async function exportPanel(tabId, panel) {
  const [exported] =
    (await runInFrame(tabId, panel.frameId, (p) => window.__panelProbe?.exportPanels([p]), [panel])) || [];
  if (exported?.dataUrl) return exported.dataUrl;
  if (panel.kind === "canvas") return capturePanel(tabId, panel);
  if (panel.url && /^https?:/i.test(panel.url)) {
    return fetchAsDataUrl(panel.url, exported?.pageUrl || (await frameUrl(tabId, panel.frameId)));
  }
  return null;
}

// ---------- screen capture (canvases the page locked against reading, e.g. Jump+) ----------

const CAPTURE_GAP_MS = 600; // Chrome allows 2 captureVisibleTab calls per second
let captureChain = Promise.resolve();
let lastCapture = 0;

const offscreenError = (message) => Object.assign(new Error(message), { offscreen: true });

/** The panel as shown on screen: a tab screenshot (our overlay hidden) cropped to the element, at
 * display size × device pixel ratio. Top frame only; rejects with `offscreen` when it can't be seen. */
function capturePanel(tabId, panel) {
  const run = captureChain.then(() => captureNow(tabId, panel));
  captureChain = run.catch(() => {});
  return run;
}

async function captureNow(tabId, panel) {
  if (panel.frameId) return null;
  const tab = await chrome.tabs.get(tabId);
  if (!tab.active) throw offscreenError("tab not visible");
  const wait = lastCapture + CAPTURE_GAP_MS - Date.now();
  if (wait > 0) await new Promise((resolve) => setTimeout(resolve, wait));
  const spot = await runInFrame(tabId, 0, (ref, url) => window.__panelProbe?.prepareCapture(ref, url), [panel.ref, panel.url]);
  if (!spot) return null;
  if (spot.offscreen) throw offscreenError("not on screen");
  let shot;
  try {
    shot = await chrome.tabs.captureVisibleTab(tab.windowId, { format: "png" });
  } finally {
    lastCapture = Date.now();
    runInFrame(tabId, 0, () => window.__panelProbe?.endCapture()).catch(() => {});
  }
  return cropDataUrl(shot, spot);
}

async function cropDataUrl(dataUrl, { rect: [x, y, w, h], viewport: [vw] }) {
  const bitmap = await createImageBitmap(await (await fetch(dataUrl)).blob());
  const scale = bitmap.width / vw;
  const sx = Math.max(0, Math.round(x * scale));
  const sy = Math.max(0, Math.round(y * scale));
  const sw = Math.min(bitmap.width - sx, Math.round(w * scale));
  const sh = Math.min(bitmap.height - sy, Math.round(h * scale));
  if (sw <= 0 || sh <= 0) throw offscreenError("not on screen");
  const canvas = new OffscreenCanvas(sw, sh);
  canvas.getContext("2d").drawImage(bitmap, sx, sy, sw, sh, 0, 0, sw, sh);
  bitmap.close();
  return blobToDataUrl(await canvas.convertToBlob({ type: "image/png" }));
}

// ---------- engine (offscreen document running webport/pipeline) ----------

const OFFSCREEN_URL = "offscreen.html";
let creatingEngine = null;

async function engineOpen() {
  const contexts = await chrome.runtime.getContexts({ contextTypes: ["OFFSCREEN_DOCUMENT"] });
  return contexts.length > 0;
}

/** Every caller waits for the same creation: a document that exists but hasn't loaded has no listener yet. */
function ensureEngine() {
  creatingEngine ??= (async () => {
    if (await engineOpen()) return;
    await chrome.offscreen.createDocument({
      url: OFFSCREEN_URL,
      reasons: ["WORKERS"],
      justification: "Runs the OCR models (WebGPU/WASM) on page images.",
    });
  })().finally(() => (creatingEngine = null));
  return creatingEngine;
}

async function closeEngine() {
  if (await engineOpen()) await chrome.offscreen.closeDocument().catch(() => {});
}

async function engine(type, body = {}, timeoutMs = ENGINE_TIMEOUT_MS) {
  await ensureEngine();
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error("engine timed out")), timeoutMs);
  });
  try {
    const res = await Promise.race([chrome.runtime.sendMessage({ ...body, target: "offscreen", type }), timeout]);
    if (!res) throw new Error("engine did not answer");
    if (!res.ok) throw new Error(res.error);
    return res.data;
  } finally {
    clearTimeout(timer);
  }
}

async function health() {
  const state = (await engineOpen()) ? await engine("status", {}, 5_000) : { state: "idle" };
  const g = await gemini.status();
  return { engine: state, has_key: g.state !== "no_key", model: gemini.models[0], gemini: g };
}

// ---------- Gemini (called directly; the key never leaves extension storage except to the API) ----------

async function checkKey() {
  const { geminiKey } = await chrome.storage.local.get("geminiKey");
  const key = (geminiKey || "").trim();
  if (!key) return { ok: true, has_key: false, stored: false };
  const [valid, why] = await gemini.checkKey(key);
  if (valid === false) return { ok: false, error: `Gemini rejected the key: ${why}`, has_key: true, stored: true };
  return {
    ok: true,
    verified: Boolean(valid),
    warning: valid ? null : `couldn't reach Gemini to verify (${why}); using it anyway`,
    has_key: true,
    stored: true,
  };
}

async function translateItems(items, urgent) {
  const flat = [];
  const spans = [];
  for (const item of items) {
    spans.push([item, flat.length]);
    flat.push(...item.texts);
  }
  const t0 = performance.now();
  const { translations, meta } = await gemini.translate(flat, { urgent });
  const results = spans.map(([item, start]) => ({ id: item.id, translations: translations.slice(start, start + item.texts.length) }));
  const done = results.map((r, i) => ({ ...r, texts: items[i].texts }));
  engine("translations", { items: done }, 10_000).catch(() => {});
  const bubbles = flat.filter((t) => t && t.trim()).length;
  console.log(
    `translate panels=${items.length} bubbles=${bubbles} urgent=${urgent} empty=${meta.empty || 0} ` +
      `missing=${meta.missing || 0}${meta.retried ? ` retried=${meta.retried}` : ""} model=${meta.model} ` +
      `api_ms=${meta.api_ms} total_ms=${Math.round(performance.now() - t0)}`
  );
  return { results, bubbles, ...meta };
}

const PICK_PROMPT =
  "A browser extension found groups of same-width images on a web page and needs to know which " +
  "group holds the comic pages the user is reading (manga, manhwa, webtoon or comic story pages / " +
  "strip segments with panels, art and speech bubbles).\n" +
  "Not the reading pages: chapter or series thumbnails, covers, banners, ads, avatars, logos, " +
  "UI icons, recommendation grids.\n" +
  "Each group is described, followed by up to 2 sample images from it.\n" +
  'Return ONLY JSON: {"group": <group number, or 0 if none is comic pages>, ' +
  '"confidence": <0..1>, "reason": "<short reason>"}';

function imagePart(dataUrl) {
  const m = /^data:(image\/[^;,]+)[^,]*,(.*)$/s.exec(dataUrl || "");
  return m ? { inline_data: { mime_type: m[1], data: m[2] } } : null;
}

async function pickGroup(req) {
  const parts = [{ text: PICK_PROMPT + (req.title ? `\nPage title: ${req.title}` : "") }];
  for (const g of req.groups) {
    parts.push({ text: `Group ${g.index}: ${g.count} ${g.kind} element(s), displayed ${g.width} px wide.` });
    for (const img of (g.images || []).slice(0, 2)) {
      const part = imagePart(img);
      if (part) parts.push(part);
    }
  }
  const { data: raw, meta } = await gemini.askJson(parts);
  const data = raw && typeof raw === "object" && !Array.isArray(raw) ? raw : {};
  const valid = new Set(req.groups.map((g) => g.index));
  let group = Math.trunc(Number(data.group ?? 0));
  if (!Number.isFinite(group) || !valid.has(group)) group = 0;
  console.log(`pick-group groups=${req.groups.length} -> ${group} (${String(data.reason || "").slice(0, 80)}) model=${meta.model}`);
  return { group, confidence: data.confidence, reason: data.reason || "", ...meta };
}

function countBubbles(regions) {
  return regions.filter((r) => (r.text || "").trim()).length;
}

const STACK_TOLERANCE_PX = 4;

/** Page ids of already-analyzed panels stacked directly above/below (webtoon slices), for the seam pass. */
function stackedNeighbours(s, panel) {
  const near = (a, b) => Math.abs(a - b) <= STACK_TOLERANCE_PX;
  const out = { above: null, below: null };
  if (panel.pageY == null || !panel.displayHeight) return out;
  for (const e of s.known.values()) {
    const p = e.panel;
    if (!e.analysis?.id || p === panel || p.frameId !== panel.frameId) continue;
    if (!near(p.left, panel.left) || !near(p.displayWidth, panel.displayWidth)) continue;
    if (near(p.pageY + p.displayHeight, panel.pageY)) out.above = e.analysis.id;
    else if (near(panel.pageY + panel.displayHeight, p.pageY)) out.below = e.analysis.id;
  }
  return out;
}

/** A neighbour's regions changed because a bubble across the strip boundary was stitched. */
async function applySeamUpdate(s, id, page) {
  const entry = [...s.known.values()].find((e) => e.analysis?.id === id);
  if (!entry) return;
  entry.analysis = page;
  entry.bubbles = countBubbles(page.regions);
  if (s.dryRun) {
    const regions = page.regions.map((r) => ({ ...r, translation: r.text || null }));
    await setState(s.tabId, entry.panel, { state: "ocr", message: `${entry.bubbles} bubbles · seam`, result: { ...page, regions } });
  } else if (entry.status === "waiting") {
    await setState(s.tabId, entry.panel, { state: "waiting", result: page });
  } else if (entry.status === "done" || entry.status === "empty") {
    if (page.translated) {
      await setState(s.tabId, entry.panel, { state: "done", result: page });
    } else {
      entry.status = "waiting";
      s.pending.push(entry);
      await setState(s.tabId, entry.panel, { state: "waiting", result: page });
    }
  }
  // "translating": translateEntries notices the swapped analysis when its call returns.
}

// ---------- pipeline ----------

function addPanels(s, panels) {
  for (const panel of panels) {
    const key = panelKey(panel);
    if (s.known.has(key)) continue;
    s.known.set(key, { panel, status: "queued", bubbles: 0, analysis: null });
    s.ocrQueue.push(panel);
    setState(s.tabId, panel, { state: "queued" });
  }
  if (!s.analyzing && s.ocrQueue.length) analyzeLoop(s);
}

/** Starts grabbing a panel's pixels; resolves to { dataUrl, grabMs } or { error, grabMs }. */
function startGrab(s, panel) {
  s.known.get(panelKey(panel)).status = "exporting";
  setState(s.tabId, panel, { state: "exporting" });
  const started = performance.now();
  return exportPanel(s.tabId, panel).then(
    (dataUrl) => ({ dataUrl, grabMs: performance.now() - started }),
    (error) => ({ error, grabMs: performance.now() - started })
  );
}

async function analyzeLoop(s) {
  s.analyzing = true;
  let prefetched = null; // { panel, grab }: the next panel's pixels, grabbed while the engine analyzes
  while (s.ocrQueue.length && isLive(s)) {
    const panel = s.ocrQueue.shift();
    const entry = s.known.get(panelKey(panel));
    const grab = prefetched?.panel === panel ? prefetched.grab : startGrab(s, panel);
    prefetched = null;
    try {
      const waitStart = performance.now();
      const { dataUrl, error, grabMs } = await grab;
      const grabWaitMs = performance.now() - waitStart;
      if (error) throw error;
      if (!dataUrl) throw new Error("could not grab pixels");

      entry.status = "reading";
      await setState(s.tabId, panel, { state: "reading" });
      if (s.ocrQueue.length && isLive(s)) {
        const next = s.ocrQueue[0];
        prefetched = { panel: next, grab: startGrab(s, next) };
      }
      const label = `${panel.kind} ${panel.naturalWidth || "?"}x${panel.naturalHeight || "?"}`;
      const analyzeStart = performance.now();
      const { settings } = await chrome.storage.local.get("settings");
      const analysis = await engine("analyze", {
        image: dataUrl,
        label,
        nocache: s.dryRun,
        fastReread: Boolean(settings?.fastReread),
        ...stackedNeighbours(s, panel),
      });
      const analyzeMs = performance.now() - analyzeStart;
      entry.analysis = analysis;
      entry.bubbles = countBubbles(analysis.regions);
      for (const [id, page] of Object.entries(analysis.seam_updates || {})) await applySeamUpdate(s, id, page);

      if (s.dryRun) {
        s.timings.push({
          grabMs,
          grabWaitMs,
          analyzeMs,
          serverMs: analysis.total_ms || 0,
          stageMs: analysis.stage_ms || {},
          bubbles: entry.bubbles,
          finishedAt: Date.now(),
        });
        entry.status = "ocr";
        const regions = analysis.regions.map((r) => ({ ...r, translation: r.text || null }));
        await setState(s.tabId, panel, {
          state: "ocr",
          message: `${entry.bubbles} bubbles · ${Math.round(grabWaitMs + analyzeMs)} ms (engine ${Math.round(analysis.total_ms || 0)})`,
          result: { ...analysis, regions },
        });
      } else if (analysis.translated) {
        entry.status = "done";
        await setState(s.tabId, panel, { state: "done", result: analysis });
      } else if (!entry.bubbles) {
        entry.status = "empty";
        await setState(s.tabId, panel, { state: "empty", result: analysis });
      } else if (s.urgentLeft > 0) {
        s.urgentLeft--;
        translateBatch(s, [entry], true);
      } else {
        entry.status = "waiting";
        s.pending.push(entry);
        await setState(s.tabId, panel, { state: "waiting", result: analysis });
      }
    } catch (err) {
      if (err?.offscreen) {
        s.known.delete(panelKey(panel));
        await runInFrame(s.tabId, panel.frameId, (p) => window.__panelProbe?.forgetPanel(p), [panel]).catch(() => {});
      } else {
        entry.status = "error";
        entry.error = String(err?.message || err);
        await setState(s.tabId, panel, { state: "error", message: entry.error });
      }
    }
    maybeDispatch(s);
  }
  s.analyzing = false;
  maybeDispatch(s);
}

function maybeDispatch(s) {
  if (!isLive(s)) return;
  if (s.dispatching || !s.pending.length) {
    endIfIdle(s);
    return;
  }
  const buffered = s.pending.reduce((n, e) => n + e.bubbles, 0);
  const nothingLeftToRead = !s.analyzing && !s.ocrQueue.length;
  if (buffered < TARGET_BUBBLES && !nothingLeftToRead) return;

  const batch = [];
  let bubbles = 0;
  while (s.pending.length && bubbles < TARGET_BUBBLES) {
    const entry = s.pending.shift();
    batch.push(entry);
    bubbles += entry.bubbles;
  }

  s.dispatching = true;
  translateBatch(s, batch, false).finally(() => {
    s.dispatching = false;
    maybeDispatch(s);
  });
}

async function translateBatch(s, entries, urgent) {
  s.inflight++;
  try {
    await translateEntries(s, entries, urgent);
  } finally {
    s.inflight--;
    maybeDispatch(s);
  }
}

async function translateEntries(s, entries, urgent) {
  for (const e of entries) {
    e.status = "translating";
    setState(s.tabId, e.panel, { state: "translating", result: e.analysis });
  }
  try {
    const sent = entries.map((e) => e.analysis);
    const res = await translateItems(
      sent.map((a) => ({
        id: a.id,
        texts: a.regions.map((r) => (r.translation != null ? "" : r.text || "")),
      })),
      urgent
    );
    const byId = new Map(res.results.map((r) => [r.id, r.translations]));
    for (const [i, e] of entries.entries()) {
      // matched by text: a seam pass may have swapped e.analysis while the call was out
      const swapped = e.analysis !== sent[i];
      const got = new Map();
      const translations = byId.get(sent[i].id) || [];
      sent[i].regions.forEach((r, j) => {
        if (r.text && translations[j] != null) got.set(r.text, translations[j]);
      });
      const regions = e.analysis.regions.map((r) => ({ ...r, translation: r.translation ?? got.get(r.text) ?? null }));
      e.analysis = { ...e.analysis, regions };
      const missing = swapped && regions.some((r) => (r.text || "").trim() && r.translation == null);
      if (missing && isLive(s)) {
        e.status = "waiting";
        s.pending.push(e);
        await setState(s.tabId, e.panel, { state: "waiting", result: e.analysis });
        continue;
      }
      e.status = "done";
      if (isLive(s)) await setState(s.tabId, e.panel, { state: "done", result: e.analysis });
    }
  } catch (err) {
    for (const e of entries) {
      e.status = "error";
      if (isLive(s)) await setState(s.tabId, e.panel, { state: "error", message: String(err?.message || err) });
    }
  }
}
