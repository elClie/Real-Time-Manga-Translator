// SPDX-License-Identifier: GPL-3.0-only
const appEl = document.getElementById("app");
const metaEl = document.getElementById("meta");
const statusEl = document.getElementById("status");
const siteEl = document.getElementById("site");
const siteTextEl = document.getElementById("site-text");
const calibrateEl = document.getElementById("calibrate");
const calHintEl = document.getElementById("cal-hint");
const calGridEl = document.getElementById("cal-grid");
const panelsBarEl = document.getElementById("panels-bar");
const panelsEl = document.getElementById("panels");
const scanBtn = document.getElementById("scan");
const autoBtn = document.getElementById("auto");
const dryBtn = document.getElementById("dry");
const translateBtn = document.getElementById("translate");
const sessionEl = document.getElementById("session");
const progressEl = document.getElementById("progress");

/** Settings → Developer mode: panel picker, Auto/OCR test, overlay/boxes buttons, engine and session details. */
let dev = false;

const WIDTH_TOLERANCE = 0.04;
const ALIGN_PX = 8;
const FULL_COVERAGE = 0.5;
const CALIBRATION_CHOICES = 8;
const PICK_GROUPS = 6;
const PICK_WIDTH = 256;
const DETECT_SAMPLES = 3;
const DETECT_WIDTH = 512;
const CAPTURE_SHOWN = 0.98; // same as content.js: locked canvases are screenshotted only when fully visible

/**
 * @type {{ tabId: number, host: string, title: string, groups: any[], selected: number | null,
 *          checked: Set<string>, thumbs: Map<string, string>, calibration: object | null,
 *          guess: number | null, userPicked: boolean } | null}
 */
let state = null;
let scanId = 0;

function setStatus(text, isError = false) {
  statusEl.hidden = !text;
  statusEl.textContent = text || "";
  statusEl.classList.toggle("error", isError);
}

async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

async function injectContent(tabId) {
  try {
    await chrome.scripting.executeScript({
      target: { tabId, allFrames: true },
      files: ["content.js"],
    });
  } catch (err) {
    console.warn("inject partial", err);
  }
}

async function runInAllFrames(func) {
  const tab = await getActiveTab();
  if (!tab?.id) return;
  await chrome.scripting.executeScript({ target: { tabId: tab.id, allFrames: true }, func }).catch(() => {});
}

// ---------- engine status ----------

function describeEngine(e) {
  if (e.state === "ready") return `ready · ${e.ep === "webgpu" ? "WebGPU" : "WASM (slow)"}${e.busy ? " · busy" : ""}`;
  if (e.state === "loading") return "loading models…";
  if (e.state === "error") return `failed — ${e.error}`;
  return "starting…";
}

/** [text, ok] */
function describeGemini(g) {
  if (!g) return ["Gemini: no answer from the extension", false];
  if (g.state === "no_key") return ["Gemini: no API key — add one in ⚙ settings", false];
  if (g.state === "invalid") return [`Gemini: key rejected — ${g.error}`, false];
  if (g.state === "unreachable") return [`Gemini: can't reach Google (${g.error})`, false];
  if (g.state === "error") return [`Gemini: last request failed — ${g.error}`, false];
  return [`Gemini: ready · ${g.model}`, true];
}

async function checkApp() {
  const h = await chrome.runtime.sendMessage({ type: "HEALTH" }).catch(() => null);
  const e = h?.engine || { state: "error", error: "no answer from the extension" };
  const [gemini, geminiOk] = describeGemini(h?.gemini);
  if (dev) {
    appEl.textContent = `Engine: ${describeEngine(e)}\n${gemini}`;
    appEl.className = e.state !== "error" && geminiOk ? "app ok" : "app off";
  } else {
    const engine = e.state === "loading" ? " · loading OCR models…" : e.state === "error" ? ` · OCR engine failed: ${e.error}` : "";
    appEl.textContent = gemini + engine;
    appEl.className = geminiOk && e.state !== "error" ? "app" : "app off";
  }
  if (e.state === "loading" || e.state === "idle") setTimeout(checkApp, 1000);
}

// ---------- session status ----------

let autoRunning = false;
let dryRunning = false;

function fmtMs(ms) {
  return ms >= 10_000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

function describeTiming(st) {
  const c = st.counts || {};
  const busy = (c.queued || 0) + (c.exporting || 0) + (c.reading || 0);
  const t = st.timing || {};
  const errors = c.error ? ` · ${c.error} errors` : "";
  if (!t.panels) return `OCR test · ${st.panels} panels · ${busy} to read${errors}`;
  const stages = Object.entries(t.stages || {})
    .map(([name, ms]) => `${name} ${Math.round(ms)}`)
    .join(" · ");
  return (
    `OCR test · ${t.panels}/${st.panels} panels read · ${t.bubbles} bubbles · ${busy} to read${errors}\n` +
    `median per panel ${fmtMs(t.medianPanelMs)} (grab ${fmtMs(t.medianGrabMs)}, waited ${fmtMs(t.medianGrabWaitMs)} · ` +
    `engine ${fmtMs(t.medianServerMs)})\n` +
    `${stages ? `engine stages (median ms): ${stages}\n` : ""}` +
    `wall ${fmtMs(t.wallMs)} since start`
  );
}

async function refreshSession() {
  const tab = await getActiveTab();
  if (!tab?.id) return;
  const st = await chrome.runtime.sendMessage({ type: "GET_STATUS", tabId: tab.id }).catch(() => null);
  const running = Boolean(st?.active && st.auto);
  autoRunning = running && !st.dryRun;
  dryRunning = running && st.dryRun;
  autoBtn.textContent = autoRunning ? "Stop ■" : "Auto ▶";
  autoBtn.disabled = !autoRunning && (dryRunning || state?.selected == null);
  dryBtn.textContent = dryRunning ? "Stop ■" : "OCR ⏱";
  dryBtn.disabled = !dryRunning && (autoRunning || state?.selected == null);
  updateTranslateBtn();

  if (!st?.active) {
    sessionEl.hidden = true;
    progressEl.hidden = true;
    return;
  }
  if (!dev) {
    const c = st.counts || {};
    progressEl.hidden = st.dryRun;
    progressEl.textContent =
      `Translating · ${(c.done || 0) + (c.empty || 0)}/${st.panels} pages done` + (c.error ? ` · ${c.error} failed` : "");
    return;
  }
  sessionEl.hidden = false;
  sessionEl.className = "app ok";
  if (st.dryRun) {
    sessionEl.textContent = describeTiming(st);
    return;
  }
  const c = st.counts || {};
  const busy = (c.queued || 0) + (c.exporting || 0) + (c.reading || 0);
  sessionEl.textContent =
    `${st.auto ? "Auto" : "Manual"} · ${st.panels} panels · ${c.done || 0} done · ${busy} to read · ` +
    `${c.waiting || 0} waiting (${st.pendingBubbles} bubbles) · ${c.translating || 0} translating` +
    `${c.empty ? ` · ${c.empty} no text` : ""}${c.error ? ` · ${c.error} errors` : ""}`;
}

// ---------- grouping + scoring (orders the calibration choices) ----------

function median(values) {
  const s = [...values].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : 0;
}

/** Cluster candidates of the same frame + kind by display width (±4%). */
function groupByWidth(candidates) {
  const buckets = new Map();
  for (const c of candidates) {
    const key = `${c.frameId}|${c.kind}`;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(c);
  }

  const groups = [];
  for (const list of buckets.values()) {
    list.sort((a, b) => a.displayWidth - b.displayWidth);
    let current = null;
    for (const c of list) {
      if (!current || c.displayWidth > current.minWidth * (1 + WIDTH_TOLERANCE) + 4) {
        current = { minWidth: c.displayWidth, members: [] };
        groups.push(current);
      }
      current.members.push(c);
    }
  }
  return groups.map((g) => ({ members: g.members.sort((a, b) => a.pageY - b.pageY) }));
}

/** 0..1: members share a left edge, stack vertically and sit under a common ancestor. */
function structureScore(members) {
  const n = members.length;
  if (n < 2) return 0;

  const midLeft = median(members.map((m) => m.left));
  const aligned = members.filter((m) => Math.abs(m.left - midLeft) <= ALIGN_PX).length / n;

  let stackedPairs = 0;
  for (let i = 1; i < n; i++) {
    const prev = members[i - 1];
    if (members[i].pageY >= prev.pageY + prev.displayHeight * 0.5) stackedPairs++;
  }
  const stacked = stackedPairs / (n - 1);

  const ancestorCounts = new Map();
  for (const m of members) {
    for (const id of new Set(m.ancestors)) {
      const key = `${m.frameId}:${id}`;
      ancestorCounts.set(key, (ancestorCounts.get(key) || 0) + 1);
    }
  }
  const shared = Math.max(0, ...ancestorCounts.values()) / n;

  const sizeFactor = Math.min(1, Math.log2(n) / 3);
  return (sizeFactor * (aligned + stacked + shared)) / 3;
}

function scoreGroup(group) {
  const { members } = group;
  const coverage = Math.min(1, members.reduce((sum, m) => sum + m.visibleFrac, 0) / FULL_COVERAGE);
  const center = members.some((m) => m.containsCenter) ? 1 : 0;
  const demotedFrac = members.filter((m) => m.demoted).length / members.length;
  const score = (0.55 * coverage + 0.15 * center + 0.3 * structureScore(members)) * (1 - 0.7 * demotedFrac);

  const widths = members.map((m) => m.displayWidth);
  const naturalWidths = [...new Set(members.map((m) => m.naturalWidth).filter(Boolean))].sort((a, b) => a - b);

  return {
    ...group,
    kind: members[0].kind,
    displayWidth: [Math.min(...widths), Math.max(...widths)],
    naturalWidths,
    score,
  };
}

function pickRepresentative(members) {
  return [...members].sort(
    (a, b) => b.visibleFrac - a.visibleFrac || b.displayWidth * b.displayHeight - a.displayWidth * a.displayHeight
  )[0];
}

/** Best-scoring group that matches one of the site's saved dimensions. */
function findCalibratedGroup(groups, calibrations) {
  for (let i = 0; i < groups.length; i++) {
    const cal = calibrations.find((c) => matchesCalibration(groups[i], c));
    if (cal) return { index: i, cal };
  }
  return null;
}

// ---------- thumbnails ----------

const keyOf = (c) => `${c.frameId}|${c.ref}`;

function byFrame(items) {
  const map = new Map();
  for (const item of items) {
    if (!map.has(item.frameId)) map.set(item.frameId, []);
    map.get(item.frameId).push(item);
  }
  return map;
}

/** Full-size screenshot crop of a canvas the page locked against reading (only while fully on screen). */
function captured(c) {
  if (c.kind !== "canvas" || !c.tainted || !(c.shownFrac >= CAPTURE_SHOWN) || c.covered) return Promise.resolve(null);
  const key = keyOf(c);
  if (!state.captures.has(key)) {
    const panel = { ref: c.ref, url: c.url, kind: c.kind, fp: c.fp, frameId: c.frameId };
    state.captures.set(
      key,
      chrome.runtime
        .sendMessage({ type: "CAPTURE_PANEL", tabId: state.tabId, panel })
        .then((res) => (res?.ok ? res.dataUrl : null))
        .catch(() => null)
    );
  }
  return state.captures.get(key);
}

/** In-page downscaled draws where allowed, else a screenshot crop (locked canvas) or the image URL itself. */
async function loadThumbs(items) {
  const missing = items.filter((c) => !state.thumbs.has(keyOf(c)));
  for (const [frameId, list] of byFrame(missing)) {
    let thumbs = {};
    try {
      const [res] = await chrome.scripting.executeScript({
        target: { tabId: state.tabId, frameIds: [frameId] },
        func: (refs) => window.__panelProbe?.thumbnails(refs) || {},
        args: [list.map(({ ref, url }) => ({ ref, url }))],
      });
      thumbs = res?.result || {};
    } catch (err) {
      console.warn("thumbnails failed", frameId, err);
    }
    for (const c of list) {
      const shot = thumbs[c.ref] ? null : await captured(c);
      const fallback = shot ? await downscale(shot, PICK_WIDTH) : /^https?:/i.test(c.url || "") ? c.url : "";
      state.thumbs.set(keyOf(c), thumbs[c.ref] || fallback);
    }
  }
}

function thumbEl(key, className) {
  const box = document.createElement("div");
  box.className = className;
  const src = state.thumbs.get(key);
  if (src) {
    const img = document.createElement("img");
    img.src = src;
    img.referrerPolicy = "no-referrer";
    img.onerror = async () => {
      img.onerror = null;
      // hotlink-protected hosts refuse the bare URL; the background fetch sends the page as Referer
      if (/^https?:/i.test(src)) {
        const res = await chrome.runtime
          .sendMessage({ type: "FETCH_AS_DATA_URL", url: src, tabId: state.tabId, frameId: Number(key.split("|")[0]) })
          .catch(() => null);
        const data = res?.ok ? await downscale(res.dataUrl, PICK_WIDTH) : null;
        if (data) {
          state.thumbs.set(key, data);
          img.src = data;
          return;
        }
      }
      img.remove();
      box.textContent = "no preview";
    };
    box.append(img);
  } else {
    box.textContent = "no preview";
  }
  return box;
}

// ---------- automatic guess: panel detector, then optional Gemini (settings) ----------

/** Up to `n` members: the representative, then on-screen ones, then spread over the group. */
function pickSamples(members, n = 2) {
  const picked = [pickRepresentative(members)];
  const pool = [
    ...members.filter((m) => m.visibleFrac > 0),
    ...Array.from({ length: n }, (_, i) => members[Math.floor(((i + 1) * members.length) / (n + 1))]),
  ];
  for (const m of pool) if (picked.length < n && m && !picked.includes(m)) picked.push(m);
  return picked;
}

function downscale(dataUrl, width) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(1, width / img.naturalWidth);
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(img.naturalWidth * scale));
      canvas.height = Math.max(1, Math.round(img.naturalHeight * scale));
      canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
      resolve(canvas.toDataURL("image/jpeg", 0.8));
    };
    img.onerror = () => resolve(null);
    img.src = dataUrl;
  });
}

/** key → small JPEG data URL; in-page draw where allowed, else background fetch + downscale here. */
async function sampleImages(tabId, members, width = PICK_WIDTH) {
  const out = new Map();
  await Promise.all(
    [...byFrame(members)].map(async ([frameId, list]) => {
      let drawn = {};
      try {
        const [res] = await chrome.scripting.executeScript({
          target: { tabId, frameIds: [frameId] },
          func: (refs, width) => window.__panelProbe?.thumbnails(refs, width) || {},
          args: [list.map(({ ref, url }) => ({ ref, url })), width],
        });
        drawn = res?.result || {};
      } catch (err) {
        console.warn("sample draw failed", frameId, err);
      }
      await Promise.all(
        list.map(async (m) => {
          let data = drawn[m.ref];
          const shot = data ? null : await captured(m);
          if (shot) data = await downscale(shot, width);
          if (!data && /^https?:/i.test(m.url || "")) {
            const res = await chrome.runtime
              .sendMessage({ type: "FETCH_AS_DATA_URL", url: m.url, tabId, frameId })
              .catch(() => null);
            if (res?.ok) data = await downscale(res.dataUrl, width);
          }
          if (data) out.set(keyOf(m), data);
        })
      );
    })
  );
  return out;
}

/** The choice whose samples the panel detector calls manga pages (at least half; ties keep the score
 * order), or null when none does. */
async function guessWithDetector(scan, choices) {
  const top = choices.slice(0, PICK_GROUPS);
  const samples = top.map((i) => pickSamples(state.groups[i].members, DETECT_SAMPLES));
  const images = await sampleImages(state.tabId, samples.flat(), DETECT_WIDTH);
  if (scan !== scanId || state.userPicked) return null;
  const sent = samples.flat().filter((m) => images.has(keyOf(m)));
  if (!sent.length) return null;

  const res = await chrome.runtime.sendMessage({ type: "DETECT_PANELS", images: sent.map((m) => images.get(keyOf(m))) });
  if (!res?.ok) throw new Error(res?.error || "no answer");
  const verdict = new Map(sent.map((m, i) => [m, res.data[i]]));
  let best = null;
  top.forEach((gi, t) => {
    const seen = samples[t].map((m) => verdict.get(m)).filter((v) => v && !v.error);
    const share = seen.length ? seen.filter((v) => v.page).length / seen.length : 0;
    if (share >= 0.5 && (!best || share > best.share)) best = { gi, share };
  });
  return best?.gi ?? null;
}

/** Ask Gemini which calibration choice is the pages; adopt it unless the user clicks first. */
async function guessWithGemini(scan, choices) {
  const top = choices.slice(0, PICK_GROUPS);
  const samples = top.map((i) => pickSamples(state.groups[i].members));
  const images = await sampleImages(state.tabId, samples.flat());
  if (scan !== scanId || state.userPicked) return;

  const groups = top
    .map((gi, i) => {
      const g = state.groups[gi];
      return {
        index: i + 1,
        kind: g.kind,
        count: g.members.length,
        width: fmtRange(g.displayWidth),
        images: samples[i].map((m) => images.get(keyOf(m))).filter(Boolean),
      };
    })
    .filter((g) => g.images.length);
  if (!groups.length) throw new Error("no previewable images");

  const res = await chrome.runtime.sendMessage({ type: "PICK_GROUP", body: { title: state.title, groups } });
  if (scan !== scanId || state.userPicked) return;
  if (!res?.ok) throw new Error(res?.error || "no answer");
  if (!res.data.group) {
    calHintEl.textContent = "Gemini couldn't tell which one is the pages — pick one below.";
    return;
  }
  state.guess = top[res.data.group - 1];
  await chooseGroup(state.guess, "gemini");
}

// ---------- calibration ----------

function fmtRange([lo, hi]) {
  return lo === hi ? `${lo}` : `${lo}–${hi}`;
}

function showSite(text) {
  siteEl.hidden = false;
  siteTextEl.textContent = text;
}

async function showCalibration(hint) {
  const scan = scanId;
  state.userPicked = false;
  state.guess = null;
  const choices = state.groups.slice(0, CALIBRATION_CHOICES).map((_, i) => i);
  const { geminiGuess } = await getSettings();

  calibrateEl.hidden = false;
  calHintEl.textContent = `${hint} Looking for manga pages…`;
  await loadThumbs(choices.map((i) => pickRepresentative(state.groups[i].members)));
  if (scan !== scanId) return;
  renderCalibration(choices);

  const detected = await guessWithDetector(scan, choices).catch((err) => {
    console.warn("panel detector failed", err);
    return null;
  });
  if (scan !== scanId || state.userPicked) return;
  if (detected != null) {
    state.guess = detected;
    await chooseGroup(detected, "detector");
    return;
  }

  if (geminiGuess && choices.length > 1) {
    calHintEl.textContent = `${hint} Gemini is guessing too (~8 s)…`;
    guessWithGemini(scan, choices).catch((err) => {
      if (scan === scanId && !state.userPicked) {
        calHintEl.textContent = `${hint} (Gemini guess failed: ${err?.message || err})`;
      }
    });
  } else {
    calHintEl.textContent = choices.length === 1 ? `${hint} Only one kind of image found.` : hint;
  }
}

function renderCalibration(choices) {
  calGridEl.innerHTML = "";
  for (const i of choices) {
    const g = state.groups[i];
    const tile = document.createElement("button");
    tile.type = "button";
    tile.className = "cal-tile";
    tile.classList.toggle("guess", i === state.guess);
    const rep = pickRepresentative(g.members);
    const caption = document.createElement("span");
    caption.className = "caption";
    const natural = pageWidths(g.naturalWidths);
    caption.textContent = `${g.members.length}× ${g.kind} · ${natural.length ? `${natural.join("/")} px` : `${fmtRange(g.displayWidth)} px wide`}`;
    tile.append(thumbEl(keyOf(rep), "cal-thumb"), caption);
    tile.addEventListener("click", () => {
      state.userPicked = true;
      chooseGroup(i, "user");
    });
    calGridEl.append(tile);
  }
}

/** Remember the chosen group's dimension for this site (replacing the one being recalibrated) and select it. */
async function chooseGroup(index, source) {
  const group = state.groups[index];
  const cal = await saveCalibration(state.host, group, state.calibration?.id);
  state.calibration = cal;
  calibrateEl.hidden = true;
  const how = { gemini: " (Gemini's guess)", detector: " (found automatically)" }[source] || "";
  showSite(`${state.host} · saved: ${describeCalibration(cal)}${how}`);
  setStatus("");
  await selectGroup(index);
}

// ---------- panel list (manual translate) ----------

function renderPanels() {
  const members = state.groups[state.selected].members;
  panelsEl.innerHTML = "";
  panelsBarEl.hidden = false;

  members.forEach((m, i) => {
    const key = keyOf(m);
    if (!dev) {
      // read-only preview of what Translate will feed
      const tile = document.createElement("div");
      tile.className = "tile static";
      const caption = document.createElement("span");
      caption.className = "caption";
      caption.textContent = `${i + 1}`;
      tile.append(thumbEl(key, "tile-thumb"), caption);
      panelsEl.append(tile);
      return;
    }
    const tile = document.createElement("label");
    tile.className = "tile";
    tile.classList.toggle("checked", state.checked.has(key));
    tile.classList.toggle("visible", m.visibleFrac > 0);

    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = state.checked.has(key);
    box.addEventListener("change", () => {
      if (box.checked) state.checked.add(key);
      else state.checked.delete(key);
      tile.classList.toggle("checked", box.checked);
      updateMeta();
    });

    const caption = document.createElement("span");
    caption.className = "caption";
    caption.textContent = `${i + 1} · ${m.naturalWidth || "?"}×${m.naturalHeight || "?"}${m.visibleFrac > 0 ? " · on screen" : ""}`;

    tile.append(box, thumbEl(key, "tile-thumb"), caption);
    panelsEl.append(tile);
  });
  updateMeta();
}

function updateMeta() {
  const n = state?.checked.size || 0;
  const total = state?.selected != null ? state.groups[state.selected].members.length : 0;
  metaEl.textContent = `${n}/${total} selected`;
  updateTranslateBtn();
}

/** Dev: translate the checked panels. Otherwise: Auto on the page group (toggles to Stop). */
function updateTranslateBtn() {
  if (dev) {
    const n = state?.checked.size || 0;
    translateBtn.disabled = n === 0;
    translateBtn.textContent = n ? `Translate ${n}` : "Translate";
  } else {
    translateBtn.disabled = !autoRunning && (dryRunning || state?.selected == null);
    translateBtn.textContent = autoRunning ? "Stop" : "Translate";
  }
}

function selectMembers(filter) {
  if (state?.selected == null) return;
  const members = state.groups[state.selected].members;
  state.checked = new Set(members.filter(filter).map(keyOf));
  renderPanels();
}

async function selectGroup(index) {
  state.selected = index;
  const members = state.groups[index].members;
  state.checked = new Set(members.filter((m) => m.visibleFrac > 0).map(keyOf));
  renderPanels();
  await loadThumbs(members);
  renderPanels();
  refreshSession();
}

// ---------- scan ----------

async function scan() {
  const scan = ++scanId;
  setStatus("Scanning…");
  siteEl.hidden = true;
  calibrateEl.hidden = true;
  calGridEl.innerHTML = "";
  panelsEl.innerHTML = "";
  panelsBarEl.hidden = true;
  state = null;
  translateBtn.disabled = true;

  try {
    const tab = await getActiveTab();
    if (!tab?.id) throw new Error("No active tab.");
    if (!tab.url || !/^https?:/i.test(tab.url)) throw new Error("Open a manga reader page first.");

    await injectContent(tab.id);
    const frames = await chrome.scripting.executeScript({
      target: { tabId: tab.id, allFrames: true },
      func: () => window.__panelProbe?.collectCandidates() || null,
    });
    if (scan !== scanId) return;

    const candidates = [];
    for (const { frameId, result } of frames) {
      if (!result) continue;
      for (const c of result.candidates) candidates.push({ ...c, frameId });
    }
    if (!candidates.length) {
      setStatus("No images found yet. Scroll a bit and press Rescan.");
      return;
    }

    const groups = groupByWidth(candidates)
      .map(scoreGroup)
      .sort((a, b) => b.score - a.score);
    const host = new URL(tab.url).hostname;
    state = {
      tabId: tab.id,
      host,
      title: tab.title || "",
      groups,
      selected: null,
      checked: new Set(),
      thumbs: new Map(),
      captures: new Map(),
      calibration: null,
      guess: null,
      userPicked: false,
    };

    const calibrations = await getCalibrations(host);
    const match = findCalibratedGroup(groups, calibrations);
    if (scan !== scanId) return;
    if (match) {
      state.calibration = match.cal;
      await touchCalibration(host, match.cal.id);
      showSite(`${host} · using saved: ${describeCalibration(match.cal)}`);
      setStatus("");
      await selectGroup(match.index);
      return;
    }

    setStatus("");
    await showCalibration(
      calibrations.length
        ? "This page's images don't match the saved dimension for this site. Pick once more — it'll be remembered too."
        : "First time on this site: click the preview that looks most like a manga page."
    );
  } catch (err) {
    setStatus(String(err?.message || err), true);
  }
}

// ---------- actions ----------

scanBtn.addEventListener("click", scan);

document.getElementById("recalibrate").addEventListener("click", () => {
  if (!state) return;
  siteEl.hidden = true;
  showCalibration("Click the preview that looks most like a manga page. It replaces the saved dimension in use.");
});

document.getElementById("settings").addEventListener("click", () => chrome.runtime.openOptionsPage());

translateBtn.addEventListener("click", async () => {
  if (!dev) return toggleAuto(false);
  if (!state?.checked.size) return;
  const panels = state.groups[state.selected].members
    .filter((m) => state.checked.has(keyOf(m)))
    .map(({ ancestors, ...rest }) => rest);

  const res = await chrome.runtime.sendMessage({ type: "TRANSLATE_PANELS", tabId: state.tabId, panels });
  setStatus(
    res?.ok
      ? `Sent ${res.queued} panel(s). Progress shows on the page — you can close this popup.`
      : "Could not queue panels.",
    !res?.ok
  );
});

document.getElementById("sel-visible").addEventListener("click", () => selectMembers((m) => m.visibleFrac > 0));
document.getElementById("sel-all").addEventListener("click", () => selectMembers(() => true));
document.getElementById("sel-none").addEventListener("click", () => selectMembers(() => false));
document.getElementById("toggle").addEventListener("click", () =>
  runInAllFrames(() => window.__panelProbe?.toggleOverlay())
);
document.getElementById("boxes").addEventListener("click", async () => {
  const tab = await getActiveTab();
  if (!tab?.id) return;
  const frames = await chrome.scripting
    .executeScript({ target: { tabId: tab.id, allFrames: true }, func: () => window.__panelProbe?.toggleBoxes() })
    .catch(() => []);
  const on = frames.some((f) => f.result === true);
  setStatus(
    on
      ? "Boxes on: pink = bubble (numbered, what gets translated) · cyan = textline OCR read · red dashed = detected but dropped (OCR empty/unsure)."
      : ""
  );
});

document.getElementById("clear").addEventListener("click", async () => {
  const tab = await getActiveTab();
  if (tab?.id) await chrome.runtime.sendMessage({ type: "STOP_AUTO", tabId: tab.id });
  await runInAllFrames(() => window.__panelProbe?.clearOverlays());
  refreshSession();
});

async function toggleAuto(dryRun) {
  const tab = await getActiveTab();
  if (!tab?.id) return;
  if (dryRun ? dryRunning : autoRunning) {
    await chrome.runtime.sendMessage({ type: "STOP_AUTO", tabId: tab.id });
    setStatus(dryRun ? "OCR test stopped." : dev ? "Auto stopped." : "Stopped.");
  } else if (state?.selected != null) {
    if (dryRun) await runInAllFrames(() => window.__panelProbe?.clearOverlays());
    const group = state.groups[state.selected];
    const signature = { kind: group.kind, minW: group.displayWidth[0], maxW: group.displayWidth[1] };
    const panels = group.members.map(({ ancestors, ...rest }) => rest);
    const res = await chrome.runtime.sendMessage({
      type: "START_AUTO",
      tabId: tab.id,
      frameId: group.members[0].frameId,
      signature,
      panels,
      dryRun,
    });
    const what = dryRun ? "OCR test (no Gemini)" : dev ? "Auto" : "Translation";
    const started = dev || dryRun
      ? `${what} on: ${group.members.length} pages now, new ones are picked up as they load.`
      : "Translating this page — new pages are picked up as you scroll. You can close this popup.";
    setStatus(res?.ok ? started : `${what} failed: ${res?.error || "unknown error"}`, !res?.ok);
  }
  refreshSession();
}

autoBtn.addEventListener("click", () => toggleAuto(false));
dryBtn.addEventListener("click", () => toggleAuto(true));

(async () => {
  dev = (await getSettings()).devMode;
  document.body.classList.toggle("dev", dev);
  // opening the popup usually precedes a translation: start loading the models now
  chrome.runtime.sendMessage({ type: "ENGINE_WARMUP" }).catch(() => {}).finally(checkApp);
  refreshSession();
  setInterval(refreshSession, 1000);
  scan();
})();
