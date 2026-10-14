/**
 * Shared by popup + options: extension storage and site calibration ("which images are the pages").
 *
 * chrome.storage.local:
 *   geminiKey     string (sent only to the Gemini API, by the service worker)
 *   settings      { geminiGuess: boolean, devMode: boolean, overlayKey: string (e.g. "Alt+O", "" = off),
 *                   fastReread: boolean (greedy manga-ocr instead of 4-beam search) }
 *   calibrations  { [hostname]: Calibration[] }  most recently used first
 *
 * Calibration = { id, kind, naturalWidths: number[], displayWidth: [min, max], savedAt, lastUsed }
 */
const DEFAULT_SETTINGS = { geminiGuess: false, devMode: false, overlayKey: "Alt+O", fastReread: false };
const MAX_CALIBRATIONS_PER_SITE = 8;
const MIN_NATURAL_WIDTH = 120; // lazy-load placeholders (1×1 gifs etc.) say nothing about the pages
const NATURAL_TOLERANCE = 0.03;
const DISPLAY_TOLERANCE = 0.06;

async function getSettings() {
  const { settings } = await chrome.storage.local.get("settings");
  return { ...DEFAULT_SETTINGS, ...settings };
}

async function setSettings(patch) {
  await chrome.storage.local.set({ settings: { ...(await getSettings()), ...patch } });
}

/** "Ctrl+Alt+Shift+Meta+Key" for a keydown, or null for a lone modifier. Same rule as content.js. */
function comboFromEvent(e) {
  if (["Control", "Alt", "Shift", "Meta", "AltGraph"].includes(e.key)) return null;
  const key = /^Key[A-Z]$/.test(e.code) ? e.code.slice(3) : /^Digit\d$/.test(e.code) ? e.code.slice(5) : e.key.length === 1 ? e.key.toUpperCase() : e.key;
  return [e.ctrlKey && "Ctrl", e.altKey && "Alt", e.shiftKey && "Shift", e.metaKey && "Meta", key].filter(Boolean).join("+");
}

async function getAllCalibrations() {
  const { calibrations } = await chrome.storage.local.get("calibrations");
  return calibrations || {};
}

async function getCalibrations(host) {
  return (await getAllCalibrations())[host] || [];
}

async function writeCalibrations(host, list) {
  const all = await getAllCalibrations();
  if (list.length) all[host] = list;
  else delete all[host];
  await chrome.storage.local.set({ calibrations: all });
}

function pageWidths(widths) {
  return [...new Set(widths.filter((w) => w >= MIN_NATURAL_WIDTH))].sort((a, b) => a - b);
}

function near(a, b, tolerance) {
  return Math.abs(a - b) <= Math.max(4, b * tolerance);
}

/** Natural (file) width is the stable signal; display width only when either side has none. */
function matchesCalibration(group, cal) {
  if (group.kind !== cal.kind) return false;
  const groupNatural = pageWidths(group.naturalWidths || []);
  if (groupNatural.length && cal.naturalWidths.length) {
    return groupNatural.some((w) => cal.naturalWidths.some((c) => near(w, c, NATURAL_TOLERANCE)));
  }
  const [lo, hi] = cal.displayWidth;
  const [glo, ghi] = group.displayWidth;
  return glo <= hi * (1 + DISPLAY_TOLERANCE) + 4 && ghi >= lo * (1 - DISPLAY_TOLERANCE) - 4;
}

function calibrationFromGroup(group) {
  const now = Date.now();
  return {
    id: `${now.toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
    kind: group.kind,
    naturalWidths: pageWidths(group.naturalWidths || []),
    displayWidth: [...group.displayWidth],
    savedAt: now,
    lastUsed: now,
  };
}

/** Remember the group's dimension for the site (refreshes an existing match instead of duplicating). */
async function saveCalibration(host, group, replaceId = null) {
  let list = (await getCalibrations(host)).filter((c) => c.id !== replaceId);
  const existing = list.find((c) => matchesCalibration(group, c));
  const cal = existing ? { ...existing, lastUsed: Date.now() } : calibrationFromGroup(group);
  list = [cal, ...list.filter((c) => c !== existing)].slice(0, MAX_CALIBRATIONS_PER_SITE);
  await writeCalibrations(host, list);
  return cal;
}

async function touchCalibration(host, id) {
  const list = await getCalibrations(host);
  const cal = list.find((c) => c.id === id);
  if (!cal) return;
  cal.lastUsed = Date.now();
  await writeCalibrations(host, [cal, ...list.filter((c) => c !== cal)]);
}

async function removeCalibration(host, id) {
  await writeCalibrations(host, (await getCalibrations(host)).filter((c) => c.id !== id));
}

function describeCalibration(cal) {
  const widths = cal.naturalWidths.length
    ? `${cal.naturalWidths.join("/")} px file width`
    : `${cal.displayWidth[0] === cal.displayWidth[1] ? cal.displayWidth[0] : cal.displayWidth.join("–")} px on screen`;
  return `${cal.kind} · ${widths}`;
}
