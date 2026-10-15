// SPDX-License-Identifier: GPL-3.0-only
const keyInput = document.getElementById("key");
const keyStatus = document.getElementById("key-status");
const appStatus = document.getElementById("app-status");
const guessBox = document.getElementById("gemini-guess");
const devBox = document.getElementById("dev-mode");
const fastRereadBox = document.getElementById("fast-reread");
const overlayKeyInput = document.getElementById("overlay-key");
const overlayKeyStatus = document.getElementById("overlay-key-status");
const sitesEl = document.getElementById("sites");
const forgetAllBtn = document.getElementById("forget-all");

function show(el, text, tone = "") {
  el.textContent = text;
  el.className = `status ${tone}`;
}

function maskKey(key) {
  return key.length > 10 ? `${key.slice(0, 4)}…${key.slice(-4)}` : "saved";
}

function describeGemini(g) {
  if (!g) return ["no answer from the extension", false];
  if (g.state === "no_key") return ["no API key yet", false];
  if (g.state === "invalid") return [`key rejected — ${g.error}`, false];
  if (g.state === "unreachable") return [`can't reach Google (${g.error})`, false];
  if (g.state === "error") return [`last request failed — ${g.error}`, false];
  return [`ready · ${g.model}`, true];
}

async function refreshAppStatus() {
  const h = await chrome.runtime.sendMessage({ type: "HEALTH" }).catch(() => null);
  const e = h?.engine || {};
  const engine =
    e.state === "ready"
      ? `ready on ${e.ep === "webgpu" ? "WebGPU" : "WASM (slow)"}`
      : e.state === "error"
        ? `failed — ${e.error}`
        : e.state === "loading"
          ? "loading models"
          : "starts with the first translation";
  const [gemini, ok] = describeGemini(h?.gemini);
  show(appStatus, `Gemini: ${gemini} · OCR engine: ${engine}`, ok && e.state !== "error" ? "ok" : "warn");
}

async function loadKey() {
  const { geminiKey } = await chrome.storage.local.get("geminiKey");
  keyInput.value = "";
  keyInput.placeholder = geminiKey ? `Saved key ${maskKey(geminiKey)} — paste a new one to replace` : "Paste your API key";
  document.getElementById("key-remove").disabled = !geminiKey;
}

function describePush(res, removed) {
  if (removed) return ["Removed.", "ok"];
  if (!res) return ["Saved, but couldn't check it (no answer from the extension).", "warn"];
  if (!res.ok) return [res.error || "Gemini refused the key.", "error"];
  if (res.warning) return [`Saved. ${res.warning}`, "warn"];
  return ["Saved and verified with Gemini ✓", "ok"];
}

document.getElementById("key-save").addEventListener("click", async () => {
  const key = keyInput.value.trim();
  if (!key) {
    show(keyStatus, "Paste a key first.", "error");
    return;
  }
  show(keyStatus, "Checking the key…");
  const { geminiKey: previous } = await chrome.storage.local.get("geminiKey");
  await chrome.storage.local.set({ geminiKey: key });
  const res = await chrome.runtime.sendMessage({ type: "CHECK_KEY" }).catch(() => null);
  if (res && res.ok === false) {
    if (previous) await chrome.storage.local.set({ geminiKey: previous });
    else await chrome.storage.local.remove("geminiKey");
  }
  show(keyStatus, ...describePush(res, false));
  await loadKey();
  refreshAppStatus();
});

document.getElementById("key-remove").addEventListener("click", async () => {
  await chrome.storage.local.remove("geminiKey");
  show(keyStatus, ...describePush(null, true));
  await loadKey();
  refreshAppStatus();
});

document.getElementById("key-show").addEventListener("click", (e) => {
  const hidden = keyInput.type === "password";
  keyInput.type = hidden ? "text" : "password";
  e.target.textContent = hidden ? "Hide" : "Show";
});

guessBox.addEventListener("change", () => setSettings({ geminiGuess: guessBox.checked }));
devBox.addEventListener("change", () => setSettings({ devMode: devBox.checked }));
fastRereadBox.addEventListener("change", () => setSettings({ fastReread: fastRereadBox.checked }));

function showOverlayKey(key) {
  overlayKeyInput.value = key || "";
  document.getElementById("overlay-key-clear").disabled = !key;
}

overlayKeyInput.addEventListener("keydown", async (e) => {
  if (e.key === "Tab") return;
  e.preventDefault();
  if (e.key === "Escape") return overlayKeyInput.blur();
  const combo = comboFromEvent(e);
  if (!combo) return;
  const bare = !e.ctrlKey && !e.altKey && !e.metaKey && combo.length === 1;
  await setSettings({ overlayKey: combo });
  showOverlayKey(combo);
  show(overlayKeyStatus, bare ? `Saved ${combo}. A single key also fires while you read — a combo like Alt+${combo} is safer.` : `Saved ${combo}.`, bare ? "warn" : "ok");
  overlayKeyInput.blur();
});

document.getElementById("overlay-key-clear").addEventListener("click", async () => {
  await setSettings({ overlayKey: "" });
  showOverlayKey("");
  show(overlayKeyStatus, "Shortcut off.", "ok");
});

function timeAgo(ms) {
  const min = Math.round((Date.now() - ms) / 60000);
  if (min < 1) return "just now";
  if (min < 60) return `${min} min ago`;
  const h = Math.round(min / 60);
  return h < 48 ? `${h} h ago` : `${Math.round(h / 24)} days ago`;
}

async function renderSites() {
  const all = await getAllCalibrations();
  const hosts = Object.keys(all).sort();
  sitesEl.innerHTML = "";
  forgetAllBtn.hidden = !hosts.length;
  if (!hosts.length) {
    sitesEl.innerHTML = '<p class="empty">No sites calibrated yet.</p>';
    return;
  }
  for (const host of hosts) {
    const site = document.createElement("div");
    site.className = "site";
    const head = document.createElement("div");
    head.className = "site-head";
    head.textContent = host;
    const forget = document.createElement("button");
    forget.type = "button";
    forget.textContent = "Forget site";
    forget.addEventListener("click", async () => {
      await writeCalibrations(host, []);
      renderSites();
    });
    head.append(forget);
    site.append(head);

    for (const cal of all[host]) {
      const row = document.createElement("div");
      row.className = "cal";
      const label = document.createElement("span");
      label.textContent = describeCalibration(cal);
      const when = document.createElement("span");
      when.className = "when";
      when.textContent = `used ${timeAgo(cal.lastUsed)}`;
      label.append(when);
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "Remove";
      remove.addEventListener("click", async () => {
        await removeCalibration(host, cal.id);
        renderSites();
      });
      row.append(label, remove);
      site.append(row);
    }
    sitesEl.append(site);
  }
}

forgetAllBtn.addEventListener("click", async () => {
  await chrome.storage.local.set({ calibrations: {} });
  renderSites();
});

chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes.calibrations) renderSites();
});

(async () => {
  const settings = await getSettings();
  guessBox.checked = settings.geminiGuess;
  devBox.checked = settings.devMode;
  fastRereadBox.checked = settings.fastReread;
  showOverlayKey(settings.overlayKey);
  await loadKey();
  renderSites();
  refreshAppStatus();
})();
