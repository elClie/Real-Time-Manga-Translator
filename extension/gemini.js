// SPDX-License-Identifier: GPL-3.0-only
// Batched Gemini translation over REST (port of app/gemini_client.py).
// Model fallback chain with a per-call timeout (single models regularly stall or 503);
// non-urgent calls start at least MIN_INTERVAL_MS apart, urgent calls skip the wait.
// The key lives in chrome.storage.local and only ever goes to the Gemini API.

export const GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-lite-latest"];
const TIMEOUT_MS = 20_000;
const THINKING_LEVEL = "minimal";
const MIN_INTERVAL_MS = 6_000;
const API = (model) => `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`;
const MODELS_API = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1";
const RETRY_NEXT_MODEL = new Set([400, 404, 408, 429, 500, 502, 503, 504]);

export class GeminiKeyMissing extends Error {
  constructor() {
    super("no Gemini API key — add one in the extension settings");
  }
}

class GeminiHTTPError extends Error {
  constructor(status, message) {
    super(`HTTP ${status}: ${String(message).slice(0, 160)}`);
    this.status = status;
  }
}

class Timeout extends Error {}

async function storedKey() {
  const { geminiKey } = await chrome.storage.local.get("geminiKey");
  return (geminiKey || "").trim();
}

async function fetchJson(url, init, timeoutMs = TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...init, signal: controller.signal });
    return { res, data: await res.json().catch(() => ({})) };
  } catch (err) {
    if (err?.name === "AbortError") throw new Timeout();
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

export class Gemini {
  models = [...GEMINI_MODELS];
  lastStart = 0;
  rateChain = Promise.resolve();
  lastCall = null; // { ok, error } of the latest request, for the status line
  verified = new Map(); // key → checkKey result (valid or rejected; unreachable isn't cached)

  async hasKey() {
    return Boolean(await storedKey());
  }

  /** [true, ""] valid · [false, why] rejected · [null, why] couldn't reach Gemini. */
  async checkKey(key, timeoutMs = TIMEOUT_MS) {
    let result;
    try {
      const { res, data } = await fetchJson(MODELS_API, { headers: { "x-goog-api-key": key } }, timeoutMs);
      result = res.status === 200 ? [true, ""] : [false, (data.error?.message || `HTTP ${res.status}`).slice(0, 200)];
    } catch (err) {
      return [null, err instanceof Timeout ? "timeout" : err?.name || "network error"];
    }
    this.verified.set(key, result);
    if (result[0]) this.lastCall = null;
    return result;
  }

  /** { state: no_key | invalid | unreachable | error | ok, error?, model } */
  async status() {
    const key = await storedKey();
    const model = this.models[0];
    if (!key) return { state: "no_key", model };
    const [valid, why] = this.verified.get(key) || (await this.checkKey(key, 4_000));
    if (valid === false) return { state: "invalid", error: why, model };
    if (valid === null) return { state: "unreachable", error: why, model };
    if (this.lastCall && !this.lastCall.ok) return { state: "error", error: this.lastCall.error, model };
    return { state: "ok", model };
  }

  /** Translates texts in one request; translations are aligned with texts (null if missing). */
  async translate(texts, { src = "Japanese", tgt = "English", urgent = false } = {}) {
    const payload = texts.map((t, id) => ({ id, text: t })).filter((p) => p.text && p.text.trim());
    if (!payload.length) return { translations: texts.map(() => null), meta: {} };

    const head = translationPrompt(src, tgt);
    const parts = [{ text: head + JSON.stringify(payload) }];

    let raw;
    let meta;
    if (urgent) {
      ({ raw, meta } = await this.generate(parts));
    } else {
      const turn = this.rateChain.then(async () => {
        const wait = this.lastStart + MIN_INTERVAL_MS - performance.now();
        if (wait > 0) await new Promise((r) => setTimeout(r, wait));
        return this.generate(parts);
      });
      this.rateChain = turn.catch(() => {});
      ({ raw, meta } = await turn);
    }

    const byId = parseTranslations(raw, payload.length);
    const missing = payload.filter((p) => !byId.has(p.id));
    if (missing.length) {
      // one retry for what the reply lost (unparseable JSON, skipped ids); not paced, it's recovery
      const retry = await this.generate([{ text: head + JSON.stringify(missing) }]);
      const recovered = parseTranslations(retry.raw, missing.length);
      for (const p of missing) if (recovered.has(p.id)) byId.set(p.id, recovered.get(p.id));
      meta = { ...meta, retried: missing.length, recovered: recovered.size };
    }
    const translations = texts.map((_, i) => (byId.has(i) ? byId.get(i) : null));
    meta = {
      ...meta,
      empty: payload.filter((p) => byId.get(p.id) === "").length,
      missing: payload.filter((p) => !byId.has(p.id)).length,
    };
    return { translations, meta };
  }

  /** One-off JSON request with arbitrary parts (text + inline images); skips the spacing gate. */
  async askJson(parts) {
    const { raw, meta } = await this.generate(parts);
    return { data: parseJson(raw), meta };
  }

  async call(key, model, parts, thinking) {
    const generationConfig = { responseMimeType: "application/json" };
    if (thinking && THINKING_LEVEL) generationConfig.thinkingConfig = { thinkingLevel: THINKING_LEVEL };
    const { res, data } = await fetchJson(API(model), {
      method: "POST",
      headers: { "x-goog-api-key": key, "content-type": "application/json" },
      body: JSON.stringify({ contents: [{ parts }], generationConfig }),
    });
    if (res.status !== 200) throw new GeminiHTTPError(res.status, data.error?.message || "");
    const texts = [];
    for (const c of data.candidates || []) {
      for (const p of c.content?.parts || []) if (p.text && !p.thought) texts.push(p.text);
    }
    return { text: texts.join("\n"), usage: data.usageMetadata || {} };
  }

  async generate(parts) {
    try {
      const out = await this.tryModels(parts);
      this.lastCall = { ok: true };
      return out;
    } catch (err) {
      if (!(err instanceof GeminiKeyMissing)) this.lastCall = { ok: false, error: String(err?.message || err).slice(0, 200) };
      throw err;
    }
  }

  /** Tries each model in order; the first one that answers moves to the front. */
  async tryModels(parts) {
    const key = await storedKey();
    if (!key) throw new GeminiKeyMissing();
    this.lastStart = performance.now();
    const failures = [];
    for (const model of [...this.models]) {
      let thinking = true;
      for (;;) {
        const t0 = performance.now();
        let reply;
        try {
          reply = await this.call(key, model, parts, thinking);
        } catch (err) {
          if (err instanceof Timeout) {
            failures.push(`${model}: timeout ${TIMEOUT_MS / 1000}s`);
            break;
          }
          if (err instanceof GeminiHTTPError) {
            if (err.status === 400 && thinking) {
              thinking = false; // model may not accept thinkingLevel
              continue;
            }
            failures.push(`${model}: ${err.message}`);
            if (RETRY_NEXT_MODEL.has(err.status)) break;
            throw err;
          }
          failures.push(`${model}: ${err?.name || "network error"}`);
          break;
        }
        if (failures.length) console.log(`gemini fallbacks: ${failures.join("; ")}`);
        this.models = [model, ...this.models.filter((m) => m !== model)];
        return {
          raw: reply.text,
          meta: {
            model,
            api_ms: Math.round(performance.now() - t0),
            thoughts: reply.usage.thoughtsTokenCount || 0,
            out_tokens: reply.usage.candidatesTokenCount ?? null,
          },
        };
      }
      console.log(`gemini failed: ${failures[failures.length - 1]}`);
    }
    throw new Error("Gemini unavailable — " + failures.join("; "));
  }
}

/** Instruction head; the JSON payload [{id, text}] follows it. */
export function translationPrompt(src = "Japanese", tgt = "English") {
  return (
    `You are translating manga/webtoon dialogue and UI text from ${src} to ${tgt}.\n` +
    "Lines come from consecutive panels in reading order; keep names, pronouns and each " +
    "character's voice consistent across them.\n" +
    "Text comes from OCR and may contain misread characters or stray digits/symbols; " +
    "infer the intended text and drop obvious junk. If a line is pure noise, return an empty translation.\n" +
    "Return ONLY a JSON array of objects with keys id and translation.\n" +
    "Keep the same ids. Do not merge or split lines.\n" +
    "Preserve tone; do not add notes or romanization unless that is the whole text.\n\n" +
    "INPUT:\n"
  );
}

export function parseJson(raw) {
  let text = raw.trim();
  const fence = /```(?:json)?\s*([\s\S]*?)```/.exec(text);
  if (fence) text = fence[1].trim();
  try {
    return JSON.parse(text);
  } catch {
    const m = /\[[\s\S]*\]|\{[\s\S]*\}/.exec(text);
    try {
      return m ? JSON.parse(m[0]) : null;
    } catch {
      return null;
    }
  }
}

// value runs to the quote that closes the object, so unescaped quotes inside it survive
const ENTRY_RE = /"id"\s*:\s*(\d+)\s*,\s*"(?:translation|text)"\s*:\s*"(.*?)"\s*}/gs;

export function parseTranslations(raw, expected) {
  const text = raw.trim();
  const data = parseJson(raw);
  const result = new Map();
  if (Array.isArray(data)) {
    for (const item of data) {
      if (item && typeof item === "object" && "id" in item) {
        result.set(Math.trunc(Number(item.id)), String(item.translation || item.text || ""));
      }
    }
  } else if (data && typeof data === "object" && Array.isArray(data.translations)) {
    for (const item of data.translations) result.set(Math.trunc(Number(item.id)), String(item.translation ?? ""));
  }

  if (result.size < expected) {
    // the array as a whole didn't parse (e.g. an unescaped quote in one line): salvage entry by entry
    for (const m of text.matchAll(ENTRY_RE)) {
      const idx = Number(m[1]);
      if (result.has(idx)) continue;
      try {
        result.set(idx, JSON.parse(`"${m[2]}"`));
      } catch {
        result.set(idx, m[2].replace(/\\"/g, '"'));
      }
    }
  }
  if (result.size < expected) {
    console.warn(`parsed ${result.size}/${expected} translations; raw head=${JSON.stringify(text.slice(0, 200))}`);
  }
  return result;
}
