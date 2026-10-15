// SPDX-License-Identifier: GPL-3.0-only
// app/manga_ocr_rec.py: manga-ocr re-reads bubbles when some Paddle text on the page looks wrong.
// Encoder/decoder ONNX exports (tools/export_mocr.py); decoding mirrors HF generate() with the model's
// generation config (4 beams, early stopping, length penalty 2, no repeated trigrams, max 300).

import { dropSpaces, isCp932, pyStrip } from "./text.js";

export const MOCR = { pad: 6, triggerMinChars: 3.0, readMinChars: 2.5, shortRatio: 0.6, meanP: 0.9, minP: 0.2, beams: 4 };

const START = 2;
const EOS = 3;
const LENGTH_PENALTY = 2.0;
const MAX_LEN = 300;
const SIZE = 224;
const f32 = Math.fround;

// ---------- app rules ----------

export function expectedChars(r) {
  if (!r.lines.length || r.fontSize <= 0) return 0;
  let spans = 0;
  for (const l of r.lines) {
    const xs = l.map((p) => p[0]);
    const ys = l.map((p) => p[1]);
    spans += Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
  }
  return spans / r.fontSize;
}

export function suspectReason(r) {
  const expected = expectedChars(r);
  if (expected < MOCR.triggerMinChars) return null;
  const text = [...pyStrip(r.text || "")];
  if (!text.length) return "empty";
  const junk = text.filter((c) => /^[A-Za-z0-9]$/.test(c) || !isCp932(c)).length;
  if (junk) return `junk ${junk}/${text.length}`;
  if (text.length < MOCR.shortRatio * expected) return `short ${text.length}/${Math.round(expected)}`;
  return null;
}

/** manga_ocr post_process + jaconv.h2z(ascii=True, digit=True) */
export function postProcess(text) {
  text = dropSpaces(text).replace(/…/g, "...");
  text = text.replace(/[・.]{2,}/g, (m) => ".".repeat(m.length));
  return text.replace(/[\x21-\x7e]/g, (c) => String.fromCharCode(c.charCodeAt(0) + 0xfee0));
}

// ---------- preprocessing: PIL convert("L") + resize((224, 224), BILINEAR) ----------

const PRECISION_BITS = 22;

function pilCoeffs(inSize, outSize) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = filterscale;
  const out = [];
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1 / filterscale;
    const xmin = Math.max(0, Math.trunc(center - support + 0.5));
    const xmax = Math.min(inSize, Math.trunc(center + support + 0.5)) - xmin;
    const k = [];
    let ww = 0;
    for (let x = 0; x < xmax; x++) {
      const t = Math.abs((x + xmin - center + 0.5) * ss);
      const w = t < 1 ? 1 - t : 0;
      k.push(w);
      ww += w;
    }
    const kk = k.map((w) => {
      const v = (ww !== 0 ? w / ww : w) * (1 << PRECISION_BITS);
      return v < 0 ? Math.trunc(-0.5 + v) : Math.trunc(0.5 + v);
    });
    out.push({ xmin, kk });
  }
  return out;
}

const clip8 = (v) => (v >= 255 * 2 ** PRECISION_BITS + 2 ** PRECISION_BITS ? 255 : v <= 0 ? 0 : Math.min(255, Math.floor(v / 2 ** PRECISION_BITS)));

function pilResize(src, sw, sh, dw, dh) {
  let img = src;
  let w = sw;
  if (dw !== sw) {
    const cx = pilCoeffs(sw, dw);
    const tmp = new Uint8Array(dw * sh);
    for (let y = 0; y < sh; y++) {
      for (let i = 0; i < dw; i++) {
        const { xmin, kk } = cx[i];
        let s = 1 << (PRECISION_BITS - 1);
        for (let k = 0; k < kk.length; k++) s += img[y * sw + xmin + k] * kk[k];
        tmp[y * dw + i] = clip8(s);
      }
    }
    img = tmp;
    w = dw;
  }
  if (dh === sh) return img;
  const cy = pilCoeffs(sh, dh);
  const out = new Uint8Array(w * dh);
  for (let j = 0; j < dh; j++) {
    const { xmin: ymin, kk } = cy[j];
    for (let i = 0; i < w; i++) {
      let s = 1 << (PRECISION_BITS - 1);
      for (let k = 0; k < kk.length; k++) s += img[(ymin + k) * w + i] * kk[k];
      out[j * w + i] = clip8(s);
    }
  }
  return out;
}

/** RGB crop → 224×224 PIL-L pixels. */
export function prepCrop(rgb, width, height) {
  const g = new Uint8Array(width * height);
  for (let p = 0, q = 0; p < g.length; p++, q += 3) g[p] = (rgb[q] * 19595 + rgb[q + 1] * 38470 + rgb[q + 2] * 7471 + 0x8000) >> 16;
  return pilResize(g, width, height, SIZE, SIZE);
}

// ---------- decoding ----------

function banNgrams(row, lp, off) {
  const n = row.length;
  if (n < 2) return;
  const a = row[n - 2];
  const b = row[n - 1];
  for (let i = 0; i + 2 < n; i++) if (row[i] === a && row[i + 1] === b) lp[off + row[i + 2]] = -Infinity;
}

export class MangaOcr {
  /** enc/dec: ort sessions; vocab: { tokens, special } (models/mocr_vocab.json). */
  constructor(ort, ep, enc, dec, vocab) {
    Object.assign(this, { ort, ep, enc, dec });
    this.tokens = vocab.tokens;
    this.special = new Set(vocab.special);
    this.V = vocab.tokens.length;
  }

  async hiddenTensor(data, dims) {
    const { ort } = this;
    if (this.ep !== "webgpu") return new ort.Tensor("float32", data, dims);
    const device = await ort.env.webgpu.device;
    const buffer = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffer, 0, data);
    return ort.Tensor.fromGpuBuffer(buffer, { dataType: "float32", dims, dispose: () => buffer.destroy() });
  }

  async step(rows, hid) {
    const L = rows[0].length;
    const ids = new BigInt64Array(rows.length * L);
    rows.forEach((r, i) => r.forEach((t, j) => (ids[i * L + j] = BigInt(t))));
    const out = await this.dec.run({ input_ids: new this.ort.Tensor("int64", ids, [rows.length, L]), encoder_hidden_states: hid });
    return out.logprobs.data;
  }

  async greedy(hidden, n) {
    const { V } = this;
    const hid = await this.hiddenTensor(hidden, [n, 197, 768]);
    const rows = Array.from({ length: n }, () => [START]);
    const tokP = rows.map(() => []);
    const done = new Array(n).fill(false);
    while (!done.every(Boolean) && rows[0].length < MAX_LEN) {
      const lp = await this.step(rows, hid);
      rows.forEach((row, b) => {
        if (done[b]) return row.push(0);
        const off = b * V;
        banNgrams(row, lp, off);
        let best = 0;
        let max = -Infinity;
        for (let t = 0; t < V; t++) if (lp[off + t] > max) (max = lp[off + t]), (best = t);
        let sum = 0;
        for (let t = 0; t < V; t++) sum += Math.exp(lp[off + t] - max);
        tokP[b].push(1 / sum);
        row.push(best);
        done[b] = best === EOS;
      });
    }
    hid.dispose();
    return rows.map((r, b) => ({ ids: r.slice(1), tokP: tokP[b] }));
  }

  async beamSearch(hidden, n, B) {
    const { V } = this;
    const per = 197 * 768;
    const expanded = new Float32Array(n * B * per);
    for (let c = 0; c < n; c++) for (let k = 0; k < B; k++) expanded.set(hidden.subarray(c * per, (c + 1) * per), (c * B + k) * per);
    const hid = await this.hiddenTensor(expanded, [n * B, 197, 768]);
    const crops = Array.from({ length: n }, () => ({
      running: Array.from({ length: B }, (_, k) => ({ ids: [START], score: k ? -1e9 : 0, lps: [] })),
      finished: [],
      done: false,
    }));
    while (!crops.every((c) => c.done) && crops[0].running[0].ids.length < MAX_LEN) {
      const rows = crops.flatMap((c) => c.running.map((r) => r.ids));
      const lp = await this.step(rows, hid);
      const curLen = rows[0].length;
      crops.forEach((c, ci) => {
        if (c.done) return c.running.forEach((r) => r.ids.push(0));
        const cands = [];
        c.running.forEach((r, k) => {
          const off = (ci * B + k) * V;
          banNgrams(r.ids, lp, off);
          const top = [];
          for (let t = 0; t < V; t++) {
            const v = lp[off + t];
            if (top.length < 2 * B || v > top[top.length - 1].v) {
              top.push({ t, v });
              top.sort((a, b) => b.v - a.v);
              if (top.length > 2 * B) top.pop();
            }
          }
          for (const { t, v } of top) cands.push({ k, t, v, acc: r.score + v });
        });
        cands.sort((a, b) => b.acc - a.acc);
        const K = cands.slice(0, 2 * B);
        K.forEach((cand, i) => {
          if (cand.t !== EOS || i >= B) return;
          const r = c.running[cand.k];
          c.finished.push({ ids: [...r.ids, EOS], score: cand.acc / curLen ** LENGTH_PENALTY, lps: [...r.lps, cand.v] });
        });
        c.finished.sort((a, b) => b.score - a.score);
        c.finished.length = Math.min(c.finished.length, B);
        c.running = K.filter((cand) => cand.t !== EOS)
          .slice(0, B)
          .map((cand) => {
            const r = c.running[cand.k];
            return { ids: [...r.ids, cand.t], score: cand.acc, lps: [...r.lps, cand.v] };
          });
        if (c.finished.length >= B) c.done = true;
      });
    }
    hid.dispose();
    return crops.map((c) => c.finished[0] || c.running[0]).map((h) => ({ ids: h.ids.slice(1), tokP: h.lps.map(Math.exp) }));
  }

  /** Encoder once and two decoder steps at beam width, on a blank crop (shader compilation). */
  async warmup(beams = MOCR.beams) {
    const out = await this.enc.run({ pixel_values: new this.ort.Tensor("float32", new Float32Array(3 * SIZE * SIZE), [1, 3, SIZE, SIZE]) });
    const per = out.hidden.data;
    const expanded = new Float32Array(beams * per.length);
    for (let k = 0; k < beams; k++) expanded.set(per, k * per.length);
    const hid = await this.hiddenTensor(expanded, [beams, 197, 768]);
    await this.step(Array.from({ length: beams }, () => [START]), hid);
    await this.step(Array.from({ length: beams }, () => [START, START]), hid);
    hid.dispose();
  }

  /** MangaOcrReader.read: [{ text, meanP, minP }] per 224×224 L crop, one batch. */
  async read(grays, beams = MOCR.beams) {
    const n = grays.length;
    const plane = SIZE * SIZE;
    const x = new Float32Array(n * 3 * plane);
    grays.forEach((g, i) => {
      for (let p = 0; p < plane; p++) {
        const v = f32(f32(f32(g[p] / 255) - 0.5) / 0.5);
        x[i * 3 * plane + p] = v;
        x[i * 3 * plane + plane + p] = v;
        x[i * 3 * plane + 2 * plane + p] = v;
      }
    });
    const hidden = (await this.enc.run({ pixel_values: new this.ort.Tensor("float32", x, [n, 3, SIZE, SIZE]) })).hidden.data;
    const seqs = beams > 1 ? await this.beamSearch(hidden, n, beams) : await this.greedy(hidden, n);
    return seqs.map(({ ids, tokP }) => {
      const kept = [];
      const p = [];
      for (let t = 0; t < ids.length; t++) {
        p.push(tokP[t]);
        if (ids[t] === EOS) break;
        kept.push(ids[t]);
      }
      const text = postProcess(kept.filter((i) => !this.special.has(i)).map((i) => this.tokens[i]).join(""));
      return { text, meanP: p.length ? p.reduce((a, b) => a + b, 0) / p.length : 0, minP: p.length ? Math.min(...p) : 0 };
    });
  }
}

/** manga_ocr_rec.reread: mutates region texts; returns the read log (like last_reads). */
export async function reread(reader, img, regions, beams = MOCR.beams) {
  const suspects = regions.map(suspectReason).filter(Boolean);
  if (!suspects.length) return [];
  const picked = regions.filter((r) => expectedChars(r) >= MOCR.readMinChars);
  const { width: W, height: H, data } = img;
  const grays = picked.map((r) => {
    const [x1, y1, x2, y2] = r.xyxy;
    const cx1 = Math.max(0, x1 - MOCR.pad);
    const cy1 = Math.max(0, y1 - MOCR.pad);
    const cw = Math.min(W, x2 + MOCR.pad) - cx1;
    const ch = Math.min(H, y2 + MOCR.pad) - cy1;
    const crop = new Uint8Array(cw * ch * 3);
    for (let y = 0; y < ch; y++) crop.set(data.subarray(((cy1 + y) * W + cx1) * 3, ((cy1 + y) * W + cx1 + cw) * 3), y * cw * 3);
    return prepCrop(crop, cw, ch);
  });
  const t0 = performance.now();
  const reads = await reader.read(grays, beams);
  const ms = performance.now() - t0;
  const log = [];
  picked.forEach((r, i) => {
    const { text, meanP, minP } = reads[i];
    const ok = meanP >= MOCR.meanP && minP >= MOCR.minP && !!text;
    log.push({ before: r.text, after: text, accepted: ok, mean_p: meanP, min_p: minP, why: suspectReason(r) || "-" });
    if (ok) r.text = text;
  });
  log.push({ batch_ms: ms, bubbles: picked.length, trigger: suspects });
  return log;
}
