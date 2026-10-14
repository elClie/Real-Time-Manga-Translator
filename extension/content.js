(() => {
  const MIN_SIDE = 120;
  const MIN_AREA = 120 * 120;
  const MAX_CANDIDATES = 200;
  const ANCESTOR_DEPTH = 4;
  const THUMB_WIDTH = 160;
  const LAZY_ATTRS = ["data-src", "data-original", "data-url", "data-lazy-src"];
  const BG_SELECTOR = "div, section, article, li, span, a";

  if (window.__panelProbe?.version === 26) return;

  const nodeIds = new WeakMap();
  let nextNodeId = 1;
  function nodeId(node) {
    if (!nodeIds.has(node)) nodeIds.set(node, nextNodeId++);
    return nodeIds.get(node);
  }

  function absUrl(raw) {
    try {
      return new URL(raw, location.href).href;
    } catch {
      return raw;
    }
  }

  function cssUrl(value) {
    if (!value || value === "none") return null;
    const m = String(value).match(/url\((['"]?)(.*?)\1\)/i);
    return m?.[2] ? absUrl(m[2]) : null;
  }

  /** Same identity the background worker uses: images by URL (indices shift), canvases by index + content. */
  function panelKey(p) {
    return p.kind === "canvas" ? `${p.ref}|${p.fp || ""}` : `${p.kind}|${p.url || ""}`;
  }

  function demoteReason(el) {
    const node = el.closest?.(
      "header, footer, nav, aside, [role='banner'], [role='navigation']"
    );
    if (node) return "chrome";

    const cls = `${el.className || ""} ${el.id || ""}`.toLowerCase();
    if (
      /(thumb|thumbnail|avatar|icon|logo|banner|promo|recommend|related|share|button)/.test(
        cls
      )
    ) {
      return "ui-class";
    }
    return null;
  }

  function ancestorIds(el) {
    const ids = [];
    let node = el.parentElement;
    while (node && ids.length < ANCESTOR_DEPTH && node !== document.body) {
      ids.push(nodeId(node));
      node = node.parentElement;
    }
    return ids;
  }

  /** Where the element sits relative to this frame's viewport and DOM. */
  function layoutInfo(el, rect) {
    const vw = window.innerWidth || 1;
    const vh = window.innerHeight || 1;
    const visW = Math.max(0, Math.min(rect.right, vw) - Math.max(rect.left, 0));
    const visH = Math.max(0, Math.min(rect.bottom, vh) - Math.max(rect.top, 0));
    return {
      displayWidth: Math.round(rect.width),
      displayHeight: Math.round(rect.height),
      left: Math.round(rect.left + window.scrollX),
      pageY: Math.round(rect.top + window.scrollY),
      visibleFrac: (visW * visH) / (vw * vh),
      containsCenter:
        rect.left <= vw / 2 && rect.right >= vw / 2 && rect.top <= vh / 2 && rect.bottom >= vh / 2,
      ancestors: ancestorIds(el),
      demoted: demoteReason(el),
    };
  }

  function isBig(w, h) {
    return w >= MIN_SIDE && h >= MIN_SIDE && w * h >= MIN_AREA;
  }

  /** Real image URL, preferring lazy-load attributes while src is still a placeholder. */
  function imgUrl(img) {
    const lazy = LAZY_ATTRS.map((a) => img.getAttribute(a)).find(Boolean);
    if (lazy && img.naturalWidth < MIN_SIDE) return absUrl(lazy);
    return img.currentSrc || img.src || (lazy ? absUrl(lazy) : "");
  }

  function collectFromImgs(list) {
    [...document.images].forEach((img, i) => {
      const rect = img.getBoundingClientRect();
      const w = img.naturalWidth || 0;
      const h = img.naturalHeight || 0;
      // Lazy placeholders count if they are big on screen even before decoding.
      if (!isBig(w, h) && !isBig(rect.width, rect.height)) return;

      const url = imgUrl(img);
      if (!url || url.startsWith("data:")) return;

      list.push({
        kind: "img",
        ref: `img:${i}`,
        url,
        naturalWidth: w,
        naturalHeight: h,
        ...layoutInfo(img, rect),
      });
    });
  }

  function collectBackgrounds(list) {
    document.querySelectorAll(BG_SELECTOR).forEach((el, i) => {
      const rect = el.getBoundingClientRect();
      if (!isBig(rect.width, rect.height)) return;

      const bg = cssUrl(getComputedStyle(el).backgroundImage);
      if (!bg || bg.startsWith("data:")) return;

      list.push({
        kind: "background",
        ref: `bg:${i}`,
        url: bg,
        naturalWidth: 0,
        naturalHeight: 0,
        ...layoutInfo(el, rect),
      });
    });
  }

  const BLANK = "blank";

  /** Tiny content hash so a redrawn canvas (page flip) counts as a new panel; null if tainted, BLANK
   * if nothing is drawn yet (lazy readers fill strip canvases only as they near the viewport). */
  function canvasFingerprint(canvas) {
    try {
      const probe = document.createElement("canvas");
      probe.width = 8;
      probe.height = 8;
      const ctx = probe.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(canvas, 0, 0, 8, 8);
      const data = ctx.getImageData(0, 0, 8, 8).data;
      let hash = 0;
      let alpha = 0;
      for (let i = 0; i < data.length; i += 4) {
        hash = (hash * 31 + (data[i] >> 3) * 1024 + (data[i + 1] >> 3) * 32 + (data[i + 2] >> 3)) | 0;
        alpha = Math.max(alpha, data[i + 3]);
      }
      return alpha ? (hash >>> 0).toString(36) : BLANK;
    } catch {
      return null;
    }
  }

  /** Share of the element itself that is inside the viewport. */
  function shownFrac(rect) {
    const w = Math.max(0, Math.min(rect.right, window.innerWidth) - Math.max(rect.left, 0));
    const h = Math.max(0, Math.min(rect.bottom, window.innerHeight) - Math.max(rect.top, 0));
    return rect.width && rect.height ? (w * h) / (rect.width * rect.height) : 0;
  }

  function collectCanvases(list) {
    document.querySelectorAll("canvas").forEach((canvas, i) => {
      const rect = canvas.getBoundingClientRect();
      const w = canvas.width || 0;
      const h = canvas.height || 0;
      if (!isBig(w, h) && !isBig(rect.width, rect.height)) return;

      const fp = canvasFingerprint(canvas);
      if (fp === BLANK) return;
      const shown = shownFrac(rect);
      list.push({
        kind: "canvas",
        ref: `canvas:${i}`,
        url: null,
        fp,
        tainted: fp === null,
        shownFrac: shown,
        covered: fp === null && shown >= CAPTURE_SHOWN && isCovered(canvas),
        naturalWidth: w,
        naturalHeight: h,
        ...layoutInfo(canvas, rect),
      });
    });
  }

  function collectCandidates(kinds = ["img", "background", "canvas"]) {
    const candidates = [];
    if (kinds.includes("img")) collectFromImgs(candidates);
    if (kinds.includes("background")) collectBackgrounds(candidates);
    if (kinds.includes("canvas")) collectCanvases(candidates);

    const byKey = new Map();
    for (const c of candidates) {
      const key = panelKey(c);
      if (!byKey.has(key)) byKey.set(key, []);
      byKey.get(key).push(c);
    }
    const unique = [...byKey.values()].map((group) => {
      if (group.length < 2 || group[0].kind !== "img") return group[0];
      const els = group.map((c) => document.images[Number(c.ref.split(":")[1])]);
      return group[els.indexOf(bestCopy(els))] || group[0];
    });

    return { candidates: unique.slice(0, MAX_CANDIDATES) };
  }

  function fixedLayer(el) {
    for (let n = el; n; n = n.parentElement) if (getComputedStyle(n).position === "fixed") return n;
    return null;
  }

  /** Of several <img>s showing one URL (a site viewer opened over its page, e.g. pixiv), the copy the
   * user actually sees: on screen and on top; else off screen but in the layer that is in front (same
   * fixed ancestor as whatever is at the viewport center); else anything rendered. Ties keep `current`. */
  function bestCopy(copies, current = null) {
    if (copies.length < 2) return copies[0] || null;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const center = document.elementFromPoint(vw / 2, vh / 2);
    const front = center ? fixedLayer(center) : null;
    const score = (img) => {
      const r = img.getBoundingClientRect();
      if (!r.width || !r.height) return 0;
      const left = Math.max(r.left, 0);
      const right = Math.min(r.right, vw);
      const top = Math.max(r.top, 0);
      const bottom = Math.min(r.bottom, vh);
      if (right <= left || bottom <= top) return fixedLayer(img) === front ? 2 : 1;
      const hit = document.elementFromPoint((left + right) / 2, (top + bottom) / 2);
      return hit && (hit === img || hit.contains(img)) ? 3 : 1;
    };
    let best = current && copies.includes(current) ? current : copies[0];
    let bestScore = score(best);
    for (const img of copies) {
      if (img === best) continue;
      const s = score(img);
      if (s > bestScore) [best, bestScore] = [img, s];
    }
    return best;
  }

  const imagesWithUrl = (url) => (url ? [...document.images].filter((i) => imgUrl(i) === url) : []);

  function elementForRef(ref, url) {
    const [kind, index] = ref.split(":");
    if (kind === "img") {
      const byIndex = document.images[Number(index)];
      const copies = imagesWithUrl(url);
      return bestCopy(copies, copies.includes(byIndex) ? byIndex : null) || byIndex || null;
    }
    if (kind === "canvas") return document.querySelectorAll("canvas")[Number(index)] || null;
    if (kind === "bg") return document.querySelectorAll(BG_SELECTOR)[Number(index)] || null;
    return null;
  }

  function drawToDataUrl(source, srcW, srcH, width, type = "image/jpeg", quality = 0.8) {
    if (!srcW || !srcH) return null;
    const scale = Math.min(1, width / srcW);
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(srcW * scale));
    canvas.height = Math.max(1, Math.round(srcH * scale));
    try {
      canvas.getContext("2d").drawImage(source, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL(type, quality);
    } catch {
      return null;
    }
  }

  /** Small previews for decoded, untainted images/canvases in this frame. */
  function thumbnails(items, width = THUMB_WIDTH) {
    const out = {};
    for (const { ref, url } of items || []) {
      const el = elementForRef(ref, url);
      if (el instanceof HTMLImageElement) {
        out[ref] = drawToDataUrl(el, el.naturalWidth, el.naturalHeight, width);
      } else if (el instanceof HTMLCanvasElement) {
        out[ref] = drawToDataUrl(el, el.width, el.height, width);
      } else {
        out[ref] = null;
      }
    }
    return out;
  }

  function blobToDataUrl(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error("FileReader failed"));
      reader.onload = () => resolve(reader.result);
      reader.readAsDataURL(blob);
    });
  }

  async function fetchInPage(url) {
    if (!url) throw new Error("No url");
    if (url.startsWith("data:")) return url;
    // blob: and same-origin/CORS work here; others fall back to the background fetch.
    const res = await fetch(url, { credentials: "include", mode: "cors" });
    if (!res.ok) throw new Error(`fetch ${res.status}`);
    return blobToDataUrl(await res.blob());
  }

  /** Full-size lossless copy of what the browser decoded: JPEG re-encoding visibly hurts OCR, and
   * toBlob encodes off the page's main thread (toDataURL would block scrolling for ~50 ms a page). */
  async function drawToPngDataUrl(source, width, height) {
    if (!width || !height) return null;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    try {
      canvas.getContext("2d").drawImage(source, 0, 0);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
      return blob ? await blobToDataUrl(blob) : null;
    } catch {
      return null; // tainted (cross-origin without CORS)
    }
  }

  async function exportOne(panel) {
    const el = elementForRef(panel.ref, panel.url);

    if (el instanceof HTMLCanvasElement) {
      return { dataUrl: await drawToPngDataUrl(el, el.width, el.height), method: "canvas" };
    }
    if (el instanceof HTMLImageElement && el.naturalWidth >= MIN_SIDE && imgUrl(el) === panel.url) {
      const dataUrl = await drawToPngDataUrl(el, el.naturalWidth, el.naturalHeight);
      if (dataUrl) return { dataUrl, method: "canvas-draw" };
    }
    if (panel.url) {
      try {
        return { dataUrl: await fetchInPage(panel.url), method: "page-fetch" };
      } catch {
        /* background will try */
      }
    }
    return { dataUrl: null, method: null };
  }

  async function exportPanels(panels) {
    const out = [];
    for (const panel of panels || []) {
      try {
        const { dataUrl, method } = await exportOne(panel);
        out.push({ ...panel, dataUrl, exportOk: Boolean(dataUrl), exportMethod: method, pageUrl: location.href });
      } catch (err) {
        out.push({ ...panel, dataUrl: null, exportOk: false, exportError: String(err?.message || err), pageUrl: location.href });
      }
    }
    return out;
  }

  // ---------- overlay ----------

  const PAD_RATIO = 0.35;
  const VERTICAL_MIN_ASPECT = 0.6;
  const MIN_FONT = 7;
  const MAX_FONT = 44;
  const READABLE_FONT = 11;
  const WIDEN_STEP = 1.15;
  const MAX_WIDEN = 1.9;
  const MERGE_GAP_RATIO = 1.3; // fragments closer than this × font size are one bubble (≈ column spacing)
  const MERGE_MIN_ALIGN = 0.5;
  const BOX_GAP = 2; // natural px kept between neighbouring boxes
  const MIN_BOX = 8;

  const OVERLAY_CSS = `
    .layer { position: absolute; box-sizing: border-box; }
    .badge {
      position: absolute; top: 6px; left: 6px; padding: 3px 8px; border-radius: 10px;
      background: rgba(20, 20, 20, 0.85); color: #fff; font: 600 12px system-ui, sans-serif;
      white-space: nowrap;
    }
    .badge[data-state="error"] { background: rgba(170, 30, 30, 0.92); }
    .badge[data-state="ocr"] { background: rgba(30, 80, 150, 0.92); }
    .badge[data-state="done"], .badge[data-state="empty"] { animation: fade 0.4s 2.5s forwards; }
    @keyframes fade { to { opacity: 0; } }
    .bubble {
      position: absolute; box-sizing: border-box; display: grid;
      grid-template-columns: minmax(0, 1fr); place-items: center;
      padding: 2px; overflow: hidden; background: #fff; color: #111; border-radius: 12px;
      font-family: "Comic Neue", "Comic Sans MS", "Segoe UI", sans-serif;
      font-weight: 700; line-height: 1.1; text-align: center;
    }
    .bubble .text { width: 100%; overflow-wrap: normal; word-break: normal; }
    .shape { position: absolute; z-index: 0; }
    .bubble { z-index: 2; }
    .badge { z-index: 3; }
    .boxes { z-index: 4; }
    .bubble.in-shape { padding: 0; border-radius: 10px; }
    .boxes { position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible; display: none; }
    .show-boxes .boxes { display: block; }
    .show-boxes .bubble, .show-boxes .shape { opacity: 0.25; }
    .boxes .bubble-poly { stroke: #22c55e; stroke-width: 2; }
    .boxes .bubble-inner { stroke: #3b82f6; stroke-width: 1.5; stroke-dasharray: 5 3; }
    .boxes polygon, .boxes rect { fill: none; vector-effect: non-scaling-stroke; }
    .boxes .region { stroke: #ff2bd6; stroke-width: 2; }
    .boxes .line { stroke: #00d8ff; stroke-width: 1.5; }
    .boxes .dropped { stroke: #ff3b30; stroke-width: 1.5; stroke-dasharray: 4 3; }
    .boxes text { fill: #ff2bd6; font-weight: 700; font-family: system-ui, sans-serif; paint-order: stroke; stroke: #fff; }
  `;

  const BADGE_TEXT = {
    queued: "queued",
    exporting: "grabbing…",
    reading: "reading…",
    translating: "translating…",
    empty: "no text",
  };

  /** key → { el, url, layer, badge, lastRect, boundSrc, lazyAtBind } */
  const overlays = new Map();
  let host = null;
  let root = null;
  let overlayVisible = true;
  let boxesVisible = false;
  let loopRunning = false;

  function ensureRoot() {
    if (!host) {
      host = document.createElement("div");
      host.style.cssText =
        "position:fixed;inset:0;pointer-events:none;z-index:2147483647;overflow:hidden;";
      root = host.attachShadow({ mode: "open" });
      const style = document.createElement("style");
      style.textContent = OVERLAY_CSS;
      root.append(style);
    }
    if (!host.isConnected) document.documentElement.append(host);
    host.style.display = overlayVisible ? "" : "none";
    return root;
  }

  function pct(value, total) {
    return `${(value / total) * 100}%`;
  }

  const boxRight = (b) => b.x + b.w;
  const boxBottom = (b) => b.y + b.h;
  const boxCenterX = (b) => b.x + b.w / 2;
  const boxCenterY = (b) => b.y + b.h / 2;
  const intersects = (a, b) =>
    a.x < boxRight(b) && b.x < boxRight(a) && a.y < boxBottom(b) && b.y < boxBottom(a);

  function boxUnion(a, b) {
    const x = Math.min(a.x, b.x);
    const y = Math.min(a.y, b.y);
    return { x, y, w: Math.max(boxRight(a), boxRight(b)) - x, h: Math.max(boxBottom(a), boxBottom(b)) - y };
  }

  /** Signed gaps between two boxes on each axis (negative = they overlap on that axis). */
  function boxGaps(a, b) {
    return {
      x: Math.max(b.x - boxRight(a), a.x - boxRight(b)),
      y: Math.max(b.y - boxBottom(a), a.y - boxBottom(b)),
    };
  }

  function spanOverlap(a0, a1, b0, b1) {
    return Math.max(0, Math.min(a1, b1) - Math.max(a0, b0)) / Math.max(1, Math.min(a1 - a0, b1 - b0));
  }

  /**
   * MIT sometimes leaves one speech bubble as several text regions (adjacent vertical columns
   * or stacked lines). Merge those so they share one overlay box instead of covering each other.
   */
  function isFragmentPair(a, b) {
    if (a.vertical !== b.vertical) return false;
    const g = boxGaps(a.core, b.core);
    const maxGap = Math.min(a.font, b.font) * MERGE_GAP_RATIO;
    if (a.vertical) {
      const aligned = spanOverlap(a.core.y, boxBottom(a.core), b.core.y, boxBottom(b.core));
      return g.x <= maxGap && g.y < 0 && aligned >= MERGE_MIN_ALIGN;
    }
    const aligned = spanOverlap(a.core.x, boxRight(a.core), b.core.x, boxRight(b.core));
    return g.y <= maxGap && g.x < 0 && aligned >= MERGE_MIN_ALIGN;
  }

  function mergeFragments(items) {
    for (let merged = true; merged; ) {
      merged = false;
      for (let i = 0; i < items.length && !merged; i++) {
        for (let j = i + 1; j < items.length && !merged; j++) {
          const [a, b] = [items[i], items[j]];
          if (!isFragmentPair(a, b)) continue;
          const aFirst = a.vertical
            ? boxCenterX(a.core) >= boxCenterX(b.core) // Japanese columns read right to left
            : boxCenterY(a.core) <= boxCenterY(b.core);
          const [first, second] = aFirst ? [a, b] : [b, a];
          items[i] = {
            text: `${first.text} ${second.text}`,
            core: boxUnion(a.core, b.core),
            font: Math.max(a.font, b.font),
            vertical: a.vertical,
          };
          items.splice(j, 1);
          merged = true;
        }
      }
    }
    return items;
  }

  /** Bubble rectangle in natural image pixels: padded text bbox, widened for vertical source text. */
  function bubbleBox(item, page) {
    const { core } = item;
    const pad = Math.max(3, item.font * PAD_RATIO);
    let left = core.x - pad;
    let right = boxRight(core) + pad;
    // bubbles stitched across a strip boundary may start/end in the neighbouring strip
    const top = core.y < 0 ? core.y - pad : Math.max(0, core.y - pad);
    const bottom = boxBottom(core) > page.height ? boxBottom(core) + pad : Math.min(page.height, boxBottom(core) + pad);

    const minWidth = item.vertical ? (bottom - top) * VERTICAL_MIN_ASPECT : 0;
    if (right - left < minWidth) {
      const grow = (minWidth - (right - left)) / 2;
      left -= grow;
      right += grow;
    }
    left = Math.max(0, left);
    right = Math.min(page.width, right);
    return { x: left, y: top, w: right - left, h: bottom - top };
  }

  function trimRight(box, edge) {
    if (boxRight(box) > edge) box.w = Math.max(MIN_BOX, edge - box.x);
  }

  function trimLeft(box, edge) {
    if (box.x >= edge) return;
    const right = boxRight(box);
    box.x = Math.min(edge, right - MIN_BOX);
    box.w = right - box.x;
  }

  function trimBottom(box, edge) {
    if (boxBottom(box) > edge) box.h = Math.max(MIN_BOX, edge - box.y);
  }

  function trimTop(box, edge) {
    if (box.y >= edge) return;
    const bottom = boxBottom(box);
    box.y = Math.min(edge, bottom - MIN_BOX);
    box.h = bottom - box.y;
  }

  /** Two colliding boxes give up space on the axis where their text is furthest apart, split at the midpoint. */
  function separate(a, b) {
    if (!intersects(a.box, b.box)) return;
    const g = boxGaps(a.core, b.core);
    if (g.x >= g.y) {
      const [l, r] = boxCenterX(a.core) <= boxCenterX(b.core) ? [a, b] : [b, a];
      const cut = (boxRight(l.core) + r.core.x) / 2;
      trimRight(l.box, cut - BOX_GAP / 2);
      trimLeft(r.box, cut + BOX_GAP / 2);
    } else {
      const [t, btm] = boxCenterY(a.core) <= boxCenterY(b.core) ? [a, b] : [b, a];
      const cut = (boxBottom(t.core) + btm.core.y) / 2;
      trimBottom(t.box, cut - BOX_GAP / 2);
      trimTop(btm.box, cut + BOX_GAP / 2);
    }
  }

  /** How far each box may widen sideways without reaching the halfway point to a neighbour on its rows. */
  function widenLimits(item, items, page) {
    const a = item.box;
    let lo = 0;
    let hi = page.width;
    for (const other of items) {
      const b = other.box;
      if (other === item || !(a.y < boxBottom(b) && b.y < boxBottom(a))) continue;
      if (boxCenterX(b) < boxCenterX(a)) lo = Math.max(lo, Math.min(a.x, (boxRight(b) + a.x) / 2 + BOX_GAP / 2));
      else hi = Math.min(hi, Math.max(boxRight(a), (boxRight(a) + b.x) / 2 - BOX_GAP / 2));
    }
    return { lo, hi };
  }

  const SHAPE_INSET_X = 0.08; // text keeps this share of the inner rect clear on each side
  const SHAPE_INSET_Y = 0.05;

  /** Regions the app found a bubble for: one text box per bubble, in the bubble's inner rectangle. */
  function layoutShapes(regions) {
    const groups = new Map();
    for (const r of regions) {
      if (!r.bubble || !r.translation) continue;
      const group = groups.get(r.bubble.id) || { bubble: r.bubble, parts: [] };
      group.parts.push(r);
      groups.set(r.bubble.id, group);
    }
    return [...groups.values()].map(({ bubble, parts }) => {
      const vertical = parts.some((r) => r.source_vertical);
      const cx = (r) => (r.bbox[0] + r.bbox[2]) / 2;
      const cy = (r) => (r.bbox[1] + r.bbox[3]) / 2;
      parts.sort((a, b) => (vertical ? cx(b) - cx(a) : cy(a) - cy(b))); // Japanese columns read right to left
      const [x1, y1, x2, y2] = bubble.inner;
      const w = x2 - x1;
      const h = y2 - y1;
      const xs = bubble.poly.map((p) => p[0]);
      return {
        bubble,
        text: parts.map((r) => r.translation).join(" "),
        box: { x: x1 + w * SHAPE_INSET_X, y: y1 + h * SHAPE_INSET_Y, w: w * (1 - 2 * SHAPE_INSET_X), h: h * (1 - 2 * SHAPE_INSET_Y) },
        limits: { lo: Math.min(...xs), hi: Math.max(...xs) },
      };
    });
  }

  function isDark([r, g, b]) {
    return 0.299 * r + 0.587 * g + 0.114 * b < 110;
  }

  /** The bubble's outline filled with its own background colour, covering the original text. */
  function shapeElement(bubble, page) {
    const xs = bubble.poly.map((p) => p[0]);
    const ys = bubble.poly.map((p) => p[1]);
    const box = { x: Math.min(...xs), y: Math.min(...ys) };
    box.w = Math.max(1, Math.max(...xs) - box.x);
    box.h = Math.max(1, Math.max(...ys) - box.y);
    const el = document.createElement("div");
    el.className = "shape";
    Object.assign(el.style, {
      left: pct(box.x, page.width),
      top: pct(box.y, page.height),
      width: pct(box.w, page.width),
      height: pct(box.h, page.height),
      background: `rgb(${bubble.fill.join(",")})`,
      clipPath: `polygon(${bubble.poly.map(([x, y]) => `${pct(x - box.x, box.w)} ${pct(y - box.y, box.h)}`).join(", ")})`,
    });
    return el;
  }

  function layoutBubbles(regions, page) {
    const items = mergeFragments(
      regions
        .filter((r) => r.translation && !r.bubble)
        .map((r) => {
          const [x1, y1, x2, y2] = r.bbox;
          return {
            text: r.translation,
            core: { x: x1, y: y1, w: x2 - x1, h: y2 - y1 },
            font: r.font_size > 0 ? r.font_size : 12,
            vertical: Boolean(r.source_vertical),
          };
        })
    );
    for (const item of items) item.box = bubbleBox(item, page);
    for (let i = 0; i < items.length; i++) {
      for (let j = i + 1; j < items.length; j++) separate(items[i], items[j]);
    }
    for (const item of items) item.limits = widenLimits(item, items, page);
    return items;
  }

  function placeBubble(bubble, box, page) {
    bubble.__overlay.current = box;
    bubble.style.left = pct(box.x, page.width);
    bubble.style.top = pct(box.y, page.height);
    bubble.style.width = pct(box.w, page.width);
    bubble.style.height = pct(box.h, page.height);
  }

  function widenBox(box, factor, limits) {
    const w = Math.min(limits.hi - limits.lo, Math.max(box.w, box.w * factor));
    const x = Math.min(Math.max(limits.lo, boxCenterX(box) - w / 2), limits.hi - w);
    return { ...box, x, w };
  }

  function largestFittingFont(bubble) {
    const text = bubble.firstElementChild;
    const style = getComputedStyle(bubble);
    const availHeight =
      bubble.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
    const fits = () =>
      text.scrollWidth <= text.clientWidth + 1 && text.offsetHeight <= availHeight + 1;

    let lo = MIN_FONT;
    let hi = Math.max(MIN_FONT, Math.min(MAX_FONT, Math.floor(availHeight)));
    while (lo < hi) {
      const mid = Math.ceil((lo + hi) / 2);
      text.style.fontSize = `${mid}px`;
      if (fits()) lo = mid;
      else hi = mid - 1;
    }
    text.style.fontSize = `${lo}px`;
    return lo;
  }

  /** Shrink font to fit; if it gets unreadably small, widen the bubble a little and retry. */
  function fitText(bubble) {
    const { box, limits, page } = bubble.__overlay;
    let factor = 1;
    placeBubble(bubble, box, page);
    let font = largestFittingFont(bubble);
    while (font < READABLE_FONT && factor * WIDEN_STEP <= MAX_WIDEN && box.w * factor < limits.hi - limits.lo) {
      factor *= WIDEN_STEP;
      placeBubble(bubble, widenBox(box, factor, limits), page);
      font = largestFittingFont(bubble);
    }
  }

  const fontOf = (bubble) => parseFloat(bubble.firstElementChild.style.fontSize) || 0;

  /** Boxes that still overlap after fitting: trim the one with the larger font away from the other. */
  function resolveOverlaps(bubbles) {
    for (let pass = 0; pass < 6; pass++) {
      let changed = false;
      for (let i = 0; i < bubbles.length; i++) {
        for (let j = i + 1; j < bubbles.length; j++) {
          const [a, b] = [bubbles[i], bubbles[j]];
          if (!intersects(a.__overlay.current, b.__overlay.current)) continue;
          const [big, other] = fontOf(a) >= fontOf(b) ? [a, b] : [b, a];
          const box = { ...big.__overlay.current };
          const o = other.__overlay.current;
          const overlapX = Math.min(boxRight(box), boxRight(o)) - Math.max(box.x, o.x);
          const overlapY = Math.min(boxBottom(box), boxBottom(o)) - Math.max(box.y, o.y);
          if (overlapX <= overlapY) {
            if (boxCenterX(box) <= boxCenterX(o)) trimRight(box, o.x - BOX_GAP);
            else trimLeft(box, boxRight(o) + BOX_GAP);
          } else if (boxCenterY(box) <= boxCenterY(o)) {
            trimBottom(box, o.y - BOX_GAP);
          } else {
            trimTop(box, boxBottom(o) + BOX_GAP);
          }
          const was = big.__overlay.current;
          if (box.x === was.x && box.y === was.y && box.w === was.w && box.h === was.h) continue;
          placeBubble(big, box, big.__overlay.page);
          largestFittingFont(big);
          changed = true;
        }
      }
      if (!changed) break;
    }
  }

  function buildBubbles(entry, result) {
    entry.layer.querySelectorAll(".bubble, .shape").forEach((b) => b.remove());
    const page = { width: result.width, height: result.height };
    for (const item of layoutShapes(result.regions)) {
      entry.layer.insertBefore(shapeElement(item.bubble, page), entry.badge);
      const bubble = document.createElement("div");
      bubble.className = "bubble in-shape";
      bubble.style.background = `rgb(${item.bubble.fill.join(",")})`;
      bubble.style.color = isDark(item.bubble.fill) ? "#fff" : "#111";
      bubble.__overlay = { box: item.box, limits: item.limits, page };
      const text = document.createElement("div");
      text.className = "text";
      text.textContent = item.text;
      bubble.append(text);
      entry.layer.insertBefore(bubble, entry.badge);
    }
    for (const item of layoutBubbles(result.regions, page)) {
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.__overlay = { box: item.box, limits: item.limits, page };
      const text = document.createElement("div");
      text.className = "text";
      text.textContent = item.text;
      bubble.append(text);
      entry.layer.insertBefore(bubble, entry.badge);
    }
    buildBoxes(entry, result);
    entry.lastRect = "";
  }

  const SVG_NS = "http://www.w3.org/2000/svg";

  function svgEl(tag, attrs) {
    const el = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
  }

  /** Debug view: merged bubbles, the textlines OCR read, and detected lines that were dropped. */
  function buildBoxes(entry, result) {
    entry.layer.querySelector(".boxes")?.remove();
    const svg = svgEl("svg", {
      class: "boxes",
      viewBox: `0 0 ${result.width} ${result.height}`,
      preserveAspectRatio: "none",
    });
    const quadKey = (q) => q.map((p) => p.join(",")).join(" ");
    const labelSize = Math.max(10, Math.round(result.width / 45));
    const kept = new Set();
    result.regions.forEach((r, i) => {
      for (const q of r.lines || []) {
        kept.add(quadKey(q));
        svg.append(svgEl("polygon", { class: "line", points: quadKey(q) }));
      }
      const [x1, y1, x2, y2] = r.bbox;
      svg.append(svgEl("rect", { class: "region", x: x1, y: y1, width: x2 - x1, height: y2 - y1 }));
      const label = svgEl("text", {
        x: x1 + 2,
        y: Math.max(labelSize, y1 - labelSize / 4),
        "font-size": labelSize,
        "stroke-width": labelSize / 5,
      });
      label.textContent = String(i + 1);
      svg.append(label);
    });
    for (const q of result.det_lines || []) {
      if (!kept.has(quadKey(q))) svg.append(svgEl("polygon", { class: "dropped", points: quadKey(q) }));
    }
    const drawn = new Set();
    for (const { bubble } of result.regions) {
      if (!bubble || drawn.has(bubble.id)) continue;
      drawn.add(bubble.id);
      svg.append(svgEl("polygon", { class: "bubble-poly", points: quadKey(bubble.poly) }));
      const [x1, y1, x2, y2] = bubble.inner;
      svg.append(svgEl("rect", { class: "bubble-inner", x: x1, y: y1, width: x2 - x1, height: y2 - y1 }));
    }
    entry.layer.insertBefore(svg, entry.badge);
  }

  /** An <img> whose src changed since binding shows a different page (flip viewers): hide its overlay. */
  function showsBoundImage(entry) {
    if (!(entry.el instanceof HTMLImageElement)) return true;
    const current = entry.el.currentSrc || entry.el.src;
    if (current === entry.boundSrc) return true;
    if (entry.lazyAtBind && entry.el.naturalWidth >= MIN_SIDE) {
      entry.boundSrc = current;
      entry.lazyAtBind = false;
      return true;
    }
    return false;
  }

  const REBIND_MS = 500;
  let lastRebind = 0;

  /** Move each <img> overlay to the copy of its image that is currently seen (viewer opened/closed). */
  function rebindCopies() {
    const byUrl = new Map();
    for (const img of document.images) {
      const url = imgUrl(img);
      if (!byUrl.has(url)) byUrl.set(url, []);
      byUrl.get(url).push(img);
    }
    for (const entry of overlays.values()) {
      if (!entry.url || !(entry.el instanceof HTMLImageElement)) continue;
      const copies = byUrl.get(entry.url) || [];
      if (copies.length < 2 && entry.el.isConnected) continue;
      const best = bestCopy(copies, entry.el.isConnected ? entry.el : null);
      if (best && best !== entry.el) {
        entry.el = best;
        entry.boundSrc = best.currentSrc || best.src;
        entry.lazyAtBind = best.naturalWidth < MIN_SIDE;
        entry.lastRect = "";
      }
    }
  }

  const SEEN_MS = 150;
  let lastSeen = 0;

  /** A real page image/canvas (not a transparent anti-save sheet or a click-catching div). */
  function isPageLike(el) {
    if (el instanceof HTMLImageElement) return el.naturalWidth >= MIN_SIDE && el.checkVisibility({ opacityProperty: true });
    if (el instanceof HTMLCanvasElement) return el.width >= MIN_SIDE && el.checkVisibility({ opacityProperty: true });
    return false;
  }

  /** Whether the user can see this element: book-style readers keep every page in place and flip by
   * hiding (opacity / visibility / display) or restacking them, so position alone isn't enough. */
  function isSeen(el) {
    if (!el.checkVisibility({ opacityProperty: true, visibilityProperty: true })) return false;
    const r = el.getBoundingClientRect();
    const left = Math.max(r.left, 0);
    const right = Math.min(r.right, window.innerWidth);
    const top = Math.max(r.top, 0);
    const bottom = Math.min(r.bottom, window.innerHeight);
    if (right <= left || bottom <= top) return true; // off screen: nothing to hide
    const hit = document.elementFromPoint((left + right) / 2, (top + bottom) / 2);
    if (!hit || hit === el || el.contains(hit) || hit.contains(el)) return true;
    return !isPageLike(hit); // another page on top of this one
  }

  /** Readable canvases redrawn with another page (fingerprint changed) hide their old text right away. */
  const isStale = (entry) => {
    if (entry.fp == null) return false;
    const r = entry.el.getBoundingClientRect();
    if (r.bottom <= 0 || r.top >= window.innerHeight || r.right <= 0 || r.left >= window.innerWidth) return entry.stale;
    return canvasFingerprint(entry.el) !== entry.fp;
  };

  function tick(now) {
    if (now - lastRebind >= REBIND_MS) {
      lastRebind = now;
      rebindCopies();
      for (const entry of overlays.values()) entry.stale = entry.el.isConnected && isStale(entry);
    }
    if (now - lastSeen >= SEEN_MS) {
      lastSeen = now;
      for (const entry of overlays.values()) entry.unseen = entry.el.isConnected && !isSeen(entry.el);
    }
    for (const entry of overlays.values()) {
      if (!entry.el.isConnected || entry.unseen || entry.stale || !showsBoundImage(entry)) {
        entry.layer.style.display = "none";
        entry.lastRect = "";
        continue;
      }
      const r = entry.el.getBoundingClientRect();
      const size = `${Math.round(r.width)}x${Math.round(r.height)}`;
      const sig = `${Math.round(r.left)},${Math.round(r.top)},${size}`;
      if (sig === entry.lastRect) continue;

      const sizeChanged = !entry.lastRect.endsWith(size);
      entry.lastRect = sig;
      const visible = r.width > 0 && r.height > 0;
      Object.assign(entry.layer.style, {
        display: visible ? "" : "none",
        left: `${r.left}px`,
        top: `${r.top}px`,
        width: `${r.width}px`,
        height: `${r.height}px`,
      });
      if (visible && sizeChanged) {
        const bubbles = [...entry.layer.querySelectorAll(".bubble")];
        bubbles.forEach(fitText);
        resolveOverlaps(bubbles);
      }
    }
    if (overlays.size) requestAnimationFrame(tick);
    else loopRunning = false;
  }

  function startLoop() {
    if (loopRunning) return;
    loopRunning = true;
    requestAnimationFrame(tick);
  }

  /** Called by the background worker as a panel moves through export → OCR → translate → done. */
  function setPanelState({ ref, url, kind, fp, state, message, result }) {
    const key = panelKey({ ref, url, kind, fp });
    let entry = overlays.get(key);
    if (!entry) {
      const el = elementForRef(ref, url);
      if (!el) return false;
      const layer = document.createElement("div");
      layer.className = "layer";
      layer.dataset.key = key;
      layer.classList.toggle("show-boxes", boxesVisible);
      const badge = document.createElement("div");
      badge.className = "badge";
      layer.append(badge);
      const isImg = el instanceof HTMLImageElement;
      entry = {
        el,
        url: isImg ? url : null,
        fp: kind === "canvas" ? fp : null, // a reused canvas keeps one overlay per page it has shown
        layer,
        badge,
        lastRect: "",
        boundSrc: isImg ? el.currentSrc || el.src : null,
        lazyAtBind: isImg && el.naturalWidth < MIN_SIDE,
      };
      overlays.set(key, entry);
    }
    ensureRoot().append(entry.layer);

    if (result) buildBubbles(entry, result);
    const translated = result ? result.regions.filter((r) => r.translation).length : 0;
    entry.badge.dataset.state = state;
    entry.badge.textContent =
      state === "done"
        ? `✓ ${translated} bubbles`
        : state === "error"
          ? `⚠ ${message || "error"}`
          : state === "waiting"
            ? `OCR ✓ · waiting for batch${message ? ` (${message})` : ""}`
            : state === "ocr"
              ? `⏱ ${message}`
              : BADGE_TEXT[state] || state;

    startLoop();
    return true;
  }

  const CAPTURE_SHOWN = 0.98;
  const CAPTURE_HIDE_MAX_MS = 300;
  const COVER_STEP = [48, 24]; // px between hit tests; bars thinner than 24 px may slip through

  /** el, or a parent of it that isn't also a parent of target, is fixed/sticky (a fixed reader
   * viewer holding both the page and its tap zones doesn't count). */
  function isPinned(el, target) {
    for (let e = el; e && e !== document.body && !e.contains(target); e = e.parentElement) {
      const { position } = getComputedStyle(e);
      if (position === "fixed" || position === "sticky") return true;
    }
    return false;
  }

  /** A fixed/sticky bar (cookie banner, reader toolbar) over part of the element would end up in its
   * screenshot. Viewport-sized layers are skipped: readers use them as transparent tap zones. */
  function isCovered(el) {
    const r = el.getBoundingClientRect();
    const left = Math.max(r.left, 0), right = Math.min(r.right, window.innerWidth);
    const top = Math.max(r.top, 0), bottom = Math.min(r.bottom, window.innerHeight);
    const cols = Math.min(12, Math.ceil((right - left) / COVER_STEP[0]));
    const rows = Math.ceil((bottom - top) / COVER_STEP[1]);
    const seen = new Set();
    for (let i = 0; i < cols; i++) {
      for (let j = 0; j < rows; j++) {
        const x = left + ((i + 0.5) / cols) * (right - left);
        const y = top + ((j + 0.5) / rows) * (bottom - top);
        for (const e of document.elementsFromPoint(x, y)) {
          if (e === el || e.contains(el)) break;
          if (e === host || seen.has(e)) continue;
          seen.add(e);
          const er = e.getBoundingClientRect();
          if (er.width * er.height >= 0.9 * window.innerWidth * window.innerHeight) continue;
          if (isPinned(e, el)) return true;
        }
      }
    }
    return false;
  }

  /** Where a panel sits for a tab screenshot (canvases the page locked against reading), with our
   * overlay hidden until endCapture so it isn't in the shot. */
  async function prepareCapture(ref, url) {
    const el = elementForRef(ref, url);
    if (!el) return null;
    if (shownFrac(el.getBoundingClientRect()) < CAPTURE_SHOWN || isCovered(el)) return { offscreen: true };
    if (host) host.style.visibility = "hidden";
    await Promise.race([
      new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))),
      new Promise((resolve) => setTimeout(resolve, CAPTURE_HIDE_MAX_MS)),
    ]);
    const r = el.getBoundingClientRect();
    return { rect: [r.left, r.top, r.width, r.height], viewport: [window.innerWidth, window.innerHeight] };
  }

  function endCapture() {
    if (host) host.style.visibility = "";
    return true;
  }

  /** The background gave up on a panel for now (flipped away before its screenshot): drop its overlay
   * and let auto mode report it again once it is back on screen. */
  function forgetPanel(item) {
    const key = panelKey(item);
    auto?.reported.delete(key);
    const entry = overlays.get(key);
    if (entry) {
      entry.layer.remove();
      overlays.delete(key);
    }
    return true;
  }

  function toggleOverlay() {
    overlayVisible = !overlayVisible;
    if (host) host.style.display = overlayVisible ? "" : "none";
    return overlayVisible;
  }

  function toggleBoxes() {
    boxesVisible = !boxesVisible;
    for (const entry of overlays.values()) entry.layer.classList.toggle("show-boxes", boxesVisible);
    return boxesVisible;
  }

  function clearOverlays() {
    for (const entry of overlays.values()) entry.layer.remove();
    overlays.clear();
  }

  // ---------- auto mode: report new panels matching the locked group ----------

  const WIDTH_SLACK = 0.06;
  const AUTO_DEBOUNCE_MS = 400;
  const AUTO_POLL_MS = 2000;
  const SCROLL_OPTS = { capture: true, passive: true };

  /** { signature, reported: Set, observer, timer, debounce } */
  let auto = null;

  function matchesSignature(c, sig) {
    return (
      c.kind === sig.kind &&
      c.displayWidth >= sig.minW * (1 - WIDTH_SLACK) - 4 &&
      c.displayWidth <= sig.maxW * (1 + WIDTH_SLACK) + 4
    );
  }

  function autoScan() {
    if (!auto) return;
    const { candidates } = collectCandidates([auto.signature.kind]);
    const fresh = candidates.filter((c) => {
      if (!matchesSignature(c, auto.signature)) return false;
      // locked canvases can only be screenshotted, so they wait until fully on screen
      if (c.kind === "canvas" && c.tainted && (c.shownFrac < CAPTURE_SHOWN || c.covered)) return false;
      const key = panelKey(c);
      if (auto.reported.has(key)) return false;
      auto.reported.add(key);
      return true;
    });
    if (fresh.length) {
      chrome.runtime
        .sendMessage({ type: "AUTO_PANELS", panels: fresh.map(({ ancestors, ...rest }) => rest) })
        .catch(() => {});
    }
  }

  function scheduleAutoScan() {
    if (!auto || auto.debounce) return;
    auto.debounce = setTimeout(() => {
      if (!auto) return;
      auto.debounce = null;
      autoScan();
    }, AUTO_DEBOUNCE_MS);
  }

  function startAuto(signature, knownKeys) {
    stopAuto();
    auto = { signature, reported: new Set(knownKeys || []), debounce: null };
    auto.observer = new MutationObserver(scheduleAutoScan);
    auto.observer.observe(document.documentElement, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["src", "srcset", "style", "class", ...LAZY_ATTRS],
    });
    document.addEventListener("load", scheduleAutoScan, true);
    // locked canvases qualify only while fully on screen, which a scroll may pass in under a second
    document.addEventListener("scroll", scheduleAutoScan, SCROLL_OPTS);
    document.addEventListener("scrollend", autoScan, SCROLL_OPTS);
    auto.timer = setInterval(autoScan, AUTO_POLL_MS);
    autoScan();
    return true;
  }

  function stopAuto() {
    if (!auto) return false;
    auto.observer.disconnect();
    clearInterval(auto.timer);
    clearTimeout(auto.debounce);
    document.removeEventListener("load", scheduleAutoScan, true);
    document.removeEventListener("scroll", scheduleAutoScan, SCROLL_OPTS);
    document.removeEventListener("scrollend", autoScan, SCROLL_OPTS);
    auto = null;
    return true;
  }

  // ---------- overlay shortcut (settings.overlayKey) ----------

  /** Same rule as comboFromEvent in settings.js. */
  function comboFromEvent(e) {
    if (["Control", "Alt", "Shift", "Meta", "AltGraph"].includes(e.key)) return null;
    const key = /^Key[A-Z]$/.test(e.code) ? e.code.slice(3) : /^Digit\d$/.test(e.code) ? e.code.slice(5) : e.key.length === 1 ? e.key.toUpperCase() : e.key;
    return [e.ctrlKey && "Ctrl", e.altKey && "Alt", e.shiftKey && "Shift", e.metaKey && "Meta", key].filter(Boolean).join("+");
  }

  let overlayKey = "Alt+O";
  chrome.storage.local.get("settings").then(({ settings }) => {
    if (settings && "overlayKey" in settings) overlayKey = settings.overlayKey;
  });
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === "local" && changes.settings) overlayKey = changes.settings.newValue?.overlayKey ?? "Alt+O";
  });

  window.addEventListener(
    "keydown",
    (e) => {
      if (!overlayKey || e.repeat || comboFromEvent(e) !== overlayKey) return;
      const t = e.target;
      if (t instanceof HTMLElement && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
      e.preventDefault();
      e.stopPropagation();
      chrome.runtime.sendMessage({ type: "TOGGLE_OVERLAY" }).catch(() => {});
    },
    true
  );

  window.__panelProbe = {
    version: 26,
    panelKey,
    collectCandidates,
    thumbnails,
    exportPanels,
    prepareCapture,
    endCapture,
    forgetPanel,
    setPanelState,
    toggleOverlay,
    toggleBoxes,
    clearOverlays,
    startAuto,
    stopAuto,
    autoActive: () => Boolean(auto),
  };
})();
