// Python / MIT text rules, driven by tables exported from the app's Python (tools/export_tables.py):
// str.strip / str.split whitespace, generic2.is_valuable_text, cp932 encodability, py3langid.

let T = null;

const inRanges = (ranges, cp) => {
  let lo = 0;
  let hi = ranges.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (cp < ranges[mid][0]) hi = mid - 1;
    else if (cp > ranges[mid][1]) lo = mid + 1;
    else return true;
  }
  return false;
};

export async function loadTextTables(base = "/models/") {
  if (T) return;
  const [tables, meta, bin] = await Promise.all([
    fetch(base + "unicode_tables.json").then((r) => r.json()),
    fetch(base + "langid.json").then((r) => r.json()),
    fetch(base + "langid.bin").then((r) => r.arrayBuffer()),
  ]);
  const part = (name, Type) => new Type(bin, meta.parts[name].offset, meta.parts[name].length);
  const outStates = part("out_states", Uint16Array);
  const outCounts = part("out_counts", Uint16Array);
  const outFeats = part("out_feats", Uint16Array);
  const outputs = new Map();
  for (let i = 0, f = 0; i < outStates.length; i++) {
    outputs.set(outStates[i], outFeats.subarray(f, f + outCounts[i]));
    f += outCounts[i];
  }
  T = {
    ...tables,
    langid: {
      classes: meta.classes,
      nextmove: part("nextmove", Uint16Array),
      outputs,
      ptc: part("ptc", Float32Array),
      pc: part("pc", Float32Array),
    },
  };
}

export const isSpace = (ch) => inRanges(T.space, ch.codePointAt(0));

/** str.strip() */
export function pyStrip(s) {
  const chars = [...s];
  let a = 0;
  let b = chars.length;
  while (a < b && isSpace(chars[a])) a++;
  while (b > a && isSpace(chars[b - 1])) b--;
  return chars.slice(a, b).join("");
}

/** "".join(s.split()) */
export const dropSpaces = (s) => [...s].filter((c) => !isSpace(c)).join("");

/** MIT generic2.is_valuable_text */
export const isValuableText = (s) => [...s].some((c) => !inRanges(T.not_valuable, c.codePointAt(0)));

/** manga_ocr_rec._jis */
export const isCp932 = (ch) => inRanges(T.cp932, ch.codePointAt(0));

/** py3langid classify(text)[0] */
export function langOf(text) {
  const { nextmove, outputs, ptc, pc, classes } = T.langid;
  const C = classes.length;
  const counts = new Map();
  let state = 0;
  for (const byte of new TextEncoder().encode(text)) {
    state = nextmove[(state << 8) + byte];
    const feats = outputs.get(state);
    if (feats) for (const f of feats) counts.set(f, (counts.get(f) || 0) + 1);
  }
  const score = Float64Array.from(pc);
  for (const [f, n] of counts) for (let c = 0; c < C; c++) score[c] += n * ptc[f * C + c];
  let best = 0;
  for (let c = 1; c < C; c++) if (score[c] > score[best]) best = c;
  return classes[best];
}
