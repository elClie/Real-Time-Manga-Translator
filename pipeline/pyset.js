// SPDX-License-Identifier: GPL-3.0-only
// Python container semantics the MIT merge code depends on. Its graph code iterates Python sets
// and networkx adjacency dicts, and ties in its sorts / majority votes are broken by that order.

/** CPython 3.11 set of non-negative ints (hash(i) == i, no deletions): same slots, same iteration order. */
export class PySet {
  constructor(items = []) {
    this.table = new Array(8).fill(null);
    this.mask = 7;
    this.fill = 0;
    for (const x of items) this.add(x);
  }

  has(key) {
    const { table, mask } = this;
    let i = key & mask;
    let perturb = key;
    for (;;) {
      const probes = i + 9 <= mask ? 9 : 0;
      for (let j = i; j <= i + probes; j++) {
        if (table[j] === null) return false;
        if (table[j] === key) return true;
      }
      perturb = Math.floor(perturb / 32);
      i = (i * 5 + 1 + perturb) & mask;
    }
  }

  add(key) {
    const { table, mask } = this;
    let i = key & mask;
    let perturb = key;
    for (;;) {
      const probes = i + 9 <= mask ? 9 : 0;
      for (let j = i; j <= i + probes; j++) {
        if (table[j] === key) return;
        if (table[j] === null) {
          table[j] = key;
          this.fill++;
          if (this.fill * 5 >= mask * 3) this.resize(this.fill > 50000 ? this.fill * 2 : this.fill * 4);
          return;
        }
      }
      perturb = Math.floor(perturb / 32);
      i = (i * 5 + 1 + perturb) & mask;
    }
  }

  resize(minused) {
    let size = 8;
    while (size <= minused) size *= 2;
    const old = this.table;
    this.table = new Array(size).fill(null);
    this.mask = size - 1;
    for (const key of old) if (key !== null) this.insertClean(key);
  }

  insertClean(key) {
    const { table, mask } = this;
    let i = key & mask;
    let perturb = key;
    for (;;) {
      if (table[i] === null) return void (table[i] = key);
      if (i + 9 <= mask) {
        for (let j = i + 1; j <= i + 9; j++) if (table[j] === null) return void (table[j] = key);
      }
      perturb = Math.floor(perturb / 32);
      i = (i * 5 + 1 + perturb) & mask;
    }
  }

  get size() {
    return this.fill;
  }

  *[Symbol.iterator]() {
    for (const key of this.table) if (key !== null) yield key;
  }
}

/** networkx.Graph subset: insertion-ordered nodes and adjacency. */
export class Graph {
  constructor() {
    this.adj = new Map();
  }

  addNode(n) {
    if (!this.adj.has(n)) this.adj.set(n, new Map());
  }

  addEdge(u, v, data = {}) {
    this.addNode(u);
    this.addNode(v);
    const d = this.adj.get(u).get(v) || data;
    Object.assign(d, data);
    this.adj.get(u).set(v, d);
    this.adj.get(v).set(u, d);
  }

  /** G.edges(data=True) order. */
  *edges() {
    const seen = new Set();
    for (const [n, nbrs] of this.adj) {
      for (const [nbr, d] of nbrs) if (!seen.has(nbr)) yield [n, nbr, d];
      seen.add(n);
    }
  }

  /** nx.connected_components: one PySet per component, built by _plain_bfs. */
  *connectedComponents() {
    const seen = new Set();
    const n = this.adj.size;
    for (const v of this.adj.keys()) {
      if (seen.has(v)) continue;
      const comp = this.plainBfs(n - seen.size, v);
      for (const x of comp) seen.add(x);
      yield comp;
    }
  }

  plainBfs(n, source) {
    const seen = new PySet([source]);
    let next = [source];
    while (next.length) {
      const level = next;
      next = [];
      for (const v of level) {
        for (const w of this.adj.get(v).keys()) {
          if (!seen.has(w)) {
            seen.add(w);
            next.push(w);
          }
        }
        if (seen.size === n) return seen;
      }
    }
    return seen;
  }

  /** nx minimum_spanning_edges(algorithm="kruskal", data=True): [u, v, data] in acceptance order. */
  kruskal(weight = "weight") {
    const edges = [...this.edges()].map((e) => ({ wt: e[2][weight] ?? 1, e }));
    edges.sort((a, b) => a.wt - b.wt); // Array.prototype.sort is stable, like Python's sorted
    const parent = new Map();
    const find = (x) => {
      if (!parent.has(x)) parent.set(x, x);
      while (parent.get(x) !== x) {
        parent.set(x, parent.get(parent.get(x)));
        x = parent.get(x);
      }
      return x;
    };
    const out = [];
    for (const { e } of edges) {
      const ru = find(e[0]);
      const rv = find(e[1]);
      if (ru !== rv) {
        out.push(e);
        parent.set(ru, rv);
      }
    }
    return out;
  }
}

/** collections.Counter(items).most_common(): ties keep first-seen order. */
export function mostCommon(items) {
  const counts = new Map();
  for (const x of items) counts.set(x, (counts.get(x) || 0) + 1);
  return [...counts].sort((a, b) => b[1] - a[1]);
}
