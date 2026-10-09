// Browser port of the `memory` package. Mirrors the Python line for line so
// the parity test (viz/parity.js) can replay the same event stream and
// compare per-query correctness. Any behavioural change belongs in Python
// first, then here.
//
// Exposes `HippoEngine` (global in the browser, module.exports in Node).
// The agent also emits a log of visual events (`agent.drain()`) so the
// renderer can animate what actually happened, rather than inventing motion.

(function (root) {
  "use strict";

  const DIM = 256;
  const TOKEN_RE = /[a-z0-9]+/g;

  // ---------------------------------------------------------------- embed
  function fnv1a(s) {
    const bytes = new TextEncoder().encode(s);
    let h = 0x811c9dc5;
    for (let i = 0; i < bytes.length; i++) {
      h ^= bytes[i];
      h = Math.imul(h, 0x01000193) >>> 0;
    }
    return h >>> 0;
  }

  function* features(text) {
    const words = text.toLowerCase().match(TOKEN_RE) || [];
    for (const w of words) yield [w, 1.0];
    for (let i = 0; i + 1 < words.length; i++) yield [words[i] + "_" + words[i + 1], 0.7];
    const packed = words.join(" ");
    for (let i = 0; i < packed.length - 2; i++) yield ["#" + packed.slice(i, i + 3), 0.25];
  }

  const embedCache = new Map();
  function embed(text) {
    const hit = embedCache.get(text);
    if (hit) return hit;
    const v = new Float64Array(DIM);
    for (const [feat, w] of features(text)) {
      const h = fnv1a(feat);
      v[h % DIM] += ((h >>> 17) & 1 ? 1.0 : -1.0) * w;
    }
    let n = 0;
    for (let i = 0; i < DIM; i++) n += v[i] * v[i];
    n = Math.sqrt(n);
    if (n > 0) for (let i = 0; i < DIM; i++) v[i] /= n;
    embedCache.set(text, v);
    return v;
  }

  function dot(a, b) {
    let s = 0;
    for (let i = 0; i < DIM; i++) s += a[i] * b[i];
    return s;
  }

  // ---------------------------------------------------------------- config
  function defaultConfig() {
    return {
      retrieval: { k: 5, wRelevance: 1.0, wRecency: 0.15, wStrength: 0.35,
                   semanticBonus: 0.10, recencyHalflife: 400.0,
                   minRelevance: 0.05, useStrength: true, useRecency: true },
      decay: { enabled: true, episodicHalflife: 900.0, semanticHalflife: 12000.0,
               strengthProtection: 1.5, pruneThreshold: 0.10,
               protectSoleSupport: true, floorStrength: 0.02 },
      credit: { lr: 0.35, traceDecay: 0.6, maxStrength: 4.0, minStrength: 0.0,
                enabled: true, negativeScale: 1.0 },
      consolidation: { enabled: true, every: 50, minSupport: 3, simThreshold: 0.55,
                       demoteTo: 0.25, semanticStartStrength: 1.4,
                       resolveContradictions: true },
    };
  }

  const slotKey = (s) => s[0] + "\u0001" + s[1];
  const claimKey = (s) => s[0] + "\u0001" + s[1] + "\u0001" + s[2];

  // ---------------------------------------------------------------- store
  class Store {
    constructor() {
      this.items = new Map();
      this.nextId = 0;
      this.clock = 0;
      this.stats = { written: 0, pruned: 0, consolidated: 0, superseded: 0 };
      this.valueReliability = new Map();
      this.emit = () => {};
    }
    tick() { return ++this.clock; }
    write(text, layer = "episodic", slot = null, support = null, strength = 1.0, pinned = false) {
      const it = {
        id: this.nextId++, text, layer,
        tCreated: this.clock, tLastAccess: this.clock, tLastDecay: this.clock,
        strength, accessCount: 0, slot, support: support ? support.slice() : [],
        supersededBy: null, pinned, alive: true, vec: embed(text),
      };
      this.items.set(it.id, it);
      this.stats.written++;
      this.emit({ type: "write", item: it });
      return it;
    }
    liveItems() {
      const out = [];
      for (const it of this.items.values()) if (it.alive) out.push(it);
      return out;
    }
    touch(it) { it.tLastAccess = this.clock; it.accessCount++; }
    kill(it, reason = "pruned") {
      if (it.alive && !it.pinned) {
        it.alive = false;
        this.stats[reason] = (this.stats[reason] || 0) + 1;
        this.emit({ type: "kill", item: it, reason });
      }
    }
    reliability(slot) {
      if (!slot || slot[2] == null) return 1.0;
      const r = this.valueReliability.get(claimKey(slot));
      return r === undefined ? 1.0 : r;
    }
  }

  // ---------------------------------------------------------------- retrieval
  function retrieve(store, query, cfg) {
    const items = store.liveItems();
    if (!items.length) return [];
    const q = embed(query);
    const now = store.clock;
    const out = [];
    for (const it of items) {
      const r = dot(it.vec, q);
      if (r < cfg.minRelevance) continue;
      let boost = 1.0;
      if (cfg.useRecency) {
        const age = Math.max(0, now - it.tLastAccess);
        boost += cfg.wRecency * Math.exp(-age / cfg.recencyHalflife);
      }
      if (cfg.useStrength) boost += cfg.wStrength * Math.tanh(it.strength);
      if (it.layer === "semantic") boost += cfg.semanticBonus;
      if (it.supersededBy !== null) boost *= 0.35;
      if (cfg.useStrength) boost *= 0.15 + 0.85 * store.reliability(it.slot);
      out.push({ item: it, score: cfg.wRelevance * r * boost, relevance: r });
    }
    out.sort((a, b) => b.score - a.score);
    const top = out.slice(0, cfg.k);
    for (const rt of top) store.touch(rt.item);
    return top;
  }

  // ---------------------------------------------------------------- credit
  class Trace {
    constructor() { this.weights = new Map(); }
    record(results, decay) {
      for (const [k, w] of [...this.weights]) {
        const nw = w * decay;
        if (nw < 1e-3) this.weights.delete(k);
        else this.weights.set(k, nw);
      }
      if (!results.length) return;
      let total = 0;
      for (const r of results) total += Math.max(r.score, 1e-6);
      for (const r of results) {
        const share = Math.max(r.score, 1e-6) / total;
        this.weights.set(r.item.id, (this.weights.get(r.item.id) || 0) + share);
      }
    }
  }

  function applyReward(store, trace, reward, cfg) {
    if (!cfg.enabled || reward === 0) return;
    const scale = cfg.lr * (reward > 0 ? reward : reward * cfg.negativeScale);
    for (const [id, w] of trace.weights) {
      const it = store.items.get(id);
      if (!it || !it.alive) continue;
      it.strength = Math.min(cfg.maxStrength, Math.max(cfg.minStrength, it.strength + scale * w));
      if (it.slot) {
        const k = claimKey(it.slot);
        const cur = store.valueReliability.has(k) ? store.valueReliability.get(k) : 1.0;
        store.valueReliability.set(k, Math.min(1.0, Math.max(0.0, cur + scale * w)));
      }
    }
  }

  // ---------------------------------------------------------------- decay
  function soleSupportIds(store) {
    const out = new Set();
    for (const it of store.liveItems()) {
      if (it.layer !== "semantic") continue;
      const living = it.support.filter((s) => store.items.has(s) && store.items.get(s).alive);
      if (living.length === 1) out.add(living[0]);
    }
    return out;
  }

  function decayStep(store, cfg, threshold) {
    if (!cfg.enabled) return 0;
    const pruneAt = threshold === undefined ? cfg.pruneThreshold : threshold;
    const now = store.clock;
    const sole = cfg.protectSoleSupport ? soleSupportIds(store) : new Set();
    let pruned = 0;
    for (const it of store.liveItems()) {
      if (it.pinned) continue;
      let hl = it.layer === "semantic" ? cfg.semanticHalflife : cfg.episodicHalflife;
      hl *= 1.0 + cfg.strengthProtection * Math.max(0.0, it.strength);
      const dt = Math.max(0, now - it.tLastDecay);
      it.tLastDecay = now;
      if (dt) it.strength = Math.max(cfg.floorStrength, it.strength * Math.exp(-Math.LN2 * dt / hl));
      if (it.strength < pruneAt && !sole.has(it.id)) { store.kill(it, "pruned"); pruned++; }
    }
    return pruned;
  }

  // ---------------------------------------------------------------- consolidation
  const LATEST_BONUS = 0.1;

  function abstractor(episodes) {
    const eps = episodes.slice().sort((a, b) => a.tCreated - b.tCreated);
    const latest = eps[eps.length - 1];
    if (latest.slot) {
      const [ent, attr, val] = latest.slot;
      return [`${ent} ${attr} is ${val} (consolidated from ${eps.length} observations)`, [ent, attr, val]];
    }
    let rep = eps[0];
    for (const e of eps) if (e.text.length > rep.text.length) rep = e;
    return [`${rep.text} (recurring, ${eps.length}x)`, null];
  }

  function similarityCluster(items, threshold) {
    const centroids = [], groups = [];
    for (const it of items) {
      const v = it.vec;
      let best = -1, bestSim = threshold;
      for (let ci = 0; ci < centroids.length; ci++) {
        const sim = dot(centroids[ci], v);
        if (sim > bestSim) { best = ci; bestSim = sim; }
      }
      if (best === -1) { centroids.push(Float64Array.from(v)); groups.push([it]); }
      else {
        groups[best].push(it);
        const n = groups[best].length;
        const c = new Float64Array(DIM);
        let nrm = 0;
        for (let i = 0; i < DIM; i++) { c[i] = centroids[best][i] * (n - 1) / n + v[i] / n; nrm += c[i] * c[i]; }
        nrm = Math.sqrt(nrm);
        if (nrm) for (let i = 0; i < DIM; i++) c[i] /= nrm;
        centroids[best] = c;
      }
    }
    return groups;
  }

  function cluster(items, threshold) {
    const keyed = new Map(), keyless = [];
    for (const it of items) {
      if (it.slot) {
        const k = slotKey(it.slot);
        if (!keyed.has(k)) keyed.set(k, []);
        keyed.get(k).push(it);
      } else keyless.push(it);
    }
    return [...keyed.values(), ...similarityCluster(keyless, threshold)];
  }

  function resolve(store, group0) {
    const byKey = new Map();
    for (const e of group0) if (e.slot) {
      const k = slotKey(e.slot);
      if (!byKey.has(k)) byKey.set(k, []);
      byKey.get(k).push(e);
    }
    let count = 0;
    for (const group of byKey.values()) {
      const values = [...new Set(group.map((e) => e.slot[2]))].sort();
      if (values.length < 2) { for (const e of group) e.supersededBy = null; continue; }
      group.sort((a, b) => a.tCreated - b.tCreated);
      const latestValue = group[group.length - 1].slot[2];
      const [ent, attr] = group[0].slot;
      const score = (v) => store.reliability([ent, attr, v]) + (v === latestValue ? LATEST_BONUS : 0);
      // Python: max(sorted(values), key=(score, is_latest)) -> first maximal.
      let best = null;
      for (const v of values) {
        if (best === null) { best = v; continue; }
        const a = [score(v), v === latestValue], b = [score(best), best === latestValue];
        if (a[0] > b[0] || (a[0] === b[0] && a[1] && !b[1])) best = v;
      }
      let winner = null;
      for (const e of group) if (e.slot[2] === best && (!winner || e.tCreated > winner.tCreated)) winner = e;
      for (const e of group) {
        if (e.slot[2] === best) e.supersededBy = null;
        else if (e.supersededBy === null) {
          e.supersededBy = winner.id; store.stats.superseded++; count++;
          store.emit({ type: "supersede", item: e, by: winner });
        } else e.supersededBy = winner.id;
      }
    }
    return count;
  }

  function findSemantic(store, slot, episodeIds) {
    if (!slot) {
      const ids = new Set(episodeIds);
      let best = null, bestOverlap = 0;
      for (const it of store.liveItems()) {
        if (it.layer !== "semantic" || it.slot) continue;
        let ov = 0;
        for (const s of new Set(it.support)) if (ids.has(s)) ov++;
        if (ov > bestOverlap) { best = it; bestOverlap = ov; }
      }
      return best;
    }
    const k = slotKey(slot);
    for (const it of store.liveItems()) if (it.layer === "semantic" && it.slot && slotKey(it.slot) === k) return it;
    return null;
  }

  function consolidate(store, cfg) {
    const rep = { abstractions: 0, superseded: 0, absorbed: 0, created: [] };
    if (!cfg.enabled) return rep;
    const episodic = store.liveItems().filter((i) => i.layer === "episodic");
    if (episodic.length < cfg.minSupport) return rep;
    for (const cl of cluster(episodic, cfg.simThreshold)) {
      if (cl.length < cfg.minSupport) continue;
      if (cfg.resolveContradictions) rep.superseded += resolve(store, cl);
      let active = cl.filter((e) => e.supersededBy === null);
      if (active.length < cfg.minSupport) {
        if (!active.length) active = cl.slice(-1);
        if (!active.length) continue;
      }
      const [text, slot] = abstractor(active);
      const ids = active.map((e) => e.id);
      const existing = findSemantic(store, slot, ids);
      if (existing) {
        const support = [...new Set([...existing.support, ...ids])].sort((a, b) => a - b);
        if (existing.text === text) {
          existing.support = support;
          existing.strength = Math.max(existing.strength, cfg.semanticStartStrength);
          existing.tLastAccess = store.clock;
        } else {
          store.kill(existing, "superseded");
          const it = store.write(text, "semantic", slot, support, Math.max(existing.strength, cfg.semanticStartStrength));
          rep.created.push(it); rep.abstractions++;
        }
      } else {
        const it = store.write(text, "semantic", slot, ids, cfg.semanticStartStrength);
        store.stats.consolidated++;
        rep.created.push(it); rep.abstractions++;
      }
      for (const e of active) { e.strength = Math.min(e.strength, cfg.demoteTo); rep.absorbed++; }
    }
    return rep;
  }

  // ---------------------------------------------------------------- agent
  class Agent {
    constructor(cfg) {
      this.cfg = cfg || defaultConfig();
      this.store = new Store();
      this.trace = new Trace();
      this.log = { retrievals: 0, contextTokens: 0, passes: 0 };
      this.events = [];
      this.store.emit = (e) => this.events.push(e);
    }
    observe(text, slot) { this.store.tick(); return this.store.write(text, "episodic", slot || null); }
    recall(query) {
      this.store.tick();
      const res = retrieve(this.store, query, this.cfg.retrieval);
      this.trace.record(res, this.cfg.credit.traceDecay);
      this.log.retrievals++;
      for (const r of res) this.log.contextTokens += r.item.text.split(/\s+/).length;
      this.events.push({ type: "recall", query, results: res });
      return res;
    }
    feedback(reward) {
      applyReward(this.store, this.trace, reward, this.cfg.credit);
      this.events.push({ type: "feedback", reward, ids: [...this.trace.weights.keys()], enabled: this.cfg.credit.enabled });
    }
    maintain() {
      decayStep(this.store, this.cfg.decay);
      const c = this.cfg.consolidation;
      if (c.enabled && this.store.clock % c.every === 0) return this.consolidateNow();
      return null;
    }
    consolidateNow() {
      this.log.passes++;
      const rep = consolidate(this.store, this.cfg.consolidation);
      this.events.push({ type: "consolidate", report: rep });
      return rep;
    }
    pruneWeak(threshold) {
      const before = this.store.stats.pruned;
      decayStep(this.store, this.cfg.decay, threshold);
      return this.store.stats.pruned - before;
    }
    drain() { const e = this.events; this.events = []; return e; }
  }

  function readAnswer(slot, results) {
    for (const r of results) {
      const s = r.item.slot;
      if (s && s[0] === slot[0] && s[1] === slot[1]) return s[2];
    }
    return null;
  }

  const api = { Agent, defaultConfig, readAnswer, embed, fnv1a, DIM };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.HippoEngine = api;
})(typeof window !== "undefined" ? window : globalThis);
