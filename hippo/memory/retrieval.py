"""Retrieval scoring.

score = w_rel * relevance * boost
boost = (1 + recency_term + strength_term + semantic_bonus)
        * (0.35 if superseded)
        * (0.15 + 0.85 * claim_reliability)

Relevance is cosine similarity. Recency is an exponential on time since last
access (not since creation — a memory you keep using stays "recent", which is
what makes reinforcement visible to the ranker). Strength is the learned term
that `credit.py` moves. Semantic items get a small bonus because one of them
stands in for many episodes; it is a constant, not a learned preference.

`score_all` is the single implementation of the formula. `retrieve` ranks with
it, and `eval/explain.py` reads its per-term breakdown, so the explanation
can never drift from what the ranker actually did.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .embed import cosine, embed
from .store import MemoryItem, MemoryStore


@dataclass
class RetrievalConfig:
    k: int = 5
    w_relevance: float = 1.0
    w_recency: float = 0.15
    w_strength: float = 0.35
    semantic_bonus: float = 0.10
    recency_halflife: float = 400.0
    min_relevance: float = 0.05        # hard floor: never return junk to pad k
    use_strength: bool = True
    use_recency: bool = True


@dataclass
class Retrieved:
    item: MemoryItem
    score: float
    relevance: float
    terms: dict | None = None      # per-term breakdown, for explanation only


SUPERSEDED_FACTOR = 0.35


def score_all(store: MemoryStore, query: str, cfg: RetrievalConfig) -> list[Retrieved]:
    """Score every live item above the relevance floor, ranked. Pure: no
    side effects on the store, so it is safe to call for inspection."""
    items = store.live_items()
    if not items:
        return []
    q = embed(query)
    rel = cosine(q, store.matrix(items))

    now = store.clock
    out = []
    for it, r in zip(items, rel):
        if r < cfg.min_relevance:
            continue
        # Modulation is MULTIPLICATIVE, not additive. With additive terms a
        # strong-but-irrelevant item gets lifted into top-k over a weak-but-
        # on-topic one, which is how a memory system ends up answering the
        # wrong question confidently. Multiplying keeps relevance as a gate:
        # nothing is promoted past something it is less relevant than by more
        # than the modulation ratio.
        #
        # Evaluation order below is load-bearing: it is mirrored exactly in
        # viz/engine.js and pinned by the parity test. Reordering the float
        # operations can flip near-ties.
        terms = {"relevance": float(r), "recency": 0.0, "strength": 0.0,
                 "semantic": 0.0, "superseded_factor": 1.0,
                 "reliability_factor": 1.0}
        boost = 1.0
        if cfg.use_recency:
            age = max(0, now - it.t_last_access)
            terms["recency"] = cfg.w_recency * math.exp(-age / cfg.recency_halflife)
            boost += terms["recency"]
        if cfg.use_strength:
            terms["strength"] = cfg.w_strength * math.tanh(it.strength)
            boost += terms["strength"]
        if it.layer == "semantic":
            terms["semantic"] = cfg.semantic_bonus
            boost += cfg.semantic_bonus
        if it.superseded_by is not None:
            terms["superseded_factor"] = SUPERSEDED_FACTOR
            boost *= SUPERSEDED_FACTOR
        if cfg.use_strength:
            terms["reliability_factor"] = 0.15 + 0.85 * store.reliability(it.slot)
            boost *= terms["reliability_factor"]
        s = cfg.w_relevance * float(r) * boost
        out.append(Retrieved(it, s, float(r), terms))

    out.sort(key=lambda x: -x.score)
    return out


def retrieve(store: MemoryStore, query: str, cfg: RetrievalConfig) -> list[Retrieved]:
    top = score_all(store, query, cfg)[: cfg.k]
    for rt in top:
        store.touch(rt.item)
    return top


def context_tokens(results: list[Retrieved]) -> int:
    return sum(len(r.item.text.split()) for r in results)
