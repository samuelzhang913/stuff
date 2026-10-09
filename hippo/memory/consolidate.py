"""Offline consolidation — the part most 'hippocampal memory' systems omit.

Store/retrieve/strengthen/fade is a cache with an eviction policy. What makes
the biological analogy worth borrowing is replay: periodically revisiting
episodes offline, extracting the regularity across them, and writing that
regularity to a slower store — after which the individual episodes become
expendable.

Two things happen in a consolidation pass:

1. ABSTRACTION. Episodes that cluster tightly in embedding space and share a
   slot key are merged into one semantic item carrying the support set. The
   episodes are demoted (strength cut), not deleted, so decay can finish them
   off while the abstraction survives.

2. CONTRADICTION RESOLUTION. Within a cluster, episodes asserting different
   values for the same (entity, attribute) key are in conflict. The most
   recent wins; older ones are marked superseded and penalised in ranking.
   This is the step that stops a keep-everything store from confidently
   serving a stale value forever.

The abstractor is pluggable. `LexicalAbstractor` below is deliberately simple
— it merges text and takes the latest value — because a synthetic benchmark
should not be graded on the cleverness of its summariser. In production this
is where an LLM call goes: `LLMAbstractor(client).summarise(episodes)`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .store import MemoryItem, MemoryStore


@dataclass
class ConsolidationConfig:
    enabled: bool = True
    every: int = 50                 # run a pass every N ticks
    min_support: int = 3            # episodes needed to form an abstraction
    sim_threshold: float = 0.55     # cluster tightness
    demote_to: float = 0.25         # strength left on absorbed episodes
    semantic_start_strength: float = 1.4
    resolve_contradictions: bool = True


class LexicalAbstractor:
    """Merge a cluster into one sentence. Latest value wins."""

    def __call__(self, episodes: list[MemoryItem]) -> tuple[str, tuple | None]:
        episodes = sorted(episodes, key=lambda e: e.t_created)
        latest = episodes[-1]
        if latest.slot is not None:
            ent, attr, val = latest.slot
            text = (f"{ent} {attr} is {val} "
                    f"(consolidated from {len(episodes)} observations)")
            return text, (ent, attr, val)
        # no slot: fall back to the longest episode as representative
        rep = max(episodes, key=lambda e: len(e.text))
        return f"{rep.text} (recurring, {len(episodes)}x)", None


def consolidate(store: MemoryStore, cfg: ConsolidationConfig,
                abstractor=None) -> dict:
    """One offline replay pass. Returns a small report."""
    if not cfg.enabled:
        return {"abstractions": 0, "superseded": 0, "absorbed": 0}

    abstractor = abstractor or LexicalAbstractor()
    # Superseded episodes are included on purpose: resolution is re-run every
    # pass, so a verdict reached on bad evidence can be reversed later.
    episodic = [i for i in store.live_items() if i.layer == "episodic"]
    if len(episodic) < cfg.min_support:
        return {"abstractions": 0, "superseded": 0, "absorbed": 0}

    clusters = _cluster(store, episodic, cfg.sim_threshold)

    made = absorbed = superseded = 0
    for cluster in clusters:
        if len(cluster) < cfg.min_support:
            continue

        if cfg.resolve_contradictions:
            superseded += _resolve(store, cluster)

        active = [e for e in cluster if e.superseded_by is None]
        if len(active) < cfg.min_support:
            # cluster was mostly stale values; still worth abstracting the
            # survivor so the current value gets a durable home
            active = active or cluster[-1:]
            if len(active) < 1:
                continue

        text, slot = abstractor(active)
        existing = _find_semantic(store, slot, [e.id for e in active])
        if existing is not None:
            support = sorted(set(existing.support) | {e.id for e in active})
            if existing.text == text:
                # unchanged: refresh rather than pile up near-duplicates
                existing.support = support
                existing.strength = max(existing.strength,
                                        cfg.semantic_start_strength)
                existing.t_last_access = store.clock
            else:
                # The abstraction itself is now stale. Replacing it is the
                # whole point of re-running consolidation: an abstraction that
                # cannot be revised is worse than no abstraction, because it
                # carries extra ranking weight while being wrong.
                store.kill(existing, "superseded")
                store.write(text, layer="semantic", slot=slot, support=support,
                            strength=max(existing.strength,
                                         cfg.semantic_start_strength))
                made += 1
        else:
            sem = store.write(text, layer="semantic", slot=slot,
                              support=[e.id for e in active],
                              strength=cfg.semantic_start_strength)
            store.stats["consolidated"] += 1
            made += 1

        for e in active:
            e.strength = min(e.strength, cfg.demote_to)
            absorbed += 1

    return {"abstractions": made, "superseded": superseded, "absorbed": absorbed}


def _cluster(store: MemoryStore, items: list[MemoryItem], threshold: float):
    """Group episodes that are about the same thing.

    When a slot key is available it is authoritative: embedding clusters drift,
    and a drifting centroid will happily merge two different entities and then
    abstract a fact that was never observed. Only keyless episodes fall through
    to similarity clustering.
    """
    keyed: dict[tuple, list[MemoryItem]] = defaultdict(list)
    keyless: list[MemoryItem] = []
    for it in items:
        (keyed[it.key] if it.key is not None else keyless).append(it)

    groups = list(keyed.values())
    groups.extend(_similarity_cluster(store, keyless, threshold))
    return groups


def _similarity_cluster(store: MemoryStore, items: list[MemoryItem],
                        threshold: float):
    """Greedy single-pass clustering. O(n * clusters), good enough here."""
    if not items:
        return []
    M = store.matrix(items)
    centroids: list[np.ndarray] = []
    groups: list[list[MemoryItem]] = []
    for it, v in zip(items, M):
        best, best_sim = -1, threshold
        for ci, c in enumerate(centroids):
            sim = float(np.dot(c, v))
            if sim > best_sim:
                best, best_sim = ci, sim
        if best == -1:
            centroids.append(v.copy())
            groups.append([it])
        else:
            groups[best].append(it)
            n = len(groups[best])
            c = centroids[best] * (n - 1) / n + v / n
            nrm = np.linalg.norm(c)
            centroids[best] = c / nrm if nrm else c
    return groups


LATEST_BONUS = 0.1


def _resolve(store: MemoryStore, cluster: list[MemoryItem]) -> int:
    """Pick the current value for each key; supersede the rest.

    Score per value = learned claim reliability + a small bonus for being the
    most recently asserted. With no feedback every claim sits at 1.0, so this
    reduces to "latest wins" (right for genuine updates). Once outcomes have
    discredited a claim by more than the bonus, an older value can win back.
    Losers are flagged, not weakened: a flag is reversible, a strength cut
    is not.
    """
    by_key = defaultdict(list)
    for e in cluster:
        if e.key is not None:
            by_key[e.key].append(e)

    count = 0
    for key, group in by_key.items():
        # sorted: max() over a set of strings breaks ties in hash order, which
        # Python randomises per process, so runs were not reproducible.
        values = sorted({e.value for e in group})
        if len(values) < 2:
            for e in group:
                e.superseded_by = None
            continue
        group.sort(key=lambda e: e.t_created)
        latest_value = group[-1].value
        score = {v: store.reliability((key[0], key[1], v))
                    + (LATEST_BONUS if v == latest_value else 0.0)
                 for v in values}
        best = max(values, key=lambda v: (score[v], v == latest_value))
        winner = max((e for e in group if e.value == best),
                     key=lambda e: e.t_created)
        for e in group:
            if e.value == best:
                e.superseded_by = None
            elif e.superseded_by is None:
                e.superseded_by = winner.id
                store.stats["superseded"] += 1
                count += 1
            else:
                e.superseded_by = winner.id
    return count


def _find_semantic(store: MemoryStore, slot, episode_ids=()):
    """Existing abstraction for this cluster, if any.

    Keyed clusters match on (entity, attribute). Keyless clusters have nothing
    to match on but their evidence, so they match the abstraction whose
    support overlaps them most; without this, every pass minted a fresh
    duplicate abstraction for every keyless cluster.
    """
    if slot is None:
        ids = set(episode_ids)
        best, best_overlap = None, 0
        for it in store.live_items():
            if it.layer != "semantic" or it.slot is not None:
                continue
            overlap = len(ids.intersection(it.support))
            if overlap > best_overlap:
                best, best_overlap = it, overlap
        return best
    key = (slot[0], slot[1])
    for it in store.live_items():
        if it.layer == "semantic" and it.key == key:
            return it
    return None
