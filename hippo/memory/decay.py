"""Forgetting.

Two rules, and the second is the one that matters:

1. Strength decays exponentially in elapsed time, and the decay *rate* is
   itself modulated by strength — a well-reinforced memory forgets slowly
   (spaced-repetition style). Semantic items decay far slower than episodic
   ones, since they already represent many episodes. Recency of *use* is
   deliberately not handled here; it is a separate term in the retrieval
   score, so that "recently seen" and "repeatedly useful" stay distinguishable
   instead of being folded into one number.

2. Pruning is relevance-blind by construction, and that is dangerous. The
   failure mode of any frequency-based policy is deleting the rare-but-critical
   item: used once, never since, catastrophic when missing. Two guards:
   `pinned` items are never pruned, and an item is protected while it is the
   sole surviving support for a semantic abstraction. Neither guard is
   sufficient in general — if the agent never learns an item is critical, no
   policy here will save it. That is a real limit of the design, not an
   implementation gap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .store import MemoryStore


@dataclass
class DecayConfig:
    enabled: bool = True
    episodic_halflife: float = 900.0     # in ticks, measured on elapsed time
    semantic_halflife: float = 12000.0
    strength_protection: float = 1.5     # how much strength slows decay
    prune_threshold: float = 0.10
    protect_sole_support: bool = True
    floor_strength: float = 0.02


def step(store: MemoryStore, cfg: DecayConfig):
    """Apply one decay step to the whole store, then prune."""
    if not cfg.enabled:
        return 0

    now = store.clock
    sole_support = _sole_support_ids(store) if cfg.protect_sole_support else set()

    pruned = 0
    for it in store.live_items():
        if it.pinned:
            continue
        halflife = (cfg.semantic_halflife if it.layer == "semantic"
                    else cfg.episodic_halflife)
        # stronger memories get a longer effective half-life
        halflife *= 1.0 + cfg.strength_protection * max(0.0, it.strength)
        # dt = elapsed since this item was last decayed. Using elapsed time
        # rather than total age is what keeps repeated calls from compounding
        # the same interval over and over.
        dt = max(0, now - it.t_last_decay)
        it.t_last_decay = now
        if dt:
            it.strength = max(cfg.floor_strength,
                              it.strength * math.exp(-math.log(2) * dt / halflife))

        if it.strength < cfg.prune_threshold and it.id not in sole_support:
            store.kill(it, "pruned")
            pruned += 1
    return pruned


def _sole_support_ids(store: MemoryStore) -> set[int]:
    """Episodic ids that are the last living evidence for some abstraction."""
    out = set()
    for it in store.live_items():
        if it.layer != "semantic":
            continue
        living = [s for s in it.support
                  if s in store.items and store.items[s].alive]
        if len(living) == 1:
            out.add(living[0])
    return out
