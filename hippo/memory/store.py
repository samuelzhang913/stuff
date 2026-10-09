"""Two-layer memory store: fast episodic buffer + slow semantic layer.

The split follows complementary learning systems: episodes are written once,
cheaply, with no interpretation. Abstractions are built later, offline, by
`consolidate.py`, and live in the semantic layer where they decay far more
slowly. Nothing here decides what is important; that is `credit.py`'s job.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .embed import DIM, embed


@dataclass
class MemoryItem:
    id: int
    text: str
    layer: str                      # 'episodic' | 'semantic'
    t_created: int
    t_last_access: int
    t_last_decay: int = 0
    strength: float = 1.0
    access_count: int = 0
    # Structured slot, when the writer happens to know one. Real systems get
    # this from an extraction model; the benchmark supplies it directly.
    slot: tuple | None = None       # (entity, attribute, value)
    support: list = field(default_factory=list)   # episodic ids behind a semantic item
    superseded_by: int | None = None
    pinned: bool = False
    alive: bool = True

    @property
    def key(self):
        return None if self.slot is None else (self.slot[0], self.slot[1])

    @property
    def value(self):
        return None if self.slot is None else self.slot[2]


class MemoryStore:
    """Holds items plus a parallel embedding matrix with free-list reuse."""

    def __init__(self):
        self.items: dict[int, MemoryItem] = {}
        self._next_id = 0
        self._vecs = np.zeros((0, DIM), dtype=np.float64)
        self._rows: dict[int, int] = {}     # item id -> row index
        self.clock = 0
        self.stats = {"written": 0, "pruned": 0, "consolidated": 0, "superseded": 0}
        # Claim-level credit: (entity, attribute, value) -> reliability in
        # [0, 1]. Starts at 1 ("innocent until punished") and is moved only by
        # outcome feedback. Lives on the claim rather than on an item, so a
        # fresh restatement of a discredited claim inherits the discredit, and
        # rewriting an abstraction does not wipe what was learned.
        self.value_reliability: dict[tuple, float] = {}

    def reliability(self, slot) -> float:
        if slot is None or slot[2] is None:
            return 1.0
        return self.value_reliability.get(tuple(slot), 1.0)

    # -- writing -----------------------------------------------------------

    def tick(self) -> int:
        self.clock += 1
        return self.clock

    def write(self, text, layer="episodic", slot=None, support=None,
              strength=1.0, pinned=False) -> MemoryItem:
        item = MemoryItem(
            id=self._next_id,
            text=text,
            layer=layer,
            t_created=self.clock,
            t_last_access=self.clock,
            t_last_decay=self.clock,
            strength=strength,
            slot=slot,
            support=list(support or []),
            pinned=pinned,
        )
        self._next_id += 1
        self.items[item.id] = item
        self._rows[item.id] = self._vecs.shape[0]
        self._vecs = np.vstack([self._vecs, embed(text)[None, :]])
        self.stats["written"] += 1
        return item

    # -- reading -----------------------------------------------------------

    def live_items(self):
        return [it for it in self.items.values() if it.alive]

    def matrix(self, items):
        if not items:
            return np.zeros((0, DIM), dtype=np.float64)
        return self._vecs[[self._rows[it.id] for it in items]]

    def touch(self, item: MemoryItem):
        item.t_last_access = self.clock
        item.access_count += 1

    # -- removal -----------------------------------------------------------

    def kill(self, item: MemoryItem, reason="pruned"):
        if item.alive and not item.pinned:
            item.alive = False
            self.stats[reason] = self.stats.get(reason, 0) + 1

    def size(self):
        live = self.live_items()
        return {
            "total": len(live),
            "episodic": sum(1 for i in live if i.layer == "episodic"),
            "semantic": sum(1 for i in live if i.layer == "semantic"),
        }
