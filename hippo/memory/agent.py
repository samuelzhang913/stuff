"""MemoryAgent — the orchestrator that wires the pieces into a loop.

    observe(text, slot)      write an episode
    recall(query)            retrieve, recording an eligibility trace
    feedback(reward)         push the outcome back onto what was retrieved
    maintain()               decay + (periodically) consolidate

Every component is individually switchable so the eval can ablate one at a
time against an otherwise identical system.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import consolidate as C
from . import credit as CR
from . import decay as D
from .retrieval import RetrievalConfig, Retrieved, context_tokens, retrieve
from .store import MemoryStore


@dataclass
class AgentConfig:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    decay: D.DecayConfig = field(default_factory=D.DecayConfig)
    credit: CR.CreditConfig = field(default_factory=CR.CreditConfig)
    consolidation: C.ConsolidationConfig = field(default_factory=C.ConsolidationConfig)
    # the naive alternative to outcome-based credit, for the ablation
    frequency_reinforcement: bool = False
    window: int | None = None          # sliding-window baseline: keep last N


class MemoryAgent:
    def __init__(self, cfg: AgentConfig | None = None):
        self.cfg = cfg or AgentConfig()
        self.store = MemoryStore()
        self.trace = CR.EligibilityTrace()
        self.last_results: list[Retrieved] = []
        self.log = {"retrievals": 0, "context_tokens": 0, "passes": 0}

    # -- loop --------------------------------------------------------------

    def observe(self, text: str, slot=None, pinned=False):
        self.store.tick()
        item = self.store.write(text, slot=slot, pinned=pinned)
        self._enforce_window()
        return item

    def recall(self, query: str) -> list[Retrieved]:
        self.store.tick()
        results = retrieve(self.store, query, self.cfg.retrieval)
        self.last_results = results
        self.trace.record(results, self.cfg.credit.trace_decay)
        if self.cfg.frequency_reinforcement:
            CR.frequency_reinforce(self.store, results)
        self.log["retrievals"] += 1
        self.log["context_tokens"] += context_tokens(results)
        return results

    def feedback(self, reward: float):
        CR.apply_reward(self.store, self.trace, reward, self.cfg.credit)

    def maintain(self):
        D.step(self.store, self.cfg.decay)
        if (self.cfg.consolidation.enabled
                and self.store.clock % self.cfg.consolidation.every == 0):
            self.log["passes"] += 1
            return C.consolidate(self.store, self.cfg.consolidation)
        return None

    # -- baselines ---------------------------------------------------------

    def _enforce_window(self):
        if self.cfg.window is None:
            return
        live = sorted((i for i in self.store.live_items()),
                      key=lambda i: i.t_created)
        excess = len(live) - self.cfg.window
        for it in live[:max(0, excess)]:
            self.store.kill(it, "pruned")

    def snapshot(self):
        s = self.store.size()
        s.update(self.store.stats)
        s.update(self.log)
        return s
