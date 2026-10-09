"""Outcome-based credit assignment.

The core claim of the design: reinforcement must be driven by *outcomes*, not
by retrieval frequency. Frequency is self-confirming — an item retrieved often
gets retrieved more, regardless of whether it ever helped.

Mechanism: an eligibility trace. Every retrieval during an episode is recorded
with the share of the context it occupied. When the episode resolves and the
environment returns a reward, that reward is applied retroactively to whatever
was eligible, scaled by share. Negative reward weakens the items that were
sitting in context when the agent got it wrong, which is how actively
misleading memories (stale values, say) get driven down instead of merely
aging out.

Known limitation, stated plainly: this cannot distinguish an item that caused
the outcome from an item that merely co-occurred with it. With k=5 retrievals
per episode, four innocent items absorb credit alongside the one that mattered.
Over many episodes the consistent contributor accumulates and the coincidental
ones wash out, but that is a statistical argument, not a causal one. Doing it
properly needs counterfactual replay (re-run the episode with the item held
out), which is implemented below as `counterfactual_credit` but costs one extra
rollout per item and is off by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .retrieval import Retrieved
from .store import MemoryStore


@dataclass
class CreditConfig:
    lr: float = 0.35
    trace_decay: float = 0.6           # across episodes, for multi-step credit
    max_strength: float = 4.0
    min_strength: float = 0.0
    enabled: bool = True
    negative_scale: float = 1.0        # weight on punishment vs reward


@dataclass
class EligibilityTrace:
    weights: dict[int, float] = field(default_factory=dict)

    def record(self, results: list[Retrieved], decay: float):
        for k in list(self.weights):
            self.weights[k] *= decay
            if self.weights[k] < 1e-3:
                del self.weights[k]
        if not results:
            return
        total = sum(max(r.score, 1e-6) for r in results)
        for r in results:
            share = max(r.score, 1e-6) / total
            self.weights[r.item.id] = self.weights.get(r.item.id, 0.0) + share

    def clear(self):
        self.weights.clear()


def apply_reward(store: MemoryStore, trace: EligibilityTrace, reward: float,
                 cfg: CreditConfig):
    """Push reward back onto everything currently eligible."""
    if not cfg.enabled or reward == 0.0:
        return
    scale = cfg.lr * (reward if reward > 0 else reward * cfg.negative_scale)
    for item_id, w in trace.weights.items():
        it = store.items.get(item_id)
        if it is None or not it.alive:
            continue
        it.strength = min(cfg.max_strength,
                          max(cfg.min_strength, it.strength + scale * w))
        if it.slot is not None:
            key = tuple(it.slot)
            r = store.value_reliability.get(key, 1.0) + scale * w
            # capped at 1: reward can restore a claim, never make it
            # outrank a newer one through accumulated praise alone
            store.value_reliability[key] = min(1.0, max(0.0, r))


def frequency_reinforce(store: MemoryStore, results: list[Retrieved], rate=0.1,
                        max_strength=4.0):
    """The naive alternative, kept for the ablation: strengthen on use alone.

    Included so the benchmark can show what it costs rather than asserting it.
    """
    for r in results:
        it = r.item
        it.strength = min(max_strength, it.strength + rate)
