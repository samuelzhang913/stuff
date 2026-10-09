"""Experimental conditions.

The baseline that matters is `keep_all`: unbounded store, plain semantic
retrieval, no decay, no strengthening, no consolidation. Most memory
architectures quietly lose to it on accuracy and only win on cost. If the
full system cannot beat it, the honest reading is that the machinery is a
compression scheme, not a performance win — and the results file should say
so rather than reaching for a flattering metric.
"""

from __future__ import annotations

import copy

from memory import AgentConfig
from memory.consolidate import ConsolidationConfig
from memory.credit import CreditConfig
from memory.decay import DecayConfig
from memory.retrieval import RetrievalConfig

K = 5


def _base() -> AgentConfig:
    return AgentConfig(
        retrieval=RetrievalConfig(k=K),
        decay=DecayConfig(),
        credit=CreditConfig(),
        consolidation=ConsolidationConfig(),
    )


def keep_all() -> AgentConfig:
    """Flat RAG over everything ever written."""
    return AgentConfig(
        retrieval=RetrievalConfig(k=K, use_strength=False, use_recency=False,
                                  semantic_bonus=0.0),
        decay=DecayConfig(enabled=False),
        credit=CreditConfig(enabled=False),
        consolidation=ConsolidationConfig(enabled=False),
    )


def keep_all_recency() -> AgentConfig:
    """Flat RAG + a recency prior. The cheap trick that often beats a
    memory architecture; included so the comparison is not a straw man."""
    cfg = keep_all()
    cfg.retrieval = RetrievalConfig(k=K, use_strength=False, use_recency=True,
                                    w_recency=0.35, semantic_bonus=0.0)
    return cfg


def sliding_window(n=300) -> AgentConfig:
    cfg = keep_all()
    cfg.window = n
    return cfg


def full() -> AgentConfig:
    return _base()


def no_decay() -> AgentConfig:
    cfg = _base()
    cfg.decay = DecayConfig(enabled=False)
    return cfg


def no_consolidation() -> AgentConfig:
    cfg = _base()
    cfg.consolidation = ConsolidationConfig(enabled=False)
    return cfg


def no_reinforcement() -> AgentConfig:
    cfg = _base()
    cfg.credit = CreditConfig(enabled=False)
    return cfg


def frequency_reinforcement() -> AgentConfig:
    """Strengthen on retrieval count instead of on outcome — the design this
    package argues against. If it matches outcome-based credit, the argument
    is wrong and the results should record that."""
    cfg = _base()
    cfg.credit = CreditConfig(enabled=False)
    cfg.frequency_reinforcement = True
    return cfg


CONDITIONS = {
    "keep_all": keep_all,
    "keep_all_recency": keep_all_recency,
    "sliding_window": sliding_window,
    "full": full,
    "no_decay": no_decay,
    "no_consolidation": no_consolidation,
    "no_reinforcement": no_reinforcement,
    "frequency_reinforcement": frequency_reinforcement,
}
