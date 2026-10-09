# hippo — agent memory that consolidates, learns from outcomes, and forgets

An external memory for AI agents, loosely modelled on the hippocampus:
episodes are written fast, replayed offline into abstractions, reinforced by
outcomes rather than by use, and decayed when they stop mattering.
See `RESULTS.md` for what each part actually contributes.

```
memory/
  embed.py        deterministic hashed embeddings (swap for a real encoder)
  store.py        episodic + semantic layers, claim-reliability ledger
  retrieval.py    relevance gated by recency, strength and reliability
  credit.py       eligibility traces; reward applied retroactively
  consolidate.py  offline replay: abstraction + revisable contradiction resolution
  decay.py        strength-protected decay, sole-support pruning guard
  agent.py        observe / recall / feedback / maintain
eval/
  tasks.py        synthetic benchmark: stable, changing, rare, rumour-contested facts
  baselines.py    keep_all, recency, sliding window, and one-off ablations
  run_ablations.py
  decay_sweep.py
  explain.py      why each wrong answer happened, factor by factor
  runlog.py       append-only experiment log -> runs/log.jsonl
tests/            unit, regression, reproducibility, headline-claim and parity tests
viz/
  engine.js       browser port, verified identical to Python (parity.js)
  page.html       live network view; build.py inlines it into hippo.html
```

## Use

```python
from memory import MemoryAgent

agent = MemoryAgent()
agent.observe("auth-service region is us-east-1", slot=("auth-service", "region", "us-east-1"))
results = agent.recall("which region does auth-service use?")
agent.feedback(+1.0)      # or -1.0 when the answer turned out wrong
agent.maintain()          # decay every step; consolidation every 50 ticks
```

`slot` is optional. Without it, consolidation falls back to similarity
clustering and cannot resolve contradictions.

## Run

```
pip install -r requirements.txt
python -m unittest discover -s tests -t .
python -m eval.run_ablations --seeds 10
python -m eval.decay_sweep
python -m eval.explain --seed 0
python -m viz.export && node viz/parity.js && python -m viz.build
```

Working on the code? Read `CLAUDE.md` first. It lists the invariants CI
enforces and the open findings.

## Where to take it next

- Swap `LexicalAbstractor` for an LLM call: same interface, real summaries.
- Use a real sentence encoder in `embed.py`.
- Add counterfactual credit: re-run an episode with one memory held out, to
  separate cause from co-occurrence.
- Learn what to pin. Rare-but-critical facts are the open failure (RESULTS §6).
