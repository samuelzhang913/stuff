"""Export benchmark streams + a Python reference trace for the browser engine.

    python -m viz.export

Writes viz/data.json: for each seed, the event stream and, for the full
system, the per-query correctness the Python engine produced. The browser
engine replays the same stream and viz/parity.js compares the two traces.

data.json doubles as a behaviour snapshot: tests/test_artifacts.py fails if
the engine's output no longer matches it, so any change in behaviour is
caught and has to be acknowledged by re-running this export.
"""

from __future__ import annotations

import json
from pathlib import Path

from memory import MemoryAgent
from eval.baselines import full
from eval.tasks import build_benchmark, read_answer

SEEDS = [0, 1, 2]
LENGTH = 2400
OUT = Path(__file__).with_name("data.json")


def reference_trace(events):
    """Per query: correct or not, plus the exact retrieved ids and scores.

    Pinning scores (not just correctness) matters: a small change to a weight
    usually flips no answers, so a correctness-only snapshot would let the
    Python and JS engines drift apart silently.
    """
    agent = MemoryAgent(full())
    out, retrieved = [], []
    for ev in events:
        if ev.kind == "obs":
            agent.observe(ev.text, slot=ev.slot)
        else:
            res = agent.recall(ev.text)
            ans = read_answer(ev, res)
            ok = ans == ev.gold
            agent.feedback(1.0 if ok else -1.0)
            out.append(int(ok))
            retrieved.append([[r.item.id, r.score] for r in res])
        agent.maintain()
    snap = agent.snapshot()
    return out, retrieved, {k: snap[k] for k in ("total", "episodic", "semantic",
                                                 "written", "pruned", "superseded")}


def render(verbose=False) -> str:
    data = {"length": LENGTH, "seeds": {}}
    for s in SEEDS:
        events, _ = build_benchmark(seed=s, length=LENGTH)
        trace, retrieved, final = reference_trace(events)
        data["seeds"][str(s)] = {
            "events": [[e.kind[0], e.text,
                        list(e.slot) if e.slot else None,
                        e.gold, e.tag] for e in events],
            "reference": trace,
            "reference_retrieved": retrieved,
            "reference_final": final,
        }
        if verbose:
            print(f"seed {s}: {len(events)} events, "
                  f"python acc {sum(trace) / len(trace):.3f}, final {final}")
    return json.dumps(data, separators=(",", ":"))


def main():
    OUT.write_text(render(verbose=True))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
