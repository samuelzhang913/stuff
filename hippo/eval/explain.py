"""Explain every wrong answer in a run.

    python -m eval.explain                         # full system, seed 0
    python -m eval.explain --condition no_reinforcement --tag unreliable
    python -m eval.explain --seed 3 --limit 5 --jsonl runs/explain.jsonl

For each failed query: what was retrieved and why (the per-term score
breakdown from memory.retrieval.score_all, the same code the ranker uses),
where the best memory holding the *correct* value ranked, and a one-line
diagnosis. Use this before touching any weight: aggregate accuracy says that
something broke, this says which mechanism did it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory import MemoryAgent
from memory.retrieval import score_all
from eval.baselines import CONDITIONS
from eval.tasks import build_benchmark, read_answer


def _row(rank, r):
    it = r.item
    return {
        "rank": rank, "id": it.id, "layer": it.layer, "text": it.text,
        "key": list(it.key) if it.key else None,
        "value": it.value, "score": round(r.score, 5),
        "terms": {k: round(v, 5) for k, v in r.terms.items()},
        "strength": round(it.strength, 4),
        "superseded": it.superseded_by is not None,
        "age": it.t_created,
    }


def _diagnose(rec, k):
    gold, best, key = rec["gold"], rec["best_gold"], rec["key"]
    if best is None:
        if rec["gold_ever_written"]:
            return "forgotten: every memory holding the correct value was pruned"
        return "never stored: no memory ever held the correct value"
    # The reader only looks at memories for the queried (entity, attribute),
    # so the competitor is the highest-ranked same-key memory with a wrong
    # value. Memories about other attributes cannot cause a wrong answer.
    rival = next((r for r in rec["retrieved"]
                  if r["key"] == key and r["value"] != gold), None)
    if rival is None:
        if best["rank"] < k:
            return "reader: correct memory was retrieved but not used"
        return (f"crowded out: no memory for this attribute reached the top-{k}; "
                f"best correct memory at rank {best['rank']}")
    lead = rival
    # Which factor separates the winner from the correct memory?
    a, b = lead["terms"], best["terms"]
    factors = {
        "relevance": a["relevance"] / max(b["relevance"], 1e-9),
        "superseded flag": a["superseded_factor"] / max(b["superseded_factor"], 1e-9),
        "claim reliability": a["reliability_factor"] / max(b["reliability_factor"], 1e-9),
        "strength+recency+layer": (1 + a["recency"] + a["strength"] + a["semantic"]) /
                                  (1 + b["recency"] + b["strength"] + b["semantic"]),
    }
    cause = max(factors, key=factors.get)
    where = f"rank {best['rank']}" + ("" if best["rank"] < k else f" (outside top-{k})")
    return (f"outranked: correct memory at {where}; '{lead['value']}' won "
            f"mainly on {cause} (x{factors[cause]:.2f})")


def explain(condition="full", seed=0, length=2400, tag=None, limit=None):
    events, _ = build_benchmark(seed=seed, length=length)
    agent = MemoryAgent(CONDITIONS[condition]())
    k = agent.cfg.retrieval.k
    written_values = {}
    records, qi = [], 0

    for ev in events:
        if ev.kind == "obs":
            agent.observe(ev.text, slot=ev.slot)
            if ev.slot:
                written_values.setdefault(ev.slot[:2], set()).add(ev.slot[2])
        else:
            # score_all is pure; call it before recall() mutates recency.
            ranked = score_all(agent.store, ev.text, agent.cfg.retrieval)
            results = agent.recall(ev.text)
            assert [r.item.id for r in results] == [r.item.id for r in ranked[:k]], \
                "explain diverged from the ranker"
            answer = read_answer(ev, results)
            ok = answer == ev.gold
            agent.feedback(1.0 if ok else -1.0)

            if not ok and (tag is None or ev.tag == tag):
                key = ev.slot[:2]
                best = next(((i, r) for i, r in enumerate(ranked)
                             if r.item.key == key and r.item.value == ev.gold), None)
                rec = {
                    "query_index": qi, "tick": agent.store.clock, "query": ev.text,
                    "tag": ev.tag, "key": list(key), "gold": ev.gold, "answer": answer,
                    "retrieved": [_row(i, r) for i, r in enumerate(ranked[:k])],
                    "best_gold": _row(*best) if best else None,
                    "gold_ever_written": ev.gold in written_values.get(key, ()),
                    "claim_reliability": {
                        v: round(agent.store.reliability((*key, v)), 4)
                        for v in sorted(written_values.get(key, ()))},
                }
                rec["diagnosis"] = _diagnose(rec, k)
                records.append(rec)
                if limit and len(records) >= limit:
                    break
            qi += 1
        agent.maintain()
    return records


def _print(records):
    for r in records:
        print(f"\n#{r['query_index']} tick {r['tick']} [{r['tag']}] {r['query']}")
        print(f"  answered {r['answer']!r}, expected {r['gold']!r}")
        print(f"  -> {r['diagnosis']}")
        print(f"  claim reliability: {r['claim_reliability']}")
        for x in r["retrieved"]:
            t = x["terms"]
            flag = " SUPERSEDED" if x["superseded"] else ""
            print(f"    {x['rank']}. [{x['layer'][:3]}] score {x['score']:.3f} = "
                  f"rel {t['relevance']:.3f} * (1+rec {t['recency']:.3f}"
                  f"+str {t['strength']:.3f}+sem {t['semantic']:.2f})"
                  f" * sup {t['superseded_factor']:.2f} * reliab {t['reliability_factor']:.2f}"
                  f"{flag}  | {x['text'][:60]}")
        b = r["best_gold"]
        if b and b["rank"] >= len(r["retrieved"]):
            print(f"    best correct memory: rank {b['rank']}, score {b['score']:.3f} | {b['text'][:60]}")
    print(f"\n{len(records)} wrong answers explained")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="full", choices=sorted(CONDITIONS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--length", type=int, default=2400)
    ap.add_argument("--tag", default=None,
                    help="stable | volatile_pre | volatile_post | rare | unreliable")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jsonl", default=None, help="also write records as JSON lines")
    args = ap.parse_args()
    records = explain(args.condition, args.seed, args.length, args.tag, args.limit)
    _print(records)
    if args.jsonl:
        Path(args.jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.jsonl, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
