"""Run every condition across seeds and report.

Usage: python -m eval.run_ablations [--seeds N] [--length N] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from collections import defaultdict

from memory import MemoryAgent
from eval import runlog
from eval.baselines import CONDITIONS
from eval.tasks import build_benchmark, read_answer


def run_one(cfg_fn, seed, length):
    events, meta = build_benchmark(seed=seed, length=length)
    agent = MemoryAgent(cfg_fn())

    by_tag = defaultdict(lambda: [0, 0])    # tag -> [correct, total]
    ctx_tokens = 0
    n_q = 0

    for ev in events:
        if ev.kind == "obs":
            agent.observe(ev.text, slot=ev.slot)
        else:
            results = agent.recall(ev.text)
            answer = read_answer(ev, results)
            correct = (answer == ev.gold)
            agent.feedback(1.0 if correct else -1.0)
            by_tag[ev.tag][1] += 1
            by_tag[ev.tag][0] += int(correct)
            ctx_tokens += sum(len(r.item.text.split()) for r in results)
            n_q += 1
        agent.maintain()

    total_c = sum(v[0] for v in by_tag.values())
    total_n = sum(v[1] for v in by_tag.values())
    snap = agent.snapshot()
    return {
        "accuracy": total_c / total_n,
        "by_tag": {k: v[0] / v[1] for k, v in by_tag.items()},
        "avg_context_tokens": ctx_tokens / max(1, n_q),
        "final_items": snap["total"],
        "semantic_items": snap["semantic"],
        "written": snap["written"],
        "pruned": snap.get("pruned", 0),
        "superseded": snap.get("superseded", 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--length", type=int, default=2400)
    ap.add_argument("--json", default=None, help="also dump full per-seed results here")
    ap.add_argument("--no-log", action="store_true", help="skip appending to runs/log.jsonl")
    args = ap.parse_args()

    results = {}
    for name, fn in CONDITIONS.items():
        runs = [run_one(fn, s, args.length) for s in range(args.seeds)]
        agg = {
            "accuracy_mean": stats.mean(r["accuracy"] for r in runs),
            "accuracy_sd": stats.pstdev([r["accuracy"] for r in runs]),
            "context_tokens": stats.mean(r["avg_context_tokens"] for r in runs),
            "final_items": stats.mean(r["final_items"] for r in runs),
            "semantic_items": stats.mean(r["semantic_items"] for r in runs),
            "written": stats.mean(r["written"] for r in runs),
        }
        tags = sorted({t for r in runs for t in r["by_tag"]})
        agg["by_tag"] = {t: stats.mean(r["by_tag"].get(t, 0.0) for r in runs)
                         for t in tags}
        agg["_runs"] = runs
        results[name] = agg
        print(f"ran {name}")

    _print_table(results)

    # paired comparison against the baseline that matters
    base = [r["accuracy"] for r in results["keep_all"]["_runs"]]
    print("\nPaired vs keep_all (per-seed accuracy delta):")
    for name, agg in results.items():
        if name == "keep_all":
            continue
        deltas = [a["accuracy"] - b for a, b in zip(agg["_runs"], base)]
        wins = sum(1 for d in deltas if d > 0)
        print(f"  {name:<26} mean {stats.mean(deltas):+.3f}  "
              f"sd {stats.pstdev(deltas):.3f}  wins {wins}/{len(deltas)}")
        agg["paired_vs_keep_all"] = {"mean": stats.mean(deltas),
                                     "sd": stats.pstdev(deltas), "wins": wins}

    if not args.no_log:
        summary = {n: {k: v for k, v in a.items() if k != "_runs"}
                   for n, a in results.items()}
        e = runlog.append("ablation", {"seeds": args.seeds, "length": args.length}, summary)
        print(f"\nlogged to runs/log.jsonl (commit {e['git_sha'] and e['git_sha'][:8]}"
              f"{', DIRTY tree' if e['dirty'] else ''})")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)


def _print_table(results):
    tags = sorted({t for a in results.values() for t in a["by_tag"]})
    head = (f"\n{'condition':<26}{'acc':>8}{'sd':>7}" +
            "".join(f"{t:>16}" for t in tags) +
            f"{'ctx tok':>10}{'items':>9}{'sem':>6}")
    print(head)
    print("-" * len(head))
    for name, a in results.items():
        row = f"{name:<26}{a['accuracy_mean']:>8.3f}{a['accuracy_sd']:>7.3f}"
        row += "".join(f"{a['by_tag'].get(t, 0):>16.3f}" for t in tags)
        row += (f"{a['context_tokens']:>10.1f}{a['final_items']:>9.0f}"
                f"{a['semantic_items']:>6.0f}")
        print(row)


if __name__ == "__main__":
    main()
