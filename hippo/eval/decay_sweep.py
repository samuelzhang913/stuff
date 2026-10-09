"""Does forgetting help? Sweep the episodic half-life on the full system.

    python -m eval.decay_sweep [--seeds N]

The main ablation showed decay contributing nothing, because at the default
half-life almost nothing is ever pruned. This sweeps it down until forgetting
actually happens, to see what forgetting buys and what it breaks.
"""
import argparse
import statistics as stats

from eval import runlog
from eval.baselines import full
from eval.run_ablations import run_one

HALFLIVES = [None, 900, 300, 150, 75, 40]
TAGS = ["stable", "volatile_post", "unreliable", "rare"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--no-log", action="store_true", help="skip appending to runs/log.jsonl")
    args = ap.parse_args()
    print(f"{'halflife':>9}{'acc':>8}" + "".join(f"{t:>15}" for t in TAGS) +
          f"{'live items':>12}{'pruned':>9}{'ctx tok':>9}")
    summary = {}
    for hl in HALFLIVES:
        def cfg(hl=hl):
            c = full()
            if hl is None:
                c.decay.enabled = False
            else:
                c.decay.episodic_halflife = hl
            return c
        runs = [run_one(cfg, s, 2400) for s in range(args.seeds)]
        row = {
            "accuracy": stats.mean(r["accuracy"] for r in runs),
            "by_tag": {t: stats.mean(r["by_tag"].get(t, 0) for r in runs) for t in TAGS},
            "final_items": stats.mean(r["final_items"] for r in runs),
            "pruned": stats.mean(r["pruned"] for r in runs),
            "context_tokens": stats.mean(r["avg_context_tokens"] for r in runs),
        }
        summary["off" if hl is None else str(hl)] = row
        print(f"{'off' if hl is None else hl:>9}{row['accuracy']:>8.3f}"
              + "".join(f"{row['by_tag'][t]:>15.3f}" for t in TAGS)
              + f"{row['final_items']:>12.0f}{row['pruned']:>9.0f}{row['context_tokens']:>9.1f}",
              flush=True)
    if not args.no_log:
        e = runlog.append("decay_sweep", {"seeds": args.seeds, "length": 2400}, summary)
        print(f"\nlogged to runs/log.jsonl (commit {e['git_sha'] and e['git_sha'][:8]}"
              f"{', DIRTY tree' if e['dirty'] else ''})")


if __name__ == "__main__":
    main()
