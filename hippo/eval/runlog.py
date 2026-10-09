"""Append-only experiment log: runs/log.jsonl.

Every ablation or sweep appends one line: when, which code (git commit and
whether the tree had uncommitted changes), the arguments, and the summary
numbers. A later session can then compare against earlier results instead of
trusting a number quoted from memory. A run from a dirty tree is logged but
flagged, because it cannot be reproduced from the commit alone.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "runs" / "log.jsonl"


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip()
    except Exception:
        return None


def code_version():
    sha = _git("rev-parse", "HEAD")
    # The log itself lives in runs/, so writing one entry must not make the
    # next run look dirty. Only code changes count.
    status = _git("status", "--porcelain", "--", ".", ":(exclude)runs")
    return {"git_sha": sha, "dirty": bool(status) if status is not None else None}


def append(kind: str, args: dict, results: dict, path: Path = LOG) -> dict:
    entry = {
        "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "kind": kind,
        **code_version(),
        "args": args,
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def read(path: Path = LOG) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
