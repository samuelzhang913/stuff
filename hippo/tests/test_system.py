"""System-level tests: reproducibility, the headline claims, committed
artifacts being current, and the browser engine matching Python.

These are slower (a few seconds each). They guard the things RESULTS.md
claims, so if one fails, either fix the code or update RESULTS.md.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from eval import runlog
from eval.baselines import full, keep_all, no_reinforcement
from eval.explain import explain
from eval.run_ablations import run_one

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (0, 1, 2)


class Reproducibility(unittest.TestCase):
    def test_identical_across_python_hash_seeds(self):
        # Regression: string-set iteration order made runs non-reproducible.
        code = ("import json; from eval.baselines import full; "
                "from eval.run_ablations import run_one; "
                "print(json.dumps(run_one(full, 0, 2400), sort_keys=True))")
        outs = []
        for hs in ("0", "12345"):
            env = {**os.environ, "PYTHONHASHSEED": hs}
            outs.append(subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                                       capture_output=True, text=True, check=True).stdout)
        self.assertEqual(outs[0], outs[1])


class HeadlineClaims(unittest.TestCase):
    """Small-sample guards on RESULTS.md. The 10-seed numbers live there."""

    @classmethod
    def setUpClass(cls):
        cls.runs = {name: [run_one(fn, s, 2400) for s in SEEDS]
                    for name, fn in [("full", full), ("keep_all", keep_all),
                                     ("no_reinforcement", no_reinforcement)]}

    def mean(self, name, key, tag=None):
        rs = self.runs[name]
        vals = [r["by_tag"][tag] if tag else r[key] for r in rs]
        return sum(vals) / len(vals)

    def test_full_beats_keep_all(self):
        self.assertGreater(self.mean("full", "accuracy"), self.mean("keep_all", "accuracy") + 0.05)

    def test_consolidation_fixes_changed_values(self):
        self.assertGreater(self.mean("full", None, "volatile_post"),
                           self.mean("keep_all", None, "volatile_post") + 0.3)

    def test_outcome_credit_is_what_handles_rumours(self):
        self.assertGreater(self.mean("full", None, "unreliable"),
                           self.mean("no_reinforcement", None, "unreliable") + 0.2)


class Artifacts(unittest.TestCase):
    def test_data_json_matches_engine_behaviour(self):
        # viz/data.json is a behaviour snapshot. If this fails the engine's
        # behaviour changed: if intended, run `python -m viz.export`,
        # `python -m viz.build`, re-run the ablations and update RESULTS.md.
        # Exact on everything discrete (events, answers, retrieved ids, final
        # counts); scores to 1e-9 relative, because a different numpy/BLAS
        # build can legitimately differ in the last floating-point bit.
        from viz import export
        got = json.loads(export.render())
        want = json.loads(export.OUT.read_text())
        msg = "engine behaviour changed; see comment in this test"
        self.assertEqual(got.keys(), want.keys(), msg)
        for seed, w in want["seeds"].items():
            g = got["seeds"][seed]
            for field in ("events", "reference", "reference_final"):
                self.assertEqual(g[field], w[field], f"{msg} (seed {seed}, {field})")
            self.assertEqual(len(g["reference_retrieved"]), len(w["reference_retrieved"]), msg)
            for qi, (gq, wq) in enumerate(zip(g["reference_retrieved"], w["reference_retrieved"])):
                self.assertEqual([x[0] for x in gq], [x[0] for x in wq],
                                 f"{msg} (seed {seed}, query {qi}: retrieved ids)")
                for (_, gs), (_, ws) in zip(gq, wq):
                    self.assertLessEqual(abs(gs - ws), 1e-9 * max(1.0, abs(ws)),
                                         f"{msg} (seed {seed}, query {qi}: score)")

    def test_hippo_html_is_built_from_current_sources(self):
        from viz import build
        self.assertEqual(build.render(), build.OUT.read_text(),
                         "viz/hippo.html is stale: run `python -m viz.build`")

    def test_js_engine_matches_python(self):
        node = shutil.which("node")
        if node is None:
            if os.environ.get("REQUIRE_NODE"):
                self.fail("node not found but REQUIRE_NODE is set")
            self.skipTest("node not installed")
        r = subprocess.run([node, "viz/parity.js"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


class Tooling(unittest.TestCase):
    def test_explain_agrees_with_the_run(self):
        recs = explain("full", seed=0)
        wrong = sum(1 - x for x in json.loads(
            (ROOT / "viz" / "data.json").read_text())["seeds"]["0"]["reference"])
        self.assertEqual(len(recs), wrong)
        for r in recs:
            self.assertNotEqual(r["answer"], r["gold"])
            self.assertTrue(r["diagnosis"])

    def test_runlog_appends_and_reads(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "log.jsonl"
            runlog.append("test", {"a": 1}, {"acc": 0.5}, path=p)
            runlog.append("test", {"a": 2}, {"acc": 0.6}, path=p)
            rows = runlog.read(p)
            self.assertEqual([r["args"]["a"] for r in rows], [1, 2])
            self.assertIn("git_sha", rows[0])

    def test_writing_the_log_does_not_make_the_tree_dirty(self):
        # Regression: the second of two back-to-back runs was flagged dirty
        # because the first run's log entry counted as an uncommitted change.
        if runlog._git("rev-parse", "HEAD") is None:
            self.skipTest("not a git checkout")
        before = runlog.code_version()["dirty"]
        probe = runlog.ROOT / "runs" / "_dirty_probe.jsonl"
        try:
            runlog.append("probe", {}, {}, path=probe)
            self.assertEqual(runlog.code_version()["dirty"], before)
        finally:
            probe.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
