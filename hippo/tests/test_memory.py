"""Unit tests for the memory engine. Each regression test names the bug it
pins, so a failure points straight at what came back.

    python -m unittest discover -s tests -v
"""

import math
import os
import subprocess
import sys
import unittest
from pathlib import Path

from memory import MemoryAgent, MemoryStore
from memory import consolidate as C
from memory import decay as D
from memory.retrieval import RetrievalConfig, retrieve, score_all

ROOT = Path(__file__).resolve().parent.parent


class DecayTests(unittest.TestCase):
    def test_repeated_step_at_same_clock_is_noop(self):
        # Regression: decay once used age-since-last-access on every call,
        # so calling step() every tick re-applied the whole interval each time.
        s = MemoryStore()
        it = s.write("auth-service region is us-east-1")
        s.clock = 100
        cfg = D.DecayConfig()
        D.step(s, cfg)
        after_first = it.strength
        D.step(s, cfg)
        self.assertEqual(it.strength, after_first)

    def test_one_step_matches_formula(self):
        s = MemoryStore()
        it = s.write("auth-service region is us-east-1")
        s.clock = 100
        cfg = D.DecayConfig()
        D.step(s, cfg)
        hl = cfg.episodic_halflife * (1 + cfg.strength_protection * 1.0)
        self.assertAlmostEqual(it.strength, math.exp(-math.log(2) * 100 / hl), places=12)

    def test_pinned_and_sole_support_survive_pruning(self):
        s = MemoryStore()
        pinned = s.write("critical fact", pinned=True)
        sole = s.write("only evidence")
        weak = s.write("unsupported")
        s.write("abstraction", layer="semantic", support=[sole.id])
        for it in (pinned, sole, weak):
            it.strength = 0.01
        D.step(s, D.DecayConfig())
        self.assertTrue(pinned.alive)
        self.assertTrue(sole.alive)
        self.assertFalse(weak.alive)


class ConsolidationTests(unittest.TestCase):
    def _store_with(self, entries):
        s = MemoryStore()
        for text, slot in entries:
            s.tick()
            s.write(text, slot=slot)
        return s

    def test_keyless_cluster_does_not_duplicate_across_passes(self):
        # Regression: keyless clusters minted a new abstraction on every pass.
        s = self._store_with([(f"auth-service restarted cleanly at {n}", None)
                              for n in (101, 102, 103, 104)])
        cfg = C.ConsolidationConfig()
        for _ in range(3):
            C.consolidate(s, cfg)
            s.clock += 50
        keyless_sem = [i for i in s.live_items() if i.layer == "semantic" and i.slot is None]
        self.assertEqual(len(keyless_sem), 1)

    def test_keyed_clusters_never_merge_entities(self):
        a = [("auth-service region is us-east-1", ("auth-service", "region", "us-east-1"))] * 3
        b = [("ledger-api region is us-east-1", ("ledger-api", "region", "us-east-1"))] * 3
        s = self._store_with(a + b)
        C.consolidate(s, C.ConsolidationConfig())
        keys = sorted(i.key for i in s.live_items() if i.layer == "semantic")
        self.assertEqual(keys, [("auth-service", "region"), ("ledger-api", "region")])

    def test_latest_value_wins_without_feedback(self):
        slot = lambda v: ("auth-service", "region", v)
        s = self._store_with([("old", slot("us-east-1"))] * 2 + [("new", slot("eu-central-1"))] * 3)
        C.consolidate(s, C.ConsolidationConfig())
        sem = [i for i in s.live_items() if i.layer == "semantic"]
        self.assertEqual([i.value for i in sem], ["eu-central-1"])

    def test_resolution_is_reversible_when_claim_is_discredited(self):
        # Regression: superseding used to cut strength permanently, so a
        # truth wrongly superseded by a rumour could never win back.
        slot = lambda v: ("auth-service", "region", v)
        s = self._store_with([("truth", slot("us-east-1"))] * 2 + [("rumour", slot("sa-east-1"))] * 3)
        cfg = C.ConsolidationConfig()
        C.consolidate(s, cfg)
        s.value_reliability[slot("sa-east-1")] = 0.5
        s.clock += 50
        C.consolidate(s, cfg)
        epi = [i for i in s.live_items() if i.layer == "episodic"]
        self.assertTrue(all(i.superseded_by is None for i in epi if i.value == "us-east-1"))
        self.assertTrue(all(i.superseded_by is not None for i in epi if i.value == "sa-east-1"))
        sem = [i for i in s.live_items() if i.layer == "semantic"]
        self.assertEqual([i.value for i in sem], ["us-east-1"])

    def test_tie_break_is_order_independent(self):
        # Regression: max() over a set of strings broke ties in hash order,
        # which Python randomises per process. A single in-process run passes
        # by luck half the time, so check across several hash seeds.
        code = (
            "from memory import MemoryStore; from memory import consolidate as C\n"
            "slot = lambda v: ('auth-service', 'region', v)\n"
            "s = MemoryStore()\n"
            "for t, v in [('a', 'us-west-2'), ('b', 'ap-south-1'), ('c', 'eu-central-1')]:\n"
            "    s.tick(); s.write(t, slot=slot(v))\n"
            "s.value_reliability[slot('eu-central-1')] = 0.0\n"   # latest, discredited
            "C.consolidate(s, C.ConsolidationConfig())\n"
            "print(sorted({i.value for i in s.live_items()"
            " if i.layer == 'episodic' and i.superseded_by is None}))\n")
        for hs in range(8):
            out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                                 text=True, check=True,
                                 env={**os.environ, "PYTHONHASHSEED": str(hs)}).stdout.strip()
            self.assertEqual(out, "['ap-south-1']", f"PYTHONHASHSEED={hs}")  # first in sorted order


class CreditTests(unittest.TestCase):
    def _agent_with_fact(self):
        a = MemoryAgent()
        a.observe("auth-service region is us-east-1", slot=("auth-service", "region", "us-east-1"))
        a.recall("what region is auth-service in?")
        return a

    def test_negative_feedback_discredits_the_claim(self):
        a = self._agent_with_fact()
        it = next(iter(a.store.items.values()))
        before = it.strength
        a.feedback(-1.0)
        self.assertLess(a.store.reliability(("auth-service", "region", "us-east-1")), 1.0)
        self.assertLess(it.strength, before)

    def test_reliability_is_capped_at_one(self):
        a = self._agent_with_fact()
        for _ in range(20):
            a.feedback(1.0)
        self.assertEqual(a.store.reliability(("auth-service", "region", "us-east-1")), 1.0)

    def test_credit_disabled_changes_nothing(self):
        a = self._agent_with_fact()
        a.cfg.credit.enabled = False
        it = next(iter(a.store.items.values()))
        before = it.strength
        a.feedback(-1.0)
        self.assertEqual(it.strength, before)
        self.assertEqual(a.store.value_reliability, {})


class RetrievalTests(unittest.TestCase):
    def test_retrieve_is_top_k_of_score_all(self):
        s = MemoryStore()
        for n in range(30):
            s.tick()
            s.write(f"service-{n % 7} queue depth is {n}")
        cfg = RetrievalConfig()
        ranked = [r.item.id for r in score_all(s, "queue depth for service-3", cfg)]
        got = [r.item.id for r in retrieve(s, "queue depth for service-3", cfg)]
        self.assertEqual(got, ranked[: cfg.k])

    def test_score_all_has_no_side_effects(self):
        s = MemoryStore()
        it = s.write("auth-service region is us-east-1")
        s.clock = 10
        score_all(s, "auth-service region", RetrievalConfig())
        self.assertEqual((it.access_count, it.t_last_access), (0, 0))

    def test_terms_reproduce_score(self):
        s = MemoryStore()
        s.write("auth-service region is us-east-1", slot=("auth-service", "region", "us-east-1"))
        s.clock = 40
        r = score_all(s, "auth-service region", RetrievalConfig())[0]
        t = r.terms
        expect = (t["relevance"] * (1 + t["recency"] + t["strength"] + t["semantic"])
                  * t["superseded_factor"] * t["reliability_factor"])
        self.assertAlmostEqual(r.score, expect, places=12)


if __name__ == "__main__":
    unittest.main()
