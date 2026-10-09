# hippo: working notes for Claude

Agent memory that writes episodes, consolidates them offline, learns from
outcomes, and forgets. Read README.md for usage and RESULTS.md for what each
mechanism measurably contributes. This file covers how to work on it without
breaking it, and what is still open.

## Commands (run from `hippo/`)

```
python -m unittest discover -s tests -t .        # all tests, ~20s, stdlib only
python -m eval.explain --seed 0 [--tag unreliable] [--condition no_reinforcement]
python -m eval.run_ablations --seeds 10          # ~3 min, appends to runs/log.jsonl
python -m eval.decay_sweep --seeds 10            # ~3 min, appends to runs/log.jsonl
python -m viz.export && node viz/parity.js && python -m viz.build
```

Only dependency: numpy. Tests use `unittest`, not pytest, on purpose: no
extra install, and CI and local runs behave the same.

## Invariants (CI enforces each one)

1. **Python is the source of truth.** `viz/engine.js` is a line-for-line
   port. Change Python first, then mirror it in JS. `viz/parity.js` must
   report OK: same retrieved ids, same scores to 1e-9, same final store.
2. **Float evaluation order in `retrieval.score_all` is load-bearing.** It is
   mirrored in JS. Reordering operations can flip near-ties.
3. **`viz/data.json` is a behaviour snapshot.** If
   `test_data_json_matches_engine_behaviour` fails, behaviour changed. If the
   change was intended: re-export, rebuild, re-run the ablations, and update
   RESULTS.md. Never re-export just to silence the test.
4. **`viz/hippo.html` is generated.** Edit `viz/page.html` and run
   `python -m viz.build`. Never edit hippo.html directly.
5. **Runs must be reproducible.** No iteration over sets of strings where
   order matters (sort first). No unseeded randomness. Tested across
   PYTHONHASHSEED values.
6. **Don't tune defaults on the benchmark.** The benchmark is also the test
   set. Report sweeps; don't adopt their best setting.
7. **Every number in RESULTS.md comes from a logged run.** Check
   `runs/log.jsonl` for the commit. Never quote a number from a dirty-tree run.

## Debugging workflow

Accuracy moved? Don't touch weights first. Run
`python -m eval.explain --seed 0` (add `--tag` to narrow it down). For each
wrong answer you get the retrieved memories, each score split into its
factors, where the best *correct* memory ranked, and a diagnosis naming the
factor that let the wrong value win. Fix the mechanism the diagnosis points
at, then add a test that fails without the fix.

If the user pastes JSON from the page's "Copy debug state" button, it holds
the run seed, the event index, any manual actions taken (`before_event`
indexes into the stream), the store stats, discredited claims, the selected
memory and the last 12 queries with what they retrieved. To reproduce it,
replay that seed's stream in Python, applying each manual action before its
event index.

## Open findings (evidence: `eval.explain --seed 0`)

- **Rumours win by being said last.** 17 of 26 errors on seed 0. Consolidation's
  "latest value wins" marks the true episodes superseded (x0.35), so a later
  rumour outranks them. Credit has to slowly discredit the rumour to undo it.
- **Single observations get promoted to semantic memory.** When contradiction
  resolution leaves fewer than `min_support` active episodes,
  `consolidate()` still abstracts them ("consolidated from 1 observations").
  A lone rumour then gets the semantic bonus and a starting strength of 1.4.
  This was intended for genuine updates but helps rumours too. Untested
  alternative: require min_support for new abstractions, and let resolution
  alone handle updates.
- **Credit punishes bystanders.** All top-k items share the blame for a
  wrong answer, including correct facts about *other* attributes (visible as
  reliability < 1 on unrelated claims). The reader knows which item it
  used, so credit could target that item. The design question is open:
  innocent co-retrieved items may still deserve a weaker signal.
- **Crowding out.** 9 of 26 errors: no memory for the queried attribute
  reaches the top 5, because noise about the same entity outranks it. This
  is mostly the hashed bag-of-words embedding. Try a real encoder before
  tuning anything else.
- **Rare facts are unprotected.** Nothing learns what to pin.
- **Decay is inert at default settings.** The stream (~820 ticks) is shorter
  than the half-life (900). See the decay sweep in RESULTS.md.

Any of these changes behaviour. Expect invariant 3 to fire, and follow it.

## Bugs fixed so far (each has a regression test)

| bug | test |
|---|---|
| decay re-applied the full age on every call | `DecayTests.test_repeated_step_at_same_clock_is_noop` |
| keyless clusters minted a duplicate abstraction each pass | `ConsolidationTests.test_keyless_cluster_does_not_duplicate_across_passes` |
| tie-break followed Python's randomised set order | `ConsolidationTests.test_tie_break_is_order_independent` |
| superseding cut strength permanently, truth could never win back | `ConsolidationTests.test_resolution_is_reversible_when_claim_is_discredited` |
| embedding clusters merged different entities | `ConsolidationTests.test_keyed_clusters_never_merge_entities` |
| parity/snapshot compared only right/wrong, missed weight drift | parity.js + data.json now pin ids and scores |

## Environment notes

- Claude cloud sessions: PyPI may be unreachable, so don't add test
  dependencies. Playwright works for screenshots (Chromium at
  /opt/pw-browsers).
- The live page is a claude.ai artifact published from `viz/hippo.html`.
  The URL is in the user's artifact gallery as "Hippo Memory Network".
