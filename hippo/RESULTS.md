# Results

10 seeds × 8 conditions, ~820 events per run (≈700 observations, ≈120 queries).
Deterministic reader, so accuracy measures retrieval ranking alone.
Reproduce with `python -m eval.run_ablations --seeds 10` and `python -m eval.decay_sweep`.
Logged in `runs/log.jsonl`: ablation at commit `ec547a57`, decay sweep at `023a50b6`,
both from clean trees. The engine code is identical at both commits.

## Ablation

| condition | acc | stable | volatile (after change) | unreliable | rare | ctx tok/query | live items |
|---|---|---|---|---|---|---|---|
| keep_all (flat RAG) | 0.639 | 0.950 | 0.358 | 0.544 | 0.520 | 28.8 | 724 |
| keep_all + recency | 0.634 | 0.950 | 0.450 | 0.491 | 0.300 | 28.6 | 724 |
| sliding window (300) | 0.586 | 0.912 | 0.775 | 0.178 | 0.000 | 29.7 | 300 |
| **full** | **0.782** | 0.946 | 0.946 | 0.525 | 0.560 | 31.9 | 760 |
| no decay | 0.781 | 0.942 | 0.938 | 0.528 | 0.580 | 31.8 | 767 |
| no consolidation | 0.662 | 0.912 | 0.479 | 0.566 | 0.420 | 28.6 | 724 |
| no reinforcement | 0.645 | 0.971 | 0.958 | 0.078 | 0.580 | 32.0 | 767 |
| frequency reinforcement | 0.648 | 0.975 | 0.954 | 0.084 | 0.600 | 31.9 | 767 |

Paired per-seed delta vs keep_all: full +0.143 (sd 0.093, better on 9/10 seeds).

## Decay sweep (full system, episodic half-life in ticks)

| half-life | acc | stable | volatile | unreliable | rare | live items | pruned |
|---|---|---|---|---|---|---|---|
| off | 0.781 | 0.942 | 0.938 | 0.528 | 0.580 | 767 | 0 |
| 900 (default) | 0.782 | 0.946 | 0.946 | 0.525 | 0.560 | 760 | 7 |
| 300 | 0.795 | 0.958 | 0.954 | 0.556 | 0.500 | 522 | 245 |
| 150 | 0.795 | 0.958 | 0.967 | 0.566 | 0.380 | 308 | 458 |
| 75 | 0.726 | 0.967 | 0.933 | 0.434 | 0.000 | 221 | 553 |
| 40 | 0.557 | 0.929 | 0.787 | 0.059 | 0.000 | 147 | 634 |

## What the numbers say

1. **The system beats the baseline that matters**: +0.14 over keep-everything
   retrieval, consistent across seeds.

2. **Consolidation does most of the work.** Its main contribution is
   resolving contradictions. After a fact changes, keep_all answers correctly
   36% of the time and the full system 95%. Without consolidation, the gain
   over keep_all shrinks to +0.02.

3. **Outcome-based credit matters, but mostly to undo damage done by
   consolidation.** "Latest value wins" hands a contested fact to whoever
   spoke last, so with credit off the unreliable-source case falls to 8%.
   Outcome feedback discredits the rumour and brings it back to 53%. That is
   only level with keep_all (54%), which never committed to a winner at all.
   The two mechanisms have to come as a pair.

4. **Strengthening on use does nothing.** Frequency reinforcement scores
   0.648 against 0.645 with no reinforcement. Reinforcement helps only when
   it is tied to outcomes. This is the self-confirmation problem, measured.

5. **Forgetting buys storage, not accuracy or context, and the price is rare
   facts.** At the default half-life nothing is pruned, so decay was inert in
   the ablation. Faster decay (150–300) shrinks the store by up to 60%, adds
   about +0.014 accuracy, and costs 20 points on rare facts. Faster still, it
   wipes out rare facts completely. Context per query never drops, because
   retrieval is top-k either way, and abstractions are wordier than episodes.
   As a result the full system uses about 10% *more* context than keep_all.

6. **Rare-but-critical facts remain unprotected.** No setting here keeps them
   above 60%. Sole-support protection and pinning exist, but nothing here
   learns which facts to pin.

## Limits of this evaluation

- Synthetic data. Facts arrive with clean (entity, attribute, value) slots.
  A real system needs an extraction model, and extraction errors would hit
  consolidation hardest.
- The reward is perfect and immediate. In real agent work it is delayed,
  sparse and noisy, and that is where credit assignment actually gets hard.
  These results show the mechanism works when the signal exists. They say
  nothing about getting the signal.
- Hashed bag-of-words embeddings and a deterministic reader. Both are the
  same across conditions, but absolute numbers would move with a real encoder
  and an LLM reader.
- Defaults were not tuned on this benchmark. The sweep's best half-life
  (150–300) is reported, not adopted, because adopting it would mean tuning
  on the test set.
