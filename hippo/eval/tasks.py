"""Synthetic multi-episode benchmark.

The point of the task is that later performance *depends on earlier episodes*,
which is the only setting where a memory system can beat a bigger context
window. Four fact types, each probing a different claim the design makes:

  stable    A fact restated many times in varied phrasings.
            -> consolidation should collapse these; the win is context cost,
               not accuracy, and the benchmark should say so.

  volatile  A fact whose value changes partway through. Queries after the
            change require the NEW value.
            -> probes contradiction handling. A keep-everything store has
               both values sitting in it with equal standing.

  rare      A fact stated exactly once, early, and queried very late, with
            thousands of ticks of unrelated traffic in between.
            -> the adversarial case for any forgetting policy. If decay is
               tuned for the stable and volatile cases, this is what it
               breaks.

  unreliable  A true value stated twice, early. A wrong value then asserted
            repeatedly and more recently by an unreliable source. Queried many
            times across the run.
            -> the only fact type where outcome feedback can help. Recency
               picks the rumour. "Latest wins" contradiction resolution picks
               the rumour. Nothing in the text marks which source is good.
               Only repeated negative reward on the rumour can fix it, so this
               is the case that tests credit assignment rather than retrieval.

  noise     Filler observations. Volume, so retrieval is not trivial.

The reader is deterministic: walk the retrieved items in rank order, take the
first one that carries a value for the queried (entity, attribute). This makes
ranking quality the thing under test, with no LLM variance in the measurement.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

ENTITIES = [
    "auth-service", "billing-worker", "ingest-pipeline", "search-index",
    "notify-daemon", "ledger-api", "image-resizer", "session-cache",
    "report-builder", "webhook-relay", "quota-manager", "trace-collector",
]

ATTRIBUTES = {
    "region": ["us-east-1", "us-west-2", "eu-central-1", "ap-south-1", "sa-east-1"],
    "owner": ["platform-team", "payments-team", "data-team", "growth-team"],
    "runtime": ["python-3.11", "go-1.22", "node-20", "rust-1.77"],
    "datastore": ["postgres-14", "dynamo", "clickhouse", "redis-7"],
}

OBS_TEMPLATES = [
    "{e} {a} is {v}",
    "noted that {e} has {a} set to {v}",
    "deployment check: {e} {a} = {v}",
    "the {a} for {e} remains {v}",
    "config audit shows {e} running {a} {v}",
    "{e} was confirmed to use {v} as its {a}",
]

CHANGE_TEMPLATES = [
    "migration complete: {e} {a} is now {v}",
    "{e} {a} changed to {v}",
    "cutover done, {e} {a} is {v} going forward",
]

QUERY_TEMPLATES = [
    "what is the {a} of {e}?",
    "{e} {a}?",
    "remind me which {a} {e} uses",
]

NOISE_TEMPLATES = [
    "{e} restarted cleanly at {n}",
    "latency on {e} measured {n} ms",
    "{e} processed {n} jobs this hour",
    "routine health check passed for {e}",
    "{e} log rotation completed, {n} files archived",
    "queue depth for {e} is {n}",
]


@dataclass
class Event:
    kind: str              # 'obs' | 'query'
    text: str
    slot: tuple | None = None
    gold: str | None = None
    tag: str | None = None


RUMOUR_TEMPLATES = [
    "unconfirmed chatter says {e} {a} is {v}",
    "someone mentioned {e} might be on {v} for {a}",
    "second-hand note: {e} {a} {v}",
]


def build_benchmark(seed=0, n_stable=6, n_volatile=6, n_rare=5,
                    n_unreliable=4, length=2400, noise_ratio=0.72):
    """Return (events, meta). Deterministic in `seed`."""
    rng = random.Random(seed)
    pool = ENTITIES[:]
    rng.shuffle(pool)

    # assign distinct (entity, attribute) keys to each fact type
    keys, used = [], set()
    for ent in pool:
        for attr in ATTRIBUTES:
            if (ent, attr) not in used:
                keys.append((ent, attr))
                used.add((ent, attr))
    rng.shuffle(keys)

    need = n_stable + n_volatile + n_rare + n_unreliable
    assert len(keys) >= need, "not enough entity/attribute pairs"
    stable_keys = keys[:n_stable]
    volatile_keys = keys[n_stable:n_stable + n_volatile]
    rare_keys = keys[n_stable + n_volatile:n_stable + n_volatile + n_rare]
    unreliable_keys = keys[n_stable + n_volatile + n_rare:need]

    timeline: dict[int, list[Event]] = {}

    def place(t, ev):
        t = max(0, min(length - 1, int(t)))
        timeline.setdefault(t, []).append(ev)

    def obs(e, a, v, template_pool=OBS_TEMPLATES):
        return Event("obs", rng.choice(template_pool).format(e=e, a=a, v=v),
                     slot=(e, a, v))

    def query(e, a, gold, tag):
        return Event("query", rng.choice(QUERY_TEMPLATES).format(e=e, a=a),
                     slot=(e, a, None), gold=gold, tag=tag)

    meta = {"stable": [], "volatile": [], "rare": [], "unreliable": []}

    # --- stable: many restatements, queried throughout -------------------
    for (e, a) in stable_keys:
        v = rng.choice(ATTRIBUTES[a])
        meta["stable"].append((e, a, v))
        for _ in range(8):
            place(rng.uniform(0.02, 0.80) * length, obs(e, a, v))
        for _ in range(4):
            place(rng.uniform(0.35, 0.99) * length, query(e, a, v, "stable"))

    # --- volatile: value flips midway ------------------------------------
    for (e, a) in volatile_keys:
        v1, v2 = rng.sample(ATTRIBUTES[a], 2)
        t_change = rng.uniform(0.40, 0.58) * length
        meta["volatile"].append((e, a, v1, v2, t_change))
        for _ in range(5):                      # old value, stated often
            place(rng.uniform(0.02, 0.38) * length, obs(e, a, v1))
        place(t_change, obs(e, a, v2, CHANGE_TEMPLATES))
        for _ in range(2):                      # new value restated, but less
            place(rng.uniform(0.60, 0.80) * length, obs(e, a, v2))
        for _ in range(2):
            place(rng.uniform(0.10, 0.35) * length, query(e, a, v1, "volatile_pre"))
        for _ in range(4):
            place(rng.uniform(0.65, 0.99) * length, query(e, a, v2, "volatile_post"))

    # --- rare: said once, early; asked once, very late -------------------
    for (e, a) in rare_keys:
        v = rng.choice(ATTRIBUTES[a])
        meta["rare"].append((e, a, v))
        place(rng.uniform(0.03, 0.10) * length, obs(e, a, v))
        place(rng.uniform(0.88, 0.99) * length, query(e, a, v, "rare"))

    # --- unreliable: truth early, a louder and later wrong claim ----------
    for (e, a) in unreliable_keys:
        truth, rumour = rng.sample(ATTRIBUTES[a], 2)
        meta["unreliable"].append((e, a, truth, rumour))
        for _ in range(2):
            place(rng.uniform(0.02, 0.15) * length, obs(e, a, truth))
        for _ in range(6):
            place(rng.uniform(0.22, 0.85) * length,
                  obs(e, a, rumour, RUMOUR_TEMPLATES))
        for _ in range(8):
            place(rng.uniform(0.30, 0.99) * length,
                  query(e, a, truth, "unreliable"))

    # --- noise ------------------------------------------------------------
    n_signal = sum(len(v) for v in timeline.values())
    n_noise = int(n_signal * noise_ratio / max(1e-9, 1 - noise_ratio))
    for _ in range(n_noise):
        e = rng.choice(ENTITIES)
        place(rng.uniform(0, 1) * length,
              Event("obs", rng.choice(NOISE_TEMPLATES).format(
                  e=e, n=rng.randint(2, 9000))))

    events = []
    for t in sorted(timeline):
        bucket = timeline[t]
        rng.shuffle(bucket)
        events.extend(bucket)

    meta["n_events"] = len(events)
    meta["n_queries"] = sum(1 for e in events if e.kind == "query")
    return events, meta


def read_answer(query_event: Event, results) -> str | None:
    """Deterministic reader: first retrieved item carrying the queried slot."""
    e, a, _ = query_event.slot
    for r in results:
        it = r.item
        if it.slot is not None and it.slot[0] == e and it.slot[1] == a:
            return it.slot[2]
    return None
