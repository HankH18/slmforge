#!/usr/bin/env python3
"""Scenario sampling for the training set: WIDE, train-split-only, disjoint.

Four properties, each of which is enforced here and re-checked from the written
file by :mod:`datagen.audit`.

TRAIN SPLIT ONLY
    ``banned_punctuation`` and ``sentence_initial_letter`` are held out so the
    eval can predict performance on constraint types the model never trained on.
    :func:`assert_train_split_only` raises on either. It is called on every set
    this module builds, again before any row is written, and a third time by the
    auditor -- three independent chances to catch it, because one training row
    of a held-out type quietly destroys the generalization measurement and
    nothing downstream would look wrong.

WIDE SAMPLING
    Type share is measured over constraint SLOTS: occurrences of a type divided
    by the total number of constraints across all examples. (Measuring it over
    examples cannot work: with mean k near 3, the per-example shares of five
    types sum to ~300%, so a 25% cap would be unsatisfiable by arithmetic rather
    than by sampling.) The types are dealt from a reshuffled round-robin, which
    puts every train type at ~1/5 = 20% of slots, comfortably inside the cap,
    and guarantees the k types within one set are distinct.

K AND PRESSURE MATCH THE EVAL DISTRIBUTION
    Both are RECOMPUTED from ``data/scenarios/corpus.jsonl`` at run time rather
    than hardcoded, so if the corpus is rebuilt with a different mix the
    training set follows it instead of silently drifting. A proportional plan
    allocates them, with one of each k forced in when the batch is big enough
    to hold one, so a small batch still spans k = 2, 3 and 4.

DISJOINTNESS
    Tasks come from :mod:`datagen.task_pool`, which is fresh. This module
    additionally refuses to emit a (task, constraint-set) pair that appears in
    the eval corpus, or twice in its own output.

A note on "negative": every train-split type is a prohibition -- do not use
these words, this letter, lists, quotation marks -- so the behaviour spec's
"at least one negative constraint" is satisfied structurally rather than by
luck. :data:`NEGATIVE_TYPES` states it explicitly anyway and every emitted set
is checked against it, so if a positive type were ever moved into the train
split the sampler would fail loudly rather than emit sets the spec does not
describe.

A note on "adversarial": in the eval corpus that label means BOTH distractor
pressure and an embedded nudge. Here it means the scenario carries an embedded
nudge naming a constraint type it actually has. Distractor pressure is reported
separately (``distractor`` in the sample meta) and comes from the lexicon match
between task and banned-words instance. The two are reported apart precisely so
neither is overstated.
"""
from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from constraints import registry
from scenarios.instances import INSTANCE_META
from scenarios.lexicons import GENERIC

from datagen import DatagenError
from datagen.task_pool import TASKS, Task, compatible_nudges, normalize_text

__all__ = ["TRAIN_TYPES", "HELDOUT_TYPES", "NEGATIVE_TYPES", "Sample",
           "assert_train_split_only", "assert_has_negative", "proportional_plan",
           "eval_k_weights", "eval_adversarial_weights", "build_scenarios",
           "pair_key", "eval_pair_keys", "type_shares"]

#: Types a training example may use, in registry order.
TRAIN_TYPES: Tuple[str, ...] = tuple(registry.types_by_split("train"))

#: Types a training example may NEVER use. The generalization measurement.
HELDOUT_TYPES: Tuple[str, ...] = tuple(registry.types_by_split("heldout_type"))

#: Types whose rendered instruction is a prohibition (Spec: >=1 per set).
#: ``sentence_initial_letter`` is the only registered type that is not, and it
#: is held out, so this is every train type -- stated rather than assumed.
NEGATIVE_TYPES: Tuple[str, ...] = tuple(
    t for t in registry.list_types() if t != "sentence_initial_letter")


class Sample(NamedTuple):
    """One sampled training scenario plus the authoring metadata behind it.

    ``scenario`` carries exactly the six keys the eval corpus uses, so it can be
    fed to the shared renderer, the checkers and the training-time verifier
    without special-casing. Everything else lives out here and lands in the
    row's ``provenance``.
    """
    scenario: Dict[str, Any]
    task_id: str
    lexicon: str
    nudge_id: Optional[str]
    types: Tuple[str, ...]
    distractor: bool


# --------------------------------------------------------------------- guards
def assert_train_split_only(constraint_ids: Sequence[str],
                            lookup: Mapping[str, Mapping[str, Any]]) -> None:
    """Raise if any constraint is of a held-out type."""
    for cid in constraint_ids:
        con = lookup.get(cid)
        if con is None:
            raise DatagenError(f"constraint id {cid!r} is not in the instance library")
        ctype = con.get("type")
        if registry.split_of(ctype) != "train":
            raise DatagenError(
                f"constraint {cid!r} is of held-out type {ctype!r}. Held-out types "
                f"({', '.join(HELDOUT_TYPES)}) are the generalization measurement; "
                f"a training example carrying one destroys it.")


def assert_has_negative(constraint_ids: Sequence[str],
                        lookup: Mapping[str, Mapping[str, Any]]) -> None:
    """Raise unless at least one constraint is a negative one (behaviour spec)."""
    types = [lookup[cid]["type"] for cid in constraint_ids]
    if not any(t in NEGATIVE_TYPES for t in types):
        raise DatagenError(
            f"constraint set {list(constraint_ids)} has no negative constraint; the "
            f"behaviour spec requires at least one. Types were {types}.")


# ------------------------------------------------------- distribution planning
def proportional_plan(n: int, weights: Mapping[Any, int],
                      force_one_each: bool = True) -> List[Any]:
    """``n`` values drawn to match ``weights``, interleaved deterministically.

    Greedy largest-deficit allocation: at each step take the key whose share so
    far has fallen furthest behind its target. That makes every PREFIX of the
    plan approximately correct too, so a run that dies half way through still
    left a representative dataset behind.

    ``force_one_each`` seeds one of every key first (when ``n`` allows), so a
    small batch still spans every k rather than dropping the rarest one.
    """
    if n < 0:
        raise DatagenError(f"plan size must be non-negative, got {n}")
    keys = list(weights)
    if not keys:
        raise DatagenError("proportional_plan needs at least one key")
    total = sum(weights[k] for k in keys)
    if total <= 0:
        raise DatagenError(f"weights must sum to a positive number, got {dict(weights)}")
    share = {k: weights[k] / total for k in keys}

    plan: List[Any] = []
    counts = {k: 0 for k in keys}
    if force_one_each and n >= len(keys):
        for k in keys:
            plan.append(k)
            counts[k] += 1
    while len(plan) < n:
        step = len(plan) + 1
        pick = max(keys, key=lambda k: (share[k] * step - counts[k], -keys.index(k)))
        plan.append(pick)
        counts[pick] += 1
    return plan


def _weights_from_corpus(corpus_rows: Sequence[Mapping[str, Any]], key: str,
                         allowed: Sequence[Any]) -> Dict[Any, int]:
    counts = {value: 0 for value in allowed}
    for row in corpus_rows:
        value = row.get(key)
        if value in counts:
            counts[value] += 1
    if sum(counts.values()) == 0:
        raise DatagenError(
            f"the eval corpus has no usable {key!r} values, so the training "
            f"distribution cannot be matched to it")
    return counts


def eval_k_weights(corpus_rows: Sequence[Mapping[str, Any]]) -> Dict[int, int]:
    """k histogram of the eval corpus. Recomputed, never hardcoded."""
    return _weights_from_corpus(corpus_rows, "k", (2, 3, 4))


def eval_adversarial_weights(corpus_rows: Sequence[Mapping[str, Any]]) -> Dict[bool, int]:
    """clean/adversarial histogram of the eval corpus, as ``{False: n, True: n}``."""
    counts = _weights_from_corpus(corpus_rows, "pressure", ("clean", "adversarial"))
    return {False: counts["clean"], True: counts["adversarial"]}


# ------------------------------------------------------------------- sampling
def _lexicon_of(cid: str) -> str:
    meta = INSTANCE_META.get(cid)
    if meta is None:
        raise DatagenError(
            f"instance {cid!r} has no authoring metadata in scenarios.instances, so "
            f"its lexicon tag is unknown and it cannot be safely paired with a task")
    return meta.lexicon


def _candidates(ctype: str, task: Task,
                by_type: Mapping[str, List[str]]) -> Tuple[List[str], bool]:
    """Instance ids usable for ``ctype`` on ``task``, and whether they are topical.

    A topic-tagged instance may only go to a task carrying the same tag. That is
    what makes a lexical ban bite: the task pulls toward the banned vocabulary.
    Where no topical instance exists for the task's tag, the generic ones are
    used and the sample is simply not counted as carrying distractor pressure.
    """
    pool = by_type.get(ctype, [])
    if not pool:
        raise DatagenError(f"no train-split instance of type {ctype!r} in the library")
    if ctype != "banned_words" or task.lexicon == GENERIC:
        return [cid for cid in pool if _lexicon_of(cid) == GENERIC] or pool, False
    topical = [cid for cid in pool if _lexicon_of(cid) == task.lexicon]
    if topical:
        return topical, True
    return [cid for cid in pool if _lexicon_of(cid) == GENERIC] or pool, False


class _TypeDealer:
    """Round-robin dealer over the train types, reshuffled every full pass.

    Deals ``k`` DISTINCT types. A type that would repeat inside one set is
    deferred to the front of the next deal rather than dropped, so the long-run
    count per type stays level and the 25% cap holds by construction.
    """

    def __init__(self, rng: random.Random, types: Sequence[str]):
        self._rng = rng
        self._types = list(types)
        self._pool: List[str] = []

    def deal(self, k: int) -> Tuple[str, ...]:
        if k > len(self._types):
            raise DatagenError(
                f"cannot deal {k} distinct types from {len(self._types)} train types")
        chosen: List[str] = []
        deferred: List[str] = []
        while len(chosen) < k:
            if not self._pool:
                nxt = list(self._types)
                self._rng.shuffle(nxt)
                self._pool.extend(nxt)
            candidate = self._pool.pop(0)
            if candidate in chosen:
                deferred.append(candidate)
                continue
            chosen.append(candidate)
        self._pool[:0] = deferred
        return tuple(chosen)


def pair_key(task_text: str, constraint_ids: Sequence[str]) -> Tuple[str, Tuple[str, ...]]:
    """The identity of a (task, constraint-set) pair, for overlap tests.

    Constraint order does not change what is being asked of the model, so the
    ids are sorted: a set reordered is not a new pair, and treating it as one
    would let contamination through under a shuffle.
    """
    return (normalize_text(task_text), tuple(sorted(str(c) for c in constraint_ids)))


def eval_pair_keys(corpus_rows: Iterable[Mapping[str, Any]]) -> set:
    """Every (task, constraint-set) pair the eval corpus contains."""
    keys = set()
    for row in corpus_rows:
        task = row.get("task")
        cids = row.get("constraints")
        if isinstance(task, str) and isinstance(cids, (list, tuple)):
            keys.add(pair_key(task, cids))
    return keys


def type_shares(rows_types: Sequence[Sequence[str]]) -> Dict[str, float]:
    """type -> share of all constraint SLOTS. See the module docstring."""
    total = sum(len(types) for types in rows_types)
    if total == 0:
        return {}
    counts: Dict[str, int] = {}
    for types in rows_types:
        for t in types:
            counts[t] = counts.get(t, 0) + 1
    return {t: counts[t] / total for t in sorted(counts)}


def build_scenarios(n: int,
                    instances: Mapping[str, Mapping[str, Any]],
                    corpus_rows: Sequence[Mapping[str, Any]],
                    seed: int = 20260820,
                    tasks: Sequence[Task] = TASKS,
                    max_type_share: float = 0.25) -> List[Sample]:
    """``n`` training scenarios: fresh tasks x wide train-split constraint sets.

    Deterministic in ``seed``: the same seed and library give byte-identical
    scenarios, so a rerun re-uses the teacher cache instead of re-billing.
    """
    if n <= 0:
        raise DatagenError(f"asked for {n} scenarios; there is no such thing as an "
                           f"empty training batch worth writing")
    if not tasks:
        raise DatagenError("the task pool is empty")

    by_type: Dict[str, List[str]] = {t: [] for t in TRAIN_TYPES}
    for cid, con in sorted(instances.items()):
        ctype = con.get("type")
        if ctype in by_type:
            by_type[ctype].append(cid)
    missing = sorted(t for t, ids in by_type.items() if not ids)
    if missing:
        raise DatagenError(f"the instance library has no instances of train types {missing}")

    rng = random.Random(seed)
    dealer = _TypeDealer(rng, TRAIN_TYPES)
    k_plan = proportional_plan(n, eval_k_weights(corpus_rows))
    adv_plan = proportional_plan(n, eval_adversarial_weights(corpus_rows))

    task_order = list(range(len(tasks)))
    rng.shuffle(task_order)

    forbidden = eval_pair_keys(corpus_rows)
    seen: set = set()
    samples: List[Sample] = []

    for i in range(n):
        task = tasks[task_order[i % len(tasks)]]
        k = k_plan[i]
        sample = None
        types: Tuple[str, ...] = ()
        for attempt in range(24):
            # Re-deal the TYPES only every eighth retry: the first thing to vary
            # on a duplicate pair is which instances fill the same type slots,
            # because re-dealing types perturbs the round-robin that keeps the
            # type shares level.
            if attempt % 8 == 0:
                types = dealer.deal(k)
            cids, distractor = [], False
            for ctype in types:
                pool, topical = _candidates(ctype, task, by_type)
                cids.append(rng.choice(pool))
                distractor = distractor or topical
            assert_train_split_only(cids, instances)
            assert_has_negative(cids, instances)

            text, nudge_id = task.text, None
            if adv_plan[i]:
                options = compatible_nudges(types)
                if options:
                    nudge = rng.choice(options)
                    text = f"{task.text} {nudge.text}"
                    nudge_id = nudge.id

            key = pair_key(text, cids)
            if key in forbidden:
                raise DatagenError(
                    f"sampled pair {key[1]} on task {task.id} appears in the EVAL "
                    f"corpus. Training on it is contamination of the graded number.")
            if key in seen:
                continue
            seen.add(key)
            sample = Sample(
                scenario={"scenario_id": f"dg-{i + 1:05d}", "task": text,
                          "constraints": list(cids), "k": k,
                          "pressure": "adversarial" if nudge_id else "clean",
                          "split": "train"},
                task_id=task.id, lexicon=task.lexicon, nudge_id=nudge_id,
                types=tuple(types), distractor=distractor)
            break
        if sample is None:
            raise DatagenError(
                f"could not find an unused (task, constraint-set) pair for task "
                f"{task.id} after 24 attempts; the requested n is too large for the "
                f"pool of {len(tasks)} tasks x this instance library")
        samples.append(sample)

    shares = type_shares([s.types for s in samples])
    over = {t: round(v, 4) for t, v in shares.items() if v > max_type_share}
    if over:
        raise DatagenError(
            f"constraint type share exceeds the {max_type_share:.0%} cap: {over}. "
            f"A dataset dominated by one type teaches that type, not the behaviour.")
    return samples
