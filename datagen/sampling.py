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

BANNED-LETTER REQUEST WEIGHTING
    Instance choice is uniform by default. For ``banned_letter`` that is a trap:
    the pipeline rejection-filters teacher output through the real checkers, and
    the teacher's success rate depends enormously on WHICH letter is banned, so
    a uniform REQUEST distribution produces a kept distribution dominated by the
    easy letters. ``letter_mix="hard"`` divides out that bias at request time.
    See :data:`LETTER_REQUEST_WEIGHTS` for the measurement and the arithmetic.

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
           "pair_key", "eval_pair_keys", "type_shares",
           "ENGLISH_LETTER_FREQUENCY", "V1_LETTER_OUTCOMES", "LETTER_KEEP_RATES",
           "LETTER_REQUEST_WEIGHTS", "LETTER_MIXES", "letter_of"]

#: Types a training example may use, in registry order.
TRAIN_TYPES: Tuple[str, ...] = tuple(registry.types_by_split("train"))

#: Types a training example may NEVER use. The generalization measurement.
HELDOUT_TYPES: Tuple[str, ...] = tuple(registry.types_by_split("heldout_type"))

#: Types whose rendered instruction is a prohibition (Spec: >=1 per set).
#: ``sentence_initial_letter`` is the only registered type that is not, and it
#: is held out, so this is every train type -- stated rather than assumed.
NEGATIVE_TYPES: Tuple[str, ...] = tuple(
    t for t in registry.list_types() if t != "sentence_initial_letter")


# ------------------------------------------------- banned-letter request weights
#: Standard English relative letter frequencies, for the ten letters the
#: ``banned_letter`` instance library covers. A property of the LANGUAGE, not of
#: this project's eval set -- which is the whole point: see
#: :data:`LETTER_REQUEST_WEIGHTS`.
ENGLISH_LETTER_FREQUENCY: Dict[str, float] = {
    "e": 0.127, "a": 0.082, "o": 0.075, "i": 0.070, "s": 0.063,
    "r": 0.060, "l": 0.040, "u": 0.028, "m": 0.024, "p": 0.019,
}

#: ``letter -> (kept, rejected)`` measured over the v1 training run
#: (``data/train/v1.jsonl`` and ``v1.rejected.jsonl``): occurrences of each
#: ``banned_letter`` instance among kept rows, and among rejection-ledger
#: records. Rejections are per ATTEMPT, not per scenario, which is why the
#: row totals differ per letter even though every letter was requested equally.
#: :func:`datagen.generate.summarise` re-measures exactly these two counts on
#: every run, so a later run can be compared against this baseline directly.
V1_LETTER_OUTCOMES: Dict[str, Tuple[int, int]] = {
    "e": (32, 166), "a": (33, 168), "i": (44, 113), "o": (48, 113),
    "r": (56, 125), "u": (44, 79), "s": (56, 64), "l": (83, 34),
    "p": (71, 27), "m": (78, 29),
}

#: ``letter -> kept / (kept + rejected)``, derived from :data:`V1_LETTER_OUTCOMES`.
LETTER_KEEP_RATES: Dict[str, float] = {
    letter: kept / (kept + rejected)
    for letter, (kept, rejected) in V1_LETTER_OUTCOMES.items()
}

#: ``letter -> probability of REQUESTING that letter`` under ``letter_mix="hard"``.
#:
#: THE PROBLEM. :func:`build_scenarios` picks an instance uniformly from the
#: pool, so every letter is requested equally often. But
#: :mod:`datagen.generate` rejection-filters the teacher's draft through the
#: real checkers, and writing without an ``e`` is far harder than writing
#: without a ``p``. Measured over the v1 run (:data:`V1_LETTER_OUTCOMES`):
#:
#:     letter   kept   rejected   reject rate   keep rate
#:       e        32       166        83.8%       0.1616
#:       a        33       168        83.6%       0.1642
#:       i        44       113        72.0%       0.2803
#:       o        48       113        70.2%       0.2981
#:       r        56       125        69.1%       0.3094
#:       u        44        79        64.2%       0.3577
#:       s        56        64        53.3%       0.4667
#:       l        83        34        29.1%       0.7094
#:       p        71        27        27.6%       0.7245
#:       m        78        29        27.1%       0.7290
#:
#: Uniform requests therefore left v1 at l=83, m=78, p=71 ... a=33, e=32. The
#: quality gate silently built a curriculum of EASY examples, and the model was
#: trained on the easy tail of the exact skill it is graded on: v1 satisfied
#: 1 of 105 ``banned_letter`` eval instances.
#:
#: THE FIX. Weight the REQUEST by how hard the letter is, so that what SURVIVES
#: the filter lands where we want it:
#:
#:     raw(letter)  = ENGLISH_LETTER_FREQUENCY[letter] / LETTER_KEEP_RATES[letter]
#:     weight       = raw(letter) / sum(raw)
#:
#: Since kept(letter) is proportional to request(letter) * keep_rate(letter), and
#: request is proportional to frequency / keep_rate, the keep rate cancels and the
#: KEPT distribution comes out proportional to English letter frequency. The
#: division is what compensates for the filter's bias against difficulty; without
#: it the gate re-imposes its own curriculum on whatever we ask for.
#:
#: WHY ENGLISH FREQUENCY, not our eval histogram. Our eval corpus happens to use
#: e(35), a(31), o(14), i(13), s(12), and copying that histogram would train
#: against our own answer key -- the improvement would be real on this eval and
#: unfalsifiable anywhere else. English letter frequency is a property of the
#: LANGUAGE: it predicts what any held-out eval built from English text will
#: emphasise, so a model trained to it transfers rather than memorises. The two
#: targets happen to agree in their ordering, which is exactly why the honest one
#: costs us nothing.
#:
#: Worked out, the weights (and the kept share each projects, at the v1 keep
#: rates) are:
#:
#:     letter   raw = freq/keep   weight    projected kept share   English share
#:       e          0.7858        0.3403           21.6%              21.6%
#:       a          0.4995        0.2163           13.9%              13.9%
#:       o          0.2516        0.1089           12.8%              12.8%
#:       i          0.2498        0.1082           11.9%              11.9%
#:       s          0.1350        0.0585           10.7%              10.7%
#:       r          0.1939        0.0840           10.2%              10.2%
#:       l          0.0564        0.0244            6.8%               6.8%
#:       u          0.0783        0.0339            4.8%               4.8%
#:       m          0.0329        0.0143            4.1%               4.1%
#:       p          0.0262        0.0114            3.2%               3.2%
#:
#: Derived here rather than pasted in, so that correcting a keep rate or adding a
#: letter to the library re-derives the weights instead of leaving a stale
#: literal behind. :func:`build_scenarios` additionally refuses to run in "hard"
#: mode if the library contains a ``banned_letter`` instance this table does not
#: cover, because a silently unweighted letter is the original bug again.
LETTER_REQUEST_WEIGHTS: Dict[str, float] = {}
_raw_letter_weights = {
    letter: ENGLISH_LETTER_FREQUENCY[letter] / LETTER_KEEP_RATES[letter]
    for letter in ENGLISH_LETTER_FREQUENCY
}
_raw_letter_total = sum(_raw_letter_weights.values())
LETTER_REQUEST_WEIGHTS.update(
    {letter: raw / _raw_letter_total
     for letter, raw in _raw_letter_weights.items()})
del _raw_letter_weights, _raw_letter_total

#: The instance-selection mixes ``build_scenarios`` accepts. ``uniform`` is the
#: default and is the historical behaviour, byte for byte.
LETTER_MIXES: Tuple[str, ...] = ("uniform", "hard")

#: The one type whose instance choice ``letter_mix`` touches. Everything else
#: stays uniform whatever the mix.
_WEIGHTED_TYPE = "banned_letter"


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


def letter_of(cid: str, lookup: Mapping[str, Mapping[str, Any]]) -> str:
    """The letter a ``banned_letter`` instance bans, lowercased.

    Raises rather than returning ``None`` for an instance without one: a letter
    that cannot be identified cannot be weighted, and silently falling back to
    uniform for it is the exact bug :data:`LETTER_REQUEST_WEIGHTS` exists to fix.
    """
    con = lookup.get(cid)
    if con is None:
        raise DatagenError(f"instance {cid!r} is not in the instance library")
    letter = (con.get("params") or {}).get("letter")
    if not isinstance(letter, str) or not letter:
        raise DatagenError(
            f"instance {cid!r} (type {con.get('type')!r}) carries no 'letter' "
            f"parameter, so its request weight cannot be determined")
    return letter.lower()


def _letter_weights(pool: Sequence[str],
                    lookup: Mapping[str, Mapping[str, Any]]) -> List[float]:
    """:data:`LETTER_REQUEST_WEIGHTS` lined up with ``pool``, or a loud failure."""
    weights: List[float] = []
    for cid in pool:
        letter = letter_of(cid, lookup)
        weight = LETTER_REQUEST_WEIGHTS.get(letter)
        if weight is None:
            known = "".join(sorted(LETTER_REQUEST_WEIGHTS))
            raise DatagenError(
                f"instance {cid!r} bans {letter!r}, which has no measured keep "
                f"rate in LETTER_REQUEST_WEIGHTS (covers {known!r}). Measure it "
                f"before requesting it: an unweighted letter under "
                f"letter_mix='hard' is the uniform-request bug all over again.")
        weights.append(weight)
    if not weights or sum(weights) <= 0:
        raise DatagenError(
            f"the banned_letter pool {list(pool)} carries no positive request weight")
    return weights


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
                    max_type_share: float = 0.25,
                    letter_mix: str = "uniform") -> List[Sample]:
    """``n`` training scenarios: fresh tasks x wide train-split constraint sets.

    Deterministic in ``seed``: the same seed, library and ``letter_mix`` give
    byte-identical scenarios, so a rerun re-uses the teacher cache instead of
    re-billing.

    ``letter_mix`` selects how a ``banned_letter`` instance is picked from its
    pool, and touches no other type:

    ``"uniform"`` (default)
        Uniform, via ``rng.choice`` -- the historical behaviour, byte for byte.
        The default is deliberately a no-op so that every caller and every test
        written before this parameter existed keeps its exact output.
    ``"hard"``
        Weighted by :data:`LETTER_REQUEST_WEIGHTS`, so that what survives the
        rejection filter in :mod:`datagen.generate` is distributed like English
        letter frequency instead of like teacher convenience.

    An unrecognised mix raises. It is never silently treated as uniform: the
    failure mode being fixed here is precisely a curriculum quietly reverting to
    the easy letters while the run still reports success.
    """
    if n <= 0:
        raise DatagenError(f"asked for {n} scenarios; there is no such thing as an "
                           f"empty training batch worth writing")
    if not tasks:
        raise DatagenError("the task pool is empty")
    if letter_mix not in LETTER_MIXES:
        raise DatagenError(
            f"unknown letter_mix {letter_mix!r}; expected one of "
            f"{', '.join(repr(m) for m in LETTER_MIXES)}. Refusing to fall back to "
            f"uniform: a typo that silently restores the easy-letter curriculum is "
            f"the bug this parameter exists to prevent.")

    by_type: Dict[str, List[str]] = {t: [] for t in TRAIN_TYPES}
    for cid, con in sorted(instances.items()):
        ctype = con.get("type")
        if ctype in by_type:
            by_type[ctype].append(cid)
    missing = sorted(t for t, ids in by_type.items() if not ids)
    if missing:
        raise DatagenError(f"the instance library has no instances of train types {missing}")

    rng = random.Random(seed)
    # Validate the whole banned_letter pool up front, not lazily as letters come
    # up: a library entry with no measured keep rate must fail the run, not the
    # one unlucky scenario that happens to draw it.
    weight_cache: Dict[Tuple[str, ...], List[float]] = {}
    if letter_mix != "uniform":
        letter_pool = by_type.get(_WEIGHTED_TYPE, [])
        weight_cache[tuple(letter_pool)] = _letter_weights(letter_pool, instances)

    def pick_instance(ctype: str, pool: List[str]) -> str:
        """One instance id from ``pool``. Weighted only for the weighted type.

        The uniform branch is the original ``rng.choice`` call, untouched, so it
        draws from the RNG stream exactly as before and reproduces prior output.
        """
        if letter_mix == "uniform" or ctype != _WEIGHTED_TYPE:
            return rng.choice(pool)
        key = tuple(pool)
        weights = weight_cache.get(key)
        if weights is None:
            weights = weight_cache[key] = _letter_weights(pool, instances)
        return rng.choices(pool, weights=weights)[0]

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
                cids.append(pick_instance(ctype, pool))
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
