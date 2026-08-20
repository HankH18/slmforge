#!/usr/bin/env python3
"""The scenario generator: authored task specs in, validated corpus out.

The generator is deliberately not a sampler. Distractor pressure is a claim
about MEANING -- that a task about espresso bars pulls a writer toward the word
"coffee" -- and no amount of random pairing produces that. So the pairing is
authored in :mod:`scenarios.tasks`, and this module's job is to make the
authoring impossible to get wrong: it derives everything derivable from the
registry, checks every invariant the ticket names, and RAISES rather than
emitting a corpus that violates one. A silently-degraded corpus is the failure
mode that matters here, because every downstream number is computed over it.

What is derived, never authored:

* ``k`` -- always ``len(constraints)``;
* ``split`` -- ``heldout_type`` if ANY constraint's type is held out, else
  ``train``. This makes the split invariant true by construction, which is why
  the test asserts it against the shipped JSONL instead of trusting this code;
* ``family`` / instance ``split`` -- from :mod:`constraints.registry`.

What is checked and can fail the build:

* the constraint set is reachable (analytic conflicts + a constructed witness
  the real checkers pass) -- :mod:`scenarios.reachability`;
* the task does not demand what a constraint forbids;
* an adversarial scenario really carries both pressure mechanisms, and a clean
  one carries neither;
* a topic-tagged word ban is only ever attached to a task on that topic, so no
  constraint can be present without biting.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from constraints.registry import TYPES, validate_constraint

from scenarios.instances import INSTANCES, INSTANCE_META
from scenarios.lexicons import GENERIC, LEXICONS, members_present
from scenarios.pressure import NUDGES
from scenarios.reachability import build_witness, find_conflicts, task_conflicts

__all__ = [
    "NEGATIVE_TYPES",
    "MIN_SCENARIOS",
    "MIN_ADVERSARIAL_SHARE",
    "MIN_PER_SPLIT",
    "GeneratorError",
    "ScenarioBuild",
    "generate",
    "distractor_mechanisms",
    "negative_constraints",
]


class GeneratorError(ValueError):
    """The authored corpus violates an invariant. Never downgraded to a warning."""


#: Spec R2: every scenario must include at least one NEGATIVE constraint.
#: Negative = forbids something. Derived from the registry rather than typed out
#: so a future type lands on the right side of the line automatically. Six of
#: the seven pinned types are prohibitions; ``sentence_initial_letter`` is the
#: only one that REQUIRES a form rather than forbidding one, so it is the only
#: positive type -- and the only type that cannot satisfy R2 on its own.
POSITIVE_TYPES: Tuple[str, ...] = ("sentence_initial_letter",)
NEGATIVE_TYPES: Tuple[str, ...] = tuple(t for t in TYPES if t not in POSITIVE_TYPES)

#: Acceptance floors from the ticket.
MIN_SCENARIOS = 120
MIN_ADVERSARIAL_SHARE = 0.40
MIN_PER_SPLIT = 30
ALLOWED_K = (2, 3, 4)

#: Distractor evidence thresholds. Two OTHER topic words in the prompt is the
#: bar for "the task is really about this topic"; three occurrences of a banned
#: mark is the bar for "this prompt's own register uses that mark".
MIN_LEXICON_WITNESSES = 2
MIN_KEY_TERM_HITS = 2
MIN_MARK_OCCURRENCES = 3


class ScenarioBuild(NamedTuple):
    scenario: Dict[str, Any]        # the pinned six-key Scenario object
    manifest: Dict[str, Any]        # provenance: pressure mechanisms, witness, metadata


# ==========================================================================
# Pressure analysis
# ==========================================================================

def _banned_tokens(constraint: Dict[str, Any]) -> List[str]:
    params = constraint.get("params") or {}
    if constraint["type"] == "banned_words":
        return list(params["words"])
    if constraint["type"] == "banned_phrases":
        return list(params["phrases"])
    return []


def distractor_mechanisms(base_task: str, lexicon: Optional[str],
                          key_terms: Sequence[str],
                          constraints: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Qualifying DISTRACTOR PRESSURE in this pairing, with its evidence.

    Every mechanism carries evidence drawn from the task text itself, so the
    claim is re-checkable against the shipped JSONL by a reader who does not
    trust this function.
    """
    found: List[Dict[str, Any]] = []
    for constraint in constraints:
        ctype = constraint["type"]
        cid = constraint["id"]
        meta = INSTANCE_META[cid]
        if ctype in ("banned_words", "banned_phrases"):
            if meta.lexicon == GENERIC or lexicon is None or meta.lexicon != lexicon:
                continue
            tokens = _banned_tokens(constraint)
            in_lexicon = [t for t in tokens if t.lower() in
                          {m.lower() for m in LEXICONS[meta.lexicon]}]
            if not in_lexicon:
                continue
            others = members_present(meta.lexicon, base_task, exclude=tuple(tokens))
            if len(others) < MIN_LEXICON_WITNESSES:
                continue
            found.append({
                "kind": "lexicon_collision",
                "constraint_id": cid,
                "lexicon": meta.lexicon,
                "banned_members": in_lexicon,
                "topic_words_in_task": others[:6],
            })
        elif ctype == "banned_letter":
            letter = constraint["params"]["letter"].lower()
            hits = [term for term in key_terms if letter in term.lower()]
            if len(hits) < MIN_KEY_TERM_HITS:
                continue
            found.append({
                "kind": "key_term_letter",
                "constraint_id": cid,
                "letter": letter,
                "key_terms_carrying_letter": hits,
            })
        elif ctype == "banned_punctuation":
            marks = constraint["params"]["marks"]
            counts = {mark: base_task.count(mark) for mark in marks}
            total = sum(counts.values())
            if total < MIN_MARK_OCCURRENCES:
                continue
            found.append({
                "kind": "punctuation_density",
                "constraint_id": cid,
                "marks": marks,
                "occurrences_in_task": counts,
            })
    return found


def negative_constraints(constraints: Sequence[Dict[str, Any]]) -> List[str]:
    return [c["id"] for c in constraints if c["type"] in NEGATIVE_TYPES]


# ==========================================================================
# One scenario
# ==========================================================================

def _resolve(task_id: str, ids: Sequence[str]) -> List[Dict[str, Any]]:
    resolved = []
    for cid in ids:
        if cid not in INSTANCES:
            raise GeneratorError(
                f"{task_id}: constraint {cid!r} is not in the instance library")
        resolved.append(INSTANCES[cid])
    return resolved


def _build_one(index: int, spec: Dict[str, Any]) -> ScenarioBuild:
    task_id = spec["id"]
    base_task = spec["text"].strip()
    constraint_ids = list(spec["constraints"])
    constraints = _resolve(task_id, constraint_ids)
    for constraint in constraints:
        validate_constraint(constraint)

    if len(set(constraint_ids)) != len(constraint_ids):
        raise GeneratorError(f"{task_id}: repeats a constraint id {constraint_ids}")
    k = len(constraint_ids)
    if k not in ALLOWED_K:
        raise GeneratorError(f"{task_id}: k={k} but the design pins k to {list(ALLOWED_K)} "
                             "(k=1 is excluded deliberately -- see Design 'Decisions')")

    types = [c["type"] for c in constraints]
    for ctype in ("banned_letter", "sentence_initial_letter"):
        if types.count(ctype) > 1:
            raise GeneratorError(f"{task_id}: two {ctype} constraints in one scenario")

    if not negative_constraints(constraints):
        raise GeneratorError(
            f"{task_id}: no negative constraint (Spec R2). Types present: {sorted(set(types))}")

    lexicon = spec.get("lexicon")
    if lexicon is not None and lexicon not in LEXICONS:
        raise GeneratorError(f"{task_id}: unknown lexicon {lexicon!r}")

    # A topic-tagged word ban attached to a different topic is the
    # "banning zygote from a transit essay" failure: present, but never bites.
    for constraint in constraints:
        meta = INSTANCE_META[constraint["id"]]
        if meta.lexicon != GENERIC and meta.lexicon != lexicon:
            raise GeneratorError(
                f"{task_id}: constraint {constraint['id']} is tagged to lexicon "
                f"{meta.lexicon!r} but the task's topic is {lexicon!r}; it would never bite")

    key_terms = tuple(spec.get("key_terms", ()))
    for term in key_terms:
        if term.lower() not in base_task.lower():
            raise GeneratorError(
                f"{task_id}: declared key term {term!r} does not appear in the task text")

    # ---- reachability ----
    conflicts = find_conflicts(constraints)
    if conflicts:
        raise GeneratorError(f"{task_id}: unreachable constraint set: {conflicts}")
    demands = task_conflicts(base_task, constraints)
    if demands:
        raise GeneratorError(f"{task_id}: task contradicts its own constraints: {demands}")
    witness = build_witness(constraints)
    if witness is None:
        raise GeneratorError(
            f"{task_id}: could not construct a passing witness for {constraint_ids}; "
            "treat as unreachable until proven otherwise")

    # ---- pressure ----
    pressure = spec["pressure"]
    if pressure not in ("clean", "adversarial"):
        raise GeneratorError(f"{task_id}: pressure must be 'clean' or 'adversarial'")
    mechanisms = distractor_mechanisms(base_task, lexicon, key_terms, constraints)
    nudge_id = spec.get("nudge")

    if pressure == "adversarial":
        if not mechanisms:
            raise GeneratorError(
                f"{task_id}: marked adversarial but no distractor pressure qualifies. "
                "A task with no pull toward the banned token is clean, not adversarial")
        if nudge_id is None:
            raise GeneratorError(f"{task_id}: marked adversarial but carries no embedded "
                                 "pressure; the ticket requires BOTH mechanisms")
        if nudge_id not in NUDGES:
            raise GeneratorError(f"{task_id}: unknown nudge {nudge_id!r}")
        targets = NUDGES[nudge_id].targets
        if not set(targets) & set(types):
            raise GeneratorError(
                f"{task_id}: nudge {nudge_id!r} undermines {list(targets)} but the scenario "
                f"has no such constraint (types: {sorted(set(types))})")
        task_text = f"{base_task} {NUDGES[nudge_id].text}"
    else:
        if nudge_id is not None:
            raise GeneratorError(f"{task_id}: clean scenarios carry no nudge")
        if mechanisms:
            raise GeneratorError(
                f"{task_id}: marked clean but distractor pressure qualifies "
                f"({[m['kind'] for m in mechanisms]}); label it adversarial or "
                "re-pair the constraints")
        task_text = base_task

    # ---- derived fields ----
    split = ("heldout_type" if any(c["split"] == "heldout_type" for c in constraints)
             else "train")
    scenario_id = f"s-{index:04d}"
    scenario = {
        "scenario_id": scenario_id,
        "task": task_text,
        "constraints": constraint_ids,
        "k": k,
        "pressure": pressure,
        "split": split,
    }
    manifest = {
        "scenario_id": scenario_id,
        "task_key": task_id,
        "base_task": base_task,
        "domain": spec["domain"],
        "register": spec["register"],
        "lexicon": lexicon,
        "key_terms": list(key_terms),
        "constraint_types": types,
        "negative_constraints": negative_constraints(constraints),
        "pressure": pressure,
        "distractor_pressure": mechanisms,
        "embedded_pressure": (None if nudge_id is None else {
            "nudge_id": nudge_id,
            "text": NUDGES[nudge_id].text,
            "targets": list(NUDGES[nudge_id].targets),
        }),
        "reachability_witness": witness,
    }
    return ScenarioBuild(scenario, manifest)


# ==========================================================================
# The whole corpus
# ==========================================================================

def _corpus_invariants(scenarios: Sequence[Dict[str, Any]],
                       manifests: Sequence[Dict[str, Any]]) -> None:
    problems: List[str] = []
    n = len(scenarios)
    if n < MIN_SCENARIOS:
        problems.append(f"{n} scenarios, need >= {MIN_SCENARIOS}")

    ids = [s["scenario_id"] for s in scenarios]
    if len(set(ids)) != len(ids):
        problems.append("duplicate scenario_id")

    tasks = [s["task"] for s in scenarios]
    if len(set(tasks)) != len(tasks):
        problems.append("duplicate task text")
    bases = [m["base_task"] for m in manifests]
    if len(set(bases)) != len(bases):
        dupes = sorted({b for b in bases if bases.count(b) > 1})
        problems.append(f"duplicate BASE task text (nudge-stripped): {dupes[:3]}")

    ks = {s["k"] for s in scenarios}
    missing_k = [k for k in ALLOWED_K if k not in ks]
    if missing_k:
        problems.append(f"k values {missing_k} unrepresented")

    adversarial = [s for s in scenarios if s["pressure"] == "adversarial"]
    share = len(adversarial) / n if n else 0.0
    if share < MIN_ADVERSARIAL_SHARE:
        problems.append(f"adversarial share {share:.1%} < {MIN_ADVERSARIAL_SHARE:.0%}")

    for split in ("train", "heldout_type"):
        count = sum(1 for s in scenarios if s["split"] == split)
        if count < MIN_PER_SPLIT:
            problems.append(f"split {split!r} has {count} scenarios, need >= {MIN_PER_SPLIT}")

    # THE type-level split invariant, enforced at build time as well as tested.
    for scenario in scenarios:
        if scenario["split"] != "train":
            continue
        leaked = [cid for cid in scenario["constraints"]
                  if INSTANCES[cid]["split"] == "heldout_type"]
        if leaked:
            problems.append(f"{scenario['scenario_id']}: held-out constraints {leaked} "
                            "in a train scenario")

    if problems:
        raise GeneratorError("corpus invariants violated: " + "; ".join(problems))


def generate(task_specs: Optional[Sequence[Dict[str, Any]]] = None
             ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build ``(scenarios, manifests)`` from the authored task specs.

    Raises :class:`GeneratorError` on any violation. There is no "best effort"
    mode: a corpus that quietly drops a bad scenario would change the split
    balance and the adversarial share without telling anyone.
    """
    if task_specs is None:
        from scenarios.tasks import TASKS
        task_specs = TASKS

    seen_task_ids: List[str] = []
    scenarios: List[Dict[str, Any]] = []
    manifests: List[Dict[str, Any]] = []
    for index, spec in enumerate(task_specs, start=1):
        if spec["id"] in seen_task_ids:
            raise GeneratorError(f"duplicate task id {spec['id']!r}")
        seen_task_ids.append(spec["id"])
        built = _build_one(index, spec)
        scenarios.append(built.scenario)
        manifests.append(built.manifest)

    _corpus_invariants(scenarios, manifests)
    return scenarios, manifests


def summarise(scenarios: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The stated distribution, computed from the data rather than asserted."""
    n = len(scenarios)
    def dist(key: str) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for scenario in scenarios:
            out[str(scenario[key])] = out.get(str(scenario[key]), 0) + 1
        return dict(sorted(out.items()))
    adversarial = sum(1 for s in scenarios if s["pressure"] == "adversarial")
    return {
        "scenarios": n,
        "k_distribution": dist("k"),
        "pressure_distribution": dist("pressure"),
        "split_distribution": dist("split"),
        "adversarial_share": round(adversarial / n, 4) if n else 0.0,
        "constraint_instances_used": len({cid for s in scenarios for cid in s["constraints"]}),
    }
