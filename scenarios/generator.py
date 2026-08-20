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

==========================================================================
DISTRACTOR PRESSURE: WHAT "MECHANICAL" MEANS HERE.
==========================================================================

The adversarial share is a headline number, so the bar for calling a scenario
adversarial has to be one a sceptic can re-derive from the shipped JSONL. Every
one of the three mechanisms below therefore has a FLOOR COMPUTED FROM THE
SHIPPED TASK TEXT. An author can decide what to attempt; the text decides
whether it counts.

``lexicon_collision``   the banned token is a member of lexicon L, the task
                        visibly uses >= MIN_LEXICON_WITNESSES OTHER members of
                        L, AND -- the derived floor -- L is the task's dominant
                        topic: no other lexicon in the table has more members
                        present in the text. Before that last clause the topic
                        tag was an author's assertion; now the text has to
                        agree with it.

``key_term_letter``     the banned letter is carried by at least
                        MIN_LETTER_CONTENT_SHARE of the task's own distinct
                        CONTENT words. This is the floor that was missing: the
                        old rule accepted any two author-declared substrings
                        containing the letter, so the mechanism could be
                        claimed for a task the letter barely touched. The
                        declared key terms are still required, are now matched
                        WHOLE-WORD rather than as substrings, and are recorded
                        as evidence -- but they can no longer carry the claim
                        on their own.

``punctuation_density`` the task uses the banned mark(s) >= MIN_MARK_OCCURRENCES
                        times AND at a rate of >= MIN_MARK_RATE_PER_100_CHARS,
                        so a long prompt cannot reach the bar on length alone.

HONEST LIMIT OF THE ADVERSARIAL/CLEAN CONTRAST. ``key_term_letter`` fires only
where the author declared key terms, so a CLEAN task whose vocabulary happens
to be dense in its banned letter is not relabelled. That asymmetry is
deliberate and it is the safe direction: undeclared pressure in the clean arm
UNDERSTATES the clean-vs-adversarial gap, whereas an unfloored adversarial
label would overstate it, and overstating is the failure that would reach a
report. The derived share is recorded per scenario in the manifest
(``derived_letter_share``) for every scenario carrying a banned_letter,
including clean ones, so the size of that asymmetry is visible in the data
rather than argued about here.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from constraints.registry import TYPES, family_of, validate_constraint

from scenarios.instances import INSTANCES, INSTANCE_META
from scenarios.lexicons import GENERIC, LEXICONS, members_present
from scenarios.pressure import NUDGES
from scenarios.reachability import (build_witness, content_words, find_conflicts,
                                    task_conflicts, whole_word_present)

__all__ = [
    "NEGATIVE_TYPES",
    "CHARACTER_FAMILY",
    "CHARACTER_TYPES",
    "MIN_SCENARIOS",
    "MIN_ADVERSARIAL_SHARE",
    "MIN_CHARACTER_CONSTRAINT_SHARE",
    "character_constraint_share",
    "carries_character_constraint",
    "MIN_PER_SPLIT",
    "MIN_LETTER_CONTENT_SHARE",
    "MIN_MARK_RATE_PER_100_CHARS",
    "QUARANTINE",
    "GeneratorError",
    "ScenarioBuild",
    "generate",
    "distractor_mechanisms",
    "letter_content_share",
    "dominant_lexicons",
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

#: The CHARACTER family, and the types the registry puts in it. Derived, never
#: typed out: ``banned_letter`` (train) and ``banned_punctuation`` (held out)
#: are character-level because :mod:`constraints.registry` says so, so a
#: character type added later is counted here automatically instead of silently
#: deflating the share below.
CHARACTER_FAMILY = "character"
CHARACTER_TYPES: Tuple[str, ...] = tuple(
    t for t in TYPES if family_of(t) == CHARACTER_FAMILY)

#: Ladder rung ``r6-char65`` (configs/g1.json): at least this fraction of
#: scenarios must carry a character-level constraint.
#:
#: WHY THIS NUMBER EXISTS AT ALL, stated here because a floor without its
#: evidence is a preference. G1 ran the full ablation and returned ESCALATE at
#: fail_rate 0.0884. The ablation did not fail to find difficulty -- it
#: LOCALISED it. Across two frontier labs essentially every violation was
#: character-level: openai violated banned_letter 39 times and no other type at
#: all; anthropic 39 of 51. On letter-ban scenarios frontier fails 35-52%
#: against 0.9-4.5% elsewhere, and Qwen3-1.7B fails 100% of them under all
#: three strategies. Fail rate was FLAT across k (0.12/0.08/0.09 at k=2/3/4),
#: so composition is not the difficulty driver the design assumed. The old
#: corpus diluted the real effect by carrying a character constraint in only
#: half its scenarios and a letter ban in under a quarter.
#:
#: The floor is enforced here, at build time, AND recomputed from the shipped
#: JSONL by tests/test_scenarios.py AND re-derived from the corpus a run
#: actually used by scripts/gate_check.py. Three independent derivations,
#: because the escalation rests on this one number.
MIN_CHARACTER_CONSTRAINT_SHARE = 0.65

#: Distractor evidence thresholds. Two OTHER topic words in the prompt is the
#: bar for "the task is really about this topic"; three occurrences of a banned
#: mark is the bar for "this prompt's own register uses that mark".
MIN_LEXICON_WITNESSES = 2
MIN_KEY_TERM_HITS = 2
MIN_MARK_OCCURRENCES = 3

#: The derived floor for ``key_term_letter``: the banned letter must be carried
#: by at least HALF of the task's distinct content words.
#:
#: This threshold is a JUDGEMENT, not an inherited ruling, so it is stated
#: rather than buried. The mechanism's claim is "the banned letter sits inside
#: the nouns the task cannot do without". At half, avoiding the letter means
#: rewriting the majority of the task's own vocabulary -- not swapping a noun
#: or two -- which is the strongest reading the phrase will bear and therefore
#: the conservative one for a headline number. On the shipped corpus it costs
#: exactly one scenario (b06-landlord-socket, 40.0%); every other declared
#: key_term_letter task sits at 50% or above, most far above.
MIN_LETTER_CONTENT_SHARE = 0.50

#: The derived floor for ``punctuation_density``: banned marks per 100
#: characters of base task. Stops a long prompt from clearing
#: MIN_MARK_OCCURRENCES on volume rather than register. It binds nothing on the
#: shipped corpus -- every clean task uses its banned marks ZERO times and the
#: lowest qualifying adversarial rate is 1.55 -- and it is here so that a
#: future long prompt cannot buy the label with length.
MIN_MARK_RATE_PER_100_CHARS = 1.0


class Quarantined(NamedTuple):
    reason: str      # why this authored task is not in the corpus
    marker: str      # a substring the GeneratorError it provokes MUST contain


#: Authored tasks that are DELIBERATELY NOT SHIPPED, with the defect each one
#: has. The task specs live in :mod:`scenarios.tasks`; quarantining here rather
#: than deleting there keeps the authored pairing, its defect and its exclusion
#: in one auditable place.
#:
#: This is not a mute button. :func:`_prove_quarantined` rebuilds every entry at
#: build time and RAISES if the task no longer provokes an error, or provokes a
#: different one than the ``marker`` records. A quarantine entry that has gone
#: stale, or that someone added to silence an unrelated failure, fails the
#: build. The list is also republished in :func:`summarise`, so it reaches
#: ``manifest.json`` and cannot be a private arrangement between this file and
#: itself.
QUARANTINE: Dict[str, Quarantined] = {
    "b03-supervisor-sessions": Quarantined(
        "Task names the weekday 'Thursdays' while c-chr-003 bans the letter 's'. "
        "The real checker rejects the bare name, and a weekday has no synonym, so "
        "the ordinary answer is unwriteable. Not a PROOF of impossibility -- a "
        "response could refuse to name the day -- but the corpus does not ship "
        "scenarios whose brief collides with their own constraints.",
        "rejects the name 'Thursdays'"),
    "b05-recruitment-update": Quarantined(
        "Task says the process applies 'from March' while c-chr-005 bans the letter "
        "'r'. Same shape as b03: the checker rejects the month name and a month has "
        "no synonym.",
        "rejects the name 'March'"),
    "b06-landlord-socket": Quarantined(
        "Marked adversarial on a key_term_letter distractor, but only 40.0% of the "
        "task's content words carry the banned letter 'l' -- below "
        "MIN_LETTER_CONTENT_SHARE. Under the old rule the label rested entirely on "
        "four author-declared terms. It has no other qualifying mechanism, so it is "
        "not an adversarial scenario and shipping it would inflate the adversarial "
        "share.",
        "no distractor pressure qualifies"),
}


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


def dominant_lexicons(text: str) -> Tuple[List[str], int]:
    """``(lexicons with the most members present in text, that count)``.

    Derived from the text alone. This is what closes the "the author declared
    the topic tag" hole: a lexicon-tagged word ban only counts as a distractor
    if the shipped prompt is measurably about that topic and not about some
    other one that happens to share a word.
    """
    counts = {name: len(members_present(name, text)) for name in LEXICONS}
    best = max(counts.values()) if counts else 0
    return sorted(name for name, value in counts.items() if value == best and value > 0), best


def letter_content_carriers(letter: str, text: str) -> Tuple[List[str], List[str]]:
    """``(content words of text, those of them carrying letter)``. Purely derived."""
    words = content_words(text)
    low = letter.casefold()
    return words, [w for w in words if low in w]


def letter_content_share(letter: str, text: str) -> float:
    """Fraction of the task's distinct content words that carry ``letter``."""
    words, carriers = letter_content_carriers(letter, text)
    return (len(carriers) / len(words)) if words else 0.0


def _term_carries_letter(term: str, letter: str) -> bool:
    """Does a declared key term carry the banned letter in a CONTENT word?

    "the tray" must not qualify for a ban on 'e' via "the": the letter has to
    sit in a word the task is about, not in the grammar around it.
    """
    low = letter.casefold()
    return any(low in word for word in content_words(term))


def distractor_mechanisms(base_task: str, lexicon: Optional[str],
                          key_terms: Sequence[str],
                          constraints: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Qualifying DISTRACTOR PRESSURE in this pairing, with its evidence.

    Every mechanism carries evidence drawn from the task text itself AND clears
    a floor recomputed from that text, so the claim is re-checkable against the
    shipped JSONL by a reader who does not trust this function. See the module
    docstring for what each floor means and what it deliberately does not
    cover.
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
            # Derived floor: the text has to agree that this is its topic.
            leaders, best = dominant_lexicons(base_task)
            if meta.lexicon not in leaders:
                continue
            found.append({
                "kind": "lexicon_collision",
                "constraint_id": cid,
                "lexicon": meta.lexicon,
                "banned_members": in_lexicon,
                "topic_words_in_task": others[:6],
                "topic_word_count": len(others),
                "dominant_lexicons": leaders,
                "dominant_member_count": best,
            })
        elif ctype == "banned_letter":
            letter = constraint["params"]["letter"].lower()
            # The author's declaration is necessary but no longer sufficient.
            hits = [term for term in key_terms
                    if whole_word_present(term, base_task)
                    and _term_carries_letter(term, letter)]
            if len(hits) < MIN_KEY_TERM_HITS:
                continue
            # Derived floor: the letter has to saturate the task's own
            # vocabulary, whatever the author declared.
            words, carriers = letter_content_carriers(letter, base_task)
            share = (len(carriers) / len(words)) if words else 0.0
            if share < MIN_LETTER_CONTENT_SHARE:
                continue
            found.append({
                "kind": "key_term_letter",
                "constraint_id": cid,
                "letter": letter,
                "key_terms_carrying_letter": hits,
                "content_words_in_task": len(words),
                "content_words_carrying_letter": carriers,
                "letter_content_share": round(share, 4),
            })
        elif ctype == "banned_punctuation":
            marks = constraint["params"]["marks"]
            counts = {mark: base_task.count(mark) for mark in marks}
            total = sum(counts.values())
            if total < MIN_MARK_OCCURRENCES:
                continue
            # Derived floor: a rate, so length cannot buy the label.
            rate = (total / len(base_task) * 100) if base_task else 0.0
            if rate < MIN_MARK_RATE_PER_100_CHARS:
                continue
            found.append({
                "kind": "punctuation_density",
                "constraint_id": cid,
                "marks": marks,
                "occurrences_in_task": counts,
                "task_chars": len(base_task),
                "marks_per_100_chars": round(rate, 4),
            })
    return found


def negative_constraints(constraints: Sequence[Dict[str, Any]]) -> List[str]:
    return [c["id"] for c in constraints if c["type"] in NEGATIVE_TYPES]


def carries_character_constraint(constraint_ids: Iterable[str]) -> bool:
    """Does this scenario carry at least one CHARACTER-family constraint?

    Takes constraint IDS rather than objects so it can be run over a row of the
    shipped ``corpus.jsonl`` without rebuilding anything.
    """
    return any(INSTANCES[cid]["type"] in CHARACTER_TYPES for cid in constraint_ids)


def character_constraint_share(scenarios: Sequence[Dict[str, Any]]) -> float:
    """Fraction of ``scenarios`` carrying a character-level constraint."""
    if not scenarios:
        return 0.0
    n = sum(1 for s in scenarios if carries_character_constraint(s["constraints"]))
    return n / len(scenarios)


# ==========================================================================
# One scenario
# ==========================================================================

def _derived_letter_share(base_task: str,
                          constraints: Sequence[Dict[str, Any]]) -> Optional[float]:
    for constraint in constraints:
        if constraint["type"] == "banned_letter":
            return round(letter_content_share(constraint["params"]["letter"], base_task), 4)
    return None


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
        # WHOLE-WORD, not substring (A1). A substring test would let "us"
        # qualify off "museum", which is how a declared distractor becomes
        # unfalsifiable.
        if not whole_word_present(term, base_task):
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
        # Disclosure, not evidence: recorded for EVERY scenario carrying a
        # banned_letter, clean ones included, so a reader can see how much
        # letter pressure the clean arm carries without being labelled for it.
        # None when the scenario has no banned_letter constraint.
        "derived_letter_share": _derived_letter_share(base_task, constraints),
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

    # Ladder rung r6-char65. Raised here rather than left to the gate because a
    # corpus that silently drifts below the floor would be labelled with a rung
    # it does not satisfy, and the label is what the escalation is argued from.
    char_share = character_constraint_share(scenarios)
    if char_share < MIN_CHARACTER_CONSTRAINT_SHARE:
        n_char = sum(1 for s in scenarios if carries_character_constraint(s["constraints"]))
        problems.append(
            f"character-constraint share {n_char}/{n} = {char_share:.1%} < "
            f"{MIN_CHARACTER_CONSTRAINT_SHARE:.0%} (ladder rung r6-char65)")

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


def _prove_quarantined(index: int, spec: Dict[str, Any]) -> None:
    """A quarantined task must still fail, and fail for the recorded reason."""
    entry = QUARANTINE[spec["id"]]
    try:
        _build_one(index, spec)
    except GeneratorError as exc:
        if entry.marker not in str(exc):
            raise GeneratorError(
                f"{spec['id']}: quarantined for {entry.marker!r} but the build now fails "
                f"for a different reason: {exc}. Quarantine entries record ONE defect; "
                "re-diagnose rather than widening the entry") from None
        return
    raise GeneratorError(
        f"{spec['id']}: quarantine entry no longer reproduces its defect "
        f"({entry.marker!r}) -- the task now builds cleanly. Delete the quarantine "
        "entry and let the scenario back into the corpus, or record the real reason "
        "it is being held out. A quarantine that does not reproduce is a scenario "
        "silently deleted from the corpus")


def generate(task_specs: Optional[Sequence[Dict[str, Any]]] = None
             ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build ``(scenarios, manifests)`` from the authored task specs.

    Raises :class:`GeneratorError` on any violation. There is no "best effort"
    mode: a corpus that quietly drops a bad scenario would change the split
    balance and the adversarial share without telling anyone.

    The single exception is :data:`QUARANTINE`, and it is not silent: every
    entry names its defect, must still provoke exactly that defect, and is
    republished into ``manifest.json`` by :func:`summarise`. Quarantined tasks
    keep their positional slot, so the surviving ``scenario_id`` values do not
    renumber and the gap in the sequence is visible in the shipped corpus.
    """
    if task_specs is None:
        from scenarios.tasks import TASKS
        task_specs = TASKS

    seen_task_ids: List[str] = []
    scenarios: List[Dict[str, Any]] = []
    manifests: List[Dict[str, Any]] = []
    held_out: List[str] = []
    for index, spec in enumerate(task_specs, start=1):
        if spec["id"] in seen_task_ids:
            raise GeneratorError(f"duplicate task id {spec['id']!r}")
        seen_task_ids.append(spec["id"])
        if spec["id"] in QUARANTINE:
            _prove_quarantined(index, spec)
            held_out.append(spec["id"])
            continue
        built = _build_one(index, spec)
        scenarios.append(built.scenario)
        manifests.append(built.manifest)

    stale = sorted(set(QUARANTINE) - set(seen_task_ids))
    if stale and task_specs is _default_task_specs():
        raise GeneratorError(
            f"quarantine names task id(s) {stale} that are not in scenarios.tasks; "
            "a quarantine entry for a task that no longer exists is dead weight that "
            "hides the next real one")

    _corpus_invariants(scenarios, manifests)
    return scenarios, manifests


def _default_task_specs() -> Sequence[Dict[str, Any]]:
    from scenarios.tasks import TASKS
    return TASKS


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
        "adversarial_share_floor": MIN_ADVERSARIAL_SHARE,
        "character_constraint_scenarios": sum(
            1 for s in scenarios if carries_character_constraint(s["constraints"])),
        "character_constraint_share": round(character_constraint_share(scenarios), 4),
        "character_constraint_share_floor": MIN_CHARACTER_CONSTRAINT_SHARE,
        "character_types": list(CHARACTER_TYPES),
        "constraint_instances_used": len({cid for s in scenarios for cid in s["constraints"]}),
        # Published so a reader of manifest.json sees what was held back and
        # why, without reading this module. See QUARANTINE.
        "quarantined_tasks": [{"task_key": key, "reason": entry.reason}
                              for key, entry in sorted(QUARANTINE.items())],
    }
