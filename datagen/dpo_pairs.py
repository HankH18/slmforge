#!/usr/bin/env python3
"""Build a verified DPO preference dataset from a ``*.dpo_negatives.jsonl`` sidecar.

    python -m datagen.dpo_pairs                     # v3, write pairs + manifest
    python -m datagen.dpo_pairs --check             # verify only, write nothing
    python -m datagen.dpo_pairs --source data/train/v2.dpo_negatives.jsonl

The sidecar written by :mod:`datagen.generate` holds one row per *fluent rejected
draft*, with ``chosen`` back-filled from the attempt that finally passed (or left
``None`` when no attempt ever did). That file is a candidate pool, not a dataset:
it contains rows with no positive side, and its ``violated`` field is a record of
what the generator believed at generation time. This module turns the pool into a
dataset by re-deriving every claim from the pinned ground-truth layer.

What "verified" means here
--------------------------
Nothing in the source row is trusted:

* ``violated`` is IGNORED. Both texts are re-run through
  ``constraints.checkers.check`` against the constraints resolved from
  ``data/scenarios/constraint_instances.jsonl`` at build time. A pair survives
  iff ``chosen`` passes EVERY constraint, ``rejected`` fails AT LEAST ONE, and
  the two texts differ.
* ``prompt`` is IGNORED as an input. The prompt is re-rendered from the row's
  stored scenario through ``scenarios.render.render_user_message`` -- the one
  renderer the SFT trainer and the eval harness both call -- and the re-rendered
  text must equal the stored string byte for byte. A DPO prompt that differs from
  the SFT prompt trains a different task, so a mismatch drops the row and is
  counted in the manifest rather than silently accepted.

Guarantee tier (docs: "make the bad state unrepresentable"). A mis-oriented pair
is caught at a RUNTIME CHOKEPOINT, not made structurally impossible: every
emitted row is built by :class:`VerifiedPair`, whose only constructor re-runs the
checkers and raises. That stops a mis-oriented pair from being written by this
module by any path, including a future refactor that forgets the filter -- but it
is not a type-level guarantee, and it cannot stop someone hand-editing the output
JSONL. ``tests/test_dpo_pairs.py`` re-verifies the shipped file independently for
exactly that reason.

Schema: conversational, and why
-------------------------------
TRL's ``DPOTrainer`` accepts two shapes. The "standard" shape is three plain
strings; the "conversational" shape is three message lists::

    {"prompt":   [{"role": "user",      "content": ...}],
     "chosen":   [{"role": "assistant", "content": ...}],
     "rejected": [{"role": "assistant", "content": ...}]}

This module emits the conversational shape. The reason is the prompt contract,
not taste. ``training/modal_app.py`` feeds the model
``tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True,
enable_thinking=...)`` -- the chat scaffolding is part of the trained prompt, and
the eval generator makes the same call. TRL applies the chat template itself for
conversational rows and does NOT apply it for standard rows, so the standard
shape would hand the trainer bare prose with no ``<|im_start|>`` scaffolding: a
prompt distribution the SFT checkpoint has never seen. The conversational shape
reproduces the SFT layer-2 prompt through the same mechanism.

One residual mismatch, stated because it is real and this module cannot fix it:
the base model's chat template emits an empty ``<think></think>`` block after the
assistant header when ``enable_thinking=False``, which is what
``configs/train.json:data.enable_thinking`` pins for SFT. TRL applies the
template with its own kwargs, so whoever runs the DPO job must pass the same
``enable_thinking`` value through TRL's chat-template kwargs. If they do not, the
DPO prompt differs from the SFT prompt by that block. This module writes the
messages; it cannot enforce the trainer's template kwargs.

Determinism
-----------
Same input file plus same constraint library -> byte-identical output file and
byte-identical manifest. Source order is preserved, key order is fixed, and no
row and no manifest field carries a timestamp. The manifest's numbers are all
derived here at build time; none is written by hand.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from constraints import registry                                   # noqa: E402
from constraints.checkers import check                             # noqa: E402
from scenarios.paths import CORPUS_PATH, load_corpus, load_instances  # noqa: E402
from scenarios.render import render_user_message                   # noqa: E402

__all__ = [
    "VerifiedPair", "PairOrientationError", "EvalLeak", "BuildResult",
    "DROP_REASONS", "ENGLISH_LETTER_FREQUENCY_PCT", "DATAGEN_ID",
    "build", "build_manifest", "classify_row", "write_dataset", "main",
    "DEFAULT_SOURCE", "output_paths", "repo_relative", "sha256_file", "sha256_text",
]

# ==========================================================================
# Pins
# ==========================================================================

DEFAULT_SOURCE = REPO_ROOT / "data" / "train" / "v3.dpo_negatives.jsonl"

#: The dataset schema this module emits. See the module docstring for the
#: reasoning; changing it changes what the trainer sees.
SCHEMA_FORM = "conversational"
SCHEMA_KEYS = ("prompt", "chosen", "rejected",
               "scenario_id", "constraints", "violated_constraint_ids",
               "violated_types", "k")

#: Datagen scenario ids. The eval corpus uses ``s-NNNN``; anything that is not a
#: ``dg-`` id has no business in a training-side preference set.
DATAGEN_ID = re.compile(r"^dg-\d+$")

#: Every reason a source row can fail to become a pair. Published so the manifest
#: and the tests enumerate the same set instead of two drifting string lists.
DROP_REASONS = (
    "non_datagen_scenario_id",
    "chosen_absent",              # teacher never produced a compliant draft
    "prompt_render_mismatch",     # stored prompt != shared renderer's output
    "eval_corpus_prompt_collision",
    "texts_identical",
    "chosen_violates_a_constraint",
    "rejected_violates_nothing",
)

#: Relative frequency of each letter in English text, percent (Lewand,
#: *Cryptological Mathematics*, MAA 2000 -- the standard table). Used ONLY to
#: describe the attrition bias; nothing in the build depends on it. Cross-checked
#: against the data at build time: a banned letter missing from this table raises
#: rather than being silently skipped from the comparison.
ENGLISH_LETTER_FREQUENCY_PCT = {
    "a": 8.167, "b": 1.492, "c": 2.782, "d": 4.253, "e": 12.702, "f": 2.228,
    "g": 2.015, "h": 6.094, "i": 6.966, "j": 0.153, "k": 0.772, "l": 4.025,
    "m": 2.406, "n": 6.749, "o": 7.507, "p": 1.929, "q": 0.095, "r": 5.987,
    "s": 6.327, "t": 9.056, "u": 2.758, "v": 0.978, "w": 2.360, "x": 0.150,
    "y": 1.974, "z": 0.074,
}


class PairOrientationError(ValueError):
    """A pair was constructed that the pinned checkers do not certify."""


class EvalLeak(RuntimeError):
    """A pair's prompt or scenario id reaches the eval corpus. Hard stop."""


# ==========================================================================
# The pair
# ==========================================================================

@dataclasses.dataclass(frozen=True)
class VerifiedPair:
    """A preference pair that the pinned checkers certify, by construction.

    The constructor re-runs ``constraints.checkers.check`` on both texts. There
    is no way to obtain a ``VerifiedPair`` whose ``chosen`` violates a constraint
    or whose ``rejected`` violates none -- ``__post_init__`` raises. This is a
    runtime chokepoint, not a type-level guarantee: it binds every construction
    site in this process, and it does not police the JSONL after it is written.
    """

    scenario_id: str
    prompt: str
    chosen: str
    rejected: str
    constraints: Tuple[Mapping[str, Any], ...]
    k: int

    def __post_init__(self) -> None:
        if not self.constraints:
            raise PairOrientationError(
                f"{self.scenario_id}: a pair with no constraints certifies nothing")
        # Distinctness first: a "pair" of one text carries no preference at all,
        # and checking it first keeps the branch reachable and testable rather
        # than shadowed by whichever orientation check fires on the duplicate.
        if self.chosen.strip() == self.rejected.strip():
            raise PairOrientationError(
                f"{self.scenario_id}: 'chosen' and 'rejected' are the same text")
        offenders = [c["id"] for c in self.constraints
                     if not check(c, self.chosen)["pass"]]
        if offenders:
            raise PairOrientationError(
                f"{self.scenario_id}: 'chosen' violates {offenders}")
        violated = tuple(c["id"] for c in self.constraints
                         if not check(c, self.rejected)["pass"])
        if not violated:
            raise PairOrientationError(
                f"{self.scenario_id}: 'rejected' violates nothing -- not a negative")
        object.__setattr__(self, "_violated_ids", violated)

    @property
    def violated_constraint_ids(self) -> Tuple[str, ...]:
        return getattr(self, "_violated_ids")

    @property
    def violated_types(self) -> Tuple[str, ...]:
        by_id = {c["id"]: c["type"] for c in self.constraints}
        return tuple(sorted({by_id[cid] for cid in self.violated_constraint_ids}))

    def to_row(self) -> "OrderedDict[str, Any]":
        """The emitted JSONL row: TRL conversational, plus provenance columns.

        ``DPOTrainer`` reads ``prompt``/``chosen``/``rejected`` and ignores the
        rest; the extra columns exist so an auditor can re-verify a row without
        joining back to the source sidecar.
        """
        row: "OrderedDict[str, Any]" = OrderedDict()
        row["prompt"] = [{"role": "user", "content": self.prompt}]
        row["chosen"] = [{"role": "assistant", "content": self.chosen}]
        row["rejected"] = [{"role": "assistant", "content": self.rejected}]
        row["scenario_id"] = self.scenario_id
        row["constraints"] = [c["id"] for c in self.constraints]
        row["violated_constraint_ids"] = list(self.violated_constraint_ids)
        row["violated_types"] = list(self.violated_types)
        row["k"] = self.k
        return row


# ==========================================================================
# Classification
# ==========================================================================

def classify_row(row: Mapping[str, Any],
                 lookup: Mapping[str, Mapping[str, Any]],
                 corpus_prompts: Mapping[str, str],
                 ) -> Tuple[Optional[VerifiedPair], Optional[str]]:
    """One source row -> (pair, None) or (None, drop_reason).

    Every rejection reason is one of :data:`DROP_REASONS`; a row is never dropped
    silently and never dropped for two reasons at once (the first that fires
    wins, in the order below).
    """
    scenario_id = row.get("scenario_id")
    scenario = row.get("scenario")
    cids = list(row.get("constraints") or [])
    constraints = tuple(lookup[cid] for cid in cids)     # KeyError = corrupt input

    if not isinstance(scenario_id, str) or not DATAGEN_ID.match(scenario_id):
        return None, "non_datagen_scenario_id"

    chosen = row.get("chosen")
    if chosen is None:
        return None, "chosen_absent"
    rejected = row.get("rejected")

    # The prompt is re-derived, never copied. A divergence is a finding.
    rendered = render_user_message(scenario, lookup)
    if rendered != row.get("prompt"):
        return None, "prompt_render_mismatch"
    if rendered in corpus_prompts:
        return None, "eval_corpus_prompt_collision"

    if str(chosen).strip() == str(rejected).strip():
        return None, "texts_identical"
    if any(not check(c, chosen)["pass"] for c in constraints):
        return None, "chosen_violates_a_constraint"
    if all(check(c, rejected)["pass"] for c in constraints):
        return None, "rejected_violates_nothing"

    pair = VerifiedPair(scenario_id=scenario_id, prompt=rendered,
                        chosen=chosen, rejected=rejected,
                        constraints=constraints,
                        k=int(scenario.get("k", len(cids))))
    return pair, None


# ==========================================================================
# Build
# ==========================================================================

@dataclasses.dataclass
class BuildResult:
    pairs: List[VerifiedPair]
    drops: "Counter[str]"
    source_rows: List[Mapping[str, Any]]
    lookup: Mapping[str, Mapping[str, Any]]
    corpus_prompts: Mapping[str, str]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from None
    return rows


def corpus_prompt_index(lookup: Mapping[str, Mapping[str, Any]]) -> "OrderedDict[str, str]":
    """``rendered eval prompt -> eval scenario_id``, through the shared renderer.

    Built from the frozen eval corpus so the leak check compares what the model
    would actually be shown, not the raw task text.
    """
    index: "OrderedDict[str, str]" = OrderedDict()
    for scenario in load_corpus():
        index[render_user_message(scenario, lookup)] = scenario["scenario_id"]
    if not index:
        raise RuntimeError(f"{CORPUS_PATH} rendered no prompts -- leak check would be vacuous")
    return index


def build(source: Path,
          lookup: Optional[Mapping[str, Mapping[str, Any]]] = None) -> BuildResult:
    """Read a negatives sidecar and return the certified pairs plus drop tally.

    Raises :class:`EvalLeak` if any pair's rendered prompt or scenario id reaches
    the eval corpus: training on an eval prompt would invalidate the SFT->DPO
    delta this dataset exists to measure, so it is a hard stop and not a drop.
    """
    lookup = load_instances() if lookup is None else lookup
    rows = load_jsonl(Path(source))
    if not rows:
        raise RuntimeError(f"{source} is empty -- nothing to build")

    corpus_prompts = corpus_prompt_index(lookup)
    corpus_ids = set(corpus_prompts.values())

    pairs: List[VerifiedPair] = []
    drops: "Counter[str]" = Counter()
    for row in rows:
        pair, reason = classify_row(row, lookup, corpus_prompts)
        if pair is None:
            if reason not in DROP_REASONS:
                raise RuntimeError(f"unenumerated drop reason {reason!r}")
            drops[reason] += 1
            continue
        pairs.append(pair)

    # Non-vacuity: every source row was accounted for exactly once.
    if len(pairs) + sum(drops.values()) != len(rows):
        raise RuntimeError("row accounting is broken: "
                           f"{len(pairs)} kept + {sum(drops.values())} dropped "
                           f"!= {len(rows)} read")
    if not pairs:
        raise RuntimeError(f"{source} yielded zero verified pairs")

    # The leak gate, part 1: a COLLISION THAT WAS DROPPED IS STILL CONTAMINATION.
    # This used to be silent. classify_row returns "eval_corpus_prompt_collision"
    # and the row was dropped like any other, so a datagen corpus that had
    # genuinely overlapped the eval set produced a clean-looking pairs file and
    # exit 0 -- the contamination erased rather than reported. An SFT->DPO delta
    # is only a delta if the DPO model never trained on an eval prompt, so the
    # fact that overlap EXISTED has to reach whoever reads the number, even
    # though the offending rows are not in the output.
    collisions = drops.get("eval_corpus_prompt_collision", 0)
    if collisions:
        raise EvalLeak(
            f"{collisions} of {len(rows)} source rows in {source} render to a "
            f"prompt that is also in the eval corpus. They were excluded from the "
            f"output, but their existence means the datagen corpus overlaps the "
            f"eval set and any SFT->DPO comparison built on it is contaminated. "
            f"Fix the generation inputs; do not silently drop these.")

    # The leak gate, part 2. Drops above already exclude collisions, so anything
    # caught here is a build-logic failure and must stop the run loudly.
    leaked_prompts = [p.scenario_id for p in pairs if p.prompt in corpus_prompts]
    leaked_ids = sorted({p.scenario_id for p in pairs} & corpus_ids)
    if leaked_prompts or leaked_ids:
        raise EvalLeak(f"eval corpus reached the pairs: prompts from "
                       f"{sorted(set(leaked_prompts))}, ids {leaked_ids}")

    return BuildResult(pairs=pairs, drops=drops, source_rows=rows,
                       lookup=lookup, corpus_prompts=corpus_prompts)


# ==========================================================================
# Disclosure arithmetic
# ==========================================================================

def _banned_letter(constraint_ids: Sequence[str],
                   lookup: Mapping[str, Mapping[str, Any]]) -> Optional[str]:
    for cid in constraint_ids:
        constraint = lookup.get(cid)
        if constraint and constraint["type"] == "banned_letter":
            return str(constraint["params"]["letter"]).lower()
    return None


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def held_out_types(lookup: Mapping[str, Mapping[str, Any]]) -> List[str]:
    """Constraint types with NO train-split instance, derived from the library."""
    splits: Dict[str, set] = {}
    for constraint in lookup.values():
        splits.setdefault(constraint["type"], set()).add(constraint.get("split"))
    return sorted(t for t, s in splits.items() if "train" not in s)


def monoculture_report(result: BuildResult) -> "OrderedDict[str, Any]":
    """What the pairs can and cannot teach, counted rather than asserted."""
    violated = Counter()
    present = Counter()
    only_letter = 0
    for pair in result.pairs:
        for t in pair.violated_types:
            violated[t] += 1
        for c in pair.constraints:
            present[c["type"]] += 1
        if pair.violated_types == ("banned_letter",):
            only_letter += 1

    all_types = sorted(registry.list_types())
    never = sorted(set(all_types) - set(violated))

    # Within the single axis this data CAN move, which letters does it move? The
    # eval corpus is graded on its own letter mix, so a pair set that covers a
    # different mix would move an axis nobody scores. Both distributions are
    # derived here, never quoted.
    pair_letters = Counter()
    seen: set = set()
    for pair in result.pairs:
        if pair.scenario_id in seen:
            continue
        seen.add(pair.scenario_id)
        letter = _banned_letter([c["id"] for c in pair.constraints], result.lookup)
        if letter:
            pair_letters[letter] += 1
    eval_letters = Counter()
    for scenario in load_corpus():
        letter = _banned_letter(scenario.get("constraints") or [], result.lookup)
        if letter:
            eval_letters[letter] += 1

    out: "OrderedDict[str, Any]" = OrderedDict()
    out["violated_constraint_types"] = OrderedDict(sorted(violated.items()))
    out["pairs_whose_violations_are_banned_letter_only"] = only_letter
    out["fraction_banned_letter_only"] = round(only_letter / len(result.pairs), 6)
    out["constraint_types_present_in_the_prompts"] = OrderedDict(sorted(present.items()))
    out["registered_types_never_violated_by_any_rejected_text"] = never
    out["held_out_types_derived_from_the_instance_library"] = held_out_types(result.lookup)
    out["banned_letter_coverage_vs_eval_corpus"] = OrderedDict([
        ("pair_scenarios_by_banned_letter",
         OrderedDict(sorted(pair_letters.items(), key=lambda kv: (-kv[1], kv[0])))),
        ("eval_corpus_scenarios_by_banned_letter",
         OrderedDict(sorted(eval_letters.items(), key=lambda kv: (-kv[1], kv[0])))),
        ("eval_letters_with_no_pair_scenario",
         sorted(set(eval_letters) - set(pair_letters))),
        ("eval_letter_scenarios_covered",
         sum(c for l, c in eval_letters.items() if l in pair_letters)),
        ("eval_letter_scenarios_total", sum(eval_letters.values())),
    ])
    out["consequence"] = (
        "The preference signal in this file is one-dimensional: the only thing a "
        "rejected text ever does wrong is use a banned letter. DPO trained on it "
        "can move the banned_letter axis and nothing else. It must NOT be "
        "expected to improve the held-out types listed above -- which is exactly "
        "where the tuned checkpoint already scores below base -- and any "
        "held-out gain observed after DPO is unexplained by this data.")
    return out


def attrition_report(result: BuildResult) -> "OrderedDict[str, Any]":
    """The selection effect: which prompts produced a pair, and which did not.

    A source row has no positive side when the teacher never produced a compliant
    draft for that scenario. The surviving pairs are therefore drawn from the
    prompts the teacher found EASIER, so the comparison below asks whether
    "easier" tracks the English frequency of the banned letter.
    """
    rows = result.source_rows
    lookup = result.lookup
    paired_ids = {p.scenario_id for p in result.pairs}
    all_ids = {r.get("scenario_id") for r in rows}
    unpaired_ids = {i for i in all_ids if i not in paired_ids}

    letter_of: Dict[str, Optional[str]] = {}
    for row in rows:
        sid = row.get("scenario_id")
        if sid not in letter_of:
            letter_of[sid] = _banned_letter(row.get("constraints") or [], lookup)

    # Cross-validation: a letter the table does not know would silently vanish
    # from the comparison and quietly bias it.
    unknown = sorted({l for l in letter_of.values()
                      if l is not None and l not in ENGLISH_LETTER_FREQUENCY_PCT})
    if unknown:
        raise RuntimeError(f"banned letters absent from the frequency table: {unknown}")

    xs: List[float] = []
    ys: List[float] = []
    per_letter: Dict[str, List[int]] = {}
    no_letter = 0
    for sid, letter in letter_of.items():
        if letter is None:
            no_letter += 1
            continue
        paired = 1 if sid in paired_ids else 0
        xs.append(ENGLISH_LETTER_FREQUENCY_PCT[letter])
        ys.append(float(paired))
        slot = per_letter.setdefault(letter, [0, 0])
        slot[paired] += 1

    def _mean(values: Sequence[float]) -> Optional[float]:
        return round(sum(values) / len(values), 4) if values else None

    freq_paired = [x for x, y in zip(xs, ys) if y == 1.0]
    freq_unpaired = [x for x, y in zip(xs, ys) if y == 0.0]

    table = []
    for letter in sorted(per_letter, key=lambda l: -ENGLISH_LETTER_FREQUENCY_PCT[l]):
        without, with_pair = per_letter[letter]
        total = with_pair + without
        table.append(OrderedDict([
            ("letter", letter),
            ("english_frequency_pct", ENGLISH_LETTER_FREQUENCY_PCT[letter]),
            ("scenarios_with_a_pair", with_pair),
            ("scenarios_without_a_pair", without),
            ("pair_rate", round(with_pair / total, 4)),
        ]))

    r = _pearson(xs, ys)
    out: "OrderedDict[str, Any]" = OrderedDict()
    out["source_rows"] = len(rows)
    out["rows_with_no_compliant_draft"] = result.drops.get("chosen_absent", 0)
    out["fraction_of_rows_with_no_compliant_draft"] = round(
        result.drops.get("chosen_absent", 0) / len(rows), 6)
    out["distinct_scenarios_in_source"] = len(all_ids)
    out["scenarios_with_at_least_one_pair"] = len(paired_ids)
    out["scenarios_with_no_pair"] = len(unpaired_ids)
    out["fraction_of_scenarios_with_no_pair"] = round(len(unpaired_ids) / len(all_ids), 6)
    out["scenarios_without_a_banned_letter_constraint"] = no_letter
    out["banned_letter_english_frequency"] = OrderedDict([
        ("table", "Lewand relative letter frequencies for English, percent"),
        ("mean_pct_scenarios_with_a_pair", _mean(freq_paired)),
        ("mean_pct_scenarios_with_no_pair", _mean(freq_unpaired)),
        ("delta_pct", None if not (freq_paired and freq_unpaired)
         else round(_mean(freq_paired) - _mean(freq_unpaired), 4)),
        ("point_biserial_r_frequency_vs_produced_a_pair",
         None if r is None else round(r, 4)),
        ("per_letter", table),
    ])
    out["consequence"] = (
        "The pairs are not a random sample of the prompts. The rarer the banned "
        "letter, the likelier the teacher produced a compliant draft and the "
        "likelier a pair exists -- the same rejection-sampling selection effect "
        "docs/BRAINLIFT.md records for v1. Difficulty is therefore correlated "
        "with absence: the hardest prompts contribute no preference signal at "
        "all, so a DPO gain measured on this data is a gain on the easier half "
        "of the distribution.")
    return out


def separation_report(result: BuildResult) -> "OrderedDict[str, Any]":
    """Proof the pairs cannot leak the eval set. Every number is measured."""
    prompts = {p.prompt for p in result.pairs}
    ids = {p.scenario_id for p in result.pairs}
    corpus_ids = set(result.corpus_prompts.values())
    out: "OrderedDict[str, Any]" = OrderedDict()
    out["eval_corpus"] = repo_relative(CORPUS_PATH)
    out["eval_corpus_scenarios"] = len(corpus_ids)
    out["eval_corpus_distinct_rendered_prompts"] = len(result.corpus_prompts)
    out["pair_distinct_rendered_prompts"] = len(prompts)
    out["rendered_prompt_overlap_with_eval_corpus"] = len(prompts & set(result.corpus_prompts))
    out["scenario_id_overlap_with_eval_corpus"] = len(ids & corpus_ids)
    out["every_pair_scenario_id_matches"] = DATAGEN_ID.pattern
    out["non_datagen_scenario_ids_in_pairs"] = sorted(
        i for i in ids if not DATAGEN_ID.match(i))
    out["consequence"] = (
        "Zero overlap on both axes, checked by rendering the eval corpus through "
        "the same renderer rather than by comparing raw task strings. A non-zero "
        "count here is a blocker, not a caveat: the SFT->DPO delta would be "
        "measured on prompts the DPO model had trained on.")
    return out


# ==========================================================================
# Manifest and output
# ==========================================================================

def repo_relative(path: Path) -> str:
    """Repo-relative path when it is inside the repo, absolute otherwise.

    ``Path.relative_to`` raises for a path outside the tree, which would make the
    builder unusable with an ``--out`` anywhere else (a test's tmp_path, for one).
    """
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dataset_text(pairs: Sequence[VerifiedPair]) -> str:
    """The exact bytes of the dataset file. Source order, fixed key order."""
    return "".join(json.dumps(p.to_row(), ensure_ascii=False) + "\n" for p in pairs)


def build_manifest(result: BuildResult, source: Path, out_path: Path,
                   payload: str) -> "OrderedDict[str, Any]":
    """Every field derived from this build. No hand-set numbers, no timestamp."""
    pairs = result.pairs
    per_prompt = Counter(p.scenario_id for p in pairs)
    sids = sorted({p.scenario_id for p in pairs})

    manifest: "OrderedDict[str, Any]" = OrderedDict()
    manifest["builder"] = "datagen.dpo_pairs"
    manifest["rebuild_command"] = (
        f"python -m datagen.dpo_pairs --source {repo_relative(source)}")
    manifest["verified_by"] = "constraints.checkers.check"
    manifest["prompt_renderer"] = "scenarios.render.render_user_message"
    manifest["constraint_library"] = OrderedDict([
        ("path", "data/scenarios/constraint_instances.jsonl"),
        ("instances", len(result.lookup)),
    ])
    manifest["schema"] = OrderedDict([
        ("form", SCHEMA_FORM),
        ("consumer", "trl.DPOTrainer"),
        ("keys", list(SCHEMA_KEYS)),
        ("note", "prompt/chosen/rejected are message lists; TRL applies the chat "
                 "template. The remaining keys are provenance and are ignored by "
                 "the trainer."),
    ])
    manifest["source"] = OrderedDict([
        ("path", repo_relative(source)),
        ("sha256", sha256_file(source)),
        ("rows", len(result.source_rows)),
    ])
    manifest["output"] = OrderedDict([
        ("path", repo_relative(out_path)),
        ("sha256", sha256_text(payload)),
        ("bytes", len(payload.encode("utf-8"))),
        ("rows", len(pairs)),
    ])
    manifest["scenarios"] = OrderedDict([
        ("distinct_scenario_ids", len(sids)),
        ("scenario_id_list_sha256", sha256_text("\n".join(sids))),
        ("pairs_per_scenario", OrderedDict(
            (str(n), c) for n, c in sorted(Counter(per_prompt.values()).items()))),
        ("max_pairs_for_one_scenario", max(per_prompt.values())),
        ("scenario_ids", sids),
    ])
    manifest["k_distribution"] = OrderedDict(
        (str(k), c) for k, c in sorted(Counter(p.k for p in pairs).items()))
    manifest["dropped"] = OrderedDict([
        ("total", sum(result.drops.values())),
        ("by_reason", OrderedDict(
            (reason, result.drops.get(reason, 0)) for reason in DROP_REASONS)),
    ])
    manifest["disclosures"] = OrderedDict([
        ("monoculture", monoculture_report(result)),
        ("attrition", attrition_report(result)),
        ("train_eval_separation", separation_report(result)),
    ])
    manifest["no_dpo_model_has_been_trained_from_this_file"] = True
    return manifest


def manifest_text(manifest: Mapping[str, Any]) -> str:
    return json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"


def output_paths(source: Path) -> Tuple[Path, Path]:
    """``<stem>.dpo_negatives.jsonl`` -> the pairs file and its manifest."""
    name = Path(source).name
    if not name.endswith(".dpo_negatives.jsonl"):
        raise ValueError(f"{name} is not a *.dpo_negatives.jsonl sidecar")
    stem = name[: -len(".dpo_negatives.jsonl")]
    directory = Path(source).parent
    return (directory / f"{stem}.dpo_pairs.jsonl",
            directory / f"{stem}.dpo_pairs.manifest.json")


def write_dataset(source: Path, out_path: Path, manifest_path: Path,
                  result: Optional[BuildResult] = None) -> Tuple[str, str]:
    result = build(source) if result is None else result
    payload = dataset_text(result.pairs)
    manifest = build_manifest(result, source, out_path, payload)
    text = manifest_text(manifest)
    out_path.write_text(payload, encoding="utf-8")
    manifest_path.write_text(text, encoding="utf-8")
    return payload, text


# ==========================================================================
# CLI
# ==========================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m datagen.dpo_pairs",
        description="Build a verified DPO preference dataset from a negatives sidecar.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE),
                        help="a *.dpo_negatives.jsonl file (default: v3)")
    parser.add_argument("--out", default=None, help="output JSONL (default: derived)")
    parser.add_argument("--manifest", default=None, help="output manifest JSON")
    parser.add_argument("--check", action="store_true",
                        help="build and report, write nothing")
    args = parser.parse_args(argv)

    source = Path(args.source).resolve()
    derived_out, derived_manifest = output_paths(source)
    out_path = Path(args.out).resolve() if args.out else derived_out
    manifest_path = (Path(args.manifest).resolve() if args.manifest
                     else derived_manifest)

    try:
        result = build(source)
    except EvalLeak as exc:
        print(f"BLOCKER: {exc}", file=sys.stderr)
        return 2

    payload = dataset_text(result.pairs)
    manifest = build_manifest(result, source, out_path, payload)

    if not args.check:
        out_path.write_text(payload, encoding="utf-8")
        manifest_path.write_text(manifest_text(manifest), encoding="utf-8")

    mono = manifest["disclosures"]["monoculture"]
    attr = manifest["disclosures"]["attrition"]
    sep = manifest["disclosures"]["train_eval_separation"]
    freq = attr["banned_letter_english_frequency"]
    print(f"source        : {manifest['source']['path']} "
          f"({manifest['source']['rows']} rows)")
    print(f"verified pairs: {manifest['output']['rows']} over "
          f"{manifest['scenarios']['distinct_scenario_ids']} scenarios")
    print(f"dropped       : {manifest['dropped']['total']} "
          f"{dict(manifest['dropped']['by_reason'])}")
    print(f"violated types: {dict(mono['violated_constraint_types'])} "
          f"(banned_letter only: {mono['fraction_banned_letter_only']:.4f})")
    print(f"attrition     : {attr['rows_with_no_compliant_draft']}/"
          f"{attr['source_rows']} rows had no compliant draft "
          f"({attr['fraction_of_rows_with_no_compliant_draft']:.4f}); "
          f"mean banned-letter frequency "
          f"{freq['mean_pct_scenarios_with_a_pair']}% paired vs "
          f"{freq['mean_pct_scenarios_with_no_pair']}% unpaired "
          f"(r={freq['point_biserial_r_frequency_vs_produced_a_pair']})")
    print(f"eval overlap  : {sep['rendered_prompt_overlap_with_eval_corpus']} prompts, "
          f"{sep['scenario_id_overlap_with_eval_corpus']} scenario ids")
    if args.check:
        print("--check: nothing written")
    else:
        print(f"wrote {repo_relative(out_path)} and {repo_relative(manifest_path)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
