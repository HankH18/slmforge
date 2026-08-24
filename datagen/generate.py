#!/usr/bin/env python3
"""Teacher-distillation pipeline: draft, check, retry, reject, write.

Modelled on :mod:`scenarios.make_exemplars`, which does this correctly at
six-example scale: the teacher drafts, the pinned checkers verify, a violating
draft is retried, and nothing unverified ships. This is the same discipline with
a sampler in front of it, a rejection ledger behind it, and a disk cache so a
rerun costs nothing.

Which scenarios (``--sampler``)
------------------------------
``default``
    :func:`datagen.sampling.build_scenarios` -- a fresh task from
    :mod:`datagen.task_pool` paired with a wide train-split constraint set.
``attack``
    :func:`datagen.sampling.build_attack_samples` -- scenarios from
    :mod:`datagen.attacks`, whose briefs COMPEL the violation rather than asking
    permission for it (format compulsion, subject compulsion, erosion recap).
    Everything downstream is identical: the same renderer, the same checkers,
    the same pre-write verification, the same auditor. An attack row is exempt
    from nothing.

    One rejection matters more here than anywhere else. Under format compulsion
    the cheapest draft that passes every checker is a REFUSAL -- "I can't give
    you that as a list, here is a paragraph" -- and a few hundred of those would
    teach the model to decline briefs it can complete. ``evasive_draft`` in
    :mod:`datagen.filter` rejects them and the count is reported as
    ``evasive_drafts``, before any GPU time is spent on the rows.

What a run does, per example
----------------------------
1. The sampler pairs a task with a train-split constraint set (never a held-out
   type, always at least one negative constraint).
2. The prompt is :func:`scenarios.render.render_user_message` -- imported, the
   one definition, shared with eval. There is no second renderer here.
3. The teacher drafts. The response is cached on disk by request hash.
4. :func:`datagen.filter.decide` keeps or rejects it on the checkers' verdict.
5. A rejected draft is retried up to ``--attempts`` times, each retry carrying
   the failed draft and the checker's own evidence back to the teacher.
6. Every rejection is recorded with a machine-readable reason code. The fluent,
   on-task, single-violation ones are additionally written to a DPO-negatives
   file as candidate preference pairs.

Before anything is written, every row is re-verified from the values that will
actually be on disk: prompt re-rendered and compared byte for byte, checkers
re-run on the exact stored completion, held-out types re-checked, and the
(task, constraint-set) pair checked against the eval corpus. A row that fails
any of those is a bug in this file, and the run dies rather than shipping it.

Exit codes
----------
``0`` kept >= --min-kept. ``2`` a material shortfall. ``3`` zero rows kept, in
which case the dataset file is NOT created -- a previous incarnation of this
pipeline exited 0 having written a 0-byte dataset, which is a success signal
that means nothing.

Usage
-----
  set -a && . ./.env && set +a
  python datagen/generate.py --n 64 --out data/train/smoke_v0.jsonl

  # an attack block, drafted by a subagent teacher rather than an API:
  python datagen/generate.py --sampler attack --n 480 \
      --out data/train/v4_attacks.jsonl \
      --teacher anthropic:claude-opus-5 --drafts drafts/v4_attacks.jsonl \
      --attempts 1

``--drafts`` is what makes the second form runnable without a network call: the
teacher generator is a file of drafts keyed by scenario id (see
:func:`drafts_generator`) rather than a provider client. It is deliberately
paired with a mandatory explicit ``--teacher``, because the ledger has to name
whoever actually wrote the text.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from constraints import registry                                  # noqa: E402
from harness import run_checkers                                  # noqa: E402
from scenarios.paths import load_corpus, load_instances, load_smoke  # noqa: E402
from scenarios.render import render_user_message                  # noqa: E402

from datagen import DatagenError                                  # noqa: E402
from datagen.attacks import ATTACK_SHAPES, POOL_NAMES, POOL_ROLE  # noqa: E402
from datagen.filter import decide, fluency_report                 # noqa: E402
from datagen.sampling import (DEFAULT_ATTACK_POOL, LETTER_MIXES,  # noqa: E402
                              SAMPLERS, assert_has_negative,
                              assert_train_split_only, attack_shape_counts,
                              build_attack_samples, build_scenarios,
                              eval_pair_keys, letter_of, pair_key, type_shares)
from datagen.task_pool import assert_disjoint_from_corpus         # noqa: E402
from datagen.teacher import (CACHE_DIR, TeacherCache, call_teacher,  # noqa: E402
                             repair_turns, resolve_teacher)

__all__ = ["generate", "write_outputs", "sidecar_paths", "summarise",
           "letter_histograms", "drafts_generator", "build_samples", "main"]

#: Fraction of --n that must survive before the batch counts as usable.
DEFAULT_KEEP_FRACTION = 0.75

VERIFIED_BY = "constraints.checkers.check"


def sidecar_paths(out: Path) -> Dict[str, Path]:
    """The three files that travel with a dataset, derived from its path."""
    out = Path(out)
    stem = out.name[: -len(out.suffix)] if out.suffix else out.name
    return {"dataset": out,
            "rejected": out.with_name(f"{stem}.rejected.jsonl"),
            "dpo_negatives": out.with_name(f"{stem}.dpo_negatives.jsonl"),
            "report": out.with_name(f"{stem}.report.json")}


def _row_for(sample, draft: str, attempt: int, teacher: str, reason: str,
             seed: int, lookup: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    scenario = sample.scenario
    provenance = {
        "teacher": teacher,
        "attempt": attempt,
        "teacher_resolved_by": reason,
        "task_id": sample.task_id,
        "lexicon": sample.lexicon,
        "nudge": sample.nudge_id,
        "distractor": sample.distractor,
        "seed": seed,
        "verified_by": VERIFIED_BY,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Only on an attack row, and only when there is something to say. A row from
    # the default sampler keeps the provenance shape every shipped dataset has.
    attack = getattr(sample, "attack", None)
    if attack:
        provenance["attack"] = dict(attack)
    return {
        "messages": [
            {"role": "user", "content": render_user_message(scenario, lookup)},
            {"role": "assistant", "content": draft},
        ],
        "constraints": list(scenario["constraints"]),
        "provenance": provenance,
        "scenario": dict(scenario),
    }


#: What is said back to a teacher that declined the brief instead of writing it.
#: Deliberately NOT a copy of :data:`datagen.teacher.REPAIR_HEADER`: that one
#: quotes a checker's evidence for a broken rule, and nothing is broken here --
#: the draft is compliant and useless. The instruction is the opposite one.
EVASION_HEADER = (
    "That response did not do the task. It declined it, or discussed the "
    "requirements, instead of writing the piece that was asked for. The "
    "requirements CAN all be satisfied at once: write the piece so that every "
    "requirement holds, in whatever form makes that possible. Do not mention the "
    "requirements, do not explain what you did or did not do, do not apologise, "
    "and do not offer an alternative version -- return only the finished piece."
)


def _evasion_repair_turns(base_turns: Sequence[Mapping[str, str]], draft: str,
                          cues: Sequence[Any]) -> List[Dict[str, str]]:
    """The turn list for a retry after an ``evasive_draft`` rejection.

    Carries the detector's own evidence back -- the rule that fired and the
    sentence that fired it -- so the teacher is told what a machine actually
    measured rather than a paraphrase of it.
    """
    lines = [f"- {cue.rule}: {cue.span}" for cue in cues]
    turns = [dict(t) for t in base_turns]
    turns.append({"role": "assistant", "content": draft})
    turns.append({"role": "user",
                  "content": EVASION_HEADER + "\n\nWhat gave it away:\n"
                             + "\n".join(lines)})
    return turns


def _verify_before_write(row: Mapping[str, Any],
                         lookup: Mapping[str, Mapping[str, Any]],
                         forbidden_pairs: set) -> None:
    """Everything the auditor will check, checked here first, on the exact values
    that are about to be written."""
    scenario = row["scenario"]
    cids = list(scenario["constraints"])
    if list(row["constraints"]) != cids:
        raise DatagenError(f"{scenario['scenario_id']}: row constraints disagree with "
                           f"its scenario's")
    assert_train_split_only(cids, lookup)
    assert_has_negative(cids, lookup)

    rendered = render_user_message(scenario, lookup)
    if row["messages"][0]["content"] != rendered:
        raise DatagenError(
            f"{scenario['scenario_id']}: the stored user message is not what the "
            f"shared renderer produces. Training on it would measure the model "
            f"under a prompt it never saw.")

    completion = row["messages"][1]["content"]
    if not isinstance(completion, str) or not completion.strip():
        raise DatagenError(f"{scenario['scenario_id']}: empty completion")
    results = run_checkers(scenario, lookup, completion)
    bad = [r for r in results if not r["pass"]]
    if bad:
        raise DatagenError(
            f"{scenario['scenario_id']}: about to write a row that FAILS "
            f"{[r['constraint_id'] for r in bad]} on the exact stored text")

    if pair_key(scenario["task"], cids) in forbidden_pairs:
        raise DatagenError(
            f"{scenario['scenario_id']}: its (task, constraint-set) pair is in the "
            f"eval corpus -- that is contamination")


def generate(samples: Sequence[Any],
             generator,
             teacher: str,
             teacher_reason: str,
             lookup: Mapping[str, Mapping[str, Any]],
             forbidden_pairs: set,
             attempts: int = 4,
             cache: Optional[TeacherCache] = None,
             seed: int = 0,
             workers: int = 8,
             log=print) -> Dict[str, Any]:
    """Run the pipeline over pre-built samples. Returns rows + ledgers + stats.

    Takes the generator as an argument rather than building one, so the whole
    pipeline is exercisable with a scripted teacher and no network.
    """
    if attempts < 1:
        raise DatagenError(f"--attempts must be at least 1, got {attempts}")
    import eval as E  # the live harness, for its prompt plumbing only

    cache = cache if cache is not None else TeacherCache()
    if not getattr(generator, "thread_safe", False):
        workers = 1
    workers = max(1, min(int(workers), len(samples) or 1))

    lock = threading.Lock()
    kept: List[Optional[Dict[str, Any]]] = [None] * len(samples)
    rejections: List[Dict[str, Any]] = []
    negatives: List[Dict[str, Any]] = []
    give_ups: List[Dict[str, Any]] = []

    def one(index_sample):
        index, sample = index_sample
        scenario = sample.scenario
        prompt = E.build_prompt("zero_shot", scenario, lookup, None)
        base_turns = list(prompt["turns"])
        turns = base_turns
        local_rejections: List[Dict[str, Any]] = []
        local_negatives: List[Dict[str, Any]] = []
        row = None
        kept_attempt = None

        for attempt in range(1, attempts + 1):
            request = {"system": prompt["system"], "turns": turns}
            try:
                raw = call_teacher(generator, scenario, request, lookup, teacher, cache)
            except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
                local_rejections.append({
                    "scenario_id": scenario["scenario_id"], "task_id": sample.task_id,
                    "attempt": attempt, "reason": "teacher_error",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "constraints": list(scenario["constraints"]),
                    "failures": [], "draft": None})
                break

            draft = raw.strip()
            verdict = decide(draft, scenario, lookup)
            if verdict.keep:
                row = _row_for(sample, draft, attempt, teacher, teacher_reason,
                               seed, lookup)
                kept_attempt = attempt
                break

            report = fluency_report(draft, scenario["task"], verdict.results)
            local_rejections.append({
                "scenario_id": scenario["scenario_id"], "task_id": sample.task_id,
                "attempt": attempt, "reason": verdict.reason,
                "constraints": list(scenario["constraints"]),
                "failures": verdict.failures,
                # The rules that fired, with the sentence that fired each, so an
                # evasive_draft rejection can be read back and argued with rather
                # than taken on trust.
                "evasion": [cue.as_dict() for cue in verdict.evasion],
                "draft": draft,
                "fluency": report.as_dict()})
            if report.fluent:
                local_negatives.append({
                    "scenario_id": scenario["scenario_id"],
                    "prompt": render_user_message(scenario, lookup),
                    "rejected": draft,
                    "chosen": None,          # filled in below if a later attempt won
                    "violated": verdict.failures,
                    "constraints": list(scenario["constraints"]),
                    "scenario": dict(scenario),
                    "fluency": report.as_dict(),
                    "provenance": {"teacher": teacher, "attempt": attempt,
                                   "task_id": sample.task_id, "seed": seed}})
            if verdict.reason == "constraint_violation":
                turns = repair_turns(base_turns, draft, verdict.failures)
            elif verdict.reason == "evasive_draft":
                # There IS something to quote back here, unlike an empty draft:
                # the sentence that declined the job. Without this branch the
                # retry re-sends the identical prompt, gets the identical
                # refusal, and the scenario is lost -- which on an attack block
                # would silently drop exactly the hardest rows.
                turns = _evasion_repair_turns(base_turns, draft, verdict.evasion)
            else:
                # Nothing to quote back at the teacher for an empty or degenerate
                # draft, so ask again from the top rather than send it its own
                # silence.
                turns = base_turns

        with lock:
            kept[index] = row
            rejections.extend(local_rejections)
            if row is not None:
                for neg in local_negatives:
                    neg["chosen"] = row["messages"][1]["content"]
            negatives.extend(local_negatives)
            if row is None:
                give_ups.append({
                    "scenario_id": scenario["scenario_id"],
                    "task_id": sample.task_id,
                    "constraints": list(scenario["constraints"]),
                    "attempts": attempts,
                    "last_reason": (local_rejections[-1]["reason"]
                                    if local_rejections else "no_attempt")})
            done = sum(1 for r in kept if r is not None) + len(give_ups)
        if done % 16 == 0 or done == len(samples):
            log(f"  {done}/{len(samples)} examples resolved "
                f"({sum(1 for r in kept if r is not None)} kept)")

    if workers == 1:
        for pair in enumerate(samples):
            one(pair)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, list(enumerate(samples))))

    rows = [r for r in kept if r is not None]
    for row in rows:
        _verify_before_write(row, lookup, forbidden_pairs)

    return {"rows": rows, "rejections": rejections, "negatives": negatives,
            "give_ups": give_ups,
            "cache": {"hits": cache.hits, "misses": cache.misses}}


def letter_histograms(rows: Sequence[Mapping[str, Any]],
                      rejections: Sequence[Mapping[str, Any]],
                      lookup: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Realised ``banned_letter`` distribution: kept, rejected, and keep rate.

    This is the evidence that a ``--letter-mix`` actually moved the curriculum.
    Without it the intervention is an assertion: the REQUEST distribution is
    knowable from the sampler alone, but what survives the rejection filter is
    not, and the survivors are what the model trains on.

    Counted the same way on both sides -- occurrences of a ``banned_letter``
    instance in a row's constraint set -- so the two are comparable and the keep
    rate is well defined. Rejections are per ATTEMPT, not per scenario, matching
    :data:`datagen.sampling.V1_LETTER_OUTCOMES`, so this run's keep rates can be
    read straight against that measured baseline.
    """
    def tally(records: Sequence[Mapping[str, Any]]) -> Counter:
        counts: Counter = Counter()
        for record in records:
            for cid in record.get("constraints", []):
                con = lookup.get(cid)
                if con is not None and con.get("type") == "banned_letter":
                    counts[letter_of(cid, lookup)] += 1
        return counts

    kept = tally(rows)
    rejected = tally(rejections)
    letters = sorted(set(kept) | set(rejected))
    return {
        "kept": {letter: kept[letter] for letter in letters},
        "rejected": {letter: rejected[letter] for letter in letters},
        "keep_rate": {letter: round(kept[letter] / (kept[letter] + rejected[letter]), 4)
                      for letter in letters
                      if kept[letter] + rejected[letter] > 0},
    }


def summarise(rows: Sequence[Mapping[str, Any]],
              rejections: Sequence[Mapping[str, Any]],
              negatives: Sequence[Mapping[str, Any]],
              give_ups: Sequence[Mapping[str, Any]],
              lookup: Mapping[str, Mapping[str, Any]],
              requested: int) -> Dict[str, Any]:
    """Every reported number, computed from the rows themselves."""
    row_types = [[lookup[c]["type"] for c in row["constraints"]] for row in rows]
    shares = type_shares(row_types)
    type_counts = Counter(t for types in row_types for t in types)
    evasive = [r for r in rejections if r["reason"] == "evasive_draft"]
    return {
        "requested": requested,
        "kept": len(rows),
        "accept_rate": (len(rows) / requested) if requested else 0.0,
        "gave_up": len(give_ups),
        "rejections": len(rejections),
        "rejection_reasons": dict(Counter(r["reason"] for r in rejections)),
        # Called out separately from rejection_reasons ON PURPOSE. This is the
        # number that says how hard the teacher tried to talk its way out of the
        # attack instead of answering it, and it has to be readable at a glance
        # BEFORE anyone spends GPU time on the block. Zero here on an attack run
        # is a claim to be suspicious of, not a relief: check that the gate is
        # wired at all (tests/test_datagen.py proves it bites).
        "evasive_drafts": len(evasive),
        "evasion_rules": dict(Counter(
            cue["rule"] for r in evasive for cue in r.get("evasion", []))),
        "attack_shapes": dict(Counter(
            row["provenance"]["attack"]["shape"] for row in rows
            if row["provenance"].get("attack"))),
        "rejected_constraint_types": dict(Counter(
            f["type"] for r in rejections for f in r.get("failures", []))),
        "dpo_negatives": len(negatives),
        "attempt_histogram": dict(Counter(row["provenance"]["attempt"] for row in rows)),
        "k_distribution": dict(Counter(row["scenario"]["k"] for row in rows)),
        "pressure_distribution": dict(Counter(row["scenario"]["pressure"] for row in rows)),
        "banned_letter_distribution": letter_histograms(rows, rejections, lookup),
        "type_counts": dict(type_counts),
        "type_shares": {t: round(v, 4) for t, v in shares.items()},
        "max_type_share": round(max(shares.values()), 4) if shares else 0.0,
        "distractor_examples": sum(1 for row in rows
                                   if row["provenance"].get("distractor")),
        "unique_tasks": len({row["provenance"]["task_id"] for row in rows}),
    }


class DraftFile:
    """A teacher generator backed by a file of drafts somebody already wrote.

    :func:`datagen.teacher.call_teacher` takes an INJECTED generator, so the
    thing on the other end of the pipeline does not have to be a provider
    client. This is the CLI's way of using that: a JSONL file of

        {"scenario_id": "atk-a-00001", "draft": "..."}

    (or ``"drafts": [first_attempt, second_attempt, ...]`` when a scenario was
    redrafted), which is what a SUBAGENT teacher can produce -- and a subagent
    can do something no API teacher can, which is import
    :mod:`constraints.checkers` and iterate its own draft against the real
    checkers before writing it down.

    It refuses to invent anything. A scenario with no draft in the file raises,
    which the pipeline records as a ``teacher_error`` against that scenario id
    rather than quietly shipping fewer rows than were asked for.
    """

    thread_safe = True

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise DatagenError(f"{self.path} does not exist, so there are no drafts "
                               f"to distil from")
        self.drafts: Dict[str, List[str]] = {}
        for number, line in enumerate(self.path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatagenError(f"{self.path}:{number} is not valid JSON: {exc}")
            scenario_id = record.get("scenario_id")
            drafts = record.get("drafts")
            if drafts is None and isinstance(record.get("draft"), str):
                drafts = [record["draft"]]
            if not isinstance(scenario_id, str) or not scenario_id.strip():
                raise DatagenError(f"{self.path}:{number} has no 'scenario_id'")
            if (not isinstance(drafts, list) or not drafts
                    or not all(isinstance(d, str) for d in drafts)):
                raise DatagenError(
                    f"{self.path}:{number} ({scenario_id}) has no 'draft' string and "
                    f"no non-empty 'drafts' list of strings")
            if scenario_id in self.drafts:
                raise DatagenError(
                    f"{self.path}:{number}: two records for scenario {scenario_id}; "
                    f"which one is the teacher's answer is then undecidable")
            self.drafts[scenario_id] = list(drafts)
        if not self.drafts:
            raise DatagenError(f"{self.path} contains no drafts")

    def __call__(self, scenario: Mapping[str, Any], prompt: Mapping[str, Any],
                 constraint_lookup=None) -> str:
        scenario_id = scenario["scenario_id"]
        drafts = self.drafts.get(scenario_id)
        if drafts is None:
            raise DatagenError(
                f"{self.path} carries no draft for scenario {scenario_id}")
        # One extra (assistant, user) pair is appended per retry, exactly as in
        # datagen.teacher.repair_turns.
        index = max(0, (len(prompt["turns"]) - 1) // 2)
        return drafts[min(index, len(drafts) - 1)]


def drafts_generator(path: Path) -> DraftFile:
    """A teacher generator reading pre-authored drafts from ``path``."""
    return DraftFile(path)


def build_samples(sampler: str, n: int,
                  instances: Mapping[str, Mapping[str, Any]],
                  corpus_rows: Sequence[Mapping[str, Any]],
                  seed: int,
                  letter_mix: str = "uniform",
                  attack_pool: str = DEFAULT_ATTACK_POOL,
                  shapes: Sequence[str] = ATTACK_SHAPES,
                  log=print) -> List[Any]:
    """The one place that chooses a sampler. Both return :class:`Sample`."""
    if sampler == "default":
        return build_scenarios(n, instances, corpus_rows, seed=seed,
                               letter_mix=letter_mix)
    if sampler == "attack":
        return build_attack_samples(n, instances, corpus_rows, seed=seed,
                                    pool=attack_pool, shapes=shapes, log=log)
    raise DatagenError(
        f"unknown sampler {sampler!r}; expected one of {', '.join(SAMPLERS)}")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_outputs(out: Path, result: Mapping[str, Any],
                  report: Mapping[str, Any]) -> Dict[str, Path]:
    """Write the dataset and its ledgers. Refuses to create an empty dataset."""
    paths = sidecar_paths(Path(out))
    if not result["rows"]:
        raise DatagenError(
            "zero rows survived the filter; refusing to write an empty dataset "
            "file. An empty file that exists is worse than no file: it looks like "
            "a successful run to everything downstream.")
    _write_jsonl(paths["dataset"], result["rows"])
    _write_jsonl(paths["rejected"], result["rejections"])
    _write_jsonl(paths["dpo_negatives"], result["negatives"])
    paths["report"].write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, required=True, help="examples to attempt")
    ap.add_argument("--out", required=True, help="dataset JSONL path")
    ap.add_argument("--teacher", default=None,
                    help="model spec; default resolves from configs/models.json")
    ap.add_argument("--attempts", type=int, default=4,
                    help="teacher attempts per example before giving up")
    ap.add_argument("--min-kept", type=int, default=None,
                    help=f"minimum kept rows; default ceil({DEFAULT_KEEP_FRACTION:.2f} * n)")
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--sampler", choices=list(SAMPLERS), default="default",
                    help="'default' draws fresh tasks from datagen.task_pool; "
                         "'attack' draws compulsion scenarios from datagen.attacks "
                         "(format compulsion, subject compulsion, erosion recap), "
                         "the attack families the shipped corpus has never carried")
    ap.add_argument("--attack-pool", choices=list(POOL_NAMES),
                    default=DEFAULT_ATTACK_POOL,
                    help="which attack pool to draw from (--sampler attack). Only a "
                         "pool whose datagen.attacks role is 'train' is accepted; "
                         "pool B is the held-out attack eval")
    ap.add_argument("--attack-shapes", default=",".join(ATTACK_SHAPES),
                    help=f"comma-separated subset of {', '.join(ATTACK_SHAPES)}; "
                         f"shapes are interleaved so a small block spans all of them")
    ap.add_argument("--drafts", default=None,
                    help="JSONL of pre-authored drafts keyed by scenario_id, used "
                         "INSTEAD of calling a provider. Requires an explicit "
                         "--teacher naming whoever wrote them")
    ap.add_argument("--letter-mix", choices=list(LETTER_MIXES), default="uniform",
                    help="how a banned_letter instance is picked: 'uniform' is one "
                         "per letter with equal probability; 'hard' weights the "
                         "REQUEST by English letter frequency divided by the "
                         "measured teacher keep rate, so the KEPT distribution "
                         "lands on English frequency instead of on whichever "
                         "letters the teacher finds easy")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-cache", action="store_true",
                    help="bypass datagen/.cache (this re-bills the teacher)")
    args = ap.parse_args(argv)

    if args.n <= 0:
        print(f"--n must be positive, got {args.n}", file=sys.stderr)
        return 2
    shapes = [s.strip() for s in str(args.attack_shapes).split(",") if s.strip()]
    unknown = [s for s in shapes if s not in ATTACK_SHAPES]
    if unknown or not shapes:
        print(f"--attack-shapes must be a non-empty subset of "
              f"{', '.join(ATTACK_SHAPES)}; got {args.attack_shapes!r}",
              file=sys.stderr)
        return 2
    if args.drafts and not args.teacher:
        # The ledger records the teacher on every row. Drafts written by
        # somebody other than the configured default must not be attributed to
        # that default, so the attribution is required rather than guessed.
        print("--drafts needs an explicit --teacher naming whoever wrote them; "
              "the resolved default would put a model's name on text it did not "
              "write.", file=sys.stderr)
        return 2
    min_kept = (args.min_kept if args.min_kept is not None
                else math.ceil(DEFAULT_KEEP_FRACTION * args.n))

    instances = load_instances()
    corpus = list(load_corpus()) + list(load_smoke())
    assert_disjoint_from_corpus(corpus)
    forbidden = eval_pair_keys(corpus)

    teacher, reason = resolve_teacher(args.teacher)
    print(f"teacher: {teacher}  ({reason})")
    print(f"held-out types never sampled: "
          f"{', '.join(registry.types_by_split('heldout_type'))}")
    print(f"sampler: {args.sampler}")
    if args.sampler == "default":
        print(f"banned_letter request mix: {args.letter_mix}")

    try:
        samples = build_samples(args.sampler, args.n, instances, corpus,
                                seed=args.seed, letter_mix=args.letter_mix,
                                attack_pool=args.attack_pool, shapes=shapes)
    except DatagenError as exc:
        print(f"the {args.sampler} sampler refused to build this batch:\n{exc}",
              file=sys.stderr)
        return 2
    print(f"sampled {len(samples)} scenarios from {len({s.task_id for s in samples})} "
          f"distinct task texts")
    if args.sampler == "attack":
        print(f"attack shapes: {attack_shape_counts(samples)}")

    import eval as E
    generator = (drafts_generator(Path(args.drafts)) if args.drafts
                 else E.make_generator(teacher))
    cache = TeacherCache(CACHE_DIR, enabled=not args.no_cache)

    result = generate(samples, generator, teacher, reason, instances, forbidden,
                      attempts=args.attempts, cache=cache, seed=args.seed,
                      workers=args.workers)

    report = summarise(result["rows"], result["rejections"], result["negatives"],
                       result["give_ups"], instances, args.n)
    report.update({"teacher": teacher, "teacher_resolved_by": reason,
                   "attempts": args.attempts, "seed": args.seed,
                   "letter_mix": args.letter_mix,
                   "sampler": args.sampler,
                   "drafts_file": str(args.drafts) if args.drafts else None,
                   "min_kept": min_kept, "cache": result["cache"],
                   "generated_utc": datetime.now(timezone.utc).isoformat()})
    if args.sampler == "attack":
        report.update({"attack_pool": args.attack_pool,
                       "attack_pool_role": POOL_ROLE[args.attack_pool],
                       "attack_shapes_requested": shapes})

    if not result["rows"]:
        print("\nZERO rows kept. No dataset written.", file=sys.stderr)
        print(f"rejection reasons: {report['rejection_reasons']}", file=sys.stderr)
        return 3

    paths = write_outputs(Path(args.out), result, report)

    print("\n--- realised distribution -------------------------------------")
    print(f"kept {report['kept']}/{report['requested']} "
          f"(accept rate {report['accept_rate']:.1%}), gave up on {report['gave_up']}")
    print(f"k: {report['k_distribution']}   pressure: {report['pressure_distribution']}")
    print(f"type shares: {report['type_shares']}  (max {report['max_type_share']:.1%})")
    if report["attack_shapes"]:
        print(f"attack shapes kept: {report['attack_shapes']}")
    # Printed on EVERY run, attack or not, and printed even when it is zero: the
    # count of drafts that passed every checker by declining the job is the one
    # number nobody should have to go looking for in a sidecar.
    print(f"evasive drafts rejected: {report['evasive_drafts']} "
          f"{report['evasion_rules']}")
    letters = report["banned_letter_distribution"]
    print(f"banned_letter ({args.letter_mix}) kept: {letters['kept']}")
    print(f"banned_letter ({args.letter_mix}) rejected: {letters['rejected']}")
    print(f"banned_letter ({args.letter_mix}) keep rate: {letters['keep_rate']}")
    print(f"attempts: {report['attempt_histogram']}")
    print(f"rejections: {report['rejections']} {report['rejection_reasons']}")
    print(f"candidate DPO negatives: {report['dpo_negatives']}")
    print(f"cache: {report['cache']}")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    if report["kept"] < min_kept:
        print(f"\nSHORTFALL: kept {report['kept']} < --min-kept {min_kept}",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
