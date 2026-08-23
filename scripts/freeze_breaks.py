#!/usr/bin/env python3
"""Freeze the partner's confirmed jailbreaks into a committed in-repo exam.

WHY THIS FILE EXISTS
--------------------
The assignment is graded on one number: *how many of the partner's breaks still
break the model, before and after the retrain*. A number like that is only worth
something if the exam it was measured against is fixed, public and tamper-evident.
An exam that lives in ``~/Downloads`` is none of those: it is one editor window
away from being "improved" between the before-run and the after-run, and nobody
reading the repo can tell.

So this script does three things and nothing else:

1. reads the partner's JSONL,
2. **validates it hard** -- 20 rows, unique ids, every row a confirmed
   ``violated: true`` / ``break_kind: "checker"`` row, the multi-turn rows
   carrying exactly three turns in the order user -> assistant -> user,
3. writes ``data/eval/partner_breaks.jsonl`` deterministically and prints its
   sha256, which is pinned in :data:`EXPECTED_SHA256` below.

NOTHING HERE IS A SECOND IMPLEMENTATION
---------------------------------------
The constraint each row is about is minted through
``constraints.registry.build_constraint``, so the type table, the family and the
train/heldout split come from the pinned registry rather than from a copy typed
out here. The re-check that proves each row really is a break runs through
``constraints.checkers.check`` -- the same pinned entry point every published
number in this repo is re-derived from.

WHAT THE FROZEN ROW CARRIES
---------------------------
Every partner field, verbatim (``prompt`` included -- string for the 15
single-turn rows, the partner's own three message dicts for the 5 multi-turn
rows), plus four fields derived from the pinned registry:

===============  ==============================================================
``constraint_type``  the registry type name (the partner writes ``banned_word``
                     and ``banned_phrase``; the registry calls those
                     ``banned_words`` / ``banned_phrases``)
``split``            ``registry.split_of(constraint_type)``
``trainable``        ``split == "train"``. Three rows are ``false``: their type
                     is ``banned_punctuation``, a held-out type, and
                     ``datagen/audit.py`` fails any dataset that uses one. They
                     can be measured forever and trained on never.
``multiturn``        derived from the shape of ``prompt``, and cross-checked
                     against the partner's own ``multiturn`` field wherever the
                     partner set one.
===============  ==============================================================

Deliberately NOT carried: any re-derived checker evidence. Evidence is a
property of today's checkers, and baking it into the artifact would make a
legitimate checker improvement look like exam tampering. The re-check runs at
freeze time and again in ``tests/test_partner_breaks.py``; the artifact stays a
record of what the partner sent.

USAGE
-----
::

    python scripts/freeze_breaks.py --src ~/Downloads/confirmed_breaks.jsonl
    python scripts/freeze_breaks.py --check     # re-validate what is committed

``--check`` is the one a grader (or CI) runs: it re-reads the committed
artifact, re-runs every validation, re-derives the sha256 and compares it with
the pin in this file. It never touches the partner's original.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from constraints import checkers                            # noqa: E402
from constraints import registry                            # noqa: E402
from constraints.registry import ConstraintError            # noqa: E402

#: Where the frozen exam lives. Committed, so the before-run and the after-run
#: provably faced the same 20 prompts.
DEFAULT_OUT = REPO_ROOT / "data" / "eval" / "partner_breaks.jsonl"

#: The partner's delivery. Outside the repo on purpose -- it is an input, not an
#: artifact, and this script exists to stop it being the thing we measure against.
DEFAULT_SRC = Path.home() / "Downloads" / "confirmed_breaks.jsonl"

#: The exam is exactly this big. Not "at least": a 19-row or 21-row file is a
#: different exam, and the headline number would silently change denominator.
EXPECTED_ROWS = 20

#: ...of which exactly this many are the multi-turn erosion attacks. They are the
#: out-of-distribution half of the headline and are reported separately.
EXPECTED_MULTITURN = 5

#: The turn shape a multi-turn row must have: the partner's opening ask, the
#: model's compliant answer, and the follow-up whose FORM requires the violation.
MULTITURN_ROLES = ("user", "assistant", "user")

#: Every row must be a confirmed break, established by a checker rather than by
#: anyone's judgement. A row that is here for another reason is not exam material.
REQUIRED_VIOLATED = True
REQUIRED_BREAK_KIND = "checker"

#: sha256 of the committed data/eval/partner_breaks.jsonl.
#:
#: Pinned HERE, in code, rather than in a sidecar file next to the artifact: a
#: sidecar is edited in the same gesture as the thing it describes, whereas
#: changing this line is a diff in a source file that reviewers read. `--check`
#: re-derives the digest and refuses any mismatch, so editing the exam without
#: editing this constant turns the check red -- and editing both is a deliberate,
#: visible act.
EXPECTED_SHA256 = "c7b1409ceb3d39266e4f1d4466b26790ed6e2e6f91bf775425d15f153aa4cefe"

#: The partner's family names -> the pinned registry's type names. Built from
#: ``registry.list_types()`` so a registry rename cannot leave a stale copy here,
#: plus the two singular aliases the partner's harness emits.
FAMILY_ALIASES = {"banned_word": "banned_words", "banned_phrase": "banned_phrases"}


def family_to_type(family: Any) -> str:
    """Partner family name -> pinned registry type. Unknown names raise."""
    known = registry.list_types()
    if isinstance(family, str):
        if family in known:
            return family
        alias = FAMILY_ALIASES.get(family)
        if alias in known:
            return alias
    raise ConstraintError(
        f"family {family!r} is not a constraint type this repo can check; "
        f"registry types are {known}, aliases {sorted(FAMILY_ALIASES)}")


def params_for(ctype: str, arg: Any) -> Dict[str, Any]:
    """``(type, arg)`` -> the params dict that type's checker expects.

    The partner ships one scalar ``arg`` per row; the registry wants a typed
    params object. Validated through ``registry.validate_params`` so a wrong
    shape raises here instead of producing a checker verdict nobody can trust.
    """
    if ctype in ("no_lists", "no_quotation_marks"):
        if arg is not None:
            raise ConstraintError(f"{ctype} takes no argument, got arg={arg!r}")
        params: Dict[str, Any] = {}
    elif ctype == "banned_words":
        params = {"words": [arg]}
    elif ctype == "banned_phrases":
        params = {"phrases": [arg]}
    elif ctype == "banned_letter":
        params = {"letter": arg}
    elif ctype == "banned_punctuation":
        params = {"marks": [arg]}
    else:  # pragma: no cover - unreachable while the registry is pinned
        raise ConstraintError(f"no params rule for registry type {ctype!r}")
    registry.validate_params(ctype, params)
    return params


# Every pinned type must have a params rule, or a row of that type would raise
# at freeze time instead of being frozen. Checked at import, where it is loud.
assert set(registry.list_types()) == {
    "banned_words", "banned_phrases", "banned_letter", "banned_punctuation",
    "sentence_initial_letter", "no_lists", "no_quotation_marks"}, (
    "constraints.registry.TYPES changed; params_for() needs the new type")


def constraint_for(row: Mapping[str, Any], index: int) -> Dict[str, Any]:
    """The ad-hoc constraint a frozen row is about.

    Minted by ``registry.build_constraint`` so id prefix, family and split are
    the registry's, never this file's. ``index`` is the row's position, which
    makes the minted id stable for a given exam.
    """
    ctype = row.get("constraint_type") or family_to_type(row.get("family"))
    return registry.build_constraint(ctype, params_for(ctype, row.get("arg")), index)


class FreezeError(Exception):
    """Validation failed. Carries every problem, not just the first."""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__("\n".join(f"  - {p}" for p in self.problems))


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _turns_of(prompt: Any) -> Optional[List[Dict[str, str]]]:
    """``prompt`` -> the message list to send, or None if it is not a valid shape.

    A string is one user turn. A list is taken verbatim. This is the ONLY place
    the two shapes are reconciled, and ``scripts/replay_breaks.py`` imports it
    rather than re-deriving, so the exam cannot be sent one way and validated
    another.
    """
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}] if prompt.strip() else None
    if isinstance(prompt, list):
        turns = []
        for msg in prompt:
            if not isinstance(msg, dict):
                return None
            role, content = msg.get("role"), msg.get("content")
            if not isinstance(role, str) or not isinstance(content, str) or not content.strip():
                return None
            turns.append({"role": role, "content": content})
        return turns or None
    return None


def turns_of(prompt: Any) -> List[Dict[str, str]]:
    """:func:`_turns_of` but raising, for callers that already validated."""
    turns = _turns_of(prompt)
    if turns is None:
        raise FreezeError([f"prompt is not a string or a list of message dicts: {prompt!r}"])
    return turns


def validate_rows(rows: Sequence[Mapping[str, Any]], source: str) -> List[Dict[str, Any]]:
    """Every partner row, validated hard and normalised. Raises FreezeError.

    Collects EVERY problem before raising: a validator that stops at the first
    bad row makes a broken delivery take twenty round trips to diagnose.
    """
    problems: List[str] = []
    notes: List[str] = []
    out: List[Dict[str, Any]] = []

    if len(rows) != EXPECTED_ROWS:
        problems.append(f"{source}: {len(rows)} rows, expected exactly {EXPECTED_ROWS}")

    seen: Dict[str, int] = {}
    n_multiturn = 0

    for i, row in enumerate(rows):
        where = f"{source} row {i}"
        if not isinstance(row, dict):
            problems.append(f"{where}: not a JSON object")
            continue
        rid = row.get("id")
        if not isinstance(rid, str) or not rid.strip():
            problems.append(f"{where}: id is missing or not a non-empty string ({rid!r})")
            rid = f"<row {i}>"
        elif rid in seen:
            problems.append(f"{where}: duplicate id {rid!r} (first seen at row {seen[rid]})")
        else:
            seen[rid] = i
        where = f"{source} {rid}"

        if row.get("violated") is not REQUIRED_VIOLATED:
            problems.append(f"{where}: violated is {row.get('violated')!r}, expected "
                            f"{REQUIRED_VIOLATED!r}; this exam is confirmed breaks only")
        if row.get("break_kind") != REQUIRED_BREAK_KIND:
            problems.append(f"{where}: break_kind is {row.get('break_kind')!r}, expected "
                            f"{REQUIRED_BREAK_KIND!r}; a break nobody's checker confirms "
                            f"is not a measurable break")

        prompt = row.get("prompt")
        turns = _turns_of(prompt)
        if turns is None:
            problems.append(f"{where}: prompt is neither a non-empty string nor a list of "
                            f"message dicts with non-empty string role/content")
            continue
        multiturn = isinstance(prompt, list)
        if multiturn:
            n_multiturn += 1
            roles = tuple(t["role"] for t in turns)
            if roles != MULTITURN_ROLES:
                problems.append(f"{where}: multi-turn prompt has {len(turns)} turn(s) "
                                f"{roles}, expected {len(MULTITURN_ROLES)} turns "
                                f"{MULTITURN_ROLES}")
        stated = row.get("multiturn")
        if stated is None:
            notes.append(f"{rid}: partner row carries no 'multiturn' field; derived "
                         f"{multiturn} from the shape of 'prompt'")
        elif bool(stated) != multiturn:
            problems.append(f"{where}: row says multiturn={stated!r} but the prompt is "
                            f"{'a message list' if multiturn else 'a string'}")

        response = row.get("response")
        if not isinstance(response, str) or not response.strip():
            problems.append(f"{where}: response is missing or empty; the recorded break "
                            f"cannot be re-checked without it")
            continue

        try:
            constraint = constraint_for(row, i)
        except ConstraintError as exc:
            problems.append(f"{where}: {exc}")
            continue

        # The load-bearing check: does the partner's own recorded response still
        # violate, under THIS repo's pinned checkers? A row that does not is not
        # evidence of anything and must never enter the exam.
        try:
            verdict = checkers.check(dict(constraint), response)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            problems.append(f"{where}: checker {constraint['type']} raised "
                            f"{type(exc).__name__}: {exc}")
            continue
        if verdict["pass"]:
            problems.append(f"{where}: row claims violated=true, but "
                            f"constraints.checkers.check({constraint['type']}) PASSES the "
                            f"recorded response. Either the response or the checker moved.")
            continue

        if "refusal" not in row:
            notes.append(f"{rid}: partner row carries no 'refusal' field; frozen as null")

        frozen = {
            # --- the partner's fields, verbatim
            "id": rid,
            "category": row.get("category"),
            "family": row.get("family"),
            "arg": row.get("arg"),
            "prompt": prompt,
            "response": response,
            "violated": row.get("violated"),
            "checker_detail": row.get("checker_detail"),
            "refusal": row.get("refusal"),
            "break_kind": row.get("break_kind"),
            # --- derived from the pinned registry / the prompt's shape
            "constraint_type": constraint["type"],
            "split": registry.split_of(constraint["type"]),
            "trainable": registry.split_of(constraint["type"]) == "train",
            "multiturn": multiturn,
        }
        out.append(frozen)

    if not problems and len(out) != EXPECTED_ROWS:
        problems.append(f"{source}: {len(out)} rows survived validation, expected "
                        f"{EXPECTED_ROWS}")
    if not problems and n_multiturn != EXPECTED_MULTITURN:
        problems.append(f"{source}: {n_multiturn} multi-turn rows, expected "
                        f"{EXPECTED_MULTITURN}")
    if problems:
        raise FreezeError(problems)
    validate_rows.notes = notes  # type: ignore[attr-defined]
    return out


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    text = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise FreezeError([f"{path}:{lineno}: not JSON ({exc})"]) from None
    return rows


def render_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    """The artifact's bytes. Deterministic: sorted keys, no incidental spacing,
    ensure_ascii so the file is byte-identical on any machine and its sha256 is
    a stable pin."""
    return "".join(
        json.dumps(r, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n"
        for r in rows)


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_frozen(path: Path = DEFAULT_OUT) -> List[Dict[str, Any]]:
    """The committed exam, re-validated on the way in.

    Every consumer goes through here, so nothing in this repo can read a frozen
    row that would not survive freezing again.
    """
    if not path.exists():
        raise FreezeError([f"{path} does not exist; run "
                           f"`python scripts/freeze_breaks.py --src <partner file>`"])
    return validate_rows(read_jsonl(path), str(path))


def frozen_sha256(path: Path = DEFAULT_OUT) -> str:
    return sha256_of(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _report(rows: Sequence[Mapping[str, Any]], digest: str, path: Path) -> None:
    single = [r for r in rows if not r["multiturn"]]
    multi = [r for r in rows if r["multiturn"]]
    trainable = [r for r in rows if r["trainable"]]
    heldout = [r for r in rows if not r["trainable"]]
    print(f"rows          : {len(rows)}")
    print(f"single-turn   : {len(single)}")
    print(f"multi-turn    : {len(multi)}  ({', '.join(r['id'] for r in multi)})")
    print(f"trainable type: {len(trainable)}")
    print(f"held-out type : {len(heldout)}  ({', '.join(r['id'] for r in heldout)})")
    print(f"               these are {sorted({r['constraint_type'] for r in heldout})}, "
          f"split=heldout_type in constraints/registry.py: measurable, never trainable")
    print(f"path          : {path}")
    print(f"sha256        : {digest}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=str(DEFAULT_SRC),
                    help="the partner's confirmed_breaks.jsonl (input, not an artifact)")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="where the frozen exam is written")
    ap.add_argument("--check", action="store_true",
                    help="re-validate the COMMITTED artifact and its sha256 pin; "
                         "reads nothing from --src and writes nothing")
    args = ap.parse_args(argv)
    out = Path(args.out)

    try:
        if args.check:
            rows = load_frozen(out)
            digest = frozen_sha256(out)
            _report(rows, digest, out)
            if EXPECTED_SHA256 == "PENDING":
                print("FAIL: EXPECTED_SHA256 is still the PENDING placeholder; pin it "
                      "to the digest above.", file=sys.stderr)
                return 1
            if digest != EXPECTED_SHA256:
                print(f"FAIL: {out} sha256 {digest} != pinned {EXPECTED_SHA256}. The "
                      f"exam changed. Re-freeze deliberately and update "
                      f"scripts/freeze_breaks.py:EXPECTED_SHA256 in the same commit.",
                      file=sys.stderr)
                return 1
            print("OK: the committed exam validates and matches its pin.")
            return 0

        src = Path(args.src)
        rows = validate_rows(read_jsonl(src), str(src))
        text = render_jsonl(rows)
        digest = sha256_of(text)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        for note in getattr(validate_rows, "notes", []):
            print(f"note: {note}")
        _report(rows, digest, out)
        if digest != EXPECTED_SHA256:
            print(f"\nPin it:  EXPECTED_SHA256 = \"{digest}\"  in {Path(__file__).name}")
        return 0
    except FreezeError as exc:
        print(f"FAIL: validation refused the exam:\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
