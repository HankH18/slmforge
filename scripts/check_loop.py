#!/usr/bin/env python3
"""
P4 verify command -- offline proof that the full loop ran.

    python scripts/check_loop.py runs/smoke

The project spec's MVP bar is "Full loop -- generate -> train -> eval -- runs end
to end, demonstrated on a small smoke-test batch". This checker re-verifies that
claim from committed artifacts alone: no GPU, no API keys, no network. That is
what makes P4 re-checkable later without re-billing anything.

Exit 0 only if every leg below holds; non-zero (and a per-leg report on stderr)
otherwise.

  leg 0  rubric integrity  -- eval.rubric_hash() still equals c0adb876c3e6c4c9
  leg 1  generate          -- train_log.json names a training dataset that exists
                              in the repo, with a matching row count and a
                              MANDATORY matching sha256, and -- read out of the
                              dataset rows themselves, never from a self-attested
                              boolean -- every row's system message is
                              byte-identical to eval.SYSTEM_CORE and every row's
                              user message round-trips through eval.user_msg
  leg 2  train             -- train_log.json records a real step count and a
                              non-empty loss curve of finite numbers
  leg 3  artifacts         -- adapter AND merged-fp16 references, each with a
                              non-empty file listing and non-zero byte count
  leg 4  eval              -- a *.results.json written by eval.write_outputs,
                              stamped rubric_hash c0adb876c3e6c4c9, scored over
                              n > 0 scenarios, against the trained checkpoint
                              (not the base model), with its per-example judge
                              transcripts JSONL beside it

train_log.json / run_manifest.json are produced by training/modal_app.py; the
results + transcripts JSON are produced by the frozen eval.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

EXPECTED_RUBRIC_HASH = "c0adb876c3e6c4c9"
REPO_ROOT = Path(__file__).resolve().parent.parent


class Leg:
    def __init__(self, name):
        self.name = name
        self.ok = True
        self.notes = []
        self.errors = []

    def note(self, msg):
        self.notes.append(msg)

    def fail(self, msg):
        self.ok = False
        self.errors.append(msg)


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _import_eval():
    """The frozen eval.py -- the single source of truth for the prompt contract."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import eval as E
    return E


def _is_int(val) -> bool:
    return isinstance(val, int) and not isinstance(val, bool)


# ------------------------------------------------------------------ leg 0
def check_rubric() -> Leg:
    leg = Leg("0 rubric integrity")
    try:
        E = _import_eval()
    except Exception as exc:  # noqa: BLE001
        leg.fail(f"cannot import the frozen eval.py from {REPO_ROOT}: {exc}")
        return leg
    actual = E.rubric_hash()
    if actual != EXPECTED_RUBRIC_HASH:
        leg.fail(f"rubric drifted: eval.rubric_hash()={actual!r}, expected "
                 f"{EXPECTED_RUBRIC_HASH!r} -- every previously reported number "
                 f"is invalid. STOP and ask.")
    else:
        leg.note(f"eval.rubric_hash() == {actual}")
    return leg


# ------------------------------------------------------------------ leg 1
def _user_msg_shape(E):
    """(prefix, separator, suffix) of eval.user_msg, derived from eval itself so
    this checker cannot drift from the frozen renderer."""
    probe = E.user_msg("\x00PASSAGE\x00", "\x00QUESTION\x00")
    prefix, rest = probe.split("\x00PASSAGE\x00", 1)
    sep, suffix = rest.split("\x00QUESTION\x00", 1)
    return prefix, sep, suffix


def check_prompt_contract(ds_file: Path, E, max_report: int = 5) -> tuple:
    """THE contract that makes the training condition equal the zero_shot eval
    condition, verified FROM THE DATASET ITSELF.

    Returns (errors, n_rows). Every row must carry a system turn byte-identical
    to eval.SYSTEM_CORE and a user turn that round-trips through eval.user_msg;
    a self-attested boolean in train_log.json is not evidence of either, because
    the thing being checked is what writes it.
    """
    prefix, sep, suffix = _user_msg_shape(E)
    errors, n_rows = [], 0
    bad_system = bad_user = bad_schema = 0
    first = []

    def note_first(msg):
        if len(first) < max_report:
            first.append(msg)

    with open(ds_file) as f:
        for idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            n_rows += 1
            try:
                row = json.loads(line)
            except Exception as exc:  # noqa: BLE001
                bad_schema += 1
                note_first(f"row {idx}: not valid JSON ({exc})")
                continue
            if not isinstance(row, dict):
                bad_schema += 1
                note_first(f"row {idx}: not a JSON object")
                continue
            msgs = row.get("messages")
            if not isinstance(msgs, list) or len(msgs) < 3:
                bad_schema += 1
                note_first(f"row {idx}: 'messages' is not a list of >= 3 turns")
                continue
            if not all(isinstance(m, dict) for m in msgs[:3]):
                bad_schema += 1
                note_first(f"row {idx}: messages are not objects")
                continue
            if msgs[0].get("role") != "system" or msgs[1].get("role") != "user":
                bad_schema += 1
                note_first(f"row {idx}: turn roles are "
                           f"{[m.get('role') for m in msgs[:2]]!r}, expected "
                           f"['system', 'user']")
                continue
            if msgs[-1].get("role") != "assistant" or not str(
                    msgs[-1].get("content") or "").strip():
                bad_schema += 1
                note_first(f"row {idx}: last turn is not a non-empty assistant turn")
                continue

            system = msgs[0].get("content")
            if system != E.SYSTEM_CORE:
                bad_system += 1
                note_first(f"row {idx}: system message is not byte-identical to "
                           f"eval.SYSTEM_CORE ({str(system)[:60]!r}...)")

            user = msgs[1].get("content")
            if not isinstance(user, str):
                bad_user += 1
                note_first(f"row {idx}: user message is not a string")
            elif not (user.startswith(prefix) and user.endswith(suffix)
                      and sep in user):
                bad_user += 1
                note_first(f"row {idx}: user message does not match eval.user_msg "
                           f"(expected {prefix!r} ... {sep!r} ...)")
            else:
                body = user[len(prefix):len(user) - len(suffix)] if suffix \
                    else user[len(prefix):]
                passage, _, question = body.rpartition(sep)
                if E.user_msg(passage, question) != user:
                    bad_user += 1
                    note_first(f"row {idx}: user message does not match "
                               f"eval.user_msg(passage, question) on round-trip")

    if not n_rows:
        errors.append("training dataset is empty -- no rows to verify")
        return errors, n_rows
    if bad_schema:
        errors.append(f"{bad_schema}/{n_rows} dataset rows do not have the "
                      f"[system, user, ..., assistant] training schema")
    if bad_system:
        errors.append(f"{bad_system}/{n_rows} dataset rows: system message is not "
                      f"byte-identical to eval.SYSTEM_CORE -- the model was "
                      f"trained under a prompt the zero_shot eval never uses, so "
                      f"every reported number is measuring prompt transfer")
    if bad_user:
        errors.append(f"{bad_user}/{n_rows} dataset rows: user message does not "
                      f"match eval.user_msg(passage, question)")
    errors.extend(first)
    return errors, n_rows


def check_generate(train_log: dict, repo_root: Path) -> Leg:
    leg = Leg("1 generate (training dataset + prompt contract)")
    try:
        E = _import_eval()
    except Exception as exc:  # noqa: BLE001
        leg.fail(f"cannot import the frozen eval.py from {REPO_ROOT}: {exc}")
        return leg

    stamped = train_log.get("rubric_hash")
    if stamped != EXPECTED_RUBRIC_HASH:
        leg.fail(f"train_log.json:rubric_hash is {stamped!r}, expected "
                 f"{EXPECTED_RUBRIC_HASH!r} -- this run was trained under a "
                 f"different rubric than the one being evaluated")

    ds = train_log.get("dataset")
    if not isinstance(ds, dict):
        leg.fail("train_log.json has no 'dataset' object")
        return leg
    path = ds.get("path")
    if not path:
        leg.fail("train_log.json:dataset.path is missing or empty")
        return leg
    rows = ds.get("rows")
    if not _is_int(rows) or rows <= 0:
        leg.fail(f"train_log.json:dataset.rows is not a positive int: {rows!r}")
    trained_on = ds.get("rows_trained_on")
    if not _is_int(trained_on) or trained_on <= 0:
        leg.fail(f"dataset.rows_trained_on is missing or not a positive int "
                 f"({trained_on!r}) -- no evidence anything was trained on")
    elif _is_int(rows) and trained_on > rows:
        leg.fail(f"dataset.rows_trained_on ({trained_on}) exceeds dataset.rows "
                 f"({rows})")

    ds_file = repo_root / path
    if not ds_file.exists():
        leg.fail(f"training dataset {path!r} referenced by train_log.json does "
                 f"not exist at {ds_file}")
        return leg
    try:
        ds_file.resolve().relative_to(repo_root.resolve())
    except ValueError:
        leg.fail(f"training dataset {path!r} resolves outside the repo "
                 f"({ds_file.resolve()}) -- it is not a committed artifact")
        return leg

    actual_rows = sum(1 for ln in ds_file.read_text().splitlines() if ln.strip())
    if _is_int(rows) and actual_rows != rows:
        leg.fail(f"dataset row drift: {path} now has {actual_rows} rows, "
                 f"train_log.json recorded {rows}")

    # MANDATORY -- a missing hash is a failure, not a skip. Without it, content
    # drift in the dataset is undetectable.
    recorded_sha = ds.get("sha256")
    if not isinstance(recorded_sha, str) or not recorded_sha.strip():
        leg.fail(f"train_log.json:dataset.sha256 is missing -- without a recorded "
                 f"hash there is no way to prove {path} is the file that was "
                 f"trained on")
    else:
        actual_sha = _sha256(ds_file)
        if actual_sha != recorded_sha:
            leg.fail(f"dataset content drift: sha256({path})={actual_sha[:16]}..., "
                     f"train_log.json recorded {recorded_sha[:16]}...")
        else:
            leg.note(f"{path}: {actual_rows} rows, sha256 {actual_sha[:16]}... matches")

    # The load-bearing contract, read out of the rows themselves.
    prompt_errors, n_rows = check_prompt_contract(ds_file, E)
    for err in prompt_errors:
        leg.fail(err)
    if not prompt_errors:
        leg.note(f"all {n_rows} rows: messages[0] == eval.SYSTEM_CORE (byte-identical), "
                 f"messages[1] == eval.user_msg(passage, question)")

    # Kept as a consistency check only: the attestation must not CONTRADICT the
    # dataset. It is never what the contract is verified from -- it is written as
    # a literal by the very code it claims to attest.
    if train_log.get("system_prompt_equals_eval_SYSTEM_CORE") is not True:
        leg.fail("train_log.json does not attest system_prompt_equals_eval_"
                 "SYSTEM_CORE -- the training condition may differ from the "
                 "zero_shot eval condition")
    return leg


# ------------------------------------------------------------------ leg 2
def check_train(train_log: dict) -> Leg:
    leg = Leg("2 train (step count + loss curve)")
    steps = train_log.get("steps")
    if not _is_int(steps) or steps <= 0:
        leg.fail(f"train_log.json:steps is not a positive int: {steps!r}")
    curve = train_log.get("loss_curve")
    if not isinstance(curve, list) or not curve:
        leg.fail("train_log.json:loss_curve is missing or empty")
        return leg
    losses = []
    for entry in curve:
        if not isinstance(entry, dict) or "loss" not in entry:
            leg.fail(f"loss_curve entry without a 'loss' key: {entry!r}")
            return leg
        val = entry["loss"]
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            leg.fail(f"non-numeric loss value in loss_curve: {val!r}")
            return leg
        if not math.isfinite(float(val)):
            leg.fail(f"non-finite loss value in loss_curve: {val!r}")
            return leg
        losses.append(float(val))
    if all(v == 0.0 for v in losses):
        leg.fail("every recorded loss is 0.0 -- that is a placeholder, not a run")
    wall = train_log.get("wall_time_s") or {}
    if not isinstance(wall, dict) or not wall.get("train"):
        leg.fail("train_log.json:wall_time_s.train is missing or zero")
    if leg.ok:
        leg.note(f"{steps} optimizer steps, {len(losses)} logged losses "
                 f"({losses[0]:.4f} -> {losses[-1]:.4f}), "
                 f"train wall time {wall.get('train')}s")
    return leg


# ------------------------------------------------------------------ leg 3
def check_artifacts(train_log: dict) -> Leg:
    leg = Leg("3 artifacts (adapter + merged fp16)")
    art = train_log.get("artifacts")
    if not isinstance(art, dict):
        leg.fail("train_log.json has no 'artifacts' object")
        return leg
    for key in ("adapter", "merged"):
        info = art.get(key)
        if not isinstance(info, dict):
            leg.fail(f"artifacts.{key} is missing")
            continue
        where = info.get("remote_path") or info.get("volume_path") or info.get("path")
        if not where:
            leg.fail(f"artifacts.{key} carries no path reference")
        files = info.get("files")
        if not isinstance(files, list) or not files:
            leg.fail(f"artifacts.{key}.files is empty -- no evidence the "
                     f"{key} was written")
            continue
        size = info.get("bytes")
        if not isinstance(size, int) or size <= 0:
            leg.fail(f"artifacts.{key}.bytes is not positive: {size!r}")
            continue
        leg.note(f"{key}: {len(files)} files, {size / 1e6:.1f} MB at {where}")
    if not art.get("volume"):
        leg.fail("artifacts.volume is missing -- no way to pull the weights back")
    return leg


# ------------------------------------------------------------------ leg 4
def _checkpoint_markers(run_dir: Path, train_log: dict) -> set:
    """Strings whose presence in a model spec means it points at THIS run's
    checkpoint rather than at some other model."""
    marks = {run_dir.name.lower(), str(train_log.get("run_name") or "").lower()}
    art = train_log.get("artifacts") or {}
    for key in ("adapter", "merged"):
        info = art.get(key) or {}
        for field in ("volume_path", "remote_path", "path", "hf_repo"):
            val = info.get(field)
            if val:
                marks.add(Path(str(val)).name.lower())
                marks.add(str(val).lower())
    marks.discard("")
    return marks


def check_eval(run_dir: Path, train_log: dict) -> Leg:
    leg = Leg("4 eval (results JSON against the trained checkpoint)")
    results_files = sorted(run_dir.rglob("*.results.json"))
    if not results_files:
        leg.fail(f"no *.results.json under {run_dir} -- run eval.py against the "
                 f"trained checkpoint with --out {run_dir}/eval")
        return leg

    base_spec = ((train_log.get("base_model") or {}).get("spec") or "").lower()
    if not base_spec:
        leg.fail("train_log.json:base_model.spec is missing -- without it there "
                 "is no way to prove the results are not just the base model")
        return leg
    markers = _checkpoint_markers(run_dir, train_log)
    rejected = []
    for rf in results_files:
        try:
            res = _load_json(rf)
        except Exception as exc:  # noqa: BLE001
            rejected.append(f"{rf.name}: not valid JSON ({exc})")
            continue
        rh = res.get("rubric_hash")
        if rh != EXPECTED_RUBRIC_HASH:
            rejected.append(f"{rf.name}: rubric_hash {rh!r} != {EXPECTED_RUBRIC_HASH!r}")
            continue
        spec = str(res.get("model") or "")
        low = spec.lower()
        if not low.startswith("hf:"):
            rejected.append(f"{rf.name}: model {spec!r} is not an hf: checkpoint spec")
            continue
        if low == base_spec:
            rejected.append(f"{rf.name}: model {spec!r} is the BASE model, not the "
                            f"trained checkpoint")
            continue
        if not any(m in low for m in markers):
            rejected.append(
                f"{rf.name}: model {spec!r} does not reference this run's "
                f"checkpoint (expected one of {sorted(markers)} in the spec)")
            continue
        metrics = res.get("metrics") or {}
        n = metrics.get("n")
        if not _is_int(n) or n <= 0:
            rejected.append(f"{rf.name}: metrics.n is not positive ({n!r})")
            continue
        sa = metrics.get("spec_adherence")
        if not isinstance(sa, (int, float)) or isinstance(sa, bool) \
                or not 0.0 <= float(sa) <= 1.0:
            rejected.append(f"{rf.name}: metrics.spec_adherence is missing or not "
                            f"a rate in [0, 1] ({sa!r}) -- nothing was scored")
            continue
        transcripts = rf.with_name(rf.name.replace(".results.json",
                                                   ".transcripts.jsonl"))
        if not transcripts.exists():
            rejected.append(f"{rf.name}: judge transcripts {transcripts.name} "
                            f"missing (the spec requires raw per-example output)")
            continue
        lines = sum(1 for ln in transcripts.read_text().splitlines() if ln.strip())
        if lines != n:
            rejected.append(f"{rf.name}: {transcripts.name} has {lines} records, "
                            f"metrics.n is {n}")
            continue
        leg.note(f"{rf.relative_to(run_dir)}: model={spec} n={n} "
                 f"spec_adherence={sa} rubric_hash={rh}")
        leg.note(f"{transcripts.relative_to(run_dir)}: {lines} judge transcripts")
        return leg

    leg.fail(f"no usable results JSON among {len(results_files)} candidate(s) "
             f"under {run_dir}")
    for r in rejected:
        leg.fail(f"  rejected {r}")
    return leg


# ------------------------------------------------------------------ driver
def check_loop(run_dir: Path, repo_root: Path) -> list:
    legs = [check_rubric()]
    if not run_dir.is_dir():
        leg = Leg("1-4 artifacts")
        leg.fail(f"{run_dir} is not a directory")
        legs.append(leg)
        return legs

    log_path = run_dir / "train_log.json"
    if not log_path.exists():
        leg = Leg("1-4 artifacts")
        leg.fail(f"{log_path} missing -- training/modal_app.py writes it; without "
                 f"it there is no evidence any training happened")
        legs.append(leg)
        return legs
    try:
        train_log = _load_json(log_path)
    except Exception as exc:  # noqa: BLE001
        leg = Leg("1-4 artifacts")
        leg.fail(f"{log_path} is not valid JSON: {exc}")
        legs.append(leg)
        return legs

    legs.append(check_generate(train_log, repo_root))
    legs.append(check_train(train_log))
    legs.append(check_artifacts(train_log))
    legs.append(check_eval(run_dir, train_log))
    return legs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Verify the committed generate -> train -> eval loop artifacts")
    ap.add_argument("run_dir", help="run directory, e.g. runs/smoke")
    ap.add_argument("--repo-root", default=str(REPO_ROOT),
                    help="root the dataset path in train_log.json resolves against")
    ap.add_argument("--json", action="store_true",
                    help="emit a machine-readable report on stdout")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    repo_root = Path(args.repo_root).resolve()
    legs = check_loop(run_dir, repo_root)
    ok = all(leg.ok for leg in legs)

    if args.json:
        print(json.dumps({
            "run_dir": str(run_dir), "ok": ok,
            "legs": [{"name": l.name, "ok": l.ok, "notes": l.notes,
                      "errors": l.errors} for l in legs],
        }, indent=2))
    else:
        print(f"check_loop {run_dir}")
        for leg in legs:
            print(f"  [{'PASS' if leg.ok else 'FAIL'}] {leg.name}")
            for note in leg.notes:
                print(f"        {note}")
            for err in leg.errors:
                print(f"        !! {err}", file=sys.stderr)
        print(f"\n{'LOOP VERIFIED' if ok else 'LOOP INCOMPLETE'}: "
              f"{sum(l.ok for l in legs)}/{len(legs)} legs pass")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
