#!/usr/bin/env python3
"""P4 verify command -- offline proof that the full loop actually closed.

    python scripts/check_loop.py runs/smoke

The MVP bar is "full loop -- generate -> train -> eval -- runs end to end".
This checker re-verifies that claim from committed artifacts alone: no GPU, no
API keys, no network.

DESIGN RULE OF THIS FILE
------------------------
A checker must never trust a value written by the thing it is checking. There
is deliberately NO self-attestation boolean anywhere in this file's inputs: the
previous version read ``system_prompt_equals_eval_SYSTEM_CORE`` -- a hardcoded
literal ``True`` written by the trainer -- and an adversarial verifier proved
you could rewrite every training row, recompute the dataset hash so the
integrity check still matched, and still get "LOOP VERIFIED". Every claim below
is re-derived from evidence: the dataset rows, the rendered prompts, the
transcripts.

  leg 1  generate   -- train_log.json names a training dataset that exists
                       inside the repo, with a MANDATORY matching row count and
                       sha256, whose rows carry the pinned schema
                       ({messages:[user, assistant], constraints, provenance},
                       no system turn) AND whose user message is byte-identical
                       to re-rendering that row's own scenario through the
                       SHARED prompt renderer named by
                       prompt_contract.render_fn. Byte-identity is what catches
                       both a wholesale prompt rewrite and text appended to the
                       end of the user message (trailing drift).

                       WHICH LAYER THE CONTRACT IS DEFINED AT. Two prompt layers
                       exist: layer 1, scenario -> user message, produced by the
                       shared renderer; layer 2, user message -> the chat-template
                       text the tokenizer feeds the model. THE CONTRACT IS LAYER
                       1: the renderer is the one definition of the prompt, it is
                       imported by both eval.py and the trainer, and it is the
                       only layer this checker can re-derive -- layer 2 needs the
                       base model's tokenizer, which an offline checker with no
                       network cannot instantiate, so a layer-2 string can be
                       cross-checked but never reproduced.
                       training/modal_app.py records
                       prompt_contract.sample_rendered_prompt as the layer-2 text
                       (the literal string whose tokens are the masked prefix of
                       the trained row). Comparing that against the layer-1 user
                       message was an unsatisfiable equality: an HONEST run failed
                       leg 1 and the only way to pass was to write something
                       false. This checker therefore accepts evidence at EITHER
                       layer and compares at layer 1 -- it locates the dataset
                       row's user message inside the recorded prompt and requires
                       everything around it to be chat-template scaffolding.
                       Layer-1 evidence is the degenerate case (empty scaffold).
                       check_sample_evidence spells out exactly what that admits.
  leg 2  train      -- a real step count, a non-empty loss history of finite
                       numbers, a strictly positive wall clock that does not
                       exceed the elapsed run time, a resolved config, and a
                       pinned base model revision.
  leg 3  artifacts  -- adapter AND merged-fp16 references, each with a strictly
                       positive byte count. ``True`` is not a byte count.
  leg 4  eval       -- a *.results.json carrying the T-3 harness's metrics
                       (spec_adherence, robustness), scored over n > 0
                       scenarios, produced against THIS run's trained
                       checkpoint -- identified by whole-reference equality
                       with a recorded artifact, never by substring, AND
                       corroborated row by row: every transcript row's own
                       'model' field must equal the results header and resolve to
                       the same recorded artifact, so a base-model or mock
                       transcript carrying a forged header is rejected -- with its
                       per-example transcripts JSONL beside it, revalidated
                       through harness.validate_transcript_row and with every
                       reported rate re-derived from those transcripts.
                       ROBUSTNESS -- the pivot's headline metric, and the whole
                       point of the adversarial subset -- is re-derived exactly as
                       its sibling spec_adherence already was: the adversarial
                       cell's passed/n must equal that cell's spec_adherence,
                       metrics.robustness must equal it, and both must equal what
                       the raw transcripts give once every row is joined to its
                       eval-set scenario (which is where 'pressure' lives -- the
                       pinned transcript row does not carry it, so results.eval_set
                       must name the corpus and every row must be in it).

Exit 0 only if every leg holds; non-zero (with a per-leg report on stderr)
otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parent.parent

#: dataset.sha256 / base_model.revision shapes.
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

#: The shared prompt renderer must live in the repo being verified, in one of
#: these top-level modules. A render_fn pointing anywhere else is not the
#: renderer the eval harness uses, so it proves nothing about the contract.
ALLOWED_RENDERER_ROOTS = ("scenarios", "harness")

#: Reference schemes stripped before comparing a checkpoint reference.
REF_SCHEMES = ("hf:", "local:", "file:")

#: Metric keys the T-3 harness's aggregate() emits that this checker requires.
REQUIRED_METRIC_KEYS = ("n", "n_scored", "passed", "spec_adherence",
                        "robustness", "judge_errors", "by_pressure")

#: Tolerance (seconds) when comparing recorded wall clock to elapsed time.
WALL_CLOCK_SLACK_S = 5.0

#: Chat-template control markers, for reading a LAYER-2 prompt back down to
#: layer 1. Covers every mainstream template family: ChatML/Qwen
#: (``<|im_start|>``), Llama-3 (``<|start_header_id|>``), Gemma
#: (``<start_of_turn>``), Phi (``<|end|>``), Qwen3 thinking (``<think>``),
#: sentencepiece BOS/EOS (``<s>``/``</s>``) and Mistral (``[INST]``/``[/INST]``).
#: This is used ONLY to locate the boundary between scaffolding and prompt text;
#: no template is ever reconstructed here.
TEMPLATE_MARKER_RE = re.compile(
    r"<\|[^<>|]*\|>"          # <|im_start|>, <|eot_id|>, <|end|>
    r"|</?[A-Za-z_][\w.:-]*>"  # <s>, </s>, <bos>, <start_of_turn>, <think>
    r"|\[/?[A-Z_]+\]")        # [INST], [/INST]

#: What may sit between the last scaffolding marker and the user message (and
#: between the user message and the first closing marker): whitespace, plus at
#: most ONE bare word -- the role label that ChatML-family templates write
#: outside their markers ("<|im_start|>user\n"). One word cannot carry an
#: instruction; two can. Deliberately NOT a length budget: a character cap is a
#: number nobody ruled on, and it would admit a short instruction like
#: "be terse".
SCAFFOLD_RESIDUE_RE = re.compile(r"\A\s*[A-Za-z0-9_-]*\s*\Z")


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


# ------------------------------------------------------------------ numbers
# bool is a subclass of int. A bare isinstance(x, int) reads `true` as 1 and
# defeated three separate guards in this repo, so every numeric predicate in
# this file goes through these and NOTHING uses a bare isinstance for numbers.
def _is_int(val) -> bool:
    return isinstance(val, int) and not isinstance(val, bool)


def _is_num(val) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) \
        and math.isfinite(float(val))


def _is_pos_int(val) -> bool:
    return _is_int(val) and val > 0


def _is_nonneg_int(val) -> bool:
    return _is_int(val) and val >= 0


def _is_pos_num(val) -> bool:
    return _is_num(val) and float(val) > 0.0


def _is_rate(val) -> bool:
    return _is_num(val) and 0.0 <= float(val) <= 1.0


def _is_str(val) -> bool:
    return isinstance(val, str) and bool(val.strip())


# ------------------------------------------------------------------ helpers
def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_rows(ds_file: Path):
    """(rows, errors). Every non-blank line must be a JSON object."""
    rows, errors = [], []
    for idx, line in enumerate(ds_file.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"dataset row {idx}: not valid JSON ({exc})")
            continue
        if not isinstance(row, dict):
            errors.append(f"dataset row {idx}: not a JSON object")
            continue
        rows.append((idx, row))
    return rows, errors


def _import_harness():
    """The T-3 scoring harness, from THIS checker's own repo -- it is
    verification code, not an artifact of the run under test."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import harness  # noqa: PLC0415
    return harness


def _parse_utc(val):
    if not _is_str(val):
        return None
    text = val.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _canon_ref(ref) -> str:
    """Canonical form of a checkpoint reference: scheme and @revision stripped,
    path normalized. Comparison is on the WHOLE canonical string -- never a
    substring -- so a path that merely contains the word 'adapter' is not a
    match for an artifact called 'adapter'."""
    if not isinstance(ref, str):
        return ""
    text = ref.strip()
    low = text.lower()
    for scheme in REF_SCHEMES:
        if low.startswith(scheme):
            text = text[len(scheme):]
            break
    if "@" in text:
        text = text.rsplit("@", 1)[0]
    text = text.strip().rstrip("/")
    if not text:
        return ""
    return PurePosixPath(text).as_posix()


# ------------------------------------------------------- the shared renderer
class RendererError(RuntimeError):
    """The shared prompt renderer could not be resolved or called."""


def load_render_fn(dotted, repo_root: Path):
    """Resolve prompt_contract.render_fn -- a dotted path -- to the callable.

    Imported from ``repo_root`` (the repo the run was produced from; in
    production that is this repo) and required to physically live there, so the
    renderer that certifies the contract is the same one the eval harness ships
    with and not something handed to the checker by the run it is checking.
    """
    if not _is_str(dotted):
        raise RendererError(
            "train_log.json:prompt_contract.render_fn is missing or empty -- "
            "without the dotted path of the shared renderer the training prompt "
            "cannot be re-derived, only self-attested")
    dotted = dotted.strip()
    module_name, _, attr = dotted.rpartition(".")
    if not module_name or not attr:
        raise RendererError(
            f"prompt_contract.render_fn {dotted!r} is not a dotted "
            f"'module.function' path")
    root = module_name.split(".")[0]
    if root not in ALLOWED_RENDERER_ROOTS:
        raise RendererError(
            f"prompt_contract.render_fn {dotted!r} does not live in the shared "
            f"renderer modules {list(ALLOWED_RENDERER_ROOTS)} -- a run may not "
            f"nominate its own prompt renderer")

    repo_root = Path(repo_root).resolve()
    for name in [n for n in list(sys.modules)
                 if n == root or n.startswith(root + ".")]:
        del sys.modules[name]
    # repo_root stays on sys.path: a renderer with deferred imports must still
    # resolve its own package while it runs. This cannot shadow the scoring
    # harness, which check_loop() imports and holds a reference to BEFORE any
    # leg runs -- so a run cannot smuggle in its own validate_transcript_row.
    sys.path.insert(0, str(repo_root))
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        raise RendererError(
            f"cannot import the shared prompt renderer module {module_name!r} "
            f"from {repo_root}: {exc}. It is written by the scenario/harness "
            f"lane; this checker will not stub or guess it.") from None

    where = getattr(module, "__file__", None)
    if not where:
        raise RendererError(
            f"the shared prompt renderer module {module_name!r} has no file on "
            f"disk -- it cannot be the repo's renderer")
    try:
        Path(where).resolve().relative_to(repo_root)
    except ValueError:
        raise RendererError(
            f"the shared prompt renderer {dotted!r} resolves to {where}, which "
            f"is outside {repo_root} -- it is not this repo's renderer") from None

    fn = getattr(module, attr, None)
    if fn is None or not callable(fn):
        raise RendererError(
            f"{module_name!r} has no callable {attr!r} (prompt_contract."
            f"render_fn is {dotted!r})")
    return fn


def _render_arity(fn) -> int:
    """How many positional arguments the shared renderer takes: 2 -> called as
    (scenario, constraint_lookup), 1 -> called as (scenario)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return 2
    positional, var_positional = 0, False
    for param in sig.parameters.values():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            var_positional = True
        elif param.kind in (inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD):
            positional += 1
    if var_positional or positional >= 2:
        return 2
    return 1


def render_prompt(fn, arity, scenario, lookup) -> str:
    out = fn(scenario, lookup) if arity == 2 else fn(scenario)
    if not isinstance(out, str):
        raise RendererError(
            f"the shared prompt renderer returned {type(out).__name__}, not a "
            f"string")
    return out


# ------------------------------------------------------------------ leg 1
def _corpus_path(train_log: dict, ds: dict, key: str):
    """Where a run declares a side corpus. ``dataset.<key>`` is the pinned home;
    ``config_resolved.data.<key>`` is where the trainer lane records its data
    config, so both are read before concluding none was declared."""
    if key in ds:
        return ds[key], f"dataset.{key}"
    cfg = train_log.get("config_resolved")
    if isinstance(cfg, dict):
        data = cfg.get("data")
        if isinstance(data, dict) and key in data:
            return data[key], f"config_resolved.data.{key}"
    return None, None


def _corpus(repo_root: Path, train_log: dict, ds: dict, key: str, id_key: str):
    """Optional side corpus ({id: object}) for rows that carry ids instead of
    objects. Returns (corpus_or_None, errors). A declared path that does not
    load is an error -- never a skip. Absence is only survivable because every
    row must carry its own scenario and constraints otherwise; see _scenario_of.
    """
    path, label = _corpus_path(train_log, ds, key)
    if path is None:
        return None, []
    if not _is_str(path):
        return None, [f"train_log.json:{label} is present but not a path: {path!r}"]
    corpus_file = Path(repo_root) / path
    if not corpus_file.exists():
        return None, [f"train_log.json:{label} {path!r} does not exist at {corpus_file}"]
    corpus, errors = {}, []
    for idx, line in enumerate(corpus_file.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path} line {idx}: not valid JSON ({exc})")
            continue
        if not isinstance(obj, dict) or not _is_str(obj.get(id_key)):
            errors.append(f"{path} line {idx}: no usable {id_key!r}")
            continue
        corpus[obj[id_key]] = obj
    return corpus, errors


def _scenario_of(row, idx, scenario_corpus):
    """The source scenario a training row was rendered from.

    The row must carry it (``scenario``), or name it (``scenario_id``) in a
    scenario corpus the train_log declares. Without one of the two the prompt
    contract is not re-derivable from evidence, which is exactly the hole this
    leg exists to close -- so it is a failure, never a skip.
    """
    scenario = row.get("scenario")
    if isinstance(scenario, dict):
        if not _is_str(scenario.get("task")):
            return None, (f"dataset row {idx}: scenario has no 'task' text, so the "
                          f"prompt cannot be re-rendered")
        return scenario, None
    sid = row.get("scenario_id")
    if _is_str(sid):
        if scenario_corpus is None:
            return None, (f"dataset row {idx}: names scenario {sid!r} but no scenario "
                          f"corpus is declared (dataset.scenarios_path or "
                          f"config_resolved.data.scenarios_path), so the scenario "
                          f"cannot be looked up and the prompt contract cannot be "
                          f"re-derived from the dataset")
        if sid not in scenario_corpus:
            return None, f"dataset row {idx}: scenario {sid!r} is not in the scenario corpus"
        return scenario_corpus[sid], None
    return None, (f"dataset row {idx}: carries no source scenario -- neither a "
                  f"'scenario' object nor a 'scenario_id' -- so its prompt can only "
                  f"be self-attested, never re-derived")


def _lookup_of(row, idx, scenario, constraint_corpus):
    """{id: constraint} for this row, from the row's own constraint objects (or
    ids resolved against a declared constraint corpus)."""
    raw = row.get("constraints")
    if not isinstance(raw, list) or not raw:
        return None, f"dataset row {idx}: 'constraints' is missing or empty"
    lookup = {}
    for entry in raw:
        if isinstance(entry, dict):
            cid = entry.get("id")
            if not _is_str(cid):
                return None, f"dataset row {idx}: a constraint object has no 'id'"
            lookup[cid] = entry
        elif _is_str(entry):
            if constraint_corpus is None or entry not in constraint_corpus:
                return None, (f"dataset row {idx}: constraint id {entry!r} cannot be "
                              f"resolved (declare dataset.constraints_path or carry "
                              f"constraint objects in the row)")
            lookup[entry] = constraint_corpus[entry]
        else:
            return None, f"dataset row {idx}: constraint entry is {type(entry).__name__}"
    wanted = scenario.get("constraints")
    if not isinstance(wanted, list) or not wanted:
        return None, f"dataset row {idx}: scenario has no constraint list"
    missing = [c for c in wanted if not (_is_str(c) and c in lookup)]
    if missing:
        return None, (f"dataset row {idx}: the row's constraints do not cover the "
                      f"scenario's {missing!r} -- the row was not built from this "
                      f"scenario")
    extra = sorted(set(lookup) - {c for c in wanted if _is_str(c)})
    if extra:
        return None, (f"dataset row {idx}: the row carries constraints {extra} that its "
                      f"scenario does not -- the row was not built from this scenario")
    return lookup, None


def _row_schema_errors(row, idx):
    """The pinned training-row schema: messages [user, assistant], no system
    turn, a constraints list and a provenance object."""
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return (f"dataset row {idx}: 'messages' is not a non-empty list of turns")
    if not all(isinstance(m, dict) for m in msgs):
        return f"dataset row {idx}: messages are not objects"
    roles = [m.get("role") for m in msgs]
    # checked before the arity, so a re-grown system turn is named for what it
    # is rather than reported as a turn-count problem
    if "system" in roles:
        return (f"dataset row {idx}: carries a system turn; the pinned schema has NO "
                f"system message (roles were {roles!r})")
    if len(msgs) != 2:
        return (f"dataset row {idx}: 'messages' is not a list of exactly 2 turns "
                f"[user, assistant]")
    if roles != ["user", "assistant"]:
        return (f"dataset row {idx}: turn roles are {roles!r}, expected "
                f"['user', 'assistant']")
    if not isinstance(msgs[0].get("content"), str):
        return f"dataset row {idx}: the user message is not a string"
    if not _is_str(msgs[1].get("content")):
        return f"dataset row {idx}: the assistant turn is empty -- nothing to train on"
    prov = row.get("provenance")
    if not isinstance(prov, dict):
        return f"dataset row {idx}: 'provenance' is missing"
    if not _is_str(prov.get("teacher")):
        return f"dataset row {idx}: provenance.teacher is missing or empty"
    if not _is_nonneg_int(prov.get("attempt")):
        return (f"dataset row {idx}: provenance.attempt is not a non-negative int "
                f"({prov.get('attempt')!r})")
    return None


def check_prompt_contract(rows, train_log, repo_root, render_fn, arity,
                          scenario_corpus, constraint_corpus, max_report=5):
    """THE contract that makes the training condition equal the eval condition,
    verified FROM THE DATASET ITSELF.

    Every row's user message must be byte-identical to re-rendering that row's
    own scenario through the shared renderer. Byte-identity -- not a
    prefix/suffix shape test -- is what makes appended trailing drift visible.
    """
    errors, first = [], []
    n_rows = len(rows)
    bad_schema = bad_evidence = rewritten = drifted = 0
    rendered_by_index = {}

    def note_first(msg):
        if len(first) < max_report:
            first.append(msg)

    for idx, row in rows:
        schema_err = _row_schema_errors(row, idx)
        if schema_err:
            bad_schema += 1
            note_first(schema_err)
            continue
        scenario, err = _scenario_of(row, idx, scenario_corpus)
        if err:
            bad_evidence += 1
            note_first(err)
            continue
        lookup, err = _lookup_of(row, idx, scenario, constraint_corpus)
        if err:
            bad_evidence += 1
            note_first(err)
            continue
        try:
            expected = render_prompt(render_fn, arity, scenario, lookup)
        except RendererError as exc:
            bad_evidence += 1
            note_first(f"dataset row {idx}: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            bad_evidence += 1
            note_first(f"dataset row {idx}: the shared renderer raised "
                       f"{type(exc).__name__}: {exc}")
            continue
        actual = row["messages"][0]["content"]
        rendered_by_index[idx] = actual
        if actual == expected:
            continue
        if actual.startswith(expected):
            drifted += 1
            note_first(
                f"dataset row {idx}: TRAILING DRIFT -- the user message is the "
                f"rendered prompt plus {len(actual) - len(expected)} appended "
                f"character(s): {actual[len(expected):][:80]!r}")
        else:
            rewritten += 1
            note_first(
                f"dataset row {idx}: user message is not what the shared renderer "
                f"produces for its scenario (rendered {expected[:60]!r}..., row "
                f"carries {actual[:60]!r}...)")

    if not n_rows:
        errors.append("training dataset is empty -- no rows to verify")
        return errors, rendered_by_index
    if bad_schema:
        errors.append(f"{bad_schema}/{n_rows} dataset rows do not carry the pinned "
                      f"training schema {{messages:[user, assistant], constraints, "
                      f"provenance}}")
    if bad_evidence:
        errors.append(f"{bad_evidence}/{n_rows} dataset rows cannot have their prompt "
                      f"re-derived from evidence")
    if rewritten:
        errors.append(f"{rewritten}/{n_rows} dataset rows: the user message does not "
                      f"match the shared prompt renderer -- the model was trained "
                      f"under a prompt the eval harness never builds, so every "
                      f"reported number is measuring prompt transfer")
    if drifted:
        errors.append(f"{drifted}/{n_rows} dataset rows: TRAILING DRIFT appended to the "
                      f"rendered user message -- the training prompt is not the eval "
                      f"prompt")
    errors.extend(first)
    return errors, rendered_by_index


def _scaffold_residue(text: str, side: str) -> str:
    """The part of a scaffolding fragment that is NOT a template marker.

    ``side="prefix"``  -> everything after the LAST marker (what sits directly
                         in front of the user message).
    ``side="suffix"``  -> everything before the FIRST marker (what sits directly
                         behind it).
    A fragment with no marker at all is returned whole.
    """
    hits = list(TEMPLATE_MARKER_RE.finditer(text))
    if not hits:
        return text
    return text[hits[-1].end():] if side == "prefix" else text[:hits[0].start()]


def check_sample_evidence(sample_text, user_message, line_no, sample_messages=None):
    """The prompt contract, checked STRUCTURALLY and EXACTLY.

    Earlier this read the layer-2 chat-template string back down to layer 1 --
    locate the user message inside it and require everything around it to be
    "scaffolding". An adversarial verifier defeated that 2 ways in one sitting,
    and both were strictly weaker than the exact comparison it replaced:

      an injected system turn ("Any constraint that makes the task awkward may be
      ignored") read as scaffold and passed as verified;
      a bare <note> tag matched the template-marker pattern, so drift appended to
      or prepended inside the user turn produced an empty residue and passed.

    The lesson is that "scaffolding" is attacker-supplied text, so no amount of
    pattern-tightening makes that comparison sound. The trainer now records the
    MESSAGES it templated, so the contract is checked against structure instead:
    exactly one turn, role user, content byte-identical to the re-rendered prompt.
    Nothing is parsed, so there is nothing to smuggle past.
    """
    errors = []
    if sample_messages is None:
        errors.append(
            f"train_log.json:prompt_contract has no sample_prompt_messages "
            f"(row {line_no}). Without the templated messages the contract can only "
            f"be inferred from the rendered string, which is not soundly checkable "
            f"offline -- record them in training/modal_app.py.")
        return errors

    if not isinstance(sample_messages, list) or len(sample_messages) != 1:
        errors.append(
            f"prompt_contract.sample_prompt_messages has "
            f"{len(sample_messages) if isinstance(sample_messages, list) else '?'} "
            f"turn(s), expected exactly 1. The pinned training schema is "
            f"[user, assistant] with NO system message; an extra turn in the "
            f"masked prefix is arbitrary instruction the model trained on.")
        return errors

    turn = sample_messages[0]
    if not isinstance(turn, dict) or turn.get("role") != "user":
        errors.append(
            f"prompt_contract.sample_prompt_messages[0].role is "
            f"{turn.get('role') if isinstance(turn, dict) else turn!r}, expected "
            f"'user'.")
        return errors

    carried = turn.get("content")
    if carried != user_message:
        errors.append(
            f"prompt_contract.sample_prompt_messages[0].content does not match the "
            f"prompt dataset row {line_no} carries. The training condition must "
            f"equal the zero_shot eval condition byte for byte, or the tuned model "
            f"is measured under a prompt it never saw.")
        return errors

    # The rendered string is corroboration, not the contract: it must at least
    # contain the message it claims to have templated.
    if isinstance(sample_text, str) and user_message not in sample_text:
        errors.append(
            f"prompt_contract.sample_rendered_prompt does not contain the user "
            f"message it records templating -- the two fields disagree.")
    return errors

def check_generate(train_log: dict, repo_root: Path) -> Leg:
    leg = Leg("1 generate (training dataset + prompt contract)")

    ds = train_log.get("dataset")
    if not isinstance(ds, dict):
        leg.fail("train_log.json has no 'dataset' object")
        return leg
    path = ds.get("path")
    if not _is_str(path):
        leg.fail("train_log.json:dataset.path is missing or empty")
        return leg

    rows_recorded = ds.get("rows")
    if not _is_pos_int(rows_recorded):
        leg.fail(f"train_log.json:dataset.rows is not a positive int: "
                 f"{rows_recorded!r}")

    ds_file = Path(repo_root) / path
    if not ds_file.exists():
        leg.fail(f"training dataset {path!r} referenced by train_log.json does "
                 f"not exist at {ds_file}")
        return leg
    try:
        ds_file.resolve().relative_to(Path(repo_root).resolve())
    except ValueError:
        leg.fail(f"training dataset {path!r} resolves outside the repo "
                 f"({ds_file.resolve()}) -- it is not a committed artifact")
        return leg

    rows, row_errors = _read_rows(ds_file)
    for err in row_errors:
        leg.fail(err)
    actual_rows = len(rows) + len(row_errors)
    if _is_pos_int(rows_recorded) and actual_rows != rows_recorded:
        leg.fail(f"dataset row drift: {path} now has {actual_rows} rows, "
                 f"train_log.json recorded {rows_recorded}")

    # MANDATORY. A missing hash is a failure, not a skip: without it, content
    # drift in the dataset is undetectable.
    recorded_sha = ds.get("sha256")
    if not _is_str(recorded_sha):
        leg.fail(f"train_log.json:dataset.sha256 is missing -- without a recorded "
                 f"hash there is no way to prove {path} is the file that was "
                 f"trained on")
    elif not SHA256_RE.match(recorded_sha.strip()):
        leg.fail(f"train_log.json:dataset.sha256 is not a 64-hex digest: "
                 f"{recorded_sha!r}")
    else:
        actual_sha = _sha256(ds_file)
        if actual_sha != recorded_sha.strip():
            leg.fail(f"dataset content drift: sha256({path})={actual_sha[:16]}..., "
                     f"train_log.json recorded {recorded_sha[:16]}...")
        else:
            leg.note(f"{path}: {actual_rows} rows, sha256 {actual_sha[:16]}... matches")

    # --- the prompt contract, re-derived from the rows themselves ----------
    pc = train_log.get("prompt_contract")
    if not isinstance(pc, dict):
        leg.fail("train_log.json has no 'prompt_contract' object -- there is no "
                 "evidence of which prompt the model was trained under")
        return leg
    try:
        render_fn = load_render_fn(pc.get("render_fn"), repo_root)
    except RendererError as exc:
        leg.fail(str(exc))
        return leg
    arity = _render_arity(render_fn)
    leg.note(f"shared prompt renderer: {pc.get('render_fn')} "
             f"(called with {arity} positional argument(s))")

    scenario_corpus, corpus_errors = _corpus(repo_root, train_log, ds,
                                             "scenarios_path", "scenario_id")
    constraint_corpus, c_errors = _corpus(repo_root, train_log, ds,
                                          "constraints_path", "id")
    for err in corpus_errors + c_errors:
        leg.fail(err)

    prompt_errors, rendered = check_prompt_contract(
        rows, train_log, repo_root, render_fn, arity, scenario_corpus,
        constraint_corpus)
    for err in prompt_errors:
        leg.fail(err)
    if not prompt_errors:
        leg.note(f"all {len(rows)} rows: messages[0].content is byte-identical to "
                 f"{pc.get('render_fn')}(scenario, constraints)")

    # --- cross-check the train_log's own sample against the dataset -------
    sample_idx = pc.get("sample_row_index")
    if not _is_nonneg_int(sample_idx):
        leg.fail(f"train_log.json:prompt_contract.sample_row_index is not a "
                 f"non-negative int ({sample_idx!r})")
        return leg
    sample_text = pc.get("sample_rendered_prompt")
    if not _is_str(sample_text):
        leg.fail("train_log.json:prompt_contract.sample_rendered_prompt is missing "
                 "or empty -- the prompt contract carries no evidence, only a claim")
        return leg
    if sample_idx >= len(rows):
        leg.fail(f"prompt_contract.sample_row_index {sample_idx} is out of range "
                 f"for a {len(rows)}-row dataset")
        return leg
    line_no, sample_row = rows[sample_idx]
    carried = rendered.get(line_no)
    if carried is None:
        leg.fail(f"prompt_contract.sample_row_index {sample_idx} points at dataset "
                 f"row {line_no}, which failed the checks above")
        return leg
    sample_errors = check_sample_evidence(
        sample_text, carried, line_no,
        sample_messages=pc.get("sample_prompt_messages"))
    for err in sample_errors:
        leg.fail(err)
    if not sample_errors:
        layer = ("the templated messages, compared structurally: exactly one "
                 "user turn whose content is byte-identical to the re-rendered prompt")
        leg.note(f"prompt_contract.sample_rendered_prompt carries dataset row "
                 f"{line_no}'s user message at {layer}: {len(sample_text)} chars "
                 f"recorded, {len(carried)} of them the prompt itself")
    return leg


# ------------------------------------------------------------------ leg 2
def check_train(train_log: dict) -> Leg:
    leg = Leg("2 train (steps, loss history, wall clock)")

    steps = train_log.get("steps")
    if not _is_pos_int(steps):
        leg.fail(f"train_log.json:steps is not a positive int: {steps!r}")

    config = train_log.get("config_resolved")
    if not isinstance(config, dict) or not config:
        leg.fail("train_log.json:config_resolved is missing or empty -- no record "
                 "of the configuration this run actually used")

    base = train_log.get("base_model")
    if not isinstance(base, dict):
        leg.fail("train_log.json:base_model is missing")
    else:
        if not _is_str(base.get("repo")):
            leg.fail("train_log.json:base_model.repo is missing or empty")
        revision = base.get("revision")
        if not _is_str(revision) or not REVISION_RE.match(str(revision).strip()):
            leg.fail(f"train_log.json:base_model.revision is not a 40-hex commit "
                     f"sha ({revision!r}) -- the base model is not pinned")

    # wall clock: a strictly positive number. Bare truthiness accepted "fast"
    # and rejected a legitimate 0.5; bare isinstance(x, int) accepted True.
    wall = train_log.get("wall_clock_seconds")
    if not _is_pos_num(wall):
        leg.fail(f"train_log.json:wall_clock_seconds is not a positive number: "
                 f"{wall!r} -- a run that took no measured time is not a run")

    started = _parse_utc(train_log.get("started_utc"))
    finished = _parse_utc(train_log.get("finished_utc"))
    if started is None:
        leg.fail(f"train_log.json:started_utc is not an ISO-8601 timestamp "
                 f"({train_log.get('started_utc')!r})")
    if finished is None:
        leg.fail(f"train_log.json:finished_utc is not an ISO-8601 timestamp "
                 f"({train_log.get('finished_utc')!r})")
    if started is not None and finished is not None:
        elapsed = (finished - started).total_seconds()
        if elapsed < 0:
            leg.fail(f"train_log.json: finished_utc precedes started_utc "
                     f"({elapsed:.1f}s)")
        elif _is_pos_num(wall) and float(wall) > elapsed + WALL_CLOCK_SLACK_S:
            leg.fail(f"train_log.json:wall_clock_seconds ({float(wall):.1f}s) exceeds "
                     f"the elapsed run time started_utc -> finished_utc "
                     f"({elapsed:.1f}s) -- the recorded timings contradict each other")

    history = train_log.get("loss_history")
    if not isinstance(history, list) or not history:
        leg.fail("train_log.json:loss_history is missing or empty")
        return leg
    losses = []
    for entry in history:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            leg.fail(f"loss_history entry is not a [step, loss] pair: {entry!r}")
            return leg
        step, val = entry
        if not _is_nonneg_int(step):
            leg.fail(f"loss_history step is not a non-negative int: {step!r}")
            return leg
        if not _is_num(val):
            leg.fail(f"non-numeric or non-finite loss value in loss_history: {val!r}")
            return leg
        if _is_pos_int(steps) and step > steps:
            leg.fail(f"loss_history records step {step}, beyond the recorded "
                     f"{steps} optimizer steps")
            return leg
        losses.append(float(val))
    if all(v == 0.0 for v in losses):
        leg.fail("every recorded loss is 0.0 -- that is a placeholder, not a run")
    if leg.ok:
        leg.note(f"{steps} optimizer steps, {len(losses)} logged losses "
                 f"({losses[0]:.4f} -> {losses[-1]:.4f}), "
                 f"wall clock {float(wall):.1f}s")
    return leg


# ------------------------------------------------------------------ leg 3
def check_artifacts(train_log: dict) -> Leg:
    leg = Leg("3 artifacts (adapter + merged fp16)")
    art = train_log.get("artifacts")
    if not isinstance(art, dict):
        leg.fail("train_log.json has no 'artifacts' object")
        return leg
    seen = {}
    for key, size_key in (("adapter", "adapter_bytes"),
                          ("merged_fp16", "merged_bytes")):
        where = art.get(key)
        if not _is_str(where):
            leg.fail(f"artifacts.{key} carries no path reference ({where!r})")
            continue
        size = art.get(size_key)
        # _is_pos_int, never isinstance(size, int): `true` is not 1 byte.
        if not _is_pos_int(size):
            leg.fail(f"artifacts.{size_key} is not a positive int: {size!r} -- this "
                     f"leg exists to prove a NON-ZERO byte count")
            continue
        canon = _canon_ref(where)
        if canon in seen:
            leg.fail(f"artifacts.{key} and artifacts.{seen[canon]} are the same "
                     f"reference ({where!r}) -- one of the two was never written")
            continue
        seen[canon] = key
        leg.note(f"{key}: {size / 1e6:.1f} MB at {where}")
    return leg


# ------------------------------------------------------------------ leg 4
def checkpoint_identities(train_log: dict) -> dict:
    """{canonical reference -> artifacts key} for THIS run's trained checkpoint.

    Whole references only. The previous version seeded this set with the bare
    literals 'adapter' and 'merged_fp16' plus the run directory name and
    accepted `any(marker in spec)`, so any path containing the word "adapter"
    passed the leg whose entire job is proving the eval ran against the trained
    weights.
    """
    art = train_log.get("artifacts")
    if not isinstance(art, dict):
        return {}
    out = {}
    for key in ("adapter", "merged_fp16", "hf_repo", "merged_hf_repo"):
        canon = _canon_ref(art.get(key))
        if canon:
            out.setdefault(canon, key)
    return out


def _eval_pressures(repo_root: Path, res: dict, label: str):
    """({scenario_id: pressure}, errors) for the eval set a results JSON names.

    Robustness is spec_adherence over the ADVERSARIAL subset, and the pinned
    transcript row does not carry ``pressure`` -- harness.aggregate reads it off
    the scenario. So re-deriving robustness from the rows means joining them to
    the corpus, and a results JSON that names no resolvable eval set leaves the
    headline metric unverifiable. That is a failure, not a skip.
    """
    raw = res.get("eval_set")
    if not _is_str(raw):
        return None, [f"{label}: names no 'eval_set' -- without the scenario "
                      f"corpus each row's pressure is unknown, so metrics."
                      f"robustness cannot be re-derived from the transcripts"]
    corpus = Path(repo_root) / raw.strip()
    if not corpus.exists():
        return None, [f"{label}: eval_set {raw!r} does not exist at {corpus}"]
    try:
        corpus.resolve().relative_to(Path(repo_root).resolve())
    except ValueError:
        return None, [f"{label}: eval_set {raw!r} resolves outside the repo "
                      f"({corpus.resolve()}) -- it is not a committed artifact"]
    pressures, errors = {}, []
    for idx, line in enumerate(corpus.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            scenario = json.loads(line)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{raw} line {idx}: not valid JSON ({exc})")
            continue
        if not isinstance(scenario, dict):
            errors.append(f"{raw} line {idx}: not a JSON object")
            continue
        sid = scenario.get("scenario_id")
        if not _is_str(sid):
            errors.append(f"{raw} line {idx}: no usable 'scenario_id'")
            continue
        pressure = scenario.get("pressure")
        if not _is_str(pressure):
            errors.append(f"{raw} line {idx}: scenario {sid!r} has no 'pressure' -- "
                          f"the adversarial subset cannot be identified")
            continue
        pressures[sid] = pressure.strip()
    if not pressures and not errors:
        errors.append(f"{raw}: eval set is empty")
    return pressures, errors


def _transcript_evidence(path: Path, harness, pressures, expected_model):
    """(counts, errors) re-derived from the raw transcripts.

    ``expected_model`` is the results header's model string. Every row must carry
    it: leg 4 proves the eval ran against the trained checkpoint, and a header is
    one string a forger writes once. Base-model or mock rows under a forged
    header are what this closes.
    """
    counts = {"rows": 0, "scored": 0, "passed": 0, "judge_errors": 0,
              "adv_scored": 0, "adv_passed": 0}
    errors = []
    for idx, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        counts["rows"] += 1
        try:
            row = json.loads(line)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path.name} line {idx}: not valid JSON ({exc})")
            continue
        if not isinstance(row, dict):
            errors.append(f"{path.name} line {idx}: not a JSON object")
            continue
        # EVERY row is revalidated, quarantined ones included -- validation is
        # what proves a judge_error row really carries no fabricated verdict.
        try:
            harness.validate_transcript_row(row)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path.name} line {idx}: not a row this harness scored "
                          f"({exc})")
            continue
        # --- the row's OWN checkpoint identity, not the header's claim about it
        if row.get("model") != expected_model:
            errors.append(
                f"{path.name} line {idx}: row model {row.get('model')!r} != the "
                f"results header's model {expected_model!r} -- the transcripts "
                f"were not produced by the checkpoint the results claim")
            continue
        sid = row.get("scenario_id")
        if sid not in pressures:
            errors.append(f"{path.name} line {idx}: scenario {sid!r} is not in the "
                          f"eval set this results JSON names, so its pressure is "
                          f"unknown and it cannot be scored into robustness")
            continue
        if row.get(harness.JUDGE_ERROR_FIELD) is not None:
            counts["judge_errors"] += 1
            continue
        if row.get("spec_pass") is None:
            continue
        counts["scored"] += 1
        adversarial = pressures[sid] == "adversarial"
        if adversarial:
            counts["adv_scored"] += 1
        if row.get("spec_pass") is True:
            counts["passed"] += 1
            if adversarial:
                counts["adv_passed"] += 1
    return counts, errors


def check_eval(run_dir: Path, train_log: dict, harness, repo_root: Path) -> Leg:
    leg = Leg("4 eval (harness metrics against the trained checkpoint)")
    results_files = sorted(run_dir.rglob("*.results.json"))
    if not results_files:
        leg.fail(f"no *.results.json under {run_dir} -- run the eval harness "
                 f"against the trained checkpoint with --out {run_dir}/eval")
        return leg

    base = train_log.get("base_model") or {}
    base_ref = _canon_ref(base.get("repo") if isinstance(base, dict) else None)
    if not base_ref:
        leg.fail("train_log.json:base_model.repo is missing -- without it there "
                 "is no way to prove the results are not just the base model")
        return leg
    identities = checkpoint_identities(train_log)
    if not identities:
        leg.fail("train_log.json:artifacts names no checkpoint reference -- there "
                 "is nothing the results could be matched against")
        return leg

    rejected = []
    for rf in results_files:
        try:
            res = _load_json(rf)
        except Exception as exc:  # noqa: BLE001
            rejected.append(f"{rf.name}: not valid JSON ({exc})")
            continue
        if not isinstance(res, dict):
            rejected.append(f"{rf.name}: not a JSON object")
            continue
        spec = res.get("model")
        if not _is_str(spec):
            rejected.append(f"{rf.name}: names no model")
            continue
        canon = _canon_ref(spec)
        if canon == base_ref:
            rejected.append(f"{rf.name}: model {spec!r} is the BASE model, not the "
                            f"trained checkpoint")
            continue
        if canon not in identities:
            rejected.append(
                f"{rf.name}: model {spec!r} does not reference this run's "
                f"checkpoint -- it must equal one of {sorted(identities)} in "
                f"whole, not merely contain a word from it")
            continue

        metrics = res.get("metrics")
        if not isinstance(metrics, dict):
            rejected.append(f"{rf.name}: carries no 'metrics' object")
            continue
        missing = [k for k in REQUIRED_METRIC_KEYS if k not in metrics]
        if missing:
            rejected.append(f"{rf.name}: metrics is missing {missing} -- these are "
                            f"what harness.aggregate() emits; nothing scored them")
            continue
        n = metrics["n"]
        if not _is_pos_int(n):
            rejected.append(f"{rf.name}: metrics.n is not positive ({n!r})")
            continue
        n_scored = metrics["n_scored"]
        if not _is_pos_int(n_scored) or n_scored > n:
            rejected.append(f"{rf.name}: metrics.n_scored is not a positive int "
                            f"<= n ({n_scored!r} of {n!r}) -- nothing was scored")
            continue
        passed = metrics["passed"]
        if not _is_nonneg_int(passed) or passed > n_scored:
            rejected.append(f"{rf.name}: metrics.passed is not an int in "
                            f"[0, n_scored] ({passed!r})")
            continue
        if not _is_nonneg_int(metrics["judge_errors"]):
            rejected.append(f"{rf.name}: metrics.judge_errors is not a non-negative "
                            f"int ({metrics['judge_errors']!r})")
            continue
        sa = metrics["spec_adherence"]
        if not _is_rate(sa):
            rejected.append(f"{rf.name}: metrics.spec_adherence is missing or not a "
                            f"rate in [0, 1] ({sa!r}) -- nothing was scored")
            continue
        if abs(float(sa) - passed / n_scored) > 1e-9:
            rejected.append(f"{rf.name}: metrics.spec_adherence {sa!r} does not equal "
                            f"passed/n_scored ({passed}/{n_scored}) -- the headline "
                            f"number was not computed from the counts beside it")
            continue
        rob = metrics["robustness"]
        if not _is_rate(rob):
            rejected.append(f"{rf.name}: metrics.robustness is missing or not a rate "
                            f"in [0, 1] ({rob!r}) -- an eval with no adversarial "
                            f"cell does not test the target behavior")
            continue
        by_pressure = metrics["by_pressure"]
        if not isinstance(by_pressure, dict) or "adversarial" not in by_pressure:
            rejected.append(f"{rf.name}: metrics.by_pressure has no 'adversarial' "
                            f"cell")
            continue
        adv = by_pressure["adversarial"]
        if not isinstance(adv, dict) or not _is_pos_int(adv.get("n")):
            rejected.append(f"{rf.name}: metrics.by_pressure.adversarial.n is not "
                            f"positive -- robustness was reported over nothing")
            continue
        # ROBUSTNESS, re-derived instead of range-checked. It is the headline
        # metric of the adversarial subset; a bare 0 <= x <= 1 test accepts any
        # number at all.
        adv_n, adv_passed = adv["n"], adv.get("passed")
        if not _is_nonneg_int(adv_passed) or adv_passed > adv_n:
            rejected.append(f"{rf.name}: metrics.by_pressure.adversarial.passed is "
                            f"not an int in [0, n] ({adv_passed!r} of {adv_n}) -- "
                            f"the adversarial cell counts nothing")
            continue
        adv_sa = adv.get("spec_adherence")
        if not _is_rate(adv_sa):
            rejected.append(f"{rf.name}: metrics.by_pressure.adversarial."
                            f"spec_adherence is missing or not a rate in [0, 1] "
                            f"({adv_sa!r})")
            continue
        if abs(float(adv_sa) - adv_passed / adv_n) > 1e-9:
            rejected.append(f"{rf.name}: metrics.by_pressure.adversarial."
                            f"spec_adherence {adv_sa!r} does not equal passed/n "
                            f"({adv_passed}/{adv_n}) -- the adversarial cell's own "
                            f"rate was not computed from the counts beside it")
            continue
        if abs(float(rob) - float(adv_sa)) > 1e-9:
            rejected.append(f"{rf.name}: metrics.robustness {rob!r} does not equal "
                            f"metrics.by_pressure.adversarial.spec_adherence "
                            f"{adv_sa!r} -- robustness IS that cell, so the two "
                            f"disagreeing means one of them was written by hand")
            continue
        if adv_n > n_scored:
            rejected.append(f"{rf.name}: metrics.by_pressure.adversarial.n ({adv_n}) "
                            f"exceeds metrics.n_scored ({n_scored})")
            continue

        pressures, perrors = _eval_pressures(repo_root, res, rf.name)
        if perrors:
            rejected.append(f"{rf.name}: {perrors[0]}")
            continue

        transcripts = rf.with_name(rf.name.replace(".results.json",
                                                   ".transcripts.jsonl"))
        if not transcripts.exists():
            rejected.append(f"{rf.name}: judge transcripts {transcripts.name} "
                            f"missing (the spec requires raw per-example output)")
            continue
        counts, terrors = _transcript_evidence(transcripts, harness, pressures, spec)
        if terrors:
            rejected.append(f"{rf.name}: {terrors[0]}")
            continue
        if counts["rows"] != n:
            rejected.append(f"{rf.name}: {transcripts.name} has {counts['rows']} "
                            f"records, metrics.n is {n}")
            continue
        if counts["scored"] != n_scored or counts["passed"] != passed:
            rejected.append(
                f"{rf.name}: the transcripts do not support the metrics -- "
                f"{counts['scored']} scored / {counts['passed']} passed in "
                f"{transcripts.name} vs n_scored={n_scored} / passed={passed}")
            continue
        if counts["judge_errors"] != metrics["judge_errors"]:
            rejected.append(f"{rf.name}: {counts['judge_errors']} quarantined rows in "
                            f"{transcripts.name} vs metrics.judge_errors="
                            f"{metrics['judge_errors']}")
            continue
        if counts["adv_scored"] != adv_n or counts["adv_passed"] != adv_passed:
            rejected.append(
                f"{rf.name}: the transcripts do not support the ADVERSARIAL cell "
                f"robustness is computed over -- {counts['adv_scored']} scored / "
                f"{counts['adv_passed']} passed adversarial rows in "
                f"{transcripts.name} vs n={adv_n} / passed={adv_passed}")
            continue
        derived_rob = counts["adv_passed"] / counts["adv_scored"] \
            if counts["adv_scored"] else None
        if derived_rob is None or abs(float(rob) - derived_rob) > 1e-9:
            rejected.append(
                f"{rf.name}: metrics.robustness {rob!r} is not what the raw "
                f"transcripts give ({counts['adv_passed']}/{counts['adv_scored']} "
                f"adversarial rows = {derived_rob!r})")
            continue

        leg.note(f"{rf.relative_to(run_dir)}: model={spec} "
                 f"(matches artifacts.{identities[canon]}) n={n} "
                 f"spec_adherence={sa} robustness={rob}")
        leg.note(f"{transcripts.relative_to(run_dir)}: {counts['rows']} transcripts, "
                 f"all carrying model={spec}; {counts['passed']}/{counts['scored']} "
                 f"overall and {counts['adv_passed']}/{counts['adv_scored']} "
                 f"adversarial re-derived from the rows against {res.get('eval_set')}")
        return leg

    leg.fail(f"no usable results JSON among {len(results_files)} candidate(s) "
             f"under {run_dir}")
    for r in rejected:
        leg.fail(f"  rejected {r}")
    return leg


# ------------------------------------------------------------------ driver
def check_loop(run_dir: Path, repo_root: Path) -> list:
    # Captured before any leg runs, so that resolving the run's prompt renderer
    # (which manipulates sys.path and sys.modules) can never substitute the
    # scoring harness leg 4 revalidates the transcripts with.
    try:
        harness = _import_harness()
    except Exception as exc:  # noqa: BLE001
        leg = Leg("1-4 artifacts")
        leg.fail(f"cannot import the scoring harness from {REPO_ROOT}: {exc}")
        return [leg]

    if not run_dir.is_dir():
        leg = Leg("1-4 artifacts")
        leg.fail(f"{run_dir} is not a directory")
        return [leg]

    log_path = run_dir / "train_log.json"
    if not log_path.exists():
        leg = Leg("1-4 artifacts")
        leg.fail(f"{log_path} missing -- the trainer writes it; without it there "
                 f"is no evidence any training happened")
        return [leg]
    try:
        train_log = _load_json(log_path)
    except Exception as exc:  # noqa: BLE001
        leg = Leg("1-4 artifacts")
        leg.fail(f"{log_path} is not valid JSON: {exc}")
        return [leg]
    if not isinstance(train_log, dict):
        leg = Leg("1-4 artifacts")
        leg.fail(f"{log_path} is not a JSON object")
        return [leg]

    return [check_generate(train_log, repo_root),
            check_train(train_log),
            check_artifacts(train_log),
            check_eval(run_dir, train_log, harness, repo_root)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Verify the committed generate -> train -> eval loop artifacts")
    ap.add_argument("run_dir", help="run directory, e.g. runs/smoke")
    ap.add_argument("--repo-root", default=str(REPO_ROOT),
                    help="the repo this run was produced from: the dataset path "
                         "in train_log.json and the shared prompt renderer both "
                         "resolve against it")
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
