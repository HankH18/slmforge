#!/usr/bin/env python3
"""
P6 verify command — decides whether the MVP submission package is real.

    python scripts/check_mvp.py

Offline: no API keys, no GPU, no network. Nothing is taken on trust:

  * each per-example verdict is re-derived from that row's own `raw_output` with
    the frozen rubric (shared with check_ablation.py's `rederive_row`), so a
    transcript edited to flatter the model fails even when it is internally
    self-consistent;
  * each aggregate is re-derived from the transcripts (frozen `eval.aggregate`);
  * each metric VALUE printed in docs/MVP_REPORT.md is matched against the
    committed results JSON to the precision it is printed at — the headline
    base-vs-tuned numbers are the only ones a grader reads, so they are checked
    against the artifacts rather than merely present;
  * the report is cross-checked against configs/models.json and README.md, and
    every path it cites must exist.

Exit 0 means a grader could pull the pinned model and reproduce the numbers from
the committed artifacts.

Derived from the project spec's "Verification Requirements" table:
  * Public model checkpoint — exact HF repo id + commit hash referenced
  * One-command eval script — the repro line, byte-consistent with README.md
  * Raw judge transcripts — full per-example judge output (verdict + reasoning)
    as JSONL, not just the aggregate table
  * Pinned versions — exact HF model commit hash AND exact eval-code commit hash
and from BUILD_PLAN P6: base-vs-tuned under one rubric on one eval set, both
tables + every transcripts path in docs/MVP_REPORT.md.
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval as E  # noqa: E402  (frozen harness: imported, never reimplemented)
# One implementation of "re-derive the verdict from raw_output", shared with the
# G1 checker so the two gates can never drift apart on what the rubric says.
from check_ablation import load_scenarios, rederive_row  # noqa: E402

PINNED_RUBRIC_HASH = "c0adb876c3e6c4c9"
REQUIRED_STRATEGY = "zero_shot"
SPEC_MIN_SCENARIOS = 30
FLOAT_TOL = 1e-6

TUNED_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*@[0-9a-f]{40}$")
SHA40_RE = re.compile(r"\b[0-9a-f]{40}\b")
TABLE_SEP_RE = re.compile(r"^\|[\s:|-]+\|$")
TRANSCRIPT_KEYS = ("scenario_id", "gold_label", "adversarial", "pred_class",
                   "passed", "failure", "raw_output",
                   "judge_verdict", "judge_reasoning")

METRIC_KEYS = ("spec_adherence", "robustness", "false_answer_rate",
               "false_refusal_rate")
# a column/row label that reads like a metric but is not one the harness emits:
# reporting it means reporting a number no results JSON can back.
METRIC_LOOKALIKE_RE = re.compile(
    r"adherence|robust|_rate$|^rate|accuracy|precision|recall|^f1|score|"
    r"refusal|abstention|hallucination")
DELTA_LABEL_RE = re.compile(r"\b(delta|difference|change|improvement|gain|lift)\b")
# first number in a cell, with the unit that tells us whether it is a percentage
NUM_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*(%|pp|percentage[ _]points?)?", re.I)
NA_CELLS = {"", "-", "--", "—", "–", "n/a", "na", "none", "null", "n\\a"}

KNOWN_DIRS = {"runs", "data", "docs", "configs", "scripts", "training",
              "datagen", "tests"}
PATHY_EXTS = (".jsonl", ".json", ".md", ".py", ".txt", ".yaml", ".yml", ".csv")
STRIP_CHARS = "`()[]{}<>,;:.!\"'*|"
# a path-like run: one or more "<segment>/" followed by a final segment. Anchored
# on the characters legal in this repo's paths, so it lifts "runs/mvp/x.jsonl"
# out of a markdown link or a table cell without dragging punctuation along.
PATH_RE = re.compile(r"(?:[A-Za-z0-9_.@-]+/)+[A-Za-z0-9_.@*?-]+")


class Gate:
    def __init__(self, title):
        self.title = title
        self.failures = []
        self.notes = []

    def require(self, cond, msg):
        if not cond:
            self.failures.append(msg)
        return bool(cond)

    def fail(self, msg):
        self.failures.append(msg)

    def note(self, msg):
        self.notes.append(msg)

    def finish(self):
        for n in self.notes:
            print(f"  ok   {n}")
        if self.failures:
            print(f"\nFAIL {self.title} — {len(self.failures)} problem(s):")
            for f in self.failures:
                print(f"  - {f}")
            return 1
        print(f"\nPASS {self.title}")
        return 0


def num_eq(a, b, tol=FLOAT_TOL):
    if a is None or b is None:
        return a is None and b is None
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def read_json(path):
    with open(path) as f:
        return json.load(f)


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                rows.append((i, json.loads(line)))
    return rows


def judge_ran(row):
    """True iff eval.score_example would have called the judge for this row."""
    return (row.get("gold_label") == "answerable"
            and row.get("pred_class") == "answer"
            and row.get("failure") in (None, "entailment_fail"))


def logical_commands(text):
    """Shell-ish lines with backslash continuations joined and whitespace collapsed.

    "byte-consistent" is checked after exactly this normalization, so a repro
    command may be wrapped across lines in the report and inline in the README
    and still count as the same command — but any changed flag or value fails.
    """
    out, buf = [], None
    for raw in text.splitlines():
        ln = raw.strip()
        if ln.startswith("```"):
            continue
        buf = ln if buf is None else f"{buf} {ln}"
        if buf.endswith("\\"):
            buf = buf[:-1].rstrip()
            continue
        out.append(" ".join(buf.split()))
        buf = None
    if buf is not None:
        out.append(" ".join(buf.split()))
    return [c for c in out if c]


def referenced_paths(text):
    """Path-like substrings in the report: must contain '/' and either start in a
    known repo directory or end in a data/doc extension. HF repo ids
    ('owner/name', no extension), URLs and globs are deliberately excluded, and
    markdown-link punctuation is stripped."""
    found = set()
    for raw in PATH_RE.findall(re.sub(r"\w+://\S+", " ", text)):
        tok = raw.strip(STRIP_CHARS).rstrip("/")
        if not tok or "*" in tok or "?" in tok or "<" in tok:
            continue
        head = tok.split("/", 1)[0]
        if head in KNOWN_DIRS or tok.endswith(PATHY_EXTS):
            found.add(tok)
    return sorted(found)


# ------------------------------------------------------ report metric values
def split_row(line):
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def norm_key(cell):
    """A markdown cell -> comparable snake_case token ('Spec-Adherence' -> ...)."""
    t = re.sub(r"[`*_]+", " ", str(cell)).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", t).strip("_")


def parse_tables(text):
    """[(header_cells, [body_row_cells, ...]), ...] for every markdown table."""
    lines = text.splitlines()
    tables, i = [], 0
    while i < len(lines) - 1:
        if lines[i].strip().startswith("|") and TABLE_SEP_RE.match(lines[i + 1].strip()):
            header = split_row(lines[i])
            body, j = [], i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                body.append(split_row(lines[j]))
                j += 1
            tables.append((header, body))
            i = j
        else:
            i += 1
    return tables


def parse_number_cell(cell):
    """-> (kind, value_as_fraction, tolerance) with kind in num | na | bad.

    A cell printed to k decimals can only be trusted to half a unit in the last
    place, so the tolerance is derived from the printed precision rather than
    being a loose constant: '90.6%' pins the true value to +-0.05pp.
    """
    t = re.sub(r"[`*]", "", str(cell)).strip()
    if t.lower() in NA_CELLS:
        return "na", None, None
    m = NUM_RE.search(t)
    if not m:
        return "bad", None, None
    digits = m.group(1)
    decimals = len(digits.split(".")[1]) if "." in digits else 0
    half_ulp = 0.5 * (10.0 ** -decimals)
    value, tol = float(digits), half_ulp * 1.05
    if m.group(2):                      # printed in percent / percentage points
        value, tol = value / 100.0, tol / 100.0
    return "num", value, tol + 1e-12


def role_of(label, tuned_pin, base_repo):
    """Which committed run a table label refers to: tuned | base | delta | None."""
    low = label.lower()
    tuned_repo = tuned_pin.split("@", 1)[0].lower()
    base_name = base_repo.split("@", 1)[0].lower()
    if tuned_pin.lower() in low or tuned_repo in low:
        return "tuned"
    if base_repo.lower() in low or base_name in low:
        return "base"
    if DELTA_LABEL_RE.search(low) or "Δ" in label or "δ" in low:
        return "delta"
    if re.search(r"\btuned\b", low):
        return "tuned"
    if re.search(r"\bbase(line)?\b", low):
        return "base"
    return None


def check_report_metrics(gate, text, expected, tuned_pin, base_repo):
    """Every metric number printed in docs/MVP_REPORT.md must equal the committed
    results JSON it claims to summarize.

    Without this the report's headline base-vs-tuned numbers -- the only numbers a
    grader actually reads -- can be arbitrarily inflated while every other gate
    stays green, because the artifacts themselves are untouched.
    """
    checked, attempted = set(), set()

    def compare(role, metric, cell, where):
        attempted.add((role, metric))
        kind, val, tol = parse_number_cell(cell)
        exp = expected.get(role, {}).get(metric)
        if kind == "bad":
            gate.fail(f"docs/MVP_REPORT.md: {where} reads {cell!r}, which is not a "
                      f"number that can be checked against the committed results "
                      f"JSON")
            return
        if kind == "na":
            if exp is not None:
                gate.fail(f"docs/MVP_REPORT.md: {where} says {cell!r} but the "
                          f"committed results JSON reports {exp!r}")
            else:
                checked.add((role, metric))
            return
        if exp is None:
            gate.fail(f"docs/MVP_REPORT.md: {where} reports {cell!r} but no "
                      f"committed results JSON has a {metric} value for the "
                      f"{role} run — the report states a number nothing backs")
            return
        if abs(val - exp) > tol:
            gate.fail(f"docs/MVP_REPORT.md: {where} reports {cell!r} but the "
                      f"committed results JSON says {exp!r} "
                      f"(|{val:.6f} - {exp:.6f}| > {tol:.6f}) — the report does "
                      f"not match the artifact it claims to summarize")
            return
        checked.add((role, metric))

    for header, body in parse_tables(text):
        hkeys = [norm_key(h) for h in header]
        metric_cols = {i: k for i, k in enumerate(hkeys) if i > 0 and k in METRIC_KEYS}
        metric_rows = [r for r in body if r and norm_key(r[0]) in METRIC_KEYS]

        if metric_cols:                                   # rows are models
            for i, key in enumerate(hkeys):
                if i > 0 and key not in METRIC_KEYS and METRIC_LOOKALIKE_RE.search(key):
                    gate.fail(f"docs/MVP_REPORT.md reports a metric column "
                              f"{header[i]!r} that the eval harness never emits, so "
                              f"no results JSON can back it (known metrics: "
                              f"{list(METRIC_KEYS)})")
            for row in body:
                if not row:
                    continue
                role = role_of(row[0], tuned_pin, base_repo)
                if role is None:
                    if any(parse_number_cell(row[i])[0] == "num"
                           for i in metric_cols if i < len(row)):
                        gate.fail(f"docs/MVP_REPORT.md prints metric numbers on the "
                                  f"row {row[0]!r}, which names neither the pinned "
                                  f"tuned model, the base model, nor a delta — "
                                  f"there is no results JSON to check it against")
                    continue
                for i, metric in metric_cols.items():
                    if i < len(row):
                        compare(role, metric, row[i], f"the {role} {metric} cell")

        elif metric_rows:                                 # rows are metrics
            col_roles = {i: role_of(h, tuned_pin, base_repo)
                         for i, h in enumerate(header) if i > 0}
            for row in body:
                if not row:
                    continue
                key = norm_key(row[0])
                if key not in METRIC_KEYS:
                    if (METRIC_LOOKALIKE_RE.search(key)
                            and any(parse_number_cell(c)[0] == "num" for c in row[1:])):
                        gate.fail(f"docs/MVP_REPORT.md reports a metric {row[0]!r} "
                                  f"that the eval harness never emits, so no results "
                                  f"JSON can back it (known metrics: "
                                  f"{list(METRIC_KEYS)})")
                    continue
                for i, role in col_roles.items():
                    if i >= len(row):
                        continue
                    if role is None:
                        if parse_number_cell(row[i])[0] == "num":
                            gate.fail(f"docs/MVP_REPORT.md prints a {key} number "
                                      f"under the column {header[i]!r}, which names "
                                      f"neither the pinned tuned model, the base "
                                      f"model, nor a delta")
                        continue
                    compare(role, key, row[i], f"the {role} {key} cell")

    for role in ("base", "tuned"):
        if (role, "spec_adherence") not in attempted:
            gate.fail(f"docs/MVP_REPORT.md never prints a checkable {role} "
                      f"spec_adherence value in a markdown table; BUILD_PLAN P6 "
                      f"requires the base-vs-tuned table, and a number that cannot "
                      f"be matched to {role}'s results JSON is not evidence")
    if checked:
        gate.note(f"{len(checked)} reported metric value(s) match the committed "
                  f"results JSONs to the precision they are printed at")


def load_models_json(gate, root):
    path = root / "configs" / "models.json"
    if not gate.require(path.exists(), "configs/models.json missing"):
        return None
    try:
        cfg = read_json(path)
    except (json.JSONDecodeError, OSError) as exc:
        gate.fail(f"configs/models.json unreadable ({exc})")
        return None
    if not isinstance(cfg, dict):
        gate.fail("configs/models.json is not a JSON object")
        return None
    tuned = cfg.get("tuned")
    if not isinstance(tuned, str) or not TUNED_RE.match(tuned):
        gate.fail(f"configs/models.json['tuned'] == {tuned!r}; expected "
                  f"'<owner>/<name>@<40-hex-sha>' (written by P5's push_model.py)")
        return None
    base = cfg.get("base")
    if not isinstance(base, str) or not base.startswith("hf:"):
        gate.fail(f"configs/models.json['base'] == {base!r}; expected "
                  f"'hf:<repo>@<revision>'")
        return None
    return cfg


def check_run(gate, root, run_dir, model_spec, role, scenarios):
    """Validate one model's committed results + transcripts. Returns (res, rows)."""
    scenario_ids = set(scenarios) if scenarios is not None else None
    stem = E.sanitize(model_spec)
    matches = sorted(run_dir.glob(f"{stem}__*.results.json"))
    if not matches:
        have = sorted(p.name for p in run_dir.glob("*.results.json"))
        gate.fail(f"no {role} results JSON for {model_spec!r} in {run_dir} "
                  f"(expected {stem}__{REQUIRED_STRATEGY}.results.json; found {have})")
        return None, None
    if len(matches) > 1:
        gate.fail(f"{role} model {model_spec!r} has {len(matches)} results JSONs "
                  f"in {run_dir}: {[p.name for p in matches]}")
    rp = matches[0]
    try:
        res = read_json(rp)
    except (json.JSONDecodeError, OSError) as exc:
        gate.fail(f"{rp.name}: unreadable ({exc})")
        return None, None

    strategy = res.get("strategy")
    if strategy != REQUIRED_STRATEGY:
        gate.fail(f"{rp.name}: strategy {strategy!r} != {REQUIRED_STRATEGY!r} — "
                  f"the MVP comparison must run the deployment (zero-shot) "
                  f"condition the model was trained on")
    if res.get("rubric_hash") != PINNED_RUBRIC_HASH:
        gate.fail(f"{rp.name}: rubric_hash {res.get('rubric_hash')!r} != "
                  f"{PINNED_RUBRIC_HASH!r}")
    if not res.get("judge_model"):
        gate.fail(f"{rp.name}: no judge_model recorded")

    metrics = res.get("metrics")
    if not isinstance(metrics, dict):
        gate.fail(f"{rp.name}: no metrics object")
        return res, None
    n = metrics.get("n")
    if not isinstance(n, int) or n < SPEC_MIN_SCENARIOS:
        gate.fail(f"{rp.name}: n == {n!r}; the project spec requires at least "
                  f"{SPEC_MIN_SCENARIOS} scenarios")

    tp = rp.with_name(rp.name.replace(".results.json", ".transcripts.jsonl"))
    if not tp.exists():
        gate.fail(f"{tp.name}: missing — the spec requires the full per-example "
                  f"judge transcript JSONL for the {role} model, not just the "
                  f"aggregate table")
        return res, None
    try:
        numbered = read_jsonl(tp)
    except (json.JSONDecodeError, OSError) as exc:
        gate.fail(f"{tp.name}: unreadable ({exc})")
        return res, None
    rows = [r for _, r in numbered]
    if len(rows) != n:
        gate.fail(f"{tp.name}: {len(rows)} transcript rows but metrics.n == {n!r}")

    well_formed = True
    rederived = rederived_bad = 0
    for lineno, row in numbered:
        missing = [k for k in TRANSCRIPT_KEYS if k not in row]
        if missing:
            gate.fail(f"{tp.name}:{lineno}: row missing {missing}")
            well_formed = False
            continue
        if scenarios is not None:
            scen = scenarios.get(row.get("scenario_id"))
            if scen is not None:
                for problem in rederive_row(scen, row):
                    gate.fail(f"{tp.name}:{lineno} (scenario "
                              f"{row.get('scenario_id')}): {problem}")
                    rederived_bad += 1
                rederived += 1
        if row.get("model") not in (None, model_spec):
            gate.fail(f"{tp.name}:{lineno}: row model {row['model']!r} != "
                      f"{model_spec!r}")
        if judge_ran(row):
            if row.get("judge_verdict") is None:
                gate.fail(f"{tp.name}:{lineno} (scenario {row.get('scenario_id')}): "
                          f"judge_verdict is null on a scored answerable example")
            if not str(row.get("judge_reasoning") or "").strip():
                gate.fail(f"{tp.name}:{lineno} (scenario {row.get('scenario_id')}): "
                          f"judge_reasoning is empty on a scored answerable example")

    if well_formed and rows:
        recomputed = E.aggregate(rows)
        for k in ("n", "spec_adherence", "robustness",
                  "false_answer_rate", "false_refusal_rate"):
            if not num_eq(recomputed.get(k), metrics.get(k)):
                gate.fail(f"{rp.name}: metrics.{k} == {metrics.get(k)!r} but the "
                          f"transcripts give {recomputed.get(k)!r}")
        if recomputed["failure_breakdown"] != metrics.get("failure_breakdown"):
            gate.fail(f"{rp.name}: failure_breakdown disagrees with the transcripts")

    if rederived and not rederived_bad:
        gate.note(f"{role}: {rederived} transcript rows re-scored from their own "
                  f"raw_output with the frozen rubric; every stored verdict agrees")

    ids = [r.get("scenario_id") for r in rows]
    if len(set(ids)) != len(ids):
        gate.fail(f"{tp.name}: duplicate scenario_ids in the transcripts")
    if scenario_ids is not None and rows and set(ids) != scenario_ids:
        only_run = sorted(set(ids) - scenario_ids)[:5]
        only_set = sorted(scenario_ids - set(ids))[:5]
        gate.fail(f"{tp.name}: scenario_ids do not match the frozen eval set "
                  f"(extra {only_run}, missing {only_set})")
    return res, rows


def check_report(gate, root, cfg, base_spec, run_dir, tuned_res, base_res):
    report_path = root / "docs" / "MVP_REPORT.md"
    if not gate.require(report_path.exists(), "docs/MVP_REPORT.md missing"):
        return
    text = report_path.read_text()
    if not text.strip():
        gate.fail("docs/MVP_REPORT.md is empty")
        return

    tuned = cfg["tuned"]
    repo_id, _, model_sha = tuned.partition("@")

    # --- pinned versions ------------------------------------------------
    if tuned not in text:
        gate.fail(f"docs/MVP_REPORT.md does not contain the pinned tuned model "
                  f"{tuned!r} from configs/models.json['tuned'] — the spec "
                  f"requires the exact HF repo id + commit hash")
    else:
        gate.note(f"tuned model pinned in the report and matching models.json: {tuned}")
    for other in SHA40_RE.findall(text):
        if other != model_sha and f"{repo_id}@{other}" in text:
            gate.fail(f"docs/MVP_REPORT.md pins {repo_id}@{other} but "
                      f"configs/models.json['tuned'] says {tuned}")

    base_repo = base_spec.split(":", 1)[1]
    if base_repo not in text and base_repo.split("@", 1)[0] not in text:
        gate.fail(f"docs/MVP_REPORT.md never mentions the base model "
                  f"{base_repo!r} — a base-vs-tuned report must name both")

    eval_commit_lines = [ln for ln in text.splitlines()
                         if SHA40_RE.search(ln) and "eval" in ln.lower()
                         and not SHA40_RE.search(ln).group(0) == model_sha]
    eval_shas = {SHA40_RE.search(ln).group(0) for ln in eval_commit_lines}
    if not eval_shas:
        gate.fail("docs/MVP_REPORT.md has no eval-code commit hash: the spec "
                  "requires the exact eval-code commit alongside the model "
                  "commit. Write it on a line that says 'eval', e.g. "
                  "`| eval-code commit | <40-hex-sha> |`")
    else:
        gate.note(f"eval-code commit recorded: {sorted(eval_shas)[0]}")

    if PINNED_RUBRIC_HASH not in text:
        gate.fail(f"docs/MVP_REPORT.md does not state the rubric hash "
                  f"{PINNED_RUBRIC_HASH!r} the numbers were scored under")

    # --- both tables ----------------------------------------------------
    n_tables = sum(1 for ln in text.splitlines() if TABLE_SEP_RE.match(ln.strip()))
    if n_tables < 2:
        gate.fail(f"docs/MVP_REPORT.md contains {n_tables} markdown table(s); "
                  f"BUILD_PLAN P6 requires both tables (base-vs-tuned results "
                  f"and the delta)")
    normalized = text.lower().replace("-", "_").replace(" ", "_")
    for metric in ("spec_adherence", "robustness"):
        if metric not in normalized:
            gate.fail(f"docs/MVP_REPORT.md never reports {metric}")

    # --- and the numbers in those tables are the committed ones ----------
    tuned_metrics = (tuned_res or {}).get("metrics")
    base_metrics = (base_res or {}).get("metrics")
    if isinstance(tuned_metrics, dict) and isinstance(base_metrics, dict):
        expected = {"tuned": tuned_metrics, "base": base_metrics,
                    "delta": {k: (None if tuned_metrics.get(k) is None
                                  or base_metrics.get(k) is None
                                  else tuned_metrics[k] - base_metrics[k])
                              for k in METRIC_KEYS}}
        check_report_metrics(gate, text, expected, tuned,
                             base_spec.split(":", 1)[-1])
    else:
        gate.fail("the committed results JSONs could not be read, so the metric "
                  "values printed in docs/MVP_REPORT.md were NOT verified")

    # --- one-command repro line ----------------------------------------
    report_cmds = logical_commands(text)
    repro = [c for c in report_cmds
             if "eval.py" in c and "--model" in c and "--eval-set" in c]
    pinned_repro = [c for c in repro if tuned in c]
    if not repro:
        gate.fail("docs/MVP_REPORT.md has no one-command repro line "
                  "(`eval.py --model ... --eval-set ...`)")
    elif not pinned_repro:
        gate.fail(f"the repro line in docs/MVP_REPORT.md does not pin "
                  f"{tuned!r}: {repro[0]!r}")
    else:
        readme_path = root / "README.md"
        if not readme_path.exists():
            gate.fail("README.md missing — BUILD_PLAN P6 requires the repro line "
                      "mirrored there")
        else:
            readme_cmds = set(logical_commands(readme_path.read_text()))
            for cmd in pinned_repro:
                if cmd not in readme_cmds:
                    gate.fail(f"the repro line in docs/MVP_REPORT.md is not "
                              f"byte-consistent with README.md.\n"
                              f"      report: {cmd}\n"
                              f"      README has: "
                              f"{[c for c in readme_cmds if 'eval.py' in c] or 'no eval.py command'}")
                else:
                    gate.note(f"repro line mirrored byte-for-byte in README.md: {cmd}")

    # --- transcripts paths ----------------------------------------------
    transcripts = sorted(run_dir.glob("*.transcripts.jsonl"))
    if not transcripts:
        gate.fail(f"no transcripts JSONL in {run_dir}")
    for tp in transcripts:
        rel = tp.relative_to(root).as_posix()
        if rel not in text:
            gate.fail(f"docs/MVP_REPORT.md does not reference the transcripts "
                      f"file {rel} — the spec requires the raw judge transcripts "
                      f"to be pointed at, not just produced")

    for rel in referenced_paths(text):
        if not (root / rel).exists():
            gate.fail(f"docs/MVP_REPORT.md references {rel} which does not exist "
                      f"on disk")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Re-check the committed MVP submission package (P6 / GATE G2)")
    ap.add_argument("--root", default=str(REPO_ROOT),
                    help="repo root holding runs/mvp, docs/, configs/, README.md")
    ap.add_argument("--run-dir", default=None,
                    help="override runs/mvp")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir) if args.run_dir else root / "runs" / "mvp"
    gate = Gate(f"MVP submission check: {root}")
    print(f"checking {root} against rubric {PINNED_RUBRIC_HASH}")

    if E.rubric_hash() != PINNED_RUBRIC_HASH:
        gate.fail(f"eval.rubric_hash() == {E.rubric_hash()!r} but this checker "
                  f"pins {PINNED_RUBRIC_HASH!r} — the rubric drifted; STOP")
        return gate.finish()

    if not root.is_dir():
        gate.fail(f"{root} does not exist")
        return gate.finish()
    if not run_dir.is_dir() or not any(run_dir.iterdir()):
        gate.fail(f"{run_dir} is missing or empty — no MVP run artifacts committed")
        return gate.finish()

    cfg = load_models_json(gate, root)
    if cfg is None:
        return gate.finish()

    tuned_spec = f"hf:{cfg['tuned']}"
    base_spec = cfg["base"]
    if tuned_spec == base_spec:
        gate.fail("configs/models.json pins the same repo@commit for 'tuned' and "
                  "'base' — there is no comparison to make")

    scenarios_path = root / "data" / "eval" / "scenarios_v1.jsonl"
    scenarios = None
    if scenarios_path.exists():
        try:
            scenarios = load_scenarios(scenarios_path)
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            gate.fail(f"{scenarios_path} is unreadable ({exc}) — the per-example "
                      f"verdicts cannot be re-derived from raw_output")
    else:
        gate.fail(f"{scenarios_path} missing — cannot confirm both models ran the "
                  f"same eval set, nor re-derive the per-example verdicts")

    tuned_res, tuned_rows = check_run(gate, root, run_dir, tuned_spec, "tuned",
                                      scenarios)
    base_res, base_rows = check_run(gate, root, run_dir, base_spec, "base",
                                    scenarios)

    extra = sorted(p.name for p in run_dir.glob("*.results.json")
                   if not p.name.startswith((E.sanitize(tuned_spec) + "__",
                                             E.sanitize(base_spec) + "__")))
    if extra:
        gate.fail(f"{run_dir} holds results JSONs that are neither the pinned "
                  f"tuned nor the pinned base model: {extra}")

    if tuned_res and base_res:
        tm, bm = tuned_res.get("metrics") or {}, base_res.get("metrics") or {}
        if tm.get("n") != bm.get("n"):
            gate.fail(f"tuned n={tm.get('n')!r} but base n={bm.get('n')!r} — the "
                      f"two models did not run the same eval set")
        if tuned_res.get("rubric_hash") != base_res.get("rubric_hash"):
            gate.fail("tuned and base were scored under different rubric hashes")
        if tuned_res.get("judge_model") != base_res.get("judge_model"):
            gate.fail(f"tuned judge {tuned_res.get('judge_model')!r} != base judge "
                      f"{base_res.get('judge_model')!r} — one rubric means one judge")
        if tuned_res.get("strategy") != base_res.get("strategy"):
            gate.fail("tuned and base were run under different strategies")
        if tuned_rows and base_rows:
            if {r.get("scenario_id") for r in tuned_rows} != \
                    {r.get("scenario_id") for r in base_rows}:
                gate.fail("tuned and base transcripts cover different scenarios")
            gate.note(f"base-vs-tuned on {len(tuned_rows)} shared scenarios, "
                      f"spec_adherence {bm.get('spec_adherence')} -> "
                      f"{tm.get('spec_adherence')}")

    check_report(gate, root, cfg, base_spec, run_dir, tuned_res, base_res)
    return gate.finish()


if __name__ == "__main__":
    sys.exit(main())
