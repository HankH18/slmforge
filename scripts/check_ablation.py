#!/usr/bin/env python3
"""
P2 verify command — re-checks the committed Prompt-Ceiling Ablation (GATE G1).

    python scripts/check_ablation.py runs/ablation

Offline: no API keys, no GPU, no network. Nothing on disk is taken on trust:

  * every per-example verdict is re-derived from that row's own `raw_output`
    with the frozen `eval.parse_output` + `eval.score_example` (imported, never
    reimplemented), so a transcript whose stored pred_class/passed/failure
    flatters the model fails even when it is internally self-consistent;
  * every aggregate is then re-derived from those rows with `eval.aggregate`,
    so a hand-edited results JSON or a stale gate_report.json fails loudly
    rather than being echoed back.

The one thing that cannot be recomputed offline is the LLM judge's entailment
verdict. It is trusted ONLY where the rubric would actually have consulted it —
gold=answerable rows whose format and verbatim-span checks pass on the stored
raw_output — and a verdict recorded on any other row is itself a failure.

Exit 0 means the ablation on disk is a valid G1 artifact.

What must hold (project spec "Ablation 1" + BUILD_PLAN P2):
  * every transcript row's pred_class/passed/failure re-derives from its own
    raw_output under the frozen rubric, against the frozen eval set
  * 3 models x 3 strategies = 9 result JSONs, n == 32 each (spec floor is 30)
  * a matching transcripts JSONL per cell with n rows carrying the per-example
    judge verdict + reasoning (spec: "raw judge transcripts, not just aggregates")
  * one rubric across every cell: rubric_hash == eval.rubric_hash()
  * >= 2 different frontier model families plus the local base model
  * strategies exactly zero_shot, few_shot, structured
  * ablation_table.md's failure-mode paragraph written (no "_TODO"), citing >= 3
    scenario IDs that really exist in the eval set
  * gate_report.json's gate_fail_rate == 1 - max(spec_adherence over the 6
    FRONTIER cells), with escalation_needed == (gate_fail_rate < 0.20). That file
    is produced by scripts/make_gate_report.py; the value is RECOMPUTED here from
    the results JSONs, so a hand-edited gate number fails.
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import eval as E  # noqa: E402  (frozen harness: imported, never reimplemented)

PINNED_RUBRIC_HASH = "c0adb876c3e6c4c9"
EXPECTED_STRATEGIES = {"zero_shot", "few_shot", "structured"}
EXPECTED_N = 32
SPEC_MIN_SCENARIOS = 30           # project spec: >= 30 per model x strategy cell
ESCALATION_THRESHOLD = 0.20       # BUILD_PLAN P2
FRONTIER_PROVIDERS = {"openai", "google", "anthropic"}
LOCAL_PROVIDERS = {"hf"}
FLOAT_TOL = 1e-6

TRANSCRIPT_KEYS = ("scenario_id", "gold_label", "adversarial", "pred_class",
                   "passed", "failure", "raw_output",
                   "judge_verdict", "judge_reasoning")
SCENARIO_ID_RE = re.compile(r"\b[A-Z]{1,4}[0-9]{1,3}\b")


class Gate:
    """Collects failures instead of stopping at the first one."""

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


def provider_of(spec):
    return spec.split(":", 1)[0] if ":" in spec else ""


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


def load_scenarios(path):
    """{scenario_id: scenario} for the frozen eval set the rows must re-derive against."""
    return {s["id"]: s for s in E.load_jsonl(str(path))}


def rederive_row(scenario, row):
    """Re-run the FROZEN rubric over this row's own raw_output.

    Returns a list of disagreements between what `eval.score_example` decides and
    what the transcript claims. `eval.aggregate` reads the stored pred_class /
    passed / adversarial fields, so without this a self-consistent hand edit can
    move every aggregate in the ablation without touching a single raw_output.

    The judge is the one input that cannot be recomputed offline, so this feeds
    `score_example` the STORED verdict — but only the rubric decides whether the
    judge is consulted at all. A row whose raw_output is malformed, refuses, or
    cites a span that is not verbatim in the passage never reaches the judge, so
    no stored verdict can rescue it.
    """
    raw = row.get("raw_output")
    if not isinstance(raw, str):
        return [f"raw_output is {type(raw).__name__}, not a string — the verdict "
                f"cannot be re-derived from it"]

    judged = {"called": False}

    def judge_fn(question, answer, spans):
        judged["called"] = True
        return bool(row.get("judge_verdict")), row.get("judge_reasoning")

    try:
        rec = E.score_example(scenario, raw, judge_fn)
    except Exception as exc:  # noqa: BLE001 - a row the rubric cannot score is a failure
        return [f"eval.score_example could not score this row ({exc})"]

    problems = []
    fields = ["gold_label", "adversarial", "pred_class", "passed", "failure"]
    if "type" in row:
        fields.insert(2, "type")
    for k in fields:
        if rec.get(k) != row.get(k):
            problems.append(
                f"{k} == {row.get(k)!r} but re-deriving from this row's own "
                f"raw_output with the frozen rubric gives {rec.get(k)!r}")
    if not judged["called"] and row.get("judge_verdict") is not None:
        problems.append(
            f"judge_verdict {row.get('judge_verdict')!r} is recorded, but the "
            f"rubric never sends this row to the judge (pred_class "
            f"{rec.get('pred_class')!r}, failure {rec.get('failure')!r}) — a "
            f"fabricated verdict")
    if problems and rec.get("nonverbatim_spans"):
        problems.append(f"cited span(s) not verbatim in the passage: "
                        f"{rec['nonverbatim_spans'][:2]}")
    return problems


def check_cells(gate, out_dir, scenarios):
    """Load the 9 result JSONs + their transcripts; verify shape and self-consistency.

    Returns {(model, strategy): results_dict} for the cells that loaded cleanly.
    """
    results_paths = sorted(out_dir.glob("*.results.json"))
    if not gate.require(len(results_paths) == 9,
                        f"expected 9 results JSONs (3 models x 3 strategies), "
                        f"found {len(results_paths)}: "
                        f"{[p.name for p in results_paths]}"):
        if not results_paths:
            return {}

    cells = {}
    any_judged_row = False
    rederived = rederived_bad = 0
    for rp in results_paths:
        try:
            res = read_json(rp)
        except (json.JSONDecodeError, OSError) as exc:
            gate.fail(f"{rp.name}: unreadable ({exc})")
            continue

        model = res.get("model")
        strategy = res.get("strategy")
        if not isinstance(model, str) or not isinstance(strategy, str):
            gate.fail(f"{rp.name}: missing 'model'/'strategy' string fields")
            continue
        key = (model, strategy)
        if key in cells:
            gate.fail(f"{rp.name}: duplicate cell {model} x {strategy}")
            continue
        cells[key] = res

        stem = f"{E.sanitize(model)}__{strategy}"
        if rp.name != f"{stem}.results.json":
            gate.fail(f"{rp.name}: filename does not match its own "
                      f"model/strategy (expected {stem}.results.json)")

        if res.get("rubric_hash") != PINNED_RUBRIC_HASH:
            gate.fail(f"{rp.name}: rubric_hash {res.get('rubric_hash')!r} != "
                      f"{PINNED_RUBRIC_HASH!r} — a mixed-rubric ablation is invalid")
        if strategy not in EXPECTED_STRATEGIES:
            gate.fail(f"{rp.name}: unknown strategy {strategy!r}")
        if not res.get("judge_model"):
            gate.fail(f"{rp.name}: no judge_model recorded")
        if not res.get("timestamp_utc"):
            gate.fail(f"{rp.name}: no timestamp_utc recorded")

        metrics = res.get("metrics")
        if not isinstance(metrics, dict):
            gate.fail(f"{rp.name}: no metrics object")
            continue
        n = metrics.get("n")
        if n != EXPECTED_N:
            gate.fail(f"{rp.name}: n == {n!r}, expected {EXPECTED_N} "
                      f"(project spec floor is {SPEC_MIN_SCENARIOS} per cell)")

        tp = out_dir / f"{stem}.transcripts.jsonl"
        if not tp.exists():
            gate.fail(f"{tp.name}: missing — the spec requires raw judge "
                      f"transcripts per cell, not just aggregates")
            continue
        try:
            numbered = read_jsonl(tp)
        except (json.JSONDecodeError, OSError) as exc:
            gate.fail(f"{tp.name}: unreadable ({exc})")
            continue
        rows = [r for _, r in numbered]
        if len(rows) != EXPECTED_N:
            gate.fail(f"{tp.name}: {len(rows)} rows, expected {EXPECTED_N}")

        well_formed = True
        for lineno, row in numbered:
            missing = [k for k in TRANSCRIPT_KEYS if k not in row]
            if missing:
                gate.fail(f"{tp.name}:{lineno}: row missing {missing}")
                well_formed = False
                continue
            if scenarios is not None:
                scen = scenarios.get(row.get("scenario_id"))
                if scen is None:
                    gate.fail(f"{tp.name}:{lineno}: scenario_id "
                              f"{row.get('scenario_id')!r} is not in the frozen "
                              f"eval set — this row cannot be re-derived")
                else:
                    for problem in rederive_row(scen, row):
                        gate.fail(f"{tp.name}:{lineno} (scenario "
                                  f"{row.get('scenario_id')}): {problem}")
                        rederived_bad += 1
                    rederived += 1
            if row.get("model") not in (None, model):
                gate.fail(f"{tp.name}:{lineno}: row model {row['model']!r} != "
                          f"cell model {model!r}")
            if row.get("strategy") not in (None, strategy):
                gate.fail(f"{tp.name}:{lineno}: row strategy "
                          f"{row['strategy']!r} != cell strategy {strategy!r}")
            if judge_ran(row):
                any_judged_row = True
                if row.get("judge_verdict") is None:
                    gate.fail(f"{tp.name}:{lineno} (scenario "
                              f"{row.get('scenario_id')}): judge_verdict is null "
                              f"on a judged answerable example")
                if not str(row.get("judge_reasoning") or "").strip():
                    gate.fail(f"{tp.name}:{lineno} (scenario "
                              f"{row.get('scenario_id')}): judge_reasoning is "
                              f"empty on a judged answerable example")

        if well_formed and len(rows) == EXPECTED_N:
            recomputed = E.aggregate(rows)
            for k in ("n", "spec_adherence", "robustness",
                      "false_answer_rate", "false_refusal_rate"):
                if not num_eq(recomputed.get(k), metrics.get(k)):
                    gate.fail(f"{rp.name}: metrics.{k} == {metrics.get(k)!r} but "
                              f"the transcripts give {recomputed.get(k)!r}")
            if recomputed["failure_breakdown"] != metrics.get("failure_breakdown"):
                gate.fail(f"{rp.name}: failure_breakdown "
                          f"{metrics.get('failure_breakdown')!r} disagrees with "
                          f"the transcripts {recomputed['failure_breakdown']!r}")
            if recomputed["confusion"] != metrics.get("confusion"):
                gate.fail(f"{rp.name}: confusion matrix disagrees with the "
                          f"transcripts")

    if cells and not any_judged_row:
        gate.fail("no transcript row anywhere in the ablation was scored by the "
                  "LLM judge — the judge never ran, so these are not judged results")
    if cells:
        gate.note(f"{len(cells)} cells loaded, aggregates re-derived from transcripts")
    if rederived and not rederived_bad:
        gate.note(f"{rederived} transcript rows re-scored from their own raw_output "
                  f"with the frozen rubric; every stored verdict agrees")
    return cells


def judge_ran(row):
    """True iff eval.score_example would have called the judge for this row."""
    return (row.get("gold_label") == "answerable"
            and row.get("pred_class") == "answer"
            and row.get("failure") in (None, "entailment_fail"))


def check_grid(gate, cells):
    """3 models spanning >= 2 frontier families + the local base; 3 fixed strategies."""
    models = sorted({m for m, _ in cells})
    strategies = sorted({s for _, s in cells})

    gate.require(len(models) == 3,
                 f"expected 3 models (2 frontier families + local base), "
                 f"got {len(models)}: {models}")
    gate.require(set(strategies) == EXPECTED_STRATEGIES,
                 f"strategies must be exactly {sorted(EXPECTED_STRATEGIES)}, "
                 f"got {strategies}")
    for m in models:
        for s in sorted(EXPECTED_STRATEGIES):
            if (m, s) not in cells:
                gate.fail(f"grid incomplete: no cell for {m} x {s}")

    mock = [m for m in models if provider_of(m) == "mock"]
    if mock:
        gate.fail(f"mock model(s) in the ablation ({mock}) — a mock sweep is not "
                  f"evidence of a prompting ceiling")

    frontier = [m for m in models if provider_of(m) in FRONTIER_PROVIDERS]
    families = {provider_of(m) for m in frontier}
    local = [m for m in models if provider_of(m) in LOCAL_PROVIDERS]
    gate.require(len(families) >= 2,
                 f"the spec requires >= 2 frontier models from DIFFERENT families; "
                 f"found families {sorted(families) or '[]'} from {frontier}")
    gate.require(len(local) >= 1,
                 f"no local base model (hf:...) row — BUILD_PLAN P2 requires the "
                 f"prompted base model as the third row; models were {models}")

    judges = {c.get("judge_model") for c in cells.values()}
    gate.require(len(judges) <= 1,
                 f"cells were scored by different judges {sorted(j for j in judges if j)} "
                 f"— one rubric means one judge")
    for j in judges:
        if j and any(j in m for m in models):
            gate.fail(f"judge_model {j!r} also appears as an ablation model — "
                      f"no model may score itself")

    if not gate.failures:
        gate.note(f"grid complete: {len(models)} models x {len(strategies)} "
                  f"strategies, families {sorted(families)} + local {local}")
    return frontier


def check_summary(gate, out_dir, cells):
    path = out_dir / "ablation_summary.json"
    if not gate.require(path.exists(), "ablation_summary.json missing"):
        return
    try:
        summary = read_json(path)
    except (json.JSONDecodeError, OSError) as exc:
        gate.fail(f"ablation_summary.json unreadable ({exc})")
        return
    if summary.get("rubric_hash") != PINNED_RUBRIC_HASH:
        gate.fail(f"ablation_summary.json: rubric_hash "
                  f"{summary.get('rubric_hash')!r} != {PINNED_RUBRIC_HASH!r}")
    if summary.get("n_scenarios") != EXPECTED_N:
        gate.fail(f"ablation_summary.json: n_scenarios "
                  f"{summary.get('n_scenarios')!r} != {EXPECTED_N}")
    rows = summary.get("cells")
    if not isinstance(rows, list):
        gate.fail("ablation_summary.json: 'cells' is not a list")
        return
    seen = {}
    for row in rows:
        key = (row.get("model"), row.get("strategy"))
        seen[key] = row
        if key not in cells:
            gate.fail(f"ablation_summary.json lists a cell with no results JSON: "
                      f"{key}")
            continue
        sa_summary = (row.get("metrics") or {}).get("spec_adherence")
        sa_result = (cells[key].get("metrics") or {}).get("spec_adherence")
        if not num_eq(sa_summary, sa_result, tol=1e-9):
            gate.fail(f"ablation_summary.json: {key} spec_adherence "
                      f"{sa_summary!r} != results JSON {sa_result!r}")
    for key in cells:
        if key not in seen:
            gate.fail(f"ablation_summary.json is missing cell {key} — the summary "
                      f"is stale relative to the committed results")


def check_table(gate, out_dir, scenarios_path, cells):
    path = out_dir / "ablation_table.md"
    if not gate.require(path.exists(), "ablation_table.md missing"):
        return
    text = path.read_text()

    if "_TODO" in text:
        gate.fail("ablation_table.md still contains the literal placeholder "
                  "'_TODO' — the failure-mode paragraph was never written")

    for model, _ in cells:
        if model not in text:
            gate.fail(f"ablation_table.md does not mention model {model!r}")
    for strat in sorted(EXPECTED_STRATEGIES):
        if strat not in text:
            gate.fail(f"ablation_table.md does not mention strategy {strat!r}")

    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("#") and "failure mode" in ln.lower():
            start = i
            break
    if start is None:
        gate.fail("ablation_table.md has no '## Failure mode ...' section")
        return
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].lstrip().startswith("#"):
            end = j
            break
    section = "\n".join(lines[start + 1:end])
    prose = " ".join(section.split())
    if len(prose) < 120:
        gate.fail(f"the failure-mode section is {len(prose)} characters of prose; "
                  f"the spec asks for a written paragraph naming the specific "
                  f"failure mode that survives the best prompting attempt")

    if not scenarios_path.exists():
        gate.fail(f"eval set {scenarios_path} not found — cannot verify the "
                  f"scenario IDs cited in the failure-mode paragraph")
        return
    known = {s["id"] for s in E.load_jsonl(str(scenarios_path))}
    cited = sorted({t for t in SCENARIO_ID_RE.findall(section) if t in known})
    if len(cited) < 3:
        gate.fail(f"the failure-mode paragraph cites {len(cited)} real scenario "
                  f"ID(s) {cited}; the spec requires >= 3 IDs that exist in "
                  f"{scenarios_path.name}")
    else:
        gate.note(f"failure-mode paragraph written, cites {len(cited)} real "
                  f"scenario IDs: {cited[:6]}")


def check_gate_report(gate, out_dir, cells, frontier_models):
    path = out_dir / "gate_report.json"
    if not gate.require(path.exists(), "gate_report.json missing"):
        return
    try:
        report = read_json(path)
    except (json.JSONDecodeError, OSError) as exc:
        gate.fail(f"gate_report.json unreadable ({exc})")
        return

    frontier_cells = {k: v for k, v in cells.items() if k[0] in frontier_models}
    if len(frontier_cells) != 6:
        gate.fail(f"expected 6 frontier cells (2 families x 3 strategies) to "
                  f"compute gate_fail_rate from, found {len(frontier_cells)}")
        return
    sas = [(k, (v.get("metrics") or {}).get("spec_adherence"))
           for k, v in frontier_cells.items()]
    if any(sa is None for _, sa in sas):
        gate.fail("a frontier cell has no spec_adherence; cannot recompute "
                  "gate_fail_rate")
        return
    best_key, best = max(sas, key=lambda kv: kv[1])
    expected_rate = 1.0 - float(best)
    expected_escalation = expected_rate < ESCALATION_THRESHOLD

    if "gate_fail_rate" not in report:
        gate.fail("gate_report.json has no 'gate_fail_rate'")
    elif not num_eq(report.get("gate_fail_rate"), expected_rate):
        gate.fail(f"gate_report.json gate_fail_rate {report.get('gate_fail_rate')!r} "
                  f"disagrees with the committed results: 1 - max(spec_adherence "
                  f"over the 6 frontier cells) = 1 - {best!r} = {expected_rate!r} "
                  f"(best frontier cell: {best_key[0]} x {best_key[1]})")

    if "escalation_needed" not in report:
        gate.fail("gate_report.json has no 'escalation_needed'")
    else:
        flag = report["escalation_needed"]
        if not isinstance(flag, bool):
            gate.fail(f"gate_report.json escalation_needed must be a boolean, got "
                      f"{flag!r}")
        elif flag != expected_escalation:
            gate.fail(f"gate_report.json escalation_needed == {flag!r} but "
                      f"gate_fail_rate {expected_rate:.4f} "
                      f"{'<' if expected_escalation else '>='} "
                      f"{ESCALATION_THRESHOLD} means it must be "
                      f"{expected_escalation!r}")

    for key in ("max_spec_adherence", "best_spec_adherence"):
        if key in report and not num_eq(report[key], best):
            gate.fail(f"gate_report.json {key} {report[key]!r} != the actual max "
                      f"frontier spec_adherence {best!r}")
    if "rubric_hash" in report and report["rubric_hash"] != PINNED_RUBRIC_HASH:
        gate.fail(f"gate_report.json rubric_hash {report['rubric_hash']!r} != "
                  f"{PINNED_RUBRIC_HASH!r}")

    if not gate.failures:
        gate.note(f"gate_fail_rate {expected_rate:.4f} recomputed from the 6 "
                  f"frontier cells (best: {best_key[0]} x {best_key[1]} at "
                  f"{best:.4f}), escalation_needed={expected_escalation}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Re-check the committed Prompt-Ceiling Ablation (P2 / GATE G1)")
    ap.add_argument("ablation_dir", nargs="?", default="runs/ablation")
    ap.add_argument("--scenarios",
                    default=str(REPO_ROOT / "data" / "eval" / "scenarios_v1.jsonl"),
                    help="eval set the cited scenario IDs must exist in")
    args = ap.parse_args(argv)

    out_dir = Path(args.ablation_dir)
    gate = Gate(f"ablation check: {out_dir}")
    print(f"checking {out_dir} against rubric {PINNED_RUBRIC_HASH}")

    if E.rubric_hash() != PINNED_RUBRIC_HASH:
        gate.fail(f"eval.rubric_hash() == {E.rubric_hash()!r} but this checker "
                  f"pins {PINNED_RUBRIC_HASH!r} — the rubric drifted; STOP")
        return gate.finish()

    if not out_dir.is_dir():
        gate.fail(f"{out_dir} does not exist or is not a directory")
        return gate.finish()
    if not any(out_dir.iterdir()):
        gate.fail(f"{out_dir} is empty — no ablation artifacts committed")
        return gate.finish()

    scenarios_path = Path(args.scenarios)
    scenarios = None
    if scenarios_path.exists():
        try:
            scenarios = load_scenarios(scenarios_path)
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            gate.fail(f"eval set {scenarios_path} is unreadable ({exc}) — the "
                      f"per-example verdicts cannot be re-derived")
    else:
        gate.fail(f"eval set {scenarios_path} not found — the per-example "
                  f"verdicts cannot be re-derived from raw_output, so the "
                  f"transcripts would only be checked against themselves")

    cells = check_cells(gate, out_dir, scenarios)
    if not cells:
        gate.fail("no usable result cells found")
        return gate.finish()
    frontier_models = check_grid(gate, cells)
    check_summary(gate, out_dir, cells)
    check_table(gate, out_dir, scenarios_path, cells)
    check_gate_report(gate, out_dir, cells, frontier_models)
    return gate.finish()


if __name__ == "__main__":
    sys.exit(main())
