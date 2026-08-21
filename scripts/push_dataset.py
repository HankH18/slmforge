#!/usr/bin/env python3
"""Publish the training dataset to a public HF dataset repo, and pin it.

The project spec calls the dataset "your real artifact", so it gets the same
treatment as the model: a public repo, an exact commit hash recorded in
`configs/models.json`, and a read-only re-check that the pin still resolves.

Publishing is irreversible in the way that matters -- once rows are public they
are mirrored, cached and indexed -- so this scans every row before uploading and
refuses on a hit. The scan is not a formality: these rows carry raw provider
error text. The v2 run's `rejected.jsonl` holds 64 rows whose `detail` is an
OpenAI 429 body, which is the author's billing state and says nothing about the
data, so `--redact-errors` replaces it with the exception class alone.

    python scripts/push_dataset.py --dataset data/train/v2.jsonl \\
        --repo qwen3-forbidden-constraint-holder-data --redact-errors

    python scripts/push_dataset.py --check configs/models.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

PIN_RE = re.compile(r"^(?P<repo>[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*)"
                    r"@(?P<sha>[0-9a-f]{40})$")

#: Anything matching here must never reach a public repo. Deliberately broader
#: than "API keys": an absolute home path names a person, and a provider error
#: body is about the author's account, not about the dataset.
FORBIDDEN = {
    "openai key": re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    "anthropic key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "google key": re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    "hugging face token": re.compile(r"hf_[A-Za-z0-9]{30,}"),
    "aws key id": re.compile(r"AKIA[0-9A-Z]{16}"),
    "bearer token": re.compile(r"[Bb]earer\s+[A-Za-z0-9._\-]{20,}"),
    "email address": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "absolute home path": re.compile(r"/(?:Users|home)/[A-Za-z0-9._\-]+"),
    "private ip": re.compile(r"\b(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d+\.\d+\b"),
}

#: Sidecars published alongside the dataset, by suffix on the dataset stem.
SIDECARS = ("rejected.jsonl", "dpo_negatives.jsonl", "report.json")


class PublishError(Exception):
    pass


def die(msg: str, code: int = 1) -> int:
    print(f"FAIL {msg}", file=sys.stderr)
    return code


def mask(secret: Optional[str]) -> str:
    return "<unset>" if not secret else f"{secret[:4]}…(len {len(secret)})"


def sidecar_paths(dataset: Path) -> Dict[str, Path]:
    stem = dataset.name[:-len(".jsonl")] if dataset.name.endswith(".jsonl") \
        else dataset.stem
    out = {"dataset": dataset}
    for suffix in SIDECARS:
        cand = dataset.with_name(f"{stem}.{suffix}")
        if cand.exists():
            out[suffix] = cand
    return out


def scan_for_secrets(paths: Mapping[str, Path]) -> List[str]:
    """Every forbidden pattern, over every line of every file about to go public."""
    problems: List[str] = []
    for label, path in paths.items():
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for name, pat in FORBIDDEN.items():
                m = pat.search(line)
                if m:
                    problems.append(
                        f"{path.name}:{lineno} contains a {name} "
                        f"({m.group()[:24]!r}...). Nothing was pushed.")
                    break
    return problems


def redact_error_details(path: Path, out: Path) -> int:
    """Replace a raw provider error body with its exception class.

    A rejection row records WHY a sample was abandoned. "RateLimitError" is the
    honest reason and belongs in the published record; the response body behind
    it is the author's account state and does not.
    """
    n = 0
    with open(out, "w") as fh:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            detail = row.get("detail")
            if isinstance(detail, str) and detail:
                cls = detail.split(":", 1)[0].strip()
                if cls != detail:
                    row["detail"] = cls
                    row["_detail_redacted"] = (
                        "provider response body removed before publication; it "
                        "described the author's account, not the data")
                    n += 1
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return n


def dataset_card(repo_id: str, dataset: Path, report: Mapping[str, Any],
                 model_repo: Optional[str]) -> str:
    letters = report.get("banned_letter_distribution", {})
    kept = letters.get("kept", {})
    keep_rate = letters.get("keep_rate", {})
    rows = report.get("kept")

    lines = [
        "---",
        "license: apache-2.0",
        "task_categories:",
        "  - text-generation",
        "language:",
        "  - en",
        "tags:",
        "  - constrained-generation",
        "  - instruction-following",
        "  - synthetic",
        "  - distillation",
        "size_categories:",
        "  - n<1K" if (rows or 0) < 1000 else "  - 1K<n<10K",
        "---",
        "",
        "# Forbidden-Constraint Holder — training data",
        "",
        "Teacher-distilled supervised fine-tuning data for one narrow behaviour:",
        "**complete an ordinary writing task while violating zero hard**",
        "**constraints**, at least one of them negative, when the task's own",
        "subject matter pulls toward the banned token.",
        "",
        "Every constraint here has a deterministic checker. That is what makes",
        "the set unusual: **every kept row was verified programmatically before",
        "it was written**, against the exact stored text, with the same checker",
        "the evaluation harness uses. A row that violated its own constraint set",
        "was rejected, not trained on.",
        "",
        "## Files",
        "",
        "| File | What |",
        "|---|---|",
        f"| `{dataset.name}` | the training rows ({rows}) |",
        "| `*.rejected.jsonl` | every rejected teacher attempt, with the checker "
        "failures that rejected it |",
        "| `*.dpo_negatives.jsonl` | (rejected, chosen) pairs for preference "
        "tuning, kept only where the rejected draft was fluent |",
        "| `*.report.json` | the generation ledger: accept rate, attempt "
        "histogram, per-type and per-letter distributions |",
        "| `constraint_instances.jsonl` | the constraint library the `constraints` "
        "ids resolve against |",
        "",
        "The rejected rows are published deliberately. The accept rate is not a",
        "footnote here — which constraints the teacher *could not satisfy* is the",
        "finding this dataset was rebuilt around (below).",
        "",
        "## Row schema",
        "",
        "```json",
        "{",
        '  "messages": [{"role": "user", "content": "..."},',
        '               {"role": "assistant", "content": "..."}],',
        '  "constraints": ["c-lex-001", "c-chr-001"],',
        '  "scenario": {"scenario_id": "...", "task": "...", "constraints": [...],',
        '               "k": 3, "pressure": "adversarial", "split": "train"},',
        '  "provenance": {"teacher": "...", "attempt": 2, "task_id": "...",',
        '                 "seed": 20260821, "verified_by": "constraints.checkers.check"}',
        "}",
        "```",
        "",
        "There is **no system message**, by design: the training condition is",
        "bound to the evaluation condition by a shared prompt renderer, and every",
        "row's user message was re-rendered through it and compared byte-for-byte",
        "before training.",
        "",
        "`provenance.attempt` is how many teacher tries that row took. It is a",
        "difficulty signal: rows at attempt 1 were easy for the teacher, rows at",
        "attempt 8 were nearly impossible.",
        "",
        "## How it was generated",
        "",
        "| | |",
        "|---|---|",
        f"| Teacher | `{report.get('teacher')}` |",
        f"| Requested | {report.get('requested')} |",
        f"| Kept | {report.get('kept')} |",
        f"| Accept rate | {report.get('accept_rate', 0):.3f} |",
        f"| Rejected attempts | {report.get('rejections')} |",
        f"| Attempts allowed per row | {report.get('attempts')} |",
        f"| Seed | {report.get('seed')} |",
        f"| Banned-letter request mix | `{report.get('letter_mix')}` |",
        "",
        "## The finding this data encodes",
        "",
        "The first version of this dataset was rejection-filtered for quality and",
        "nothing else. That silently **selected against difficulty**: every one of",
        "the 918 rejected rows was a banned-letter row, and the per-letter reject",
        "rates were not uniform — banning `e` or `a` failed ~84% of the time while",
        "banning `m` or `p` failed ~27%. The quality gate therefore built a",
        "curriculum out of whichever letters the teacher already found easy, while",
        "the evaluation set uses the hard ones.",
        "",
        "This version corrects it at **request** time rather than at filter time:",
        "each letter is requested with probability proportional to its English",
        "frequency divided by its measured keep rate, so what *survives* the filter",
        "lands on English frequency instead of on teacher convenience.",
        "",
        "Realised kept distribution:",
        "",
        "| letter | kept rows | keep rate |",
        "|---|---|---|",
    ]
    for letter in sorted(kept, key=lambda k: -kept[k]):
        rate = keep_rate.get(letter)
        lines.append(f"| `{letter}` | {kept[letter]} | "
                     f"{'—' if rate is None else f'{rate:.3f}'} |")

    lines += [
        "",
        "## Held-out constraint types",
        "",
        "`banned_punctuation` and `sentence_initial_letter` appear in **zero**",
        "rows. They are the generalization test: the structural family is entirely",
        "absent from training, so evaluating on it measures transfer to a",
        "constraint type the model never saw.",
        "",
        "Be aware of what that measurement found: the model fine-tuned on this",
        "data got **worse** on both held-out families than the base model. Buying",
        "compliance on the trained families cost generalization on the untrained",
        "ones. That result is in the project's write-up, not hidden in it.",
        "",
        "## Contamination",
        "",
        "No (task, constraint-set) pair in these rows appears in the evaluation",
        "corpus; the generator refuses to write a row whose pair collides, and",
        "checks it per row before writing.",
        "",
        "## Limitations",
        "",
        "* Synthetic throughout — one teacher model, one task pool, English only.",
        "* Short-form writing tasks; no code, no dialogue, no long context.",
        "* The rejected file is skewed toward hard letters by construction, so it",
        "  is not a random sample of teacher behaviour.",
    ]
    if model_repo:
        lines += ["", "## Model trained on this data", "",
                  f"[`{model_repo}`](https://huggingface.co/{model_repo})"]
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------- check
def check_pin(models_json: Path, use_hub: bool = True) -> int:
    if not models_json.exists():
        return die(f"{models_json} does not exist")
    cfg = json.loads(models_json.read_text())
    pin = cfg.get("dataset")
    if pin is None:
        return die(f"{models_json} has no 'dataset' key — no dataset published yet")
    if not isinstance(pin, str) or not PIN_RE.match(pin):
        return die(f"{models_json}['dataset'] == {pin!r}; expected "
                   f"'<owner>/<name>@<40-hex-sha>'")
    repo_id, _, sha = pin.partition("@")
    print(f"  ok   dataset pin well formed: {repo_id} @ {sha}")
    if not use_hub:
        print("\nPASS structural check only (--no-hub): repo existence and "
              "visibility were NOT verified")
        return 0
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        return die(f"huggingface_hub is required for the Hub half ({exc})")
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        info = api.dataset_info(repo_id)
    except Exception as exc:  # noqa: BLE001
        return die(f"{repo_id} could not be read from the Hub: {exc}")
    if getattr(info, "private", False) or getattr(info, "gated", False):
        return die(f"{repo_id} is private or gated — the spec requires a PUBLIC "
                   f"dataset")
    print(f"  ok   {repo_id} exists and is public (not gated)")
    shas = {c.commit_id for c in api.list_repo_commits(repo_id, repo_type="dataset")}
    if sha not in shas:
        return die(f"{sha} is not a commit of {repo_id}")
    print(f"  ok   commit {sha} is a real commit of {repo_id}")
    print(f"\nPASS dataset published, public and pinned: {pin}")
    return 0


# ---------------------------------------------------------------------- push
def push(args) -> int:
    dataset = Path(args.dataset).resolve()
    if not dataset.is_file():
        return die(f"--dataset {dataset} is not a file")
    paths = sidecar_paths(dataset)
    report_path = paths.get("report.json")
    if report_path is None:
        return die(f"no report JSON beside {dataset.name}. The generation ledger "
                   f"is what makes the accept rate checkable; refusing to publish "
                   f"rows without it.")

    instances = REPO_ROOT / "data" / "scenarios" / "constraint_instances.jsonl"
    if not instances.is_file():
        return die(f"{instances} is missing — without it the published rows' "
                   f"constraint ids resolve to nothing")

    print(f"  scanning {len(paths) + 1} file(s) before publishing")
    problems = scan_for_secrets({**paths, "constraint_instances.jsonl": instances})
    if problems and not args.redact_errors:
        for p in problems[:10]:
            print(f"FAIL {p}", file=sys.stderr)
        if len(problems) > 10:
            print(f"FAIL ... and {len(problems) - 10} more", file=sys.stderr)
        return 1

    staging = Path(tempfile.mkdtemp(prefix="slmforge-dataset-"))
    try:
        for label, path in paths.items():
            shutil.copy2(path, staging / path.name)
        shutil.copy2(instances, staging / instances.name)

        redacted = 0
        if args.redact_errors and "rejected.jsonl" in paths:
            redacted = redact_error_details(paths["rejected.jsonl"],
                                            staging / paths["rejected.jsonl"].name)
            print(f"  ok   redacted {redacted} raw provider error bodies")

        residual = scan_for_secrets(
            {p.name: p for p in staging.iterdir() if p.is_file()})
        if residual:
            for p in residual[:10]:
                print(f"FAIL {p}", file=sys.stderr)
            return die("the staged files still match a forbidden pattern after "
                       "redaction; nothing was pushed")
        print("  ok   no secret, credential, email, home path or private address "
              "in any staged file")

        if not os.environ.get("HF_TOKEN") or not os.environ.get("HF_USERNAME"):
            return die("HF_TOKEN / HF_USERNAME are not set; nothing was pushed")
        token, username = os.environ["HF_TOKEN"], os.environ["HF_USERNAME"]
        print(f"  HF_TOKEN {mask(token)}, HF_USERNAME {username}")

        try:
            from huggingface_hub import HfApi
            from huggingface_hub.errors import HfHubHTTPError
        except ImportError as exc:
            return die(f"huggingface_hub is required to publish ({exc})")

        import push_model  # reuse the write-scope decision, do not re-implement it
        api = HfApi(token=token)
        try:
            who = api.whoami()
        except Exception as exc:  # noqa: BLE001
            return die(f"HF_TOKEN was rejected by the Hub: {exc}")
        ok, why = push_model._token_can_write(who, username)
        if not ok:
            return die(f"HF_TOKEN lacks write scope ({why}); nothing was pushed")
        print(f"  ok   HF_TOKEN can write ({why})")

        repo_id = args.repo if "/" in args.repo else f"{username}/{args.repo}"
        report = json.loads(report_path.read_text())
        (staging / "README.md").write_text(
            dataset_card(repo_id, dataset, report, args.model_repo))

        try:
            api.create_repo(repo_id, repo_type="dataset", private=False,
                            exist_ok=True)
            api.upload_folder(repo_id=repo_id, repo_type="dataset",
                              folder_path=str(staging),
                              commit_message=args.note or
                              f"{dataset.name}: {report.get('kept')} verified rows")
            info = api.dataset_info(repo_id)
        except HfHubHTTPError as exc:
            return die(f"push to {repo_id} failed: {exc}")

        sha = getattr(info, "sha", None)
        if not sha or not re.fullmatch(r"[0-9a-f]{40}", sha):
            return die(f"could not determine the resulting commit for {repo_id} "
                       f"(got {sha!r})")
        if getattr(info, "private", False) or getattr(info, "gated", False):
            return die(f"{repo_id} came back private/gated")
        print(f"  ok   pushed {repo_id} @ {sha} (public)")

        models_json = Path(args.models_json)
        cfg = json.loads(models_json.read_text())
        before = {k: v for k, v in cfg.items() if k != "dataset"}
        cfg["dataset"] = f"{repo_id}@{sha}"
        models_json.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
        after = json.loads(models_json.read_text())
        lost = sorted(k for k, v in before.items() if after.get(k) != v)
        if lost:
            return die(f"{models_json} lost or changed key(s) {lost}")
        print(f"  ok   {models_json} updated: dataset = {cfg['dataset']}")
        print(f"\nPASS published https://huggingface.co/datasets/{repo_id} @ {sha}")
        return 0
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", metavar="MODELS_JSON",
                    help="read-only validation of the 'dataset' pin")
    ap.add_argument("--no-hub", action="store_true")
    ap.add_argument("--dataset", help="training JSONL to publish")
    ap.add_argument("--repo", help="dataset repo name (or owner/name)")
    ap.add_argument("--model-repo", default=None,
                    help="model repo trained on this data, linked from the card")
    ap.add_argument("--note", default=None, help="commit message")
    ap.add_argument("--redact-errors", action="store_true",
                    help="replace raw provider error bodies in the rejected rows "
                         "with the exception class before publishing")
    ap.add_argument("--models-json", default=str(REPO_ROOT / "configs" / "models.json"))
    args = ap.parse_args(argv)

    if args.check:
        return check_pin(Path(args.check), use_hub=not args.no_hub)
    if not args.dataset or not args.repo:
        ap.error("--dataset and --repo are required to publish")
    return push(args)


if __name__ == "__main__":
    raise SystemExit(main())
