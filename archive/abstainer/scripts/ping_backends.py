#!/usr/bin/env python3
"""P1 -- Backends live, IDs pinned.

Two modes:

  live (default)
      2-scenario round-trip through every backend the project depends on --
      frontier family A (openai:), frontier family B (google:), the judge
      (anthropic, exercised AS a judge), and the base model (hf:, local
      transformers) -- all through the frozen harness in eval.py. On success,
      writes configs/models.json per the BUILD_PLAN contract.

  --check <path>
      Offline re-validation of the committed configs/models.json. No API calls,
      no billing. Exits 0 only if every contract invariant holds.

A missing or invalid API key stops with a non-zero exit naming the env var.
Key values are never printed; provider errors are scrubbed of any secret
substring before display.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SCENARIOS_PATH = REPO_ROOT / "data" / "eval" / "scenarios_v1.jsonl"
DEFAULT_MODELS_JSON = REPO_ROOT / "configs" / "models.json"

# ---------------------------------------------------------------------------
# Pinned IDs. Each was verified live against the provider doc page recorded
# beside it; --check re-validates the shape, not the liveness. Override on the
# command line when a provider retires an id.
# ---------------------------------------------------------------------------
PIN_OPENAI = "gpt-5.6-sol"
PIN_GOOGLE = "gemini-3.1-pro-preview"
PIN_JUDGE = "claude-sonnet-4-6"  # frozen AnthropicJudge sends temperature=0;
#   Opus 5 / Sonnet 5 / Opus 4.7+ reject `temperature` with a 400, which the
#   judge would silently convert into a conservative FAIL on every scenario.
PIN_BASE_REPO = "Qwen/Qwen3-1.7B"
PIN_BASE_SHA = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

SOURCES = [
    "https://developers.openai.com/api/docs/models/gpt-5.6-sol",
    "https://ai.google.dev/gemini-api/docs/models",
    "https://platform.claude.com/docs/en/about-claude/models/overview",
    "https://huggingface.co/Qwen/Qwen3-1.7B",
]

# env var -> which backend needs it
REQUIRED_KEYS = {
    "OPENAI_API_KEY": "frontier family A (openai:)",
    "GOOGLE_API_KEY": "frontier family B (google:)",
    "ANTHROPIC_API_KEY": "the LLM judge (anthropic)",
}

KNOWN_PROVIDERS = {"openai", "google", "anthropic", "hf", "mock"}
SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
HF_SPEC_RE = re.compile(r"^hf:([A-Za-z0-9._\-]+/[A-Za-z0-9._\-]+)@([0-9a-f]{40})$")
ANTHROPIC_ID_RE = re.compile(r"^claude-[A-Za-z0-9._\-]+$")
AUTH_HINTS = ("401", "403", "api key", "api_key", "apikey", "unauthorized",
              "authentication", "invalid_api_key", "permission_denied",
              "credential", "not valid", "invalid key")


# ---------------------------------------------------------------------------
# secret hygiene
# ---------------------------------------------------------------------------
def _secret_values() -> list[str]:
    out = []
    for name in list(REQUIRED_KEYS) + ["HF_TOKEN", "GEMINI_API_KEY"]:
        val = os.environ.get(name) or ""
        if len(val) >= 8:
            out.append((name, val))
    return out


def scrub(text: str) -> str:
    """Never let a key value reach stdout, a log, or an artifact."""
    for name, val in _secret_values():
        text = text.replace(val, f"<redacted:{name}>")
    return text


def die(msg: str, code: int = 1) -> None:
    print(f"\nFAIL: {scrub(msg)}", file=sys.stderr)
    sys.exit(code)


# ---------------------------------------------------------------------------
# live mode
# ---------------------------------------------------------------------------
def load_env() -> None:
    env_path = REPO_ROOT / ".env"
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("  note: python-dotenv not installed; relying on the ambient environment")
        return
    if env_path.exists():
        load_dotenv(str(env_path))


def check_keys_present() -> None:
    """Missing/blank key -> stop, naming the variable. Never print a value."""
    missing = []
    for name, who in REQUIRED_KEYS.items():
        val = (os.environ.get(name) or "").strip()
        if not val:
            missing.append(f"{name} (needed by {who})")
        else:
            print(f"  key {name:<20} present (length {len(val)})")
    if missing:
        die("missing or empty API key(s) in the environment / .env:\n    - "
            + "\n    - ".join(missing)
            + "\n  Set them in " + str(REPO_ROOT / ".env")
            + " (see .env.example). No fallback is attempted.")


def classify_failure(backend: str, env_key: str | None, exc: BaseException) -> str:
    detail = scrub(f"{type(exc).__name__}: {exc}")
    if env_key and any(h in detail.lower() for h in AUTH_HINTS):
        return (f"{backend}: the API key in {env_key} was rejected by the provider "
                f"(invalid, revoked, or lacking access to this model).\n  provider said: {detail}")
    return f"{backend}: live round-trip failed.\n  provider said: {detail}"


def probe_generator(label: str, spec: str, scenarios: list[dict],
                    env_key: str | None) -> list[dict]:
    """2-scenario round-trip through eval.make_generator / build_prompt / parse_output."""
    import eval as E

    print(f"\n[{label}] {spec}")
    try:
        gen = E.make_generator(spec)
    except Exception as exc:  # noqa: BLE001
        die(classify_failure(label, env_key, exc))

    rows = []
    for sc in scenarios:
        prompt = E.build_prompt("zero_shot", sc, None)
        t0 = time.time()
        try:
            raw = gen(sc, prompt)
        except Exception as exc:  # noqa: BLE001
            die(classify_failure(f"{label} (scenario {sc['id']})", env_key, exc))
        dt = time.time() - t0
        if not (raw or "").strip():
            die(f"{label}: scenario {sc['id']} returned an empty completion -- "
                "the backend answered but produced no text.")
        parsed = E.parse_output(raw)
        cls = parsed["class"]
        expect = "refusal" if sc["gold_label"] == "unanswerable" else "answer"
        print(f"  {sc['id']:<4} gold={sc['gold_label']:<12} parse={cls:<9} "
              f"{'match' if cls == expect else 'differs'}  {dt:5.2f}s")
        rows.append({"id": sc["id"], "class": cls, "seconds": round(dt, 2)})
    print(f"  -> {label} LIVE")
    return rows


def probe_judge(judge_model: str, scenario: dict) -> list[dict]:
    """Exercise Anthropic AS A JUDGE via eval.make_judge -- not as a generator.

    Two probes: one entailed pair (expect YES) and one blatantly contradicted
    pair (expect NO). AnthropicJudge swallows API errors and returns a
    conservative False, so a bare False is not proof the judge ran -- the
    reason string is checked for the conservative-fail sentinel.
    """
    import eval as E

    print(f"\n[judge] anthropic:{judge_model}  (as judge, via eval.make_judge)")
    try:
        judge = E.make_judge("anthropic", judge_model)
    except Exception as exc:  # noqa: BLE001
        die(classify_failure("judge", "ANTHROPIC_API_KEY", exc))

    question = scenario["question"]
    spans = list(scenario["gold_spans"])
    probes = [
        ("entailed", scenario["gold_answer"], spans, True),
        ("contradicted", "It decreased by 90 percent.", spans, False),
    ]
    rows = []
    for name, answer, sp, expected in probes:
        t0 = time.time()
        verdict, reason = judge(question, answer, sp)
        dt = time.time() - t0
        reason = scrub(reason or "")
        if reason.startswith("judge_error_conservative_fail"):
            die("judge: eval.AnthropicJudge exhausted its retries and fell back to a "
                "conservative FAIL -- the judge is NOT live.\n  reason: " + reason)
        print(f"  {name:<13} verdict={'YES' if verdict else 'NO':<3} "
              f"expected={'YES' if expected else 'NO':<3} "
              f"{'ok' if verdict is expected else 'UNEXPECTED'}  {dt:5.2f}s")
        if verdict is not expected:
            die(f"judge: the {name} probe returned "
                f"{'YES' if verdict else 'NO'} but the pair is unambiguous. "
                "The pinned judge model is not usable for entailment scoring.\n"
                f"  judge said: {reason[:300]}")
        rows.append({"probe": name, "verdict": verdict, "seconds": round(dt, 2)})
    print("  -> judge LIVE")
    return rows


def write_models_json(path: Path, frontier: list[str], judge_model: str,
                      base: str, sources: list[str]) -> dict:
    """Write the contract artifact, preserving keys this phase does not own
    (notably `tuned`, appended in P5) so a P5 rerun is not clobbered."""
    managed = {"frontier", "judge_model", "base", "verified_utc", "sources"}
    preserved: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            die(f"{path} exists but is not valid JSON ({exc}); refusing to overwrite "
                "it blind. Delete or fix it, then rerun.")
        if not isinstance(existing, dict):
            die(f"{path} exists but is not a JSON object; refusing to overwrite it blind.")
        preserved = {k: v for k, v in existing.items() if k not in managed}
        if preserved:
            print(f"\n  preserving existing non-P1 key(s): {', '.join(sorted(preserved))}")

    payload = {
        "frontier": frontier,
        "judge_model": judge_model,
        "base": base,
        "verified_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "sources": sources,
    }
    payload.update(preserved)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def run_live(args: argparse.Namespace) -> int:
    print("=" * 72)
    print("P1 ping_backends -- LIVE round-trip (this bills the frontier APIs)")
    print("=" * 72)

    load_env()
    print("\n[keys]")
    check_keys_present()

    import eval as E
    print(f"\n  rubric_hash = {E.rubric_hash()}")

    if not SCENARIOS_PATH.exists():
        die(f"eval set missing at {SCENARIOS_PATH}")
    scenarios = E.load_jsonl(str(SCENARIOS_PATH))
    answerable = next(s for s in scenarios if s["gold_label"] == "answerable")
    unanswerable = next(s for s in scenarios if s["gold_label"] == "unanswerable")
    pair = [answerable, unanswerable]
    print(f"  probe scenarios: {answerable['id']} (answerable), "
          f"{unanswerable['id']} (unanswerable)")

    openai_spec = f"openai:{args.openai_model}"
    google_spec = f"google:{args.google_model}"
    base_spec = f"hf:{args.base_repo}@{args.base_sha}"

    probe_generator("openai", openai_spec, pair, "OPENAI_API_KEY")
    probe_generator("google", google_spec, pair, "GOOGLE_API_KEY")
    probe_judge(args.judge_model, answerable)
    print("\n  (loading the base model locally -- first run downloads ~3.4 GB)")
    probe_generator("hf-base", base_spec, pair, None)

    out = Path(args.out)
    payload = write_models_json(out, [openai_spec, google_spec],
                                args.judge_model, base_spec, list(SOURCES))
    print(f"\n[write] {out}")
    print(json.dumps(payload, indent=2))

    print("\nALL BACKENDS LIVE. configs/models.json written.")
    print(f"Re-check offline with: python scripts/ping_backends.py --check {out}")
    return 0


# ---------------------------------------------------------------------------
# check mode (offline, no billing)
# ---------------------------------------------------------------------------
class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
        if not ok:
            self.failures.append(label + (f" -- {detail}" if detail else ""))
        return ok


def validate_config(cfg: dict, rep: Report) -> None:
    # ---- frontier ----------------------------------------------------------
    frontier = cfg.get("frontier")
    if not rep.check(isinstance(frontier, list), "frontier is a list",
                     f"got {type(frontier).__name__}"):
        frontier = []
    rep.check(len(frontier) == 2, "frontier has exactly 2 entries", f"got {len(frontier)}")

    families = []
    for spec in frontier:
        if not isinstance(spec, str) or ":" not in spec:
            rep.check(False, f"frontier entry {spec!r} is 'provider:model'")
            continue
        provider, _, model = spec.partition(":")
        ok = rep.check(bool(provider) and bool(model.strip()),
                       f"frontier entry {spec!r} has a non-empty provider and model")
        rep.check(provider in KNOWN_PROVIDERS,
                  f"frontier provider {provider!r} is a provider eval.make_generator knows")
        rep.check(provider != "mock", f"frontier entry {spec!r} is not a mock")
        if ok:
            families.append(provider)
    rep.check(len(set(families)) == 2,
              "the 2 frontier entries come from 2 DIFFERENT provider families",
              f"families={sorted(set(families))}")

    # ---- judge -------------------------------------------------------------
    judge = cfg.get("judge_model")
    if rep.check(isinstance(judge, str) and bool(judge.strip()),
                 "judge_model is a non-empty string"):
        rep.check(bool(ANTHROPIC_ID_RE.match(judge)),
                  "judge_model is an Anthropic model id (claude-*)", repr(judge))
        rep.check(":" not in judge,
                  "judge_model is a bare id, not a provider:model spec", repr(judge))
        rep.check("anthropic" not in families,
                  "the judge family (anthropic) is NOT one of the frontier families "
                  "-- nothing scores itself")
        frontier_models = {s.partition(":")[2] for s in frontier if isinstance(s, str)}
        rep.check(judge not in frontier_models and judge not in set(frontier),
                  "judge_model is not one of the frontier models", repr(judge))

    # ---- base --------------------------------------------------------------
    base = cfg.get("base")
    if rep.check(isinstance(base, str) and bool(base), "base is a non-empty string"):
        m = HF_SPEC_RE.match(base)
        if rep.check(bool(m), "base matches hf:<owner>/<repo>@<40-hex-sha>", repr(base)):
            repo, sha = m.group(1), m.group(2)
            rep.check(bool(SHA40_RE.match(sha)),
                      "base revision is a pinned 40-hex commit sha, not a branch name", sha)
            rep.check(sha not in ("main", "master"), "base revision is not a floating ref")
            print(f"         base repo={repo} revision={sha}")

    # ---- verified_utc ------------------------------------------------------
    stamp = cfg.get("verified_utc")
    if rep.check(isinstance(stamp, str) and bool(stamp), "verified_utc is a non-empty string"):
        try:
            parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            rep.check(True, "verified_utc parses as ISO8601", stamp)
            rep.check(parsed.tzinfo is not None, "verified_utc carries a timezone offset", stamp)
        except ValueError as exc:
            rep.check(False, "verified_utc parses as ISO8601", f"{stamp!r}: {exc}")

    # ---- sources -----------------------------------------------------------
    sources = cfg.get("sources")
    if rep.check(isinstance(sources, list), "sources is a list"):
        rep.check(len(sources) > 0, "sources is non-empty", f"got {len(sources)}")
        rep.check(all(isinstance(s, str) and s.startswith("http") for s in sources),
                  "every source is an http(s) doc URL")

    # ---- tuned (optional; P5) ---------------------------------------------
    if "tuned" in cfg:
        tuned = cfg["tuned"]
        rep.check(isinstance(tuned, str) and "@" in tuned,
                  "optional 'tuned' (P5) is '<repo>@<commit>'", repr(tuned))
    else:
        print("  [ -- ] optional 'tuned' absent (added in P5)")


def run_check(path_str: str) -> int:
    path = Path(path_str)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    print("=" * 72)
    print(f"P1 verify -- offline contract check of {path}")
    print("=" * 72)
    rep = Report()

    if not path.exists():
        rep.check(False, "configs/models.json exists", str(path))
        print("\nRESULT: FAIL (1 problem)")
        return 1
    rep.check(True, "configs/models.json exists", str(path))

    try:
        cfg = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        rep.check(False, "file parses as JSON", str(exc))
        print("\nRESULT: FAIL (1 problem)")
        return 1
    rep.check(True, "file parses as JSON")
    if not rep.check(isinstance(cfg, dict), "top level is a JSON object"):
        print("\nRESULT: FAIL (1 problem)")
        return 1

    for key in ("frontier", "judge_model", "base", "verified_utc", "sources"):
        rep.check(key in cfg, f"contract key {key!r} present")

    validate_config(cfg, rep)

    if rep.failures:
        print(f"\nRESULT: FAIL ({len(rep.failures)} problem(s))")
        for f in rep.failures:
            print(f"  - {f}")
        return 1
    print("\nRESULT: PASS -- configs/models.json satisfies the P1 contract "
          "(no API was called).")
    return 0


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="P1: live round-trip every backend and pin the IDs; "
                    "or re-validate the committed configs/models.json offline.")
    ap.add_argument("--check", metavar="PATH", default=None,
                    help="offline: validate an existing models.json and exit (no API calls)")
    ap.add_argument("--out", default=str(DEFAULT_MODELS_JSON),
                    help="where to write models.json in live mode")
    ap.add_argument("--openai-model", default=PIN_OPENAI)
    ap.add_argument("--google-model", default=PIN_GOOGLE)
    ap.add_argument("--judge-model", default=PIN_JUDGE)
    ap.add_argument("--base-repo", default=PIN_BASE_REPO)
    ap.add_argument("--base-sha", default=PIN_BASE_SHA)
    args = ap.parse_args(argv)

    if args.check:
        return run_check(args.check)
    if not SHA40_RE.match(args.base_sha):
        die(f"--base-sha must be a 40-hex commit sha, got {args.base_sha!r}")
    return run_live(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        print(scrub(traceback.format_exc()), file=sys.stderr)
        sys.exit(1)
