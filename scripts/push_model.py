#!/usr/bin/env python3
"""
P5 publish + verify — put the tuned model on the Hub and pin it, then re-check.

Live push (writes to the Hub, writes configs/models.json):

    python scripts/push_model.py --adapter runs/train_v1/adapter \\
                                 --merged  runs/train_v1/merged \\
                                 --repo    grounded-abstainer-qwen3-1.7b

  Pushes the merged fp16 model to a PUBLIC repo under $HF_USERNAME, pushes the
  LoRA adapter under adapter/, reads back the exact resulting commit hash, and
  appends "tuned": "<owner>/<name>@<40-hex-sha>" to configs/models.json without
  clobbering any existing key. If HF_TOKEN is missing or lacks write scope it
  stops before touching the Hub and says so, naming the key.

Offline / read-only re-check (the P5 verify command):

    python scripts/push_model.py --check configs/models.json

  Asserts the "tuned" entry is well formed ('<owner>/<name>@<40-hex-sha>') and —
  using huggingface_hub read-only metadata only — that the repo exists, is PUBLIC
  (not private, not gated), and that the pinned commit really is a commit of that
  repo. `--no-hub` restricts it to the structural half (used by the test suite,
  which is network-free).
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TUNED_RE = re.compile(r"^(?P<repo>[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*)"
                      r"@(?P<sha>[0-9a-f]{40})$")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt")


def die(msg, code=1):
    print(f"FAIL {msg}", file=sys.stderr)
    return code


def mask(secret):
    if not secret:
        return "<unset>"
    return f"{secret[:4]}…(len {len(secret)})"


def load_env_file(path):
    """Best-effort .env load; never prints or returns values, just populates env."""
    p = Path(path)
    if not p.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(str(p))


# --------------------------------------------------------------- --check
def check_models_json(path, use_hub=True):
    path = Path(path)
    if not path.exists():
        return die(f"{path} does not exist")
    try:
        with open(path) as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return die(f"{path} is not readable JSON: {exc}")
    if not isinstance(cfg, dict):
        return die(f"{path} is not a JSON object")

    tuned = cfg.get("tuned")
    if tuned is None:
        return die(f"{path} has no 'tuned' key — P5 has not published a model yet")
    if not isinstance(tuned, str):
        return die(f"{path}['tuned'] is {type(tuned).__name__}, expected a string")
    m = TUNED_RE.match(tuned)
    if not m:
        return die(f"{path}['tuned'] == {tuned!r} is not well formed; expected "
                   f"'<owner>/<name>@<40-hex-sha>' (lowercase hex, full 40 chars, "
                   f"no 'hf:' prefix)")
    repo_id, sha = m.group("repo"), m.group("sha")
    print(f"  ok   tuned entry well formed: {repo_id} @ {sha}")

    for key in ("base", "frontier", "judge_model"):
        if key not in cfg:
            print(f"  note {path.name} has no {key!r} key (written in P1)")

    if not use_hub:
        print("\nPASS structural check only (--no-hub): repo existence, public "
              "visibility and commit reality were NOT verified")
        return 0

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError
    except ImportError as exc:
        return die(f"huggingface_hub is required for the Hub half of --check "
                   f"({exc}); re-run with --no-hub for the structural check only")

    api = HfApi()
    try:
        info = api.model_info(repo_id)                       # read-only metadata
    except RepositoryNotFoundError:
        return die(f"https://huggingface.co/{repo_id} does not exist or is not "
                   f"publicly visible — the spec requires a PUBLIC checkpoint a "
                   f"grader can pull")
    except HfHubHTTPError as exc:
        return die(f"could not read metadata for {repo_id}: {exc}")
    if getattr(info, "private", False):
        return die(f"{repo_id} is PRIVATE — graders cannot pull it")
    if getattr(info, "gated", False):
        return die(f"{repo_id} is GATED ({info.gated!r}) — graders cannot pull it "
                   f"without requesting access")
    print(f"  ok   {repo_id} exists and is public (not gated)")

    try:
        at_rev = api.model_info(repo_id, revision=sha)
    except (RevisionNotFoundError, HfHubHTTPError) as exc:
        return die(f"commit {sha} is not a commit of {repo_id}: {exc}")
    resolved = getattr(at_rev, "sha", None)
    if resolved and resolved != sha:
        return die(f"{repo_id}@{sha} resolves to {resolved} — the pin is not an "
                   f"immutable commit hash")
    print(f"  ok   commit {sha} is a real commit of {repo_id}")
    print(f"\nPASS tuned model pinned, public, and resolvable: {tuned}")
    return 0


# ---------------------------------------------------------------- push
def _validate_artifact_dirs(adapter, merged):
    problems = []
    for label, d in (("--adapter", adapter), ("--merged", merged)):
        if not d.is_dir():
            problems.append(f"{label} {d} is not a directory")
            continue
        if not any(d.iterdir()):
            problems.append(f"{label} {d} is empty")
    if problems:
        return problems
    if not (adapter / "adapter_config.json").exists():
        problems.append(f"--adapter {adapter} has no adapter_config.json — that is "
                        f"not a PEFT/LoRA adapter directory")
    if not (merged / "config.json").exists():
        problems.append(f"--merged {merged} has no config.json — that is not a "
                        f"merged transformers model directory")
    if not any(p.suffix in WEIGHT_SUFFIXES for p in merged.iterdir() if p.is_file()):
        problems.append(f"--merged {merged} holds no weight file "
                        f"({', '.join(WEIGHT_SUFFIXES)})")
    return problems


def _token_can_write(who, username):
    """Read huggingface whoami() and decide whether the token may push under
    `username`. Returns (ok, explanation)."""
    auth = (who or {}).get("auth") or {}
    tok = auth.get("accessToken") or {}
    role = tok.get("role")
    if role in ("write", "admin"):
        return True, f"token role {role!r}"
    if role == "fineGrained":
        for scope in (tok.get("fineGrained") or {}).get("scoped") or []:
            ent = scope.get("entity") or {}
            perms = scope.get("permissions") or []
            if "repo.write" in perms and (
                    ent.get("name") == username or ent.get("type") == "org"):
                return True, f"fine-grained token scoped repo.write on {ent.get('name')}"
        return False, ("fine-grained token has no repo.write scope for "
                       f"{username!r}")
    return False, f"token role is {role!r} (read-only)"


def update_models_json(models_json, repo_id, commit):
    """Append "tuned": "<repo>@<commit>" WITHOUT clobbering any existing key.

    Returns None on success, or a message describing why it refused / failed.
    The write is read back and compared key by key, so a silently mangled config
    is caught here rather than at P6.
    """
    models_json = Path(models_json)
    cfg = {}
    if models_json.exists():
        try:
            with open(models_json) as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            return f"{models_json} is not readable JSON ({exc}); refusing to overwrite"
        if not isinstance(cfg, dict):
            return f"{models_json} is not a JSON object; refusing to overwrite"
    else:
        models_json.parent.mkdir(parents=True, exist_ok=True)

    before = {k: v for k, v in cfg.items() if k != "tuned"}
    pin = f"{repo_id}@{commit}"
    cfg["tuned"] = pin
    with open(models_json, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

    with open(models_json) as f:
        after = json.load(f)
    lost = sorted(k for k, v in before.items() if after.get(k) != v)
    if lost:
        return (f"{models_json} lost or changed pre-existing key(s) {lost} while "
                f"appending 'tuned'")
    if after.get("tuned") != pin:
        return f"{models_json} did not persist the tuned pin {pin!r}"
    return None


def push(args):
    adapter, merged = Path(args.adapter).resolve(), Path(args.merged).resolve()
    problems = _validate_artifact_dirs(adapter, merged)
    if problems:
        for p in problems:
            print(f"FAIL {p}", file=sys.stderr)
        return 1

    if not os.environ.get("HF_TOKEN") or not os.environ.get("HF_USERNAME"):
        load_env_file(args.env_file)
    token = os.environ.get("HF_TOKEN")
    username = os.environ.get("HF_USERNAME")
    if not token:
        return die("HF_TOKEN is not set (checked the environment and "
                   f"{args.env_file}). Publishing needs a token with WRITE scope; "
                   "set HF_TOKEN and re-run. Nothing was pushed.")
    if not username:
        return die("HF_USERNAME is not set (checked the environment and "
                   f"{args.env_file}); it names the namespace to publish under. "
                   "Nothing was pushed.")
    print(f"  HF_TOKEN {mask(token)}, HF_USERNAME {username}")

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError as exc:
        return die(f"huggingface_hub is required to publish ({exc})")

    api = HfApi(token=token)
    try:
        who = api.whoami()
    except Exception as exc:  # noqa: BLE001 - any auth failure is fatal here
        return die(f"HF_TOKEN was rejected by the Hub: {exc}")
    ok, why = _token_can_write(who, username)
    if not ok:
        return die(f"HF_TOKEN lacks write scope ({why}). Publishing the tuned "
                   f"model needs a token with write access to the {username!r} "
                   f"namespace. Nothing was pushed — fix HF_TOKEN and re-run.")
    if who.get("name") and who["name"] != username:
        print(f"  note HF_TOKEN belongs to {who['name']!r} but HF_USERNAME is "
              f"{username!r}; pushing under {username!r}")
    print(f"  ok   HF_TOKEN can write ({why})")

    repo_id = args.repo if "/" in args.repo else f"{username}/{args.repo}"
    # The note lands in the PUBLIC commit log, which is where someone pulling a
    # pinned revision looks to find out which checkpoint they have. "tuned" alone
    # stops meaning anything the moment there are two of them.
    note = f" — {args.note}" if args.note else ""
    card = Path(args.card).resolve() if args.card else None
    if card is not None and not card.is_file():
        return die(f"--card {card} is not a file; nothing was pushed")
    try:
        api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
        merged_commit = api.upload_folder(
            repo_id=repo_id, folder_path=str(merged),
            commit_message=f"merged fp16 model (forbidden-constraint-holder{note})")
        adapter_commit = api.upload_folder(
            repo_id=repo_id, folder_path=str(adapter), path_in_repo="adapter",
            commit_message=f"LoRA adapter (forbidden-constraint-holder{note})")
        if card is not None:
            adapter_commit = api.upload_file(
                repo_id=repo_id, path_or_fileobj=str(card), path_in_repo="README.md",
                commit_message=f"model card (forbidden-constraint-holder{note})")
            print(f"  ok   model card uploaded from {card}")
    except HfHubHTTPError as exc:
        return die(f"push to {repo_id} failed: {exc}")

    commit = getattr(adapter_commit, "oid", None) or getattr(merged_commit, "oid", None)
    try:
        info = api.model_info(repo_id)
    except HfHubHTTPError as exc:
        return die(f"pushed, but could not read {repo_id} back: {exc}")
    commit = getattr(info, "sha", None) or commit
    if not commit or not re.fullmatch(r"[0-9a-f]{40}", commit):
        return die(f"could not determine the resulting commit hash for {repo_id} "
                   f"(got {commit!r})")
    if getattr(info, "private", False) or getattr(info, "gated", False):
        return die(f"{repo_id} came back private/gated — the spec requires a "
                   f"PUBLIC checkpoint")
    print(f"  ok   pushed {repo_id} @ {commit} (public)")

    models_json = Path(args.models_json)
    err = update_models_json(models_json, repo_id, commit)
    if err:
        return die(err)
    print(f"  ok   {models_json} updated: tuned = {repo_id}@{commit}")
    print(f"\nPASS published https://huggingface.co/{repo_id} @ {commit}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Publish the tuned model to a public HF repo (P5), or "
                    "re-check the pin in configs/models.json")
    ap.add_argument("--check", metavar="MODELS_JSON",
                    help="offline/read-only validation of the 'tuned' pin")
    ap.add_argument("--no-hub", action="store_true",
                    help="with --check: skip the read-only Hub metadata calls")
    ap.add_argument("--adapter", help="LoRA adapter directory to publish")
    ap.add_argument("--merged", help="merged fp16 model directory to publish")
    ap.add_argument("--repo", help="repo name (or owner/name) to publish to")
    ap.add_argument("--note", default=None,
                    help="short description of WHICH checkpoint this is (e.g. "
                         "'v2 dataset, 846 rows'); goes in the public commit log")
    ap.add_argument("--card", default=None,
                    help="model card to upload as README.md; render one with "
                         "scripts/model_card.py so it cannot drift from the run")
    ap.add_argument("--models-json", default=str(REPO_ROOT / "configs" / "models.json"),
                    help="models.json to append the 'tuned' pin to")
    ap.add_argument("--env-file", default=str(REPO_ROOT / ".env"),
                    help="fallback .env for HF_TOKEN / HF_USERNAME")
    args = ap.parse_args(argv)

    if args.check:
        if any((args.adapter, args.merged, args.repo)):
            return die("--check is a read-only mode; do not combine it with "
                       "--adapter/--merged/--repo")
        return check_models_json(args.check, use_hub=not args.no_hub)

    missing = [f for f, v in (("--adapter", args.adapter), ("--merged", args.merged),
                              ("--repo", args.repo)) if not v]
    if missing:
        return die(f"live push needs {', '.join(missing)} "
                   f"(or use --check <models.json> for the offline re-check)")
    return push(args)


if __name__ == "__main__":
    sys.exit(main())
