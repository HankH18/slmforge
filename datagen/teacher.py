#!/usr/bin/env python3
"""The teacher: which model, how it is called, and why a rerun does not re-bill.

Model choice
------------
The teacher is resolved at RUN TIME, never hardcoded into a call site:

1. ``--teacher`` wins outright.
2. otherwise ``configs/models.json`` may name one in a ``teacher`` key.
3. otherwise :data:`DEFAULT_TEACHER`, a MID-TIER model, provided its provider is
   one the config's ``frontier`` list still uses.
4. otherwise ``frontier[0]``.

Step 3 is the interesting one. The checkers are the quality gate here, not the
teacher: a flagship draft and a mid-tier draft face the identical pass/fail, and
a violating flagship draft is thrown away exactly as fast. Teacher tier buys
prose polish, which the judge scores at eval time, not compliance -- and the
flagship costs roughly 3x. So the default is mid tier ON PURPOSE, and the
frontier list is read anyway: if the config's providers have moved on, the
pinned mid-tier default is no longer a verified model id and the flagship the
config actually names is the safer thing to fall back to. Both the resolved spec
and the reason are recorded in every row's provenance.

Caching
-------
Responses are cached on disk under ``datagen/.cache`` (gitignored), keyed by a
SHA-256 of the exact request: teacher spec, system message and every turn. A
rerun with the same seed replays from disk and bills nothing. The key covers the
whole message list, so a retry -- which appends the violating draft and a repair
instruction -- is a DIFFERENT request and is not served the first answer back.

Retries
-------
A retry that re-sends the identical prompt at temperature 0 gets the identical
draft, so the pipeline does not do that. Attempt 2 and up send the failed draft
back with the checker's own evidence and an instruction to fix exactly that.
That is what makes ``--attempts`` worth spending money on.
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from datagen import DatagenError

__all__ = ["DEFAULT_TEACHER", "CACHE_DIR", "MODELS_JSON", "resolve_teacher",
           "TeacherCache", "repair_turns", "request_hash"]

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_JSON = REPO_ROOT / "configs" / "models.json"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"

#: Mid tier by design -- see the module docstring.
DEFAULT_TEACHER = "openai:gpt-5.6-terra"


def _provider(spec: str) -> str:
    return str(spec).partition(":")[0]


def resolve_teacher(cli_teacher: Optional[str] = None,
                    models_json: Path = MODELS_JSON) -> Tuple[str, str]:
    """-> (model spec, one-line reason). Raises rather than guessing a model id."""
    if cli_teacher:
        return cli_teacher, "--teacher"

    try:
        config = json.loads(Path(models_json).read_text())
    except FileNotFoundError:
        raise DatagenError(
            f"{models_json} does not exist, so the teacher cannot be resolved at run "
            f"time. Pass --teacher explicitly or restore the config.") from None
    except json.JSONDecodeError as exc:
        raise DatagenError(f"{models_json} is not valid JSON: {exc}") from None
    if not isinstance(config, dict):
        raise DatagenError(f"{models_json} must contain a JSON object")

    named = config.get("teacher")
    if isinstance(named, str) and named.strip():
        return named.strip(), f"{models_json.name}:teacher"

    frontier = config.get("frontier")
    if not isinstance(frontier, list) or not frontier or not isinstance(frontier[0], str):
        raise DatagenError(
            f"{models_json} has no usable 'frontier' list, so there is no verified "
            f"model id to fall back to. Pass --teacher explicitly.")

    providers = {_provider(spec) for spec in frontier if isinstance(spec, str)}
    if _provider(DEFAULT_TEACHER) in providers:
        return DEFAULT_TEACHER, (
            f"mid-tier default (provider {_provider(DEFAULT_TEACHER)} verified in "
            f"{models_json.name}:frontier)")
    return frontier[0], (
        f"{models_json.name}:frontier[0] -- the mid-tier default's provider "
        f"({_provider(DEFAULT_TEACHER)}) is no longer in the verified frontier list")


def request_hash(teacher: str, system: Optional[str],
                 turns: Sequence[Mapping[str, str]]) -> str:
    """SHA-256 over the whole request. Two identical requests share a key; a
    retry, which carries extra turns, does not."""
    payload = json.dumps(
        {"teacher": teacher, "system": system,
         "turns": [{"role": t["role"], "content": t["content"]} for t in turns]},
        sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class TeacherCache:
    """Disk cache of teacher responses. Thread-safe; a corrupt entry is a miss.

    A cache entry stores the request alongside the response, so an auditor can
    see what was asked as well as what came back.
    """

    def __init__(self, directory: Path = CACHE_DIR, enabled: bool = True):
        self.dir = Path(directory)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> Optional[str]:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            with self._lock:
                self.misses += 1
            return None
        try:
            entry = json.loads(path.read_text())
            response = entry["response"]
        except (json.JSONDecodeError, KeyError, OSError):
            # A half-written or hand-edited entry is treated as absent rather
            # than as an empty response: an empty response would be filtered as
            # empty_draft and silently blamed on the teacher.
            with self._lock:
                self.misses += 1
            return None
        if not isinstance(response, str):
            with self._lock:
                self.misses += 1
            return None
        with self._lock:
            self.hits += 1
        return response

    def put(self, key: str, teacher: str, system: Optional[str],
            turns: Sequence[Mapping[str, str]], response: str) -> None:
        if not self.enabled:
            return
        entry = {
            "request_hash": key, "teacher": teacher, "system": system,
            "turns": [{"role": t["role"], "content": t["content"]} for t in turns],
            "response": response,
            "cached_utc": datetime.now(timezone.utc).isoformat(),
        }
        tmp = self._path(key).with_suffix(".json.tmp")
        with self._lock:
            tmp.write_text(json.dumps(entry, ensure_ascii=False))
            tmp.replace(self._path(key))


REPAIR_HEADER = (
    "That response violated a requirement. Rewrite it so it completes the same "
    "task and satisfies EVERY requirement. Do not explain the change, do not "
    "apologise, and do not mention the requirements -- return only the rewritten "
    "response."
)


def repair_turns(base_turns: Sequence[Mapping[str, str]], draft: str,
                 failures: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    """The turn list for a retry: the original ask, the failed draft, and the
    checker's own evidence for what broke.

    The evidence comes from the checkers, so the teacher is told exactly what a
    machine will measure next time rather than a paraphrase of it.
    """
    lines = [f"- {f['constraint_id']} ({f['type']}): {f['evidence']}" for f in failures]
    turns = [dict(t) for t in base_turns]
    turns.append({"role": "assistant", "content": draft})
    turns.append({"role": "user",
                  "content": REPAIR_HEADER + "\n\nWhat failed:\n" + "\n".join(lines)})
    return turns


def call_teacher(generator: Callable[..., str], scenario: Mapping[str, Any],
                 prompt: Mapping[str, Any], lookup: Mapping[str, Mapping[str, Any]],
                 teacher: str, cache: TeacherCache) -> str:
    """One teacher call, cached. Returns the raw response text."""
    key = request_hash(teacher, prompt.get("system"), prompt["turns"])
    hit = cache.get(key)
    if hit is not None:
        return hit
    response = generator(scenario, dict(prompt), lookup)
    if not isinstance(response, str):
        raise DatagenError(
            f"the teacher generator returned {type(response).__name__}, not a string")
    cache.put(key, teacher, prompt.get("system"), prompt["turns"], response)
    return response
