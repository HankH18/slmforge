#!/usr/bin/env python3
"""Where the scenario artefacts live, plus a JSONL reader.

Kept in its own module so tests and downstream tickets (T-3's eval smoke run,
T-5's dataset writer) can import the paths without importing the authoring
data or the generator.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

__all__ = ["REPO_ROOT", "DATA_DIR", "INSTANCES_PATH", "CORPUS_PATH", "SMOKE_PATH",
           "MANIFEST_PATH", "load_jsonl", "load_json", "load_instances",
           "load_corpus", "load_smoke", "load_manifest"]

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "scenarios"

INSTANCES_PATH = DATA_DIR / "constraint_instances.jsonl"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
SMOKE_PATH = DATA_DIR / "smoke.jsonl"
MANIFEST_PATH = DATA_DIR / "manifest.json"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file. A missing file raises -- it is never an empty corpus."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from None
    return rows


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_instances() -> Dict[str, Dict[str, Any]]:
    """The shipped instance library as ``id -> constraint``."""
    return {row["id"]: row for row in load_jsonl(INSTANCES_PATH)}


def load_corpus() -> List[Dict[str, Any]]:
    return load_jsonl(CORPUS_PATH)


def load_smoke() -> List[Dict[str, Any]]:
    return load_jsonl(SMOKE_PATH)


def load_manifest() -> Dict[str, Any]:
    return load_json(MANIFEST_PATH)
