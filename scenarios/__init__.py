"""Scenario generator and corpus (ticket T-2).

Layout mirrors the other top-level packages (``datagen/``, ``training/``):
authoring data and generator code live here, emitted artefacts live under
``data/scenarios/``.

    from scenarios.paths import load_corpus, load_instances

    corpus = load_corpus()                 # data/scenarios/corpus.jsonl
    library = load_instances()             # id -> constraint object

Rebuild the artefacts with ``python -m scenarios.build``; verify the shipped
files still match their source with ``python -m scenarios.build --check``.
"""
from scenarios.paths import (CORPUS_PATH, INSTANCES_PATH, MANIFEST_PATH, SMOKE_PATH,
                             load_corpus, load_instances, load_jsonl, load_manifest,
                             load_smoke)

__all__ = ["CORPUS_PATH", "INSTANCES_PATH", "MANIFEST_PATH", "SMOKE_PATH",
           "load_corpus", "load_instances", "load_jsonl", "load_manifest", "load_smoke"]
