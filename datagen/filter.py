#!/usr/bin/env python3
"""Rejection filter for generated training examples.

Binding contract (docs/BUILD_PLAN.md, "Datagen verifier"). Keep an example iff

  * its assistant content passes ``eval.parse_output`` -- IMPORTED from the
    frozen harness, never reimplemented -- with the class the gold label
    demands: exactly ``refusal`` for gold unanswerable, ``answer`` with every
    cited span a verbatim substring of the passage for gold answerable;
  * the passage contains no double-quote character (a quote inside a passage
    round-trips into a ``Citation: "<span>"`` line and breaks parsing);
  * neither the passage nor the question shares a 6-gram (lowercased,
    whitespace-normalized) with any passage or question in the frozen eval
    set or the few-shot exemplars -- the leakage guard;
  * it is not a duplicate of an already-kept example, on normalized
    (passage, question).

Every rejection carries a machine-readable reason code from ``REASONS``.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval as E  # noqa: E402  (needs the sys.path insert above)

EVAL_SCENARIOS = REPO_ROOT / "data" / "eval" / "scenarios_v1.jsonl"
EVAL_EXEMPLARS = REPO_ROOT / "data" / "eval" / "fewshot_exemplars.jsonl"

TAXONOMY_TYPES = ("clean_answerable", "support_deleted", "distractor",
                  "parametric_trap", "near_miss")
GOLD_LABELS = ("answerable", "unanswerable")
NGRAM_N = 6

#: reason code -> human explanation. Codes are the stable machine-readable
#: contract; the prose is for audit output only.
REASONS = {
    # --- verifier contract (applied to every candidate, in this order) ---
    "bad_schema": "candidate is missing a required field or a field has the wrong type",
    "unknown_type": "type is not one of the five taxonomy types",
    "unknown_gold_label": "gold_label is neither 'answerable' nor 'unanswerable'",
    "quote_in_passage": "passage contains a double-quote character, which breaks Citation parsing",
    "quote_in_question": "question contains a double-quote character",
    "parse_malformed": "eval.parse_output classified the assistant content as malformed",
    "class_mismatch": "eval.parse_output class does not match the gold label",
    "nonverbatim_citation": "a cited span is not a verbatim substring of the passage",
    "leakage_6gram": "shares a 6-gram with an eval scenario/exemplar passage or question",
    "duplicate": "normalized (passage, question) was already kept in this batch",
    # --- upstream teacher/derivation failures recorded by generate.py ---
    "teacher_json_unparseable": "teacher reply was not parseable as the requested JSON array",
    "teacher_item_invalid": "teacher item lacked required fields or used the wrong shape",
    "gold_span_nonverbatim": "teacher's declared gold span is not a substring of its own passage",
    "gold_label_mismatch": "teacher item's gold_label contradicts the requested taxonomy bucket",
    "twin_no_parent": "no accepted answerable example was available to delete support from",
    "twin_support_remains": "sentence deletion left a cited span still present in the passage",
    "twin_too_short": "sentence deletion left fewer than the minimum number of sentences",
    "over_quota": "example passed every check but exceeded its bucket quota or the balance budget",
}

_WS_RE = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def normalize(text: str) -> str:
    """Lowercase + whitespace-normalize. Used for n-grams and for dedupe."""
    return _WS_RE.sub(" ", str(text).strip().lower())


def ngrams(text: str, n: int = NGRAM_N) -> set[str]:
    words = normalize(text).split()
    if len(words) < n:
        return set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def dedupe_key(passage: str, question: str) -> tuple[str, str]:
    return (normalize(passage), normalize(question))


def sentence_offsets(passage: str) -> list[tuple[int, int]]:
    """(start, end) character offsets of each sentence, in order."""
    spans, start = [], 0
    for m in _SENT_SPLIT_RE.finditer(passage):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(passage)))
    return [(a, b) for a, b in spans if passage[a:b].strip()]


def split_sentences(passage: str) -> list[str]:
    return [passage[a:b].strip() for a, b in sentence_offsets(passage.strip())]


class LeakageIndex:
    """Every 6-gram of every passage and question in the frozen eval files."""

    def __init__(self, texts, n: int = NGRAM_N):
        self.n = n
        self.grams: set[str] = set()
        self.sources = 0
        for t in texts:
            if t:
                self.grams |= ngrams(t, n)
                self.sources += 1

    @classmethod
    def from_eval_files(cls, paths=None, n: int = NGRAM_N) -> "LeakageIndex":
        paths = [Path(p) for p in (paths or (EVAL_SCENARIOS, EVAL_EXEMPLARS))]
        texts: list[str] = []
        for p in paths:
            for row in E.load_jsonl(str(p)):
                texts.append(row.get("passage", ""))
                texts.append(row.get("question", ""))
        return cls(texts, n=n)

    def hits(self, *texts: str) -> list[str]:
        found: set[str] = set()
        for t in texts:
            if t:
                found |= (ngrams(t, self.n) & self.grams)
        return sorted(found)


_DEFAULT_INDEX: LeakageIndex | None = None


def default_leakage_index() -> LeakageIndex:
    """Lazily built index over the two frozen eval files (process-cached)."""
    global _DEFAULT_INDEX
    if _DEFAULT_INDEX is None:
        _DEFAULT_INDEX = LeakageIndex.from_eval_files()
    return _DEFAULT_INDEX


@dataclass
class Verdict:
    ok: bool
    reason: str | None = None
    detail: str = ""
    parse_class: str | None = None
    checks: list[str] = field(default_factory=list)


REQUIRED_FIELDS = ("type", "gold_label", "passage", "question", "assistant")


class ExampleFilter:
    """Stateful filter: holds the leakage index, the dedupe set and counters."""

    def __init__(self, leakage_index: LeakageIndex | None = None):
        self.leakage = leakage_index if leakage_index is not None else default_leakage_index()
        self.seen: set[tuple[str, str]] = set()
        self.kept = 0
        self.rejected: dict[str, int] = {}
        self.samples: dict[str, str] = {}

    # ---------------------------------------------------------------- core
    def check(self, cand: dict, record: bool = True) -> Verdict:
        """Apply the verifier contract to one candidate.

        ``cand`` needs: type, gold_label, passage, question, assistant.
        A kept candidate is registered in the dedupe set (so the next
        identical candidate is rejected as ``duplicate``).
        """
        v = self._check(cand)
        if record:
            if v.ok:
                self.kept += 1
                self.seen.add(dedupe_key(cand["passage"], cand["question"]))
            else:
                self.rejected[v.reason] = self.rejected.get(v.reason, 0) + 1
                self.samples.setdefault(v.reason, v.detail)
        return v

    def note_rejection(self, reason: str, detail: str = "") -> None:
        """Record a rejection raised upstream of check() (teacher/derivation)."""
        self.rejected[reason] = self.rejected.get(reason, 0) + 1
        self.samples.setdefault(reason, detail)

    def _check(self, cand: dict) -> Verdict:
        checks: list[str] = []

        for f in REQUIRED_FIELDS:
            if f not in cand:
                return Verdict(False, "bad_schema", f"missing field {f!r}")
            if not isinstance(cand[f], str):
                return Verdict(False, "bad_schema", f"field {f!r} is not a string")
        passage, question = cand["passage"], cand["question"]
        assistant = cand["assistant"]
        if not passage.strip():
            return Verdict(False, "bad_schema", "empty passage")
        if not question.strip():
            return Verdict(False, "bad_schema", "empty question")
        if not assistant.strip():
            return Verdict(False, "bad_schema", "empty assistant content")
        checks.append("schema")

        if cand["type"] not in TAXONOMY_TYPES:
            return Verdict(False, "unknown_type", repr(cand["type"]))
        if cand["gold_label"] not in GOLD_LABELS:
            return Verdict(False, "unknown_gold_label", repr(cand["gold_label"]))
        checks.append("taxonomy")

        if '"' in passage:
            return Verdict(False, "quote_in_passage",
                           f"passage holds {passage.count(chr(34))} double-quote(s)")
        if '"' in question:
            return Verdict(False, "quote_in_question", question[:80])
        checks.append("no_double_quote")

        parsed = E.parse_output(assistant)          # frozen parser, imported
        cls = parsed["class"]
        if cls == "malformed":
            return Verdict(False, "parse_malformed", parsed.get("why", "")[:160],
                           parse_class=cls)
        if cand["gold_label"] == "unanswerable":
            if cls != "refusal":
                return Verdict(False, "class_mismatch",
                               f"gold unanswerable but parse class is {cls!r}",
                               parse_class=cls)
            checks.append("eval.parse_output:refusal")
        else:
            if cls != "answer":
                return Verdict(False, "class_mismatch",
                               f"gold answerable but parse class is {cls!r}",
                               parse_class=cls)
            nonverbatim = [s for s in parsed["spans"] if s not in passage]
            if nonverbatim:
                return Verdict(False, "nonverbatim_citation", nonverbatim[0][:160],
                               parse_class=cls)
            checks += ["eval.parse_output:answer", "verbatim_spans"]

        leak = self.leakage.hits(passage, question)
        if leak:
            return Verdict(False, "leakage_6gram", leak[0], parse_class=cls)
        checks.append(f"no_{self.leakage.n}gram_eval_leakage")

        if dedupe_key(passage, question) in self.seen:
            return Verdict(False, "duplicate", normalize(question)[:120],
                           parse_class=cls)
        checks.append("dedupe")

        return Verdict(True, None, "", parse_class=cls, checks=checks)


# ---------------------------------------------------------------- row shapes
def build_row(cand: dict, source_batch: str, verdict: Verdict) -> dict:
    """The training-example schema (P3 output, P4/P5 input).

    system is ``eval.SYSTEM_CORE`` verbatim and user is built by
    ``eval.user_msg`` so the training condition equals the zero_shot eval
    condition exactly.
    """
    return {
        "messages": [
            {"role": "system", "content": E.SYSTEM_CORE},
            {"role": "user", "content": E.user_msg(cand["passage"], cand["question"])},
            {"role": "assistant", "content": cand["assistant"]},
        ],
        "meta": {
            "type": cand["type"],
            "gold_label": cand["gold_label"],
            "source_batch": source_batch,
            "verifier": {
                "checks": verdict.checks,
                "parse_class": verdict.parse_class,
                "rubric_hash": E.rubric_hash(),
            },
        },
    }


USER_PREFIX = "Context passage:\n"
USER_SEP = "\n\nQuestion: "


def split_user_msg(content: str) -> tuple[str, str]:
    """Inverse of ``eval.user_msg``; raises ValueError if it does not round-trip."""
    if not content.startswith(USER_PREFIX):
        raise ValueError("user content does not start with the eval.user_msg prefix")
    if USER_SEP not in content:
        raise ValueError("user content has no eval.user_msg question separator")
    body = content[len(USER_PREFIX):]
    passage, _, question = body.rpartition(USER_SEP)
    if E.user_msg(passage, question) != content:
        raise ValueError("user content does not round-trip through eval.user_msg")
    return passage, question
