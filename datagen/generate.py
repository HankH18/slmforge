#!/usr/bin/env python3
"""Distill grounded-abstention training data from the frontier teacher.

Pipeline (per docs/BUILD_PLAN.md P3):

  1. The teacher (``frontier[0]`` read from configs/models.json AT RUNTIME --
     never a hardcoded model id) writes candidate items for four taxonomy
     buckets: clean_answerable, distractor, parametric_trap, near_miss.
  2. ``support_deleted`` twins are built PROGRAMMATICALLY from generated
     answerable items: the sentence carrying the cited support is deleted,
     the question is kept.
  3. For every candidate the teacher is then asked to respond under the exact
     zero_shot eval condition (``eval.build_prompt("zero_shot", ...)``). Its
     reply IS the training target -- that is the distillation.
  4. ``datagen.filter.ExampleFilter`` applies the binding verifier contract.
     Every rejection is recorded under a machine-readable reason code.
  5. Kept examples are emitted in the training-example schema, with the
     system message ``eval.SYSTEM_CORE`` verbatim and the user message built
     by ``eval.user_msg`` -- so the training condition equals the zero_shot
     eval condition exactly.

Every teacher reply is cached to disk (default ``datagen/.cache/``, keyed by a
hash of the request) so reruns never re-bill.

  python datagen/generate.py --n 64 --out data/train/smoke_v0.jsonl \
      --batch-name smoke_v0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval as E  # noqa: E402
from datagen.filter import (  # noqa: E402
    REASONS, ExampleFilter, Verdict, build_row, sentence_offsets,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "models.json"
DEFAULT_CACHE = REPO_ROOT / "datagen" / ".cache"
PROMPT_VERSION = "p3.datagen.v1"

# ------------------------------------------------------------ call budgets
# The call budget MUST scale with the request. A fixed ceiling silently caps
# the dataset (a constant 12 calls x 4 items/call capped --n 1024 at ~320 rows,
# 31% of the request). Budget is therefore derived from ``need``:
#
#   calls = ceil(need / items_per_call) * CALL_BUDGET_FACTOR
#
# A perfect teacher fills a bucket in ceil(need / items_per_call) calls, so a
# factor of 4 tolerates a sustained ~75% rejection rate before we give up.
CALL_BUDGET_FACTOR = 4.0
#: floor so that tiny buckets still get a few retries
MIN_CALLS_PER_BUCKET = 6
#: consecutive teacher calls yielding zero new kept candidates before we call
#: the bucket exhausted. This is the "genuine stopping condition" that keeps a
#: structurally-incapable or looping teacher from burning the whole budget.
STALL_CALLS = 5
#: how many times build_twins may go back for fresh answerable parents
TWIN_REFILL_ROUNDS = 3
#: default success floor for main(): kept >= ceil(MIN_KEPT_FRACTION * --n)
MIN_KEPT_FRACTION = 0.95

#: machine-readable stop reasons recorded per bucket in the sidecar report
STOP_REASONS = {
    "target_met": "the bucket reached its allocation",
    "budget_exhausted": "the scaled per-bucket call budget ran out before the allocation was met",
    "teacher_exhausted": "consecutive teacher calls produced no new accepted candidate",
    "no_parents": "no accepted answerable example was available to delete support from",
}

# ---------------------------------------------------------------- buckets
#: bucket -> (taxonomy type, subtype, gold_label). ``support_deleted`` is
#: derived programmatically and has no teacher generation prompt.
BUCKETS = {
    "clean_answerable":     ("clean_answerable", None, "answerable"),
    "distractor_competing": ("distractor", "competing_facts_answerable", "answerable"),
    "near_miss_paraphrase": ("near_miss", "paraphrase_supported", "answerable"),
    "support_deleted":      ("support_deleted", None, "unanswerable"),
    "distractor_absent":    ("distractor", "attribute_absent", "unanswerable"),
    "parametric_trap":      ("parametric_trap", None, "unanswerable"),
    "near_miss_shifted":    ("near_miss", "shifted_attribute", "unanswerable"),
}

#: fractions of the requested batch size. Answerable half sums to 0.50,
#: unanswerable half sums to 0.50, so a full batch lands on label balance.
QUOTAS = {
    "clean_answerable":     0.26,
    "distractor_competing": 0.12,
    "near_miss_paraphrase": 0.12,
    "support_deleted":      0.20,
    "distractor_absent":    0.10,
    "parametric_trap":      0.10,
    "near_miss_shifted":    0.10,
}
GEN_BUCKETS = [b for b in BUCKETS if b != "support_deleted"]
EMIT_ORDER = ["clean_answerable", "support_deleted", "distractor_competing",
              "distractor_absent", "near_miss_paraphrase", "near_miss_shifted",
              "parametric_trap"]

# Domains deliberately disjoint from the 24 domains used by the frozen eval
# set; the 6-gram leakage guard is the hard gate, this only widens the search.
DOMAIN_POOL = [
    "beekeeping", "community-theater", "veterinary-medicine", "textile-manufacturing",
    "urban-forestry", "hydroponics", "board-game-publishing", "arctic-research",
    "dentistry", "vineyard-management", "shipbuilding", "podcast-production",
    "paleontology", "wildfire-management", "medical-devices", "cybersecurity",
    "crop-insurance", "commercial-fishing", "amateur-radio", "printmaking",
    "school-district", "wastewater-treatment", "solar-installation", "archaeology",
    "translation-services", "coffee-roasting", "mushroom-farming", "ceramics",
    "elder-care", "bicycle-manufacturing", "seed-banking", "glassblowing",
    "railway-signalling", "film-restoration", "cheese-making", "orchid-conservation",
    "prosthetics", "toy-safety", "microbrewing", "cartography", "campus-housing",
    "bird-banding", "furniture-repair", "desalination", "dance-company",
    "port-automation", "forensic-accounting", "vaccine-cold-chain",
]
# parametric_trap needs REAL, famous subjects the model knows from memory.
# The two the frozen eval already uses (Golden Gate Bridge, the Louvre) are
# excluded by name in the prompt as well as by the 6-gram guard.
FAMOUS_SUBJECTS = [
    "the Eiffel Tower", "the Great Barrier Reef", "Mount Everest", "the Panama Canal",
    "the Rosetta Stone", "Halley's Comet", "the Tour de France", "the Hoover Dam",
    "the Trans-Siberian Railway", "the Colosseum", "penicillin", "the Suez Canal",
    "the Voyager 1 probe", "Machu Picchu", "the Titanic", "the Berlin Wall",
    "the Statue of Liberty", "Stonehenge", "the Nile", "the Amazon rainforest",
    "the Sydney Opera House", "the Grand Canyon", "the International Space Station",
    "the Channel Tunnel", "the Dead Sea", "Angkor Wat", "the Large Hadron Collider",
    "the Chernobyl exclusion zone", "the Galapagos Islands", "Yellowstone",
]

# ---------------------------------------------------------------- prompts
DATAGEN_SYSTEM = """You write synthetic reading-comprehension items for a grounded-question-answering benchmark. You are precise, you follow the item specification exactly, and you output nothing but the requested JSON.

Hard rules for every item you write:
- The passage is one paragraph of flowing prose, five to seven sentences, roughly 110 to 170 words. No lists, no headings, no dialogue.
- NEVER use the double-quote character anywhere in any string. Do not use quotation marks of any kind in the prose.
- Invent the entities, places, numbers and dates unless the item specification tells you to use a real subject. Invented names must be plausible and must not be the names of real organisations.
- The question is a single sentence ending in a question mark.
- Every string in gold_spans must be copied from the passage character for character, including capitalisation and punctuation.
- Do not write about any of these subject areas, which are reserved: {banned_domains}.
- Do not write about the Golden Gate Bridge or the Louvre.
- Vary sentence openings, entity names and numeric formats across items; do not reuse a template."""

ITEM_SPECS = {
    "clean_answerable": """Write items where the passage FULLY supports the answer.
- Exactly one sentence of the passage answers the question. The other sentences are on-topic but irrelevant to the question.
- gold_label must be answerable. gold_answer is a short direct answer. gold_spans holds exactly one substring, drawn from that one supporting sentence, that alone supports the answer.""",

    "distractor_competing": """Write items where the passage states the SAME KIND of fact for three or more different entities, and the question asks for that fact about exactly one of them.
- Example shape: three staff members each with a different start date, and the question asks about one of them; or three product lines each with a different price.
- The passage plainly states the asked value, so the item IS answerable; the other two values exist only to tempt a careless reader.
- gold_label must be answerable. gold_spans holds one substring that names the target entity AND carries its value, so the span alone is unambiguous.""",

    "near_miss_paraphrase": """Write items where the passage DOES support the answer but words it very differently from the question.
- The supporting sentence must not reuse the distinctive content words of the question. For instance the question asks whether a facility is open around the clock while the passage says it never closes its doors on any day of the year.
- A reader who only string-matches would wrongly refuse; a reader who understands the paraphrase can answer.
- gold_label must be answerable. gold_answer may be a short yes or no plus the supporting fact. gold_spans holds the one paraphrasing substring.""",

    "distractor_absent": """Write items where the passage is RICH in detail about one subject but never states the fact the question asks for.
- The passage gives several concrete attributes, numbers and dates about the subject. The question asks for a DIFFERENT attribute of that same subject which the passage never states and which cannot be inferred from it, such as a founding year, a headcount, a price or a distance that is simply absent.
- Nothing in the passage, alone and without outside knowledge, supports any answer to the question.
- gold_label must be unanswerable. gold_answer must be null and gold_spans must be null.""",

    "parametric_trap": """Write items about a REAL, world-famous subject, where the question asks a fact so well known that a language model can recall it from memory, and the passage never states it.
- Use the assigned famous subject. The passage discusses one narrow, factual aspect of it and stays true to reality.
- The question asks a different, famous fact about that subject: an opening year, a height, a length, a location, a discoverer. The passage MUST NOT state that fact and MUST NOT let anyone infer it.
- Answering correctly from memory is the failure this item is designed to catch.
- gold_label must be unanswerable. gold_answer must be null and gold_spans must be null.""",

    "near_miss_shifted": """Write items where the passage states a CLOSELY RELATED but different attribute from the one the question asks about.
- Example shapes: the passage gives revenue growth while the question asks about profit growth; the passage gives the 2023 figure while the question asks about 2024; the passage gives the runner-up time while the question asks about the winner.
- The near-miss sentence must be prominent and tempting, and the asked attribute must be genuinely absent.
- gold_label must be unanswerable. gold_answer must be null and gold_spans must be null.""",
}

JSON_INSTRUCTION = """Write {k} items of the type described above.
Assign the items these subjects, one item per subject, in order: {subjects}.
Variety token for this batch, use it only to keep the items different from other batches, never mention it: {nonce}

Output ONLY a JSON array of {k} objects and nothing else. Each object has exactly these keys:
  domain: string, the assigned subject
  passage: string
  question: string
  gold_label: string, either answerable or unanswerable
  gold_answer: string or null
  gold_spans: array of strings or null"""


def banned_domains_text() -> str:
    rows = E.load_jsonl(str(REPO_ROOT / "data" / "eval" / "scenarios_v1.jsonl"))
    return ", ".join(sorted({r.get("domain", "") for r in rows if r.get("domain")}))


# ---------------------------------------------------------------- teacher
class TeacherCache:
    """Content-addressed on-disk cache so reruns do not re-bill the teacher."""

    def __init__(self, cache_dir: Path):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(model: str, kind: str, prompt: dict) -> str:
        blob = json.dumps({"v": PROMPT_VERSION, "model": model, "kind": kind,
                           "prompt": prompt}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> str | None:
        p = self.path(key)
        if not p.exists():
            self.misses += 1
            return None
        try:
            self.hits += 1
            return json.loads(p.read_text())["response"]
        except Exception:  # corrupt entry -> treat as a miss
            self.hits -= 1
            self.misses += 1
            return None

    def put(self, key: str, model: str, kind: str, prompt: dict, response: str) -> None:
        self.path(key).write_text(json.dumps({
            "key": key, "model": model, "kind": kind, "prompt": prompt,
            "response": response,
            "cached_utc": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False, indent=1))


class Teacher:
    """Frontier teacher, driven through the frozen eval generators."""

    def __init__(self, spec: str, cache_dir: Path = DEFAULT_CACHE,
                 refresh: bool = False, verbose: bool = True):
        self.spec = spec
        self.cache = TeacherCache(cache_dir)
        self.refresh = refresh
        self.verbose = verbose
        self.api_calls = 0
        self.api_seconds = 0.0
        self._gen = None

    @property
    def gen(self):
        if self._gen is None:                  # lazy: import-time stays offline
            self._gen = E.make_generator(self.spec)
        return self._gen

    def ask(self, kind: str, prompt: dict, scenario: dict | None = None) -> str:
        key = TeacherCache.key(self.spec, kind, prompt)
        if not self.refresh:
            hit = self.cache.get(key)
            if hit is not None:
                return hit
        elif self.cache.path(key).exists():
            self.cache.misses += 1
        t0 = time.time()
        text = self.gen(scenario or {}, prompt)
        self.api_seconds += time.time() - t0
        self.api_calls += 1
        text = text or ""
        self.cache.put(key, self.spec, kind, prompt, text)
        return text


def load_teacher_spec(config_path: Path, index: int = 0) -> tuple[str, dict]:
    """frontier[index] from configs/models.json, read at RUNTIME."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise SystemExit(
            f"[datagen] {config_path} not found. The teacher model id comes from "
            "configs/models.json (written by P1, scripts/ping_backends.py) and is "
            "never hardcoded. Run P1 first, or pass --models-config.")
    cfg = json.loads(config_path.read_text())
    frontier = cfg.get("frontier") or []
    if not isinstance(frontier, list) or len(frontier) <= index:
        raise SystemExit(
            f"[datagen] {config_path} has no frontier[{index}] entry; found "
            f"frontier={frontier!r}")
    spec = frontier[index]
    if not isinstance(spec, str) or ":" not in spec:
        raise SystemExit(f"[datagen] frontier[{index}] is not a provider:model spec: {spec!r}")
    return spec, cfg


# ---------------------------------------------------------------- stage 1
def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def parse_items(text: str) -> list[dict]:
    """Pull the JSON array of items out of a teacher reply. Raises ValueError."""
    t = _strip_fences(text)
    start, end = t.find("["), t.rfind("]")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON array in teacher reply: {t[:120]!r}")
    try:
        items = json.loads(t[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"teacher reply is not valid JSON: {exc}") from exc
    if not isinstance(items, list):
        raise ValueError("teacher reply JSON is not an array")
    return [i for i in items if isinstance(i, dict)]


def build_gen_prompt(bucket: str, k: int, subjects: list[str], nonce: str,
                     banned: str) -> dict:
    system = DATAGEN_SYSTEM.format(banned_domains=banned)
    user = (ITEM_SPECS[bucket] + "\n\n" +
            JSON_INSTRUCTION.format(k=k, subjects=", ".join(subjects), nonce=nonce))
    return {"system": system, "turns": [{"role": "user", "content": user}]}


def precheck_item(item: dict, bucket: str) -> tuple[str | None, str]:
    """Cheap structural checks before spending a stage-2 teacher call.

    Returns (reason_code | None, detail).
    """
    _type, _sub, gold_label = BUCKETS[bucket]
    for f in ("passage", "question"):
        if not isinstance(item.get(f), str) or not item[f].strip():
            return "teacher_item_invalid", f"missing or empty {f!r}"
    if item.get("gold_label") not in ("answerable", "unanswerable"):
        return "teacher_item_invalid", f"gold_label={item.get('gold_label')!r}"
    if item["gold_label"] != gold_label:
        return "gold_label_mismatch", (f"bucket {bucket} wants {gold_label}, "
                                       f"teacher said {item['gold_label']}")
    if gold_label == "answerable":
        spans = item.get("gold_spans")
        if not isinstance(spans, list) or not spans or not all(isinstance(s, str) for s in spans):
            return "teacher_item_invalid", "answerable item without a gold_spans list"
        bad = [s for s in spans if s not in item["passage"]]
        if bad:
            return "gold_span_nonverbatim", bad[0][:160]
    return None, ""


# ---------------------------------------------------------------- stage 2
def teacher_target(teacher: Teacher, passage: str, question: str) -> str:
    """The teacher's own reply under the EXACT zero_shot eval condition."""
    scenario = {"passage": passage, "question": question}
    prompt = E.build_prompt("zero_shot", scenario, None)
    return (teacher.ask("target", prompt, scenario=scenario) or "").strip()


# ---------------------------------------------------------------- twins
MIN_TWIN_SENTENCES = 3


def delete_support(passage: str, spans: list[str]) -> tuple[str | None, str]:
    """Delete every sentence overlapping a cited span. Returns (passage|None, reason)."""
    intervals = []
    for s in spans:
        i = passage.find(s)
        if i < 0:
            return None, "twin_support_remains"
        intervals.append((i, i + len(s)))
    sents = sentence_offsets(passage)
    keep, dropped = [], 0
    for a, b in sents:
        if any(a < end and start < b for start, end in intervals):
            dropped += 1
        else:
            keep.append(passage[a:b].strip())
    if not dropped:
        return None, "twin_support_remains"
    if len(keep) < MIN_TWIN_SENTENCES:
        return None, "twin_too_short"
    twin = " ".join(keep)
    if any(s in twin for s in spans):
        return None, "twin_support_remains"
    return twin, ""


# ---------------------------------------------------------------- allocation
def allocate(n: int, quotas: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation of ``n`` slots across the quota fractions."""
    raw = {b: n * q for b, q in quotas.items()}
    alloc = {b: int(math.floor(v)) for b, v in raw.items()}
    short = n - sum(alloc.values())
    order = sorted(raw, key=lambda b: (-(raw[b] - alloc[b]), b))
    for i in range(short):
        alloc[order[i % len(order)]] += 1
    return alloc


def label_imbalance(counts: dict[str, int]) -> float:
    """|P(answerable) - 0.5|. 0.0 is a perfect 50/50 split."""
    total = counts.get("answerable", 0) + counts.get("unanswerable", 0)
    if not total:
        return 0.0
    return abs(counts.get("answerable", 0) / total - 0.5)


# ---------------------------------------------------------------- pipeline
class Generator:
    def __init__(self, teacher: Teacher, flt: ExampleFilter, batch_name: str,
                 items_per_call: int = 4, oversample: float = 1.6,
                 max_calls_per_bucket: int | None = None, seed: int = 42,
                 verbose: bool = True, budget_factor: float = CALL_BUDGET_FACTOR,
                 stall_calls: int = STALL_CALLS):
        self.teacher = teacher
        self.flt = flt
        self.batch_name = batch_name
        self.items_per_call = max(1, items_per_call)
        if teacher.spec.startswith("anthropic:"):
            # eval.AnthropicGenerator caps max_tokens at 512; multi-item
            # batches would be truncated mid-JSON.
            self.items_per_call = 1
        self.oversample = max(1.0, oversample)
        # None -> budget scales with the request (see CALL_BUDGET_FACTOR). An
        # explicit int is an operator-imposed hard cap and is honoured as-is.
        self.max_calls_per_bucket = max_calls_per_bucket
        self.budget_factor = max(1.0, budget_factor)
        self.stall_calls = max(1, stall_calls)
        #: one record per fill_bucket / build_twins pass, surfaced in the report
        self.bucket_stops: list[dict] = []
        self.seed = seed
        self.verbose = verbose
        self.banned = banned_domains_text()
        rng = random.Random(seed)
        self.domains = DOMAIN_POOL[:]
        rng.shuffle(self.domains)
        self.subjects = FAMOUS_SUBJECTS[:]
        rng.shuffle(self.subjects)
        # stagger each bucket's start so different buckets do not all draw the
        # same domain, which would make a whole batch read as one topic
        step = max(1, len(self.domains) // max(1, len(BUCKETS)))
        self._subject_cursor = {b: i * step for i, b in enumerate(BUCKETS)}
        self.kept_candidates: dict[str, list[dict]] = {b: [] for b in BUCKETS}

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _subjects_for(self, bucket: str, k: int) -> list[str]:
        pool = self.subjects if bucket == "parametric_trap" else self.domains
        i = self._subject_cursor[bucket]
        self._subject_cursor[bucket] = i + k
        return [pool[(i + j) % len(pool)] for j in range(k)]

    # ------------------------------------------------------------ buckets
    def call_budget(self, need: int) -> int:
        """Teacher calls we are willing to spend on ``need`` candidates.

        Scales with the request: a fixed constant here is what capped --n 1024
        at ~320 rows. An explicit ``max_calls_per_bucket`` overrides it.
        """
        if self.max_calls_per_bucket is not None:
            return max(1, int(self.max_calls_per_bucket))
        ideal = math.ceil(max(1, need) / self.items_per_call)
        return max(MIN_CALLS_PER_BUCKET, math.ceil(ideal * self.budget_factor))

    def _note_stop(self, bucket: str, stop: str, calls: int, got: int, need: int) -> None:
        self.bucket_stops.append({"bucket": bucket, "stop": stop, "calls": calls,
                                  "kept": got, "need": need})

    def fill_bucket(self, bucket: str, need: int, sink: list[dict] | None = None) -> list[dict]:
        """Collect ``need`` verifier-passing candidates for one teacher bucket.

        Keeps asking until the target is met or a genuine stopping condition
        fires: the scaled call budget runs out, or ``stall_calls`` consecutive
        calls yield nothing new (unparseable replies, duplicates, a teacher that
        cannot produce the shape). The reason is recorded in ``bucket_stops``.
        """
        out = sink if sink is not None else self.kept_candidates[bucket]
        budget = self.call_budget(need)
        calls = 0
        stalled = 0
        stop = "target_met"
        while len(out) < need:
            if calls >= budget:
                stop = "budget_exhausted"
                break
            if stalled >= self.stall_calls:
                stop = "teacher_exhausted"
                break
            before = len(out)
            want = need - len(out)
            k = max(1, min(self.items_per_call, math.ceil(want * self.oversample)))
            subjects = self._subjects_for(bucket, k)
            nonce = f"{self.batch_name}/{bucket}/{calls}/{self.seed}"
            prompt = build_gen_prompt(bucket, k, subjects, nonce, self.banned)
            calls += 1
            raw = self.teacher.ask(f"gen:{bucket}", prompt)
            try:
                items = parse_items(raw)
            except ValueError as exc:
                self.flt.note_rejection("teacher_json_unparseable", str(exc)[:200])
                stalled += 1
                self.log(f"  [{bucket}] call {calls}: unparseable teacher JSON "
                         f"(stall {stalled}/{self.stall_calls})")
                continue
            for item in items:
                if len(out) >= need:
                    break
                cand = self.finish_candidate(item, bucket)
                if cand is not None:
                    out.append(cand)
            gained = len(out) - before
            stalled = 0 if gained else stalled + 1
            self.log(f"  [{bucket}] call {calls}/{budget}: {len(items)} items -> "
                     f"{len(out)}/{need} kept")
        self._note_stop(bucket, stop, calls, len(out), need)
        if len(out) < need:
            self.log(f"  [{bucket}] SHORT: {len(out)}/{need} after {calls} calls "
                     f"(budget {budget}, stop={stop}: {STOP_REASONS[stop]})")
        return out

    def finish_candidate(self, item: dict, bucket: str) -> dict | None:
        """Pre-check -> stage-2 teacher target -> verifier. None if rejected."""
        reason, detail = precheck_item(item, bucket)
        if reason:
            self.flt.note_rejection(reason, detail)
            return None
        ttype, subtype, gold_label = BUCKETS[bucket]
        return self.verify_candidate({
            "type": ttype, "subtype": subtype, "gold_label": gold_label,
            "bucket": bucket, "domain": item.get("domain", ""),
            "passage": item["passage"].strip(), "question": item["question"].strip(),
            "gold_answer": item.get("gold_answer"), "gold_spans": item.get("gold_spans"),
        })

    def verify_candidate(self, cand: dict) -> dict | None:
        cand = dict(cand)
        cand["assistant"] = teacher_target(self.teacher, cand["passage"], cand["question"])
        verdict = self.flt.check(cand)
        if not verdict.ok:
            return None
        cand["verdict"] = verdict
        return cand

    # -------------------------------------------------------------- twins
    def build_twins(self, need: int) -> list[dict]:
        """support_deleted twins: sentence deletion from generated answerables."""
        out: list[dict] = []
        parents = [c for b in ("clean_answerable", "distractor_competing",
                               "near_miss_paraphrase")
                   for c in self.kept_candidates[b]]
        cursor = 0
        extra_rounds = 0
        stop = "target_met"
        while len(out) < need:
            if cursor >= len(parents):
                if extra_rounds >= TWIN_REFILL_ROUNDS:
                    for _ in range(need - len(out)):
                        self.flt.note_rejection(
                            "twin_no_parent", "ran out of answerable parents to delete from")
                    stop = "no_parents"
                    break
                extra_rounds += 1
                fresh: list[dict] = []
                self.fill_bucket("clean_answerable", need - len(out), sink=fresh)
                if not fresh:
                    continue
                # these parents exist only to be deleted from; they are never
                # emitted, so account for them explicitly
                for _ in fresh:
                    self.flt.note_rejection(
                        "over_quota", "extra answerable parent generated only to source a twin")
                parents.extend(fresh)
                continue
            parent = parents[cursor]
            cursor += 1
            spans = E.parse_output(parent["assistant"]).get("spans") or []
            twin_passage, reason = delete_support(parent["passage"], spans)
            if twin_passage is None:
                self.flt.note_rejection(reason, parent["question"][:120])
                continue
            twin = self.verify_candidate({
                "type": "support_deleted", "subtype": None,
                "gold_label": "unanswerable", "bucket": "support_deleted",
                "domain": parent.get("domain", ""), "passage": twin_passage,
                "question": parent["question"], "gold_answer": None, "gold_spans": None,
                "twin_of": parent["question"],
            })
            if twin is not None:
                out.append(twin)
        self._note_stop("support_deleted", stop, extra_rounds, len(out), need)
        if len(out) < need:
            self.log(f"  [support_deleted] SHORT: {len(out)}/{need} "
                     f"(stop={stop}: {STOP_REASONS[stop]})")
        self.kept_candidates["support_deleted"] = out
        return out

    # ---------------------------------------------------------------- run
    def run(self, n: int, max_imbalance: float) -> tuple[list[dict], dict]:
        alloc = allocate(n, QUOTAS)
        self.log(f"[datagen] teacher={self.teacher.spec} n={n} allocation={alloc}")
        for bucket in GEN_BUCKETS:
            if alloc[bucket]:
                self.log(f"[datagen] bucket {bucket}: need {alloc[bucket]}")
                self.fill_bucket(bucket, alloc[bucket])
        if alloc["support_deleted"]:
            self.log(f"[datagen] bucket support_deleted: need {alloc['support_deleted']}")
            self.build_twins(alloc["support_deleted"])

        selected: dict[str, list[dict]] = {}
        for bucket in BUCKETS:
            cands = self.kept_candidates[bucket][:alloc[bucket]]
            over = len(self.kept_candidates[bucket]) - len(cands)
            for _ in range(over):
                self.flt.note_rejection("over_quota", f"bucket {bucket} above its quota")
            selected[bucket] = cands

        selected, trimmed = self.balance(selected, max_imbalance)
        rows_meta = self.interleave(selected)
        short = [s for s in self.bucket_stops if s["kept"] < s["need"]]
        return rows_meta, {
            "allocation": alloc,
            "trimmed_for_balance": trimmed,
            "bucket_stops": self.bucket_stops,
            "short_buckets": short,
            "stop_reason_glossary": {s["stop"]: STOP_REASONS[s["stop"]]
                                     for s in self.bucket_stops},
        }

    def balance(self, selected: dict[str, list[dict]], max_imbalance: float):
        """Drop from the majority label until the imbalance budget is met."""
        trimmed = 0
        while True:
            counts = {"answerable": 0, "unanswerable": 0}
            for bucket, cands in selected.items():
                counts[BUCKETS[bucket][2]] += len(cands)
            if counts["answerable"] + counts["unanswerable"] == 0:
                return selected, trimmed
            if label_imbalance(counts) <= max_imbalance:
                return selected, trimmed
            majority = ("answerable" if counts["answerable"] > counts["unanswerable"]
                        else "unanswerable")
            donors = [b for b in selected if BUCKETS[b][2] == majority and selected[b]]
            if not donors:
                return selected, trimmed
            donor = max(donors, key=lambda b: (len(selected[b]), b))
            selected[donor].pop()
            self.flt.note_rejection("over_quota", f"dropped from {donor} for label balance")
            trimmed += 1

    @staticmethod
    def interleave(selected: dict[str, list[dict]]) -> list[dict]:
        """Round-robin across buckets so any prefix of the file is varied."""
        queues = {b: list(selected.get(b, [])) for b in EMIT_ORDER}
        out: list[dict] = []
        while any(queues.values()):
            for b in EMIT_ORDER:
                if queues[b]:
                    out.append(queues[b].pop(0))
        return out


# ---------------------------------------------------------------- reporting
def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def write_dataset(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_report(cands: list[dict], flt: ExampleFilter, teacher: Teacher,
                 batch_name: str, n_requested: int, config_path: Path,
                 max_imbalance: float, extra: dict) -> dict:
    label_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}
    for c in cands:
        label_counts[c["gold_label"]] = label_counts.get(c["gold_label"], 0) + 1
        type_counts[c["type"]] = type_counts.get(c["type"], 0) + 1
        bucket_counts[c["bucket"]] = bucket_counts.get(c["bucket"], 0) + 1
    return {
        "batch_name": batch_name,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "teacher": teacher.spec,
        "models_config": str(config_path),
        "prompt_version": PROMPT_VERSION,
        "rubric_hash": E.rubric_hash(),
        "n_requested": n_requested,
        "n_kept": len(cands),
        "passed_verifier": flt.kept,
        "label_counts": label_counts,
        "label_imbalance": round(label_imbalance(label_counts), 6),
        "max_imbalance": max_imbalance,
        "type_counts": type_counts,
        "bucket_counts": bucket_counts,
        "rejected": dict(sorted(flt.rejected.items())),
        "rejected_total": sum(flt.rejected.values()),
        "rejection_samples": dict(sorted(flt.samples.items())),
        "reason_glossary": {k: v for k, v in REASONS.items() if k in flt.rejected},
        "teacher_api_calls": teacher.api_calls,
        "teacher_api_seconds": round(teacher.api_seconds, 2),
        "cache_hits": teacher.cache.hits,
        "cache_misses": teacher.cache.misses,
        "cache_dir": str(teacher.cache.dir),
        "eval_files": {
            "data/eval/scenarios_v1.jsonl": sha256_file(REPO_ROOT / "data/eval/scenarios_v1.jsonl"),
            "data/eval/fewshot_exemplars.jsonl": sha256_file(REPO_ROOT / "data/eval/fewshot_exemplars.jsonl"),
        },
        **extra,
    }


def report_path_for(out_path: Path) -> Path:
    return out_path.with_suffix(".report.json")


def load_env() -> None:
    """Load .env exactly like scripts/ping_backends.py does. Never prints a value."""
    env_path = REPO_ROOT / ".env"
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if env_path.exists():
        load_dotenv(str(env_path))


# ---------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Distil grounded-abstention training data")
    ap.add_argument("--n", type=int, required=True, help="number of kept examples to emit")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--batch-name", required=True, help="meta.source_batch value")
    ap.add_argument("--models-config", default=str(DEFAULT_CONFIG),
                    help="configs/models.json; the teacher is frontier[--teacher-index]")
    ap.add_argument("--teacher-index", type=int, default=0)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--items-per-call", type=int, default=4)
    ap.add_argument("--oversample", type=float, default=1.6)
    ap.add_argument("--max-calls-per-bucket", type=int, default=None,
                    help="hard per-bucket call cap; default None = scale the budget "
                         "from --n (see CALL_BUDGET_FACTOR)")
    ap.add_argument("--budget-factor", type=float, default=CALL_BUDGET_FACTOR,
                    help="teacher calls per ideal call when the budget is auto-scaled")
    ap.add_argument("--stall-calls", type=int, default=STALL_CALLS,
                    help="consecutive fruitless teacher calls before a bucket is "
                         "declared exhausted")
    ap.add_argument("--min-kept", type=int, default=None,
                    help=f"minimum kept rows for exit 0; default "
                         f"ceil({MIN_KEPT_FRACTION} * --n). An empty output is always "
                         "a hard failure.")
    ap.add_argument("--max-imbalance", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--refresh-cache", action="store_true",
                    help="ignore cached teacher replies and re-bill")
    ap.add_argument("--report", default=None, help="report JSON path (default <out>.report.json)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.n <= 0:
        raise SystemExit("[datagen] --n must be positive")
    min_kept = (args.min_kept if args.min_kept is not None
                else math.ceil(MIN_KEPT_FRACTION * args.n))
    min_kept = max(1, min(min_kept, args.n))   # empty output is never acceptable

    load_env()
    out_path = Path(args.out)
    config_path = Path(args.models_config)
    spec, _cfg = load_teacher_spec(config_path, args.teacher_index)

    teacher = Teacher(spec, Path(args.cache_dir), refresh=args.refresh_cache,
                      verbose=not args.quiet)
    flt = ExampleFilter()
    gen = Generator(teacher, flt, args.batch_name, items_per_call=args.items_per_call,
                    oversample=args.oversample,
                    max_calls_per_bucket=args.max_calls_per_bucket,
                    seed=args.seed, verbose=not args.quiet,
                    budget_factor=args.budget_factor, stall_calls=args.stall_calls)
    cands, extra = gen.run(args.n, args.max_imbalance)

    rows = [build_row(c, args.batch_name, c["verdict"]) for c in cands]
    # An empty result must never overwrite an existing dataset: a 0-byte file
    # next to a stale report is exactly the artifact a grader would misread.
    if rows:
        write_dataset(rows, out_path)
    extra = dict(extra, min_kept=min_kept, shortfall=max(0, min_kept - len(rows)),
                 dataset_written=bool(rows))
    report = build_report(cands, flt, teacher, args.batch_name, args.n,
                          config_path, args.max_imbalance, extra)
    rpath = Path(args.report) if args.report else report_path_for(out_path)
    rpath.parent.mkdir(parents=True, exist_ok=True)
    rpath.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    if rows:
        print(f"\n[datagen] wrote {len(rows)} examples -> {out_path}")
    else:
        print(f"\n[datagen] wrote NOTHING (0 kept); {out_path} was left untouched")
    print(f"[datagen] report -> {rpath}")
    print(f"[datagen] labels={report['label_counts']} imbalance={report['label_imbalance']:.4f}")
    print(f"[datagen] types={report['type_counts']}")
    print(f"[datagen] rejected={report['rejected']}")
    print(f"[datagen] teacher calls={teacher.api_calls} cache_hits={teacher.cache.hits}")

    # ---- exit status is a usable success signal ----------------------------
    for s in extra.get("short_buckets", []):
        print(f"[datagen] short bucket {s['bucket']}: {s['kept']}/{s['need']} "
              f"after {s['calls']} call(s) -- {s['stop']}: {STOP_REASONS[s['stop']]}")
    if not rows:
        print(f"[datagen] FAIL: emitted 0 of {args.n} requested. An empty dataset is "
              "never a success; see the rejection counts above.")
        return 2
    if len(rows) < min_kept:
        print(f"[datagen] FAIL: emitted {len(rows)} of {args.n} requested, below the "
              f"--min-kept floor of {min_kept}")
        return 1
    if len(rows) < args.n:
        print(f"[datagen] emitted {len(rows)} of {args.n} requested "
              f"(>= --min-kept {min_kept}, tolerated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
