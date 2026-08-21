"""The data-efficiency curve rests on the subsets being honest miniatures.

Every assertion here binds a way the curve could silently lie: a point that got
lucky rows, a point that is not contained in the next one up, or a small point
that quietly dropped the hard letters and posted a flattering number.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "make_subset.py"
INSTANCES = REPO_ROOT / "data" / "scenarios" / "constraint_instances.jsonl"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("make_subset", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["make_subset"] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def instances():
    return {json.loads(l)["id"]: json.loads(l)
            for l in INSTANCES.read_text().splitlines() if l.strip()}


def _letter_instance_ids(instances, letter):
    return [cid for cid, r in instances.items()
            if r["type"] == "banned_letter" and r["params"]["letter"] == letter]


def _write_dataset(tmp_path, instances, n=400):
    """A dataset that is lopsided AND CLUSTERED: every 'e' row first, then every
    'p' row.

    The clustering is the point. An earlier version of this fixture interleaved
    the two letters, and an unshuffled subsetter -- one that just takes the first
    N rows -- passed the composition check, because a prefix of an interleaved
    file preserves the ratio by construction. Real datasets are written in
    generation order, which correlates with task, constraint and letter, so the
    interleaved fixture was testing an ordering no real dataset has. Clustered,
    a prefix-without-shuffle subset comes out 100% 'e' and the check fires.
    """
    heavy = _letter_instance_ids(instances, "e")
    light = _letter_instance_ids(instances, "p")
    assert heavy and light, "fixture needs both an 'e' and a 'p' instance"
    n_heavy = (n * 3) // 4                          # 75% 'e', 25% 'p'
    rows = []
    for i in range(n):
        cid = heavy[0] if i < n_heavy else light[0]
        rows.append({"messages": [{"role": "user", "content": f"task {i}"},
                                  {"role": "assistant", "content": f"answer {i}"}],
                     "constraints": [cid],
                     "provenance": {"i": i}})
    p = tmp_path / "ds.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p, rows


def _run(dataset, sizes, out_dir, seed=20260821):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(dataset), "--sizes", *[str(s) for s in sizes],
         "--out-dir", str(out_dir), "--seed", str(seed)],
        capture_output=True, text=True, cwd=str(REPO_ROOT))


def _load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def test_subsets_are_nested(tmp_path, instances):
    """N/4 must be contained in N/2. Independent draws per point would let a
    small point win or lose on which rows it happened to get, and that wiggle
    would be read as a property of dataset size."""
    ds, _ = _write_dataset(tmp_path, instances)
    r = _run(ds, [50, 100, 200], tmp_path / "out")
    assert r.returncode == 0, r.stderr

    def key(row):
        return row["provenance"]["i"]

    small = {key(x) for x in _load(tmp_path / "out" / "ds.n50.jsonl")}
    mid = {key(x) for x in _load(tmp_path / "out" / "ds.n100.jsonl")}
    big = {key(x) for x in _load(tmp_path / "out" / "ds.n200.jsonl")}
    assert small < mid < big
    assert len(small) == 50 and len(mid) == 100 and len(big) == 200


def test_the_same_seed_reproduces_the_same_subset(tmp_path, instances):
    """A curve you cannot regenerate is not a curve a grader can rerun."""
    ds, _ = _write_dataset(tmp_path, instances)
    assert _run(ds, [100], tmp_path / "a").returncode == 0
    assert _run(ds, [100], tmp_path / "b").returncode == 0
    assert (tmp_path / "a" / "ds.n100.jsonl").read_bytes() == \
           (tmp_path / "b" / "ds.n100.jsonl").read_bytes()


def test_a_different_seed_gives_a_different_subset(tmp_path, instances):
    ds, _ = _write_dataset(tmp_path, instances)
    assert _run(ds, [100], tmp_path / "a", seed=1).returncode == 0
    assert _run(ds, [100], tmp_path / "b", seed=2).returncode == 0
    assert (tmp_path / "a" / "ds.n100.jsonl").read_bytes() != \
           (tmp_path / "b" / "ds.n100.jsonl").read_bytes()


def test_the_hard_letters_survive_subsetting(tmp_path, instances):
    """THE ONE THAT MATTERS. The curve exists to locate the smallest N that holds
    the behavior. A subset that dropped the hard letters would post a flattering
    number at small N and misplace that answer entirely."""
    ds, _ = _write_dataset(tmp_path, instances)   # 75% 'e', 25% 'p'
    assert _run(ds, [40, 80, 160], tmp_path / "out").returncode == 0
    for n in (40, 80, 160):
        rows = _load(tmp_path / "out" / f"ds.n{n}.jsonl")
        letters = Counter()
        for row in rows:
            for cid in row["constraints"]:
                inst = instances[cid]
                if inst["type"] == "banned_letter":
                    letters[inst["params"]["letter"]] += 1
        share = letters["e"] / sum(letters.values())
        assert 0.60 <= share <= 0.90, (
            f"n={n}: 'e' share {share:.2f} drifted far from the parent's 0.75; "
            f"the subset is not a fair miniature")


def test_the_manifest_records_what_a_reader_needs_to_check(tmp_path, instances):
    ds, _ = _write_dataset(tmp_path, instances)
    assert _run(ds, [50, 100], tmp_path / "out").returncode == 0
    man = json.loads((tmp_path / "out" / "ds.subsets.json").read_text())
    assert man["nested"] is True
    assert man["source_rows"] == 400 and man["seed"] == 20260821
    assert len(man["source_sha256"]) == 64
    for n in ("50", "100"):
        entry = man["subsets"][n]
        assert entry["rows"] == int(n)
        assert len(entry["sha256"]) == 64
        assert entry["letter_counts"], "letter composition must be recorded"
        # the sha256 must be of the file actually written, not of anything else
        import hashlib
        assert entry["sha256"] == hashlib.sha256(
            Path(entry["path"]).read_bytes()).hexdigest()


def test_a_subset_larger_than_the_dataset_is_refused(tmp_path, instances):
    """Silently returning the whole dataset would put a duplicate point on the
    curve wearing a smaller N's label."""
    ds, _ = _write_dataset(tmp_path, instances, n=100)
    r = _run(ds, [500], tmp_path / "out")
    assert r.returncode != 0
    assert "cannot take subsets" in r.stderr


def test_a_nonpositive_subset_is_refused(tmp_path, instances):
    ds, _ = _write_dataset(tmp_path, instances)
    assert _run(ds, [0], tmp_path / "out").returncode != 0
    assert _run(ds, [-10], tmp_path / "out").returncode != 0


def test_a_missing_dataset_is_refused(tmp_path):
    r = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "nope.jsonl"),
         "--sizes", "10", "--out-dir", str(tmp_path / "out")],
        capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert r.returncode != 0
    assert "not found" in r.stderr
