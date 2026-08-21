"""``letter_mix``: weighting banned_letter REQUESTS so the KEPT set is not junk.

The bug these tests pin down is not a crash. v1 requested every banned letter
equally, the rejection filter in :mod:`datagen.generate` threw away 83.8% of the
``e`` drafts and 27.1% of the ``m`` drafts, and the surviving training set came
out l=83, m=78, p=71 ... a=33, e=32. The quality gate had quietly built a
curriculum of easy examples, and the model then satisfied 1 of 105
``banned_letter`` eval instances. Nothing in the pipeline looked wrong.

So the tests here are mostly about DISTRIBUTIONS rather than return values:

* the weights are derived, normalised, and cover every letter in the library;
* ``uniform`` is byte-for-byte what the sampler did before the parameter
  existed -- pinned against digests captured from the pre-change code, because
  "the default is a no-op" is a claim about exact output, not about intent;
* ``hard`` actually moves the request histogram, and only for ``banned_letter``;
* multiplying the ``hard`` request histogram by the MEASURED keep rates lands on
  English letter frequency, which is the number the intervention is for;
* both mixes stay deterministic under a fixed seed;
* an unknown mix name raises instead of quietly reverting to the easy letters.

Network-free: :func:`build_scenarios` never calls a teacher, and the one
end-to-end CLI test scripts the teacher with the corpus witness builder.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scenarios.paths import load_corpus, load_instances, load_smoke
from scenarios.reachability import build_witness

from datagen import DatagenError
from datagen import generate as G
from datagen import sampling as S

INSTANCES = load_instances()
CORPUS = list(load_corpus()) + list(load_smoke())

#: Letters in descending English frequency, for readable failure output.
ORDER = ("e", "a", "o", "i", "s", "r", "l", "u", "m", "p")

#: Big enough that a 10-way histogram is not noise, cheap enough to run in the
#: suite: build_scenarios(6000) takes about 0.1s and calls nothing external.
BIG = 6000


def digest(samples):
    """A stable fingerprint of a whole sampled set -- every field of every Sample."""
    blob = json.dumps(
        [[s.scenario, s.task_id, s.lexicon, s.nudge_id, list(s.types), s.distractor]
         for s in samples],
        sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def letter_histogram(samples):
    """``letter -> times requested`` across a sampled set."""
    counts: Counter = Counter()
    for sample in samples:
        for cid in sample.scenario["constraints"]:
            if INSTANCES[cid]["type"] == "banned_letter":
                counts[S.letter_of(cid, INSTANCES)] += 1
    return counts


def shares(counts):
    total = sum(counts.values())
    return {key: counts[key] / total for key in counts}


@pytest.fixture(scope="module")
def histograms():
    """The two request histograms, built once, from one seed, at scale."""
    return {
        mix: letter_histogram(
            S.build_scenarios(BIG, INSTANCES, CORPUS, seed=42, letter_mix=mix))
        for mix in ("uniform", "hard")
    }


# --------------------------------------------------------------------------
# the constant and its derivation
# --------------------------------------------------------------------------

def test_letter_request_weights_are_normalised():
    total = sum(S.LETTER_REQUEST_WEIGHTS.values())
    assert total == pytest.approx(1.0), f"weights sum to {total}, not 1"
    assert all(w > 0 for w in S.LETTER_REQUEST_WEIGHTS.values()), \
        "a zero or negative weight would silently drop a letter from the curriculum"


def test_every_banned_letter_instance_in_the_library_has_a_request_weight():
    """A letter with no weight is the original bug: unweighted means uniform."""
    library = {S.letter_of(cid, INSTANCES)
               for cid, con in INSTANCES.items()
               if con.get("type") == "banned_letter"}
    assert library, "the library has no banned_letter instances at all"
    missing = sorted(library - set(S.LETTER_REQUEST_WEIGHTS))
    assert not missing, (
        f"banned_letter instances ban {missing}, which LETTER_REQUEST_WEIGHTS does "
        f"not cover. Under letter_mix='hard' those letters would need a measured "
        f"keep rate before they can be requested.")


def test_the_weight_tables_line_up_with_each_other():
    assert set(S.ENGLISH_LETTER_FREQUENCY) == set(S.V1_LETTER_OUTCOMES)
    assert set(S.LETTER_KEEP_RATES) == set(S.V1_LETTER_OUTCOMES)
    assert set(S.LETTER_REQUEST_WEIGHTS) == set(S.V1_LETTER_OUTCOMES)
    assert set(S.ENGLISH_LETTER_FREQUENCY) == set(ORDER)


@pytest.mark.parametrize("letter,reject_rate", [
    ("e", 0.838), ("a", 0.836), ("i", 0.720), ("o", 0.702), ("r", 0.691),
    ("u", 0.642), ("s", 0.533), ("l", 0.291), ("p", 0.276), ("m", 0.271),
])
def test_keep_rates_reproduce_the_measured_reject_rates(letter, reject_rate):
    """The docstring's table and the code's numbers are the same measurement."""
    assert S.LETTER_KEEP_RATES[letter] == pytest.approx(1.0 - reject_rate, abs=0.001)


def test_weights_are_frequency_divided_by_keep_rate_then_normalised():
    raw = {letter: S.ENGLISH_LETTER_FREQUENCY[letter] / S.LETTER_KEEP_RATES[letter]
           for letter in S.ENGLISH_LETTER_FREQUENCY}
    total = sum(raw.values())
    for letter, value in raw.items():
        assert S.LETTER_REQUEST_WEIGHTS[letter] == pytest.approx(value / total), letter


def test_dividing_by_the_keep_rate_is_what_makes_the_weights_non_trivial():
    """Without the division the weights would just be English frequency.

    Stated as a test because it is the whole mechanism: the correction for the
    filter's bias against difficulty is the division, not the frequency table.
    """
    english = {letter: freq / sum(S.ENGLISH_LETTER_FREQUENCY.values())
               for letter, freq in S.ENGLISH_LETTER_FREQUENCY.items()}
    assert S.LETTER_REQUEST_WEIGHTS["e"] > english["e"] * 1.5, \
        "the hard letters must be over-requested relative to their frequency"
    assert S.LETTER_REQUEST_WEIGHTS["m"] < english["m"] * 0.5, \
        "the easy letters must be under-requested relative to their frequency"


def test_the_weights_project_exactly_onto_english_frequency():
    """kept ~ request x keep_rate, and request ~ frequency / keep_rate, so the
    keep rate cancels. In expectation this is exact, not approximate."""
    projected = {letter: S.LETTER_REQUEST_WEIGHTS[letter] * S.LETTER_KEEP_RATES[letter]
                 for letter in S.LETTER_REQUEST_WEIGHTS}
    total = sum(projected.values())
    english_total = sum(S.ENGLISH_LETTER_FREQUENCY.values())
    for letter in ORDER:
        assert projected[letter] / total == pytest.approx(
            S.ENGLISH_LETTER_FREQUENCY[letter] / english_total, abs=1e-9), letter


# --------------------------------------------------------------------------
# uniform is the default, and it is a no-op
# --------------------------------------------------------------------------

#: Digests of ``build_scenarios(n, INSTANCES, CORPUS, seed=seed)`` captured by
#: RUNNING THE SAMPLER BEFORE ``letter_mix`` existed. This is the evidence that
#: adding the parameter changed nothing for every caller that does not pass it:
#: "the default is uniform" is a claim about exact output, and only a pin taken
#: from the old code can test it. If a deliberate change to the sampler ever
#: makes these fail, that change is not backward compatible and the seeds in
#: data/train are no longer reproducible -- re-pin knowingly, never reflexively.
PRE_CHANGE_DIGESTS = {
    (99, 32): "8c862b217ff005599fe9b78e19791fd05670d23ec4961c2f3e1a7d795d21f495",
    (99, 256): "b0f0af31b036944eec58002e7068464102b4eeb98d936451d36ccfcd8e1c69b2",
    (20260820, 32): "3cfabe5e0fa50e036335933a0132e8cbf15174cc57a2e1ecf5738cd77356cf51",
    (20260820, 256): "1a787650d34be14e34487b6078e40f99ad36c45f487311ccc010dcf5f7f142f6",
    (7, 32): "4bd360489e885ebd14a9f86b5efb627a909346d8f7475dac262f0371c0c0d03b",
    (7, 256): "1207a13c10a686193d7dc4c53a0f6f30477466f268b66f1eab84cd2b3b34cede",
    (3, 32): "3df1a2ce488f0871005455c23e275bbe0f1518db1959947c493aa186c8b79dbe",
    (3, 256): "62f3eb730ad09643408d697aeb39666fa16ad99366b63625c08617fda9a7ca47",
}


@pytest.mark.parametrize("seed,n", sorted(PRE_CHANGE_DIGESTS))
def test_the_default_reproduces_the_pre_change_sampler_byte_for_byte(seed, n):
    got = digest(S.build_scenarios(n, INSTANCES, CORPUS, seed=seed))
    assert got == PRE_CHANGE_DIGESTS[(seed, n)], (
        f"seed={seed} n={n}: the default no longer reproduces the sampler as it "
        f"behaved before letter_mix was added")


@pytest.mark.parametrize("seed,n", sorted(PRE_CHANGE_DIGESTS))
def test_passing_uniform_explicitly_is_the_same_as_the_default(seed, n):
    assert (digest(S.build_scenarios(n, INSTANCES, CORPUS, seed=seed,
                                     letter_mix="uniform"))
            == digest(S.build_scenarios(n, INSTANCES, CORPUS, seed=seed)))


def test_uniform_requests_every_letter_about_equally(histograms):
    counts = histograms["uniform"]
    assert set(counts) == set(ORDER)
    assert max(counts.values()) / min(counts.values()) < 1.4, (
        f"uniform is meant to be flat, got {dict(counts)}")


# --------------------------------------------------------------------------
# hard actually bites
# --------------------------------------------------------------------------

def test_hard_shifts_the_request_histogram_towards_the_hard_letters(histograms):
    hard, uniform = histograms["hard"], histograms["uniform"]
    assert set(hard) == set(ORDER), f"a letter was dropped entirely: {sorted(hard)}"

    # e and a -- the two the eval leans on and the filter destroys -- must
    # dominate; l, m and p -- the teacher's favourites -- must recede.
    for hard_letter in ("e", "a"):
        for easy_letter in ("l", "m", "p"):
            assert hard[hard_letter] > 4 * hard[easy_letter], (
                f"{hard_letter}={hard[hard_letter]} is not far enough above "
                f"{easy_letter}={hard[easy_letter]}")
    assert hard["e"] == max(hard.values()), "e must be the most requested letter"

    # And each letter must move in the direction its weight says.
    for letter in ORDER:
        moved_up = hard[letter] > uniform[letter]
        wanted_up = S.LETTER_REQUEST_WEIGHTS[letter] > 1.0 / len(ORDER)
        assert moved_up == wanted_up, (
            f"{letter}: uniform={uniform[letter]} hard={hard[letter]} but weight "
            f"{S.LETTER_REQUEST_WEIGHTS[letter]:.4f} vs uniform 0.1")


def test_the_hard_request_histogram_tracks_the_declared_weights(histograms):
    observed = shares(histograms["hard"])
    for letter in ORDER:
        assert observed[letter] == pytest.approx(
            S.LETTER_REQUEST_WEIGHTS[letter], abs=0.02), (
            f"{letter}: sampled {observed[letter]:.4f}, weight "
            f"{S.LETTER_REQUEST_WEIGHTS[letter]:.4f}")


def test_the_projected_kept_distribution_lands_on_english_frequency(histograms):
    """The number that matters: request histogram x MEASURED keep rate.

    v1's uniform requests projected to l=17.6% / m=16.9% / p=16.0% kept against
    e=3.9%, which is the curriculum that failed. Under 'hard' the same
    arithmetic lands within a couple of points of English frequency.
    """
    hard = histograms["hard"]
    projected = {letter: hard[letter] * S.LETTER_KEEP_RATES[letter] for letter in ORDER}
    projected_share = shares(projected)
    english_total = sum(S.ENGLISH_LETTER_FREQUENCY.values())

    for letter in ORDER:
        english = S.ENGLISH_LETTER_FREQUENCY[letter] / english_total
        assert projected_share[letter] == pytest.approx(english, abs=0.03), (
            f"{letter}: projected kept share {projected_share[letter]:.2%} vs "
            f"English {english:.2%}")

    # The comparison that says the intervention was needed at all.
    uniform_projected = shares({letter: histograms["uniform"][letter]
                                * S.LETTER_KEEP_RATES[letter] for letter in ORDER})
    assert uniform_projected["e"] < 0.06, "sanity: uniform starves e"
    assert projected_share["e"] > 3 * uniform_projected["e"]


def test_hard_leaves_every_other_constraint_type_uniform(histograms):
    """Only banned_letter is weighted. banned_phrases has eight interchangeable
    instances and a task-independent pool, so it is the clean control: it must
    stay as flat under 'hard' as it is under 'uniform'."""
    spread = {}
    for mix in ("uniform", "hard"):
        per_type = defaultdict(Counter)
        for sample in S.build_scenarios(BIG, INSTANCES, CORPUS, seed=42,
                                        letter_mix=mix):
            for cid in sample.scenario["constraints"]:
                per_type[INSTANCES[cid]["type"]][cid] += 1
        spread[mix] = {ctype: max(c.values()) / min(c.values())
                       for ctype, c in per_type.items() if len(c) > 1}

    assert spread["hard"]["banned_phrases"] < 1.4, spread["hard"]
    assert spread["hard"]["banned_phrases"] == pytest.approx(
        spread["uniform"]["banned_phrases"], abs=0.25)
    assert spread["hard"]["banned_letter"] > 5, \
        "banned_letter is the one type that must NOT be flat under 'hard'"


# --------------------------------------------------------------------------
# determinism, and refusing to guess
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mix", ["uniform", "hard"])
def test_both_mixes_are_deterministic_under_a_fixed_seed(mix):
    first = S.build_scenarios(128, INSTANCES, CORPUS, seed=2026, letter_mix=mix)
    second = S.build_scenarios(128, INSTANCES, CORPUS, seed=2026, letter_mix=mix)
    assert digest(first) == digest(second)
    assert [s.scenario for s in first] == [s.scenario for s in second]

    other_seed = S.build_scenarios(128, INSTANCES, CORPUS, seed=2027, letter_mix=mix)
    assert digest(other_seed) != digest(first), "the seed must still matter"


def test_the_two_mixes_differ_at_the_same_seed():
    """Guards the failure where letter_mix is accepted and then ignored."""
    assert (digest(S.build_scenarios(256, INSTANCES, CORPUS, seed=5, letter_mix="hard"))
            != digest(S.build_scenarios(256, INSTANCES, CORPUS, seed=5,
                                        letter_mix="uniform")))


@pytest.mark.parametrize("mix", ["easy", "Uniform", "HARD", "", "default", None, 1])
def test_an_unknown_letter_mix_raises_rather_than_falling_back(mix):
    with pytest.raises(DatagenError, match="unknown letter_mix"):
        S.build_scenarios(8, INSTANCES, CORPUS, seed=1, letter_mix=mix)


def test_hard_refuses_a_letter_it_has_no_measured_keep_rate_for(monkeypatch):
    """A new instance banning an unmeasured letter must stop the run, not be
    quietly handed a uniform share.

    The instance is given authoring metadata too, so the run gets far enough to
    reach the weighting: without it the sampler's pre-existing lexicon guard
    would fire first and the test would pass for the wrong reason.
    """
    library = dict(INSTANCES)
    library["c-chr-999"] = {"id": "c-chr-999", "type": "banned_letter",
                            "params": {"letter": "z"}, "family": "character",
                            "split": "train"}
    monkeypatch.setitem(S.INSTANCE_META, "c-chr-999", S.INSTANCE_META["c-chr-001"])

    with pytest.raises(DatagenError, match="no measured keep rate"):
        S.build_scenarios(8, library, CORPUS, seed=1, letter_mix="hard")
    # ...and uniform is unaffected, because it was never claiming to weight.
    assert len(S.build_scenarios(8, library, CORPUS, seed=1,
                                 letter_mix="uniform")) == 8


def test_letter_of_refuses_an_instance_with_no_letter():
    with pytest.raises(DatagenError, match="no 'letter' parameter"):
        S.letter_of("c-fmt-001", INSTANCES)
    with pytest.raises(DatagenError, match="not in the instance library"):
        S.letter_of("c-nope-000", INSTANCES)


def test_letter_mixes_is_the_advertised_set():
    assert S.LETTER_MIXES == ("uniform", "hard")
    assert S.LETTER_MIXES[0] == "uniform", "the default must be listed first"


# --------------------------------------------------------------------------
# the realised-distribution report
# --------------------------------------------------------------------------

def _row(cids):
    return {"constraints": list(cids)}


def test_letter_histograms_counts_kept_rejected_and_the_keep_rate():
    #  c-chr-001 bans 'e', c-chr-009 bans 'm', c-fmt-001 is no_lists.
    rows = [_row(["c-chr-001", "c-fmt-001"]),
            _row(["c-chr-009", "c-fmt-001"]),
            _row(["c-chr-009", "c-fmt-001"]),
            _row(["c-fmt-001", "c-fmt-002"])]
    rejections = [_row(["c-chr-001", "c-fmt-001"])] * 6 + [_row(["c-fmt-001"])]

    got = G.letter_histograms(rows, rejections, INSTANCES)
    assert got["kept"] == {"e": 1, "m": 2}
    assert got["rejected"] == {"e": 6, "m": 0}
    assert got["keep_rate"] == {"e": pytest.approx(1 / 7, abs=1e-4), "m": 1.0}


def test_letter_histograms_is_empty_rather_than_wrong_with_no_letter_rows():
    got = G.letter_histograms([_row(["c-fmt-001"])], [], INSTANCES)
    assert got == {"kept": {}, "rejected": {}, "keep_rate": {}}


def test_letter_histograms_reproduces_the_v1_measurement_shape():
    """Counted the same way V1_LETTER_OUTCOMES was, so the two are comparable."""
    letters = {cid for cid, con in INSTANCES.items()
               if con.get("type") == "banned_letter"}
    rows, rejections = [], []
    for letter, (kept, rejected) in S.V1_LETTER_OUTCOMES.items():
        cid = next(c for c in letters if INSTANCES[c]["params"]["letter"] == letter)
        rows.extend([_row([cid])] * kept)
        rejections.extend([_row([cid])] * rejected)

    got = G.letter_histograms(rows, rejections, INSTANCES)
    assert got["kept"] == {L: S.V1_LETTER_OUTCOMES[L][0] for L in sorted(ORDER)}
    assert got["rejected"] == {L: S.V1_LETTER_OUTCOMES[L][1] for L in sorted(ORDER)}
    for letter in ORDER:
        assert got["keep_rate"][letter] == pytest.approx(
            S.LETTER_KEEP_RATES[letter], abs=1e-4), letter


def test_summarise_carries_the_banned_letter_distribution():
    rows = [{"constraints": ["c-chr-001", "c-fmt-001"],
             "provenance": {"attempt": 1, "task_id": "t-1", "distractor": False},
             "scenario": {"k": 2, "pressure": "clean"}}]
    report = G.summarise(rows, [], [], [], INSTANCES, requested=1)
    assert report["banned_letter_distribution"]["kept"] == {"e": 1}
    assert report["banned_letter_distribution"]["keep_rate"] == {"e": 1.0}


# --------------------------------------------------------------------------
# the CLI flag, end to end, with a scripted teacher
# --------------------------------------------------------------------------

class WitnessTeacher:
    """Returns a genuinely constraint-satisfying paragraph. No network."""

    thread_safe = True

    def __call__(self, scenario, prompt, constraint_lookup=None):
        cons = [constraint_lookup[c] for c in scenario["constraints"]]
        witness = build_witness(cons, sentences=4)
        assert witness is not None, f"no witness for {scenario['constraints']}"
        return witness


@pytest.mark.parametrize("mix", ["uniform", "hard"])
def test_the_cli_threads_letter_mix_through_and_records_it(tmp_path, monkeypatch, mix):
    import eval as E
    monkeypatch.setattr(E, "make_generator", lambda spec: WitnessTeacher())
    out = tmp_path / f"{mix}.jsonl"
    code = G.main(["--n", "12", "--out", str(out), "--teacher", "mock:compliant",
                   "--attempts", "1", "--no-cache", "--workers", "1",
                   "--letter-mix", mix])
    assert code == 0
    report = json.loads(G.sidecar_paths(out)["report"].read_text())
    assert report["letter_mix"] == mix
    assert set(report["banned_letter_distribution"]) == {"kept", "rejected", "keep_rate"}

    # The report's histogram is the realised one, recomputed from the rows.
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert (report["banned_letter_distribution"]["kept"]
            == G.letter_histograms(rows, [], INSTANCES)["kept"])


def test_the_cli_defaults_to_uniform(tmp_path, monkeypatch):
    import eval as E
    monkeypatch.setattr(E, "make_generator", lambda spec: WitnessTeacher())
    out = tmp_path / "default.jsonl"
    assert G.main(["--n", "12", "--out", str(out), "--teacher", "mock:compliant",
                   "--attempts", "1", "--no-cache", "--workers", "1"]) == 0
    report = json.loads(G.sidecar_paths(out)["report"].read_text())
    assert report["letter_mix"] == "uniform"


def test_the_cli_rejects_an_unknown_letter_mix(tmp_path):
    with pytest.raises(SystemExit) as exc:
        G.main(["--n", "4", "--out", str(tmp_path / "x.jsonl"),
                "--letter-mix", "easy"])
    assert exc.value.code == 2
