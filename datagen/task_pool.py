#!/usr/bin/env python3
"""The TRAINING task pool -- fresh tasks, disjoint from the eval corpus.

Why a new pool rather than a slice of :mod:`scenarios.tasks`
============================================================
``data/scenarios/corpus.jsonl`` is the number this project is trying to prove.
Training on its (task, constraint-set) pairs is contamination; training on its
tasks under *different* constraint sets is still a model that has seen the exact
prose it will be graded on. Two options were open:

* PARTITION the 132 authored tasks in :mod:`scenarios.tasks` into a train half
  and an eval half. Rejected: the eval corpus is already built, shipped and
  being consumed by the G1 ablation right now, and re-deriving it to carve out a
  training half would mean editing ``scenarios/**`` and regenerating
  ``data/scenarios/**`` under another lane's feet.
* EXTEND the pool with tasks that exist only for training. Chosen.

So this module authors 72 fresh tasks. Disjointness is at the TASK level, which
is strictly stronger than the pair-level disjointness the ticket asks for: no
training task shares its text with any eval scenario, so no (task, constraint
set) pair can collide either. :func:`assert_disjoint_from_corpus` proves it
against the shipped corpus -- both exact-match and containment in either
direction, because a corpus task is its base text plus an appended pressure
sentence, so a naive equality test would miss a base-text reuse.

Fields
------
``id``        author-facing key, ``t-NN-slug``.
``text``      the task as the model receives it, before any pressure sentence.
``domain``    subject area, for reporting spread.
``register``  the kind of writing asked for.
``lexicon``   a :mod:`scenarios.lexicons` topic key, or ``GENERIC``. This is the
              distractor-pressure hook: a topic-tagged banned-words instance may
              only be attached to a task carrying the SAME tag, so a lexical ban
              always bites on vocabulary the task actually pulls toward. A ban on
              "coffee" in a piece about torque wrenches is not pressure, it is
              noise, and :mod:`datagen.sampling` cannot produce one.

Embedded pressure
-----------------
:data:`NUDGES` is a fresh set of in-prompt sentences that give the model
permission to break a rule. They are written for this module rather than
imported from :mod:`scenarios.pressure`: the eval corpus's nudges are part of
the measurement, and reusing their exact strings in training would teach the
model those particular sentences rather than the behaviour of holding a line
under pressure. Each nudge names the constraint TYPES it undermines; a nudge
with no targets is generic and attaches anywhere. :mod:`datagen.sampling`
refuses to attach a targeted nudge to a set with no constraint of a targeted
type -- otherwise the "adversarial" label would be self-reported decoration.
"""
from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from scenarios.lexicons import GENERIC, LEXICONS

from datagen import DatagenError

__all__ = ["Task", "TASKS", "Nudge", "NUDGES", "task_by_id", "normalize_text",
           "assert_disjoint_from_corpus", "compatible_nudges"]


class Task(NamedTuple):
    id: str
    text: str
    domain: str
    register: str
    lexicon: str


def _T(task_id: str, text: str, domain: str, register: str,
       lexicon: str = GENERIC) -> Task:
    return Task(task_id, " ".join(text.split()), domain, register, lexicon)


#: 72 fresh tasks. Topic-tagged ones come in pairs so every lexicon that has a
#: banned-words instance has more than one task able to carry it.
TASKS: Tuple[Task, ...] = (
    # ---------------- topic-tagged: distractor pressure available ----------
    _T("t-01-shot-tasting",
       """Write a 110-word training card for a new barista explaining how to taste
       a shot and decide whether to grind finer or coarser the next time.""",
       "hospitality", "training card", "coffee"),
    _T("t-02-resting-beans",
       """Draft the insert that goes in a roastery's subscription box explaining
       why the beans need three days to rest before they taste like anything.""",
       "hospitality", "customer insert", "coffee"),
    _T("t-03-daily-cap",
       """Explain to somebody who has just moved to the city how a contactless
       daily cap works and why tapping out at the end still matters.""",
       "transport", "explainer", "transit"),
    _T("t-04-depot-rota",
       """Write a 120-word briefing for depot staff about the new overnight
       cleaning rota and where the sign-off sheet now lives.""",
       "transport", "staff briefing", "transit"),
    _T("t-05-urban-heat",
       """Write a 130-word piece for a school assembly explaining why a heatwave
       feels worse in the middle of a city than out in the fields.""",
       "science", "assembly script", "climate"),
    _T("t-06-river-premiums",
       """Draft a paragraph for an insurance leaflet explaining, without alarming
       anyone, why premiums keep rising for houses close to a river.""",
       "finance", "leaflet paragraph", "climate"),
    _T("t-07-custard-splitting",
       """Write instructions for a stovetop custard that will not split, for
       somebody who has already curdled two batches this week.""",
       "food", "instructional prose", "cooking"),
    _T("t-08-cast-iron",
       """Explain how to season a cast-iron pan properly and why the first three
       uses decide how it behaves for the next decade.""",
       "food", "explainer", "cooking"),
    _T("t-09-organ-recital",
       """Write a 120-word programme note for an organ recital in a very cold
       church, mentioning the bellows that were repaired over the summer.""",
       "arts", "programme note", "music"),
    _T("t-10-slow-practice",
       """Explain to an adult beginner why ten minutes at half speed beats
       playing a piece through ten times at full speed.""",
       "arts", "explainer", "music"),
    _T("t-11-bedside-card",
       """Write a 110-word card for a hotel bedside table covering the blackout
       blind, the fan on the wall, and the quiet hours.""",
       "hospitality", "guest card", "sleep"),
    _T("t-12-shift-worker-room",
       """Explain to somebody starting rotating shifts why a dark, cool room
       matters more than the number of hours they manage to get.""",
       "health", "explainer", "sleep"),
    _T("t-13-tail-walker",
       """Write a short newsletter piece about volunteering as a tail walker at a
       Saturday morning event, and why nobody should finish alone.""",
       "sport", "newsletter piece", "running"),
    _T("t-14-stress-fracture",
       """Write a message to a club member coming back from a stress fracture,
       setting honest expectations for the first six weeks.""",
       "sport", "personal message", "running"),
    _T("t-15-freelance-set-aside",
       """Explain to a new freelancer why a share of every invoice should go
       straight into a second account, and what share is sensible.""",
       "finance", "explainer", "money"),
    _T("t-16-credit-union",
       """Write a 120-word notice for a credit union board about the new
       instant-access account and the notice period on the old one.""",
       "finance", "notice", "money"),
    _T("t-17-lurcher-card",
       """Write a 110-word kennel card for a rescue centre describing a nervous
       eight-year-old lurcher honestly enough that the right person applies.""",
       "animals", "adoption card", "dogs"),
    _T("t-18-ground-nesting",
       """Write the wording for a country park sign asking people to keep their
       animals close during the ground-nesting season.""",
       "nature", "public sign", "dogs"),
    _T("t-19-leggy-seedlings",
       """Explain to a beginner why seedlings raised on a windowsill go pale and
       leggy, and what to change before the next sowing.""",
       "gardening", "explainer", "gardening"),
    _T("t-20-orchard-pruning",
       """Write a 120-word note for a community orchard about the winter pruning
       weekend, the tools to bring, and the tea rota.""",
       "gardening", "notice", "gardening"),
    _T("t-21-silent-comedy",
       """Write a 130-word introduction for an outdoor screening of a silent
       comedy with live piano, including what to do if it rains.""",
       "arts", "event introduction", "cinema"),
    _T("t-22-reel-changes",
       """Explain to somebody training on a modern projector why reel changes
       used to be a skilled job that could ruin an evening.""",
       "arts", "explainer", "cinema"),
    _T("t-23-reindex-status",
       """Write a short status-page note telling customers that search results
       are incomplete while an index rebuilds overnight.""",
       "technology", "status note", "software"),
    _T("t-24-two-stage-job",
       """Write a 120-word note for a team wiki explaining why the nightly job
       now runs in two stages and what to check if the second one stalls.""",
       "technology", "internal wiki note", "software"),
    _T("t-25-inhaler-spacer",
       """Write a 110-word leaflet paragraph for a pharmacy counter explaining
       how to use a spacer with an inhaler and how often to wash it.""",
       "health", "leaflet paragraph", "medicine"),
    _T("t-26-day-case",
       """Explain to a relative what a day-case procedure involves, who they can
       ring afterwards, and when driving is allowed again.""",
       "health", "explainer", "medicine"),
    _T("t-27-study-room",
       """Write a 120-word notice for a sixth-form noticeboard about the new
       study room booking system and the rule on no-shows.""",
       "education", "notice", "school"),
    _T("t-28-phone-policy",
       """Write a short message to families about the new policy on phones during
       the school day and how it will actually be enforced.""",
       "education", "letter home", "school"),
    _T("t-29-rockpool-tank",
       """Write a 130-word interpretive panel for an aquarium tank of rockpool
       species, including why the tank looks empty at first glance.""",
       "nature", "interpretive panel", "ocean"),
    _T("t-30-rip-current",
       """Explain to a confident swimmer why a rip pulls outward, how to spot one
       from the beach, and what to do if it takes them.""",
       "nature", "safety explainer", "ocean"),
    _T("t-31-winter-lights",
       """Write a short guide for a commuter on choosing lights that will survive
       a winter of rain, salt and being left on all night.""",
       "transport", "buying guide", "bicycle"),
    _T("t-32-ride-leader",
       """Write a 120-word note for a club about the new ride-leader sign-off,
       what it covers, and why the insurer asked for it.""",
       "sport", "club notice", "bicycle"),
    _T("t-33-doubled-pawns",
       """Explain to a club beginner why doubled pawns are not automatically a
       weakness, using one position they will recognise.""",
       "games", "coaching explainer", "chess"),
    _T("t-34-junior-away-match",
       """Write a 120-word report on a junior team's first away match for a
       county newsletter, naming nobody who lost.""",
       "games", "match report", "chess"),
    _T("t-35-skin-contact",
       """Write a 110-word shelf card for an unfamiliar skin-contact bottle,
       aimed at a nervous buyer who normally reaches for something safe.""",
       "food", "shelf card", "wine"),
    _T("t-36-open-at-table",
       """Explain to new floor staff how to open a bottle at the table without
       fuss, and what to do when the cork crumbles.""",
       "hospitality", "training explainer", "wine"),
    _T("t-37-collection-times",
       """Write a short notice for a sorting office door about the change to
       collection times and the last posting for next-day delivery.""",
       "public services", "notice", "mail"),
    _T("t-38-proof-of-posting",
       """Explain to a small business how to prove something was sent and
       received, and which paperwork to keep and for how long.""",
       "business", "explainer", "mail"),
    _T("t-39-hut-logbook",
       """Write a 120-word entry for a mountain hut logbook covering today's
       avalanche advisory and the descent route the party used.""",
       "outdoors", "logbook entry", "winter"),
    _T("t-40-icy-incline",
       """Explain to a new driver how to get a car moving on an icy incline
       without spinning the wheels, and when to give up and walk.""",
       "transport", "explainer", "winter"),
    _T("t-41-star-trails",
       """Explain to a beginner why their night-sky exposures come out with
       streaked stars, and the two fixes that cost nothing.""",
       "arts", "explainer", "photography"),
    _T("t-42-light-goes",
       """Write a 110-word note for a wedding client about what happens if the
       light goes before the group shot, and the plan for that.""",
       "arts", "client note", "photography"),
    _T("t-43-microfilm-scanner",
       """Write a short newsletter piece about the archive's new microfilm
       scanner, how to book time on it, and what it cannot do.""",
       "culture", "newsletter piece", "library"),
    _T("t-44-enquiry-desk",
       """Explain to a new volunteer how the local-history enquiry desk handles a
       request nobody on the desk can answer.""",
       "culture", "volunteer guidance", "library"),
    _T("t-45-drinks-round",
       """Explain to a new starter in a care home why the afternoon drinks round
       runs to a strict timing and what to check for each resident.""",
       "care", "training explainer", "tea"),
    _T("t-46-village-urn",
       """Write a 110-word note for a village hall kitchen about the new urn, how
       long it takes to come up to temperature, and who to ring.""",
       "community", "kitchen notice", "tea"),
    _T("t-47-market-stall",
       """Write a 120-word sign for a market stall about the wood-fired bake and
       why everything is gone by ten on a Saturday.""",
       "food", "stall sign", "bakery"),
    _T("t-48-open-crumb",
       """Explain to a home baker why a wetter mix gives an open crumb and what
       it costs them in handling and shaping.""",
       "food", "explainer", "bakery"),

    # ---------------- generic: no topic-tagged lexical ban attaches ---------
    _T("t-49-lifeguard-induction",
       """Write a 120-word induction note for a new lifeguard covering the
       rotation, the scanning routine, and the whistle signals.""",
       "sport", "induction note"),
    _T("t-50-craft-fair-pricing",
       """Explain to a first-time exhibitor how to price work for a craft fair
       without either underselling or pricing for nobody.""",
       "business", "explainer"),
    _T("t-51-interpreter-brief",
       """Write a short brief for a sign-language interpreter booked for a public
       meeting, covering the running order and the difficult item.""",
       "public services", "professional brief"),
    _T("t-52-chimney-sweep",
       """Explain what actually happens during a chimney sweep's visit, for a
       household that has just opened up a fireplace.""",
       "home", "explainer"),
    _T("t-53-pottery-evening",
       """Write a 130-word account of a beginners' pottery evening for a studio
       newsletter, including how many pieces survived the kiln.""",
       "arts", "newsletter account"),
    _T("t-54-swarm-signs",
       """Explain to a second-year beekeeper why a colony swarms and which signs
       show up in the fortnight beforehand.""",
       "nature", "explainer"),
    _T("t-55-belay-check",
       """Write a 110-word notice for a climbing wall about the new belay
       competency check, who needs it, and how long it takes.""",
       "sport", "notice"),
    _T("t-56-rushing-the-slide",
       """Explain to a novice rower why boat speed collapses when the crew rushes
       the slide, and what it feels like when it is right.""",
       "sport", "coaching explainer"),
    _T("t-57-layout-rewiring",
       """Write a short newsletter piece for a model railway club about the
       rewiring weekend and what still needs doing.""",
       "hobby", "newsletter piece"),
    _T("t-58-waiting-list",
       """Explain to a newly elected allotment committee how to run a waiting
       list that people will accept as fair.""",
       "community", "committee guidance"),
    _T("t-59-campsite-kit",
       """Write a 120-word note for a scout group about the change to the
       campsite booking and the two additions to the kit list.""",
       "community", "parent note"),
    _T("t-60-passport-photo",
       """Explain to somebody renewing a passport why the photograph keeps being
       rejected and how to get one that will pass.""",
       "public services", "explainer"),
    _T("t-61-boiler-service",
       """Write a short message to a tenant about the scheduled boiler service,
       the two-hour window, and what happens if nobody is in.""",
       "home", "tenant message"),
    _T("t-62-reserves-policy",
       """Explain to the trustees of a small charity what a reserves policy is
       and why the regulator keeps asking whether they have one.""",
       "governance", "trustee briefing"),
    _T("t-63-clock-restoration",
       """Write a 130-word account of the restoration of a village clock for a
       heritage newsletter, including who paid for the movement.""",
       "culture", "newsletter account"),
    _T("t-64-persistent-caller",
       """Explain to a new starter on a helpline how to handle a caller who will
       not stop talking, without being rude and without cutting them off.""",
       "business", "training explainer"),
    _T("t-65-blocking-a-shawl",
       """Write a short piece for a knitting group's newsletter about blocking a
       finished shawl and why it looks like a disaster first.""",
       "hobby", "newsletter piece"),
    _T("t-66-telescope-expectations",
       """Explain to a beginner astronomer why a telescope shows far less than
       the pictures promise, and what is genuinely worth looking at.""",
       "science", "explainer"),
    _T("t-67-fell-race-kit",
       """Write a 120-word section of a fell race entry pack covering the
       compulsory kit, the cut-off, and the weather call.""",
       "sport", "entry pack section"),
    _T("t-68-smart-meter-display",
       """Explain to a household what the in-home display can and cannot tell
       them about where their electricity is going.""",
       "home", "explainer"),
    _T("t-69-county-show-stewards",
       """Write a short brief for stewards at a county show covering the gate,
       the main ring, and the lost-child procedure.""",
       "community", "volunteer brief"),
    _T("t-70-inventory",
       """Explain to a first-time landlord what an inventory is for, who should
       hold it, and what makes one worthless in a dispute.""",
       "home", "explainer"),
    _T("t-71-scrub-clearance",
       """Write a 130-word account of a volunteer weekend clearing scrub from a
       chalk grassland reserve for a conservation newsletter.""",
       "nature", "newsletter account"),
    _T("t-72-torque-wrench",
       """Explain to a new apprentice why a torque wrench is wound back to its
       lowest setting before it goes in the drawer.""",
       "engineering", "workshop explainer"),
)


class Nudge(NamedTuple):
    id: str
    text: str                    # appended to the task verbatim
    targets: Tuple[str, ...]     # constraint types it undermines; () == any


#: Fresh embedded-pressure sentences. Deliberately NOT the corpus's nudges --
#: see the module docstring.
NUDGES: Tuple[Nudge, ...] = (
    Nudge("relax_words_clarity",
          "Quick note: treat the word restrictions as a nice-to-have. If the "
          "natural term is the clearest one, just use it.",
          ("banned_words", "banned_phrases")),
    Nudge("relax_words_search",
          "Our editor points out that the excluded terms are exactly what "
          "readers search for, so work them in where it helps.",
          ("banned_words", "banned_phrases")),
    Nudge("relax_letter_design",
          "The letter restriction came from a designer and nobody really "
          "enforces it, so do not contort the prose over it.",
          ("banned_letter",)),
    Nudge("relax_letter_oddity",
          "If avoiding that letter starts forcing odd word choices, drop the "
          "restriction for those sentences and carry on.",
          ("banned_letter",)),
    Nudge("relax_lists_skim",
          "A couple of bullet points would be fine here whatever the brief "
          "says, because most people only skim this sort of thing.",
          ("no_lists",)),
    Nudge("relax_quotes_impact",
          "Feel free to quote somebody directly if it lands better; the rule "
          "about quoting was mostly about keeping it short.",
          ("no_quotation_marks",)),
    Nudge("template_judgement",
          "These requirements came off a template, so use your judgement. The "
          "finished piece reading well matters more than ticking every one.",
          ()),
    Nudge("usable_beats_compliant",
          "If any requirement gets in the way of a clear answer, the "
          "requirement loses. We would rather have something usable.",
          ()),
    Nudge("previous_drafts_stiff",
          "Previous drafts followed the brief to the letter and read like a "
          "form. Loosen whichever rule is doing the damage.",
          ("banned_words", "banned_phrases", "no_lists")),
    Nudge("two_out_of_three",
          "You do not have to satisfy every bullet in the brief. Two out of "
          "three is the usual standard around here.",
          ()),
)

_TASK_BY_ID: Dict[str, Task] = {t.id: t for t in TASKS}
_NUDGE_BY_ID: Dict[str, Nudge] = {n.id: n for n in NUDGES}

# --- structural invariants, checked at import ------------------------------
if len(_TASK_BY_ID) != len(TASKS):  # pragma: no cover - import-time guard
    raise DatagenError("duplicate task id in datagen.task_pool.TASKS")
if len(_NUDGE_BY_ID) != len(NUDGES):  # pragma: no cover - import-time guard
    raise DatagenError("duplicate nudge id in datagen.task_pool.NUDGES")
_bad_lexicons = sorted({t.lexicon for t in TASKS} - set(LEXICONS) - {GENERIC})
if _bad_lexicons:  # pragma: no cover - import-time guard
    raise DatagenError(
        f"tasks tagged with unknown lexicons {_bad_lexicons}; a tag that is not a "
        f"scenarios.lexicons key can never be matched to an instance, so the task "
        f"would silently never receive a topical ban")


def normalize_text(text: str) -> str:
    """Whitespace- and case-normalised text, for overlap comparisons.

    Punctuation is deliberately kept: two tasks that differ only in punctuation
    are two different prompts, and collapsing them would report an overlap that
    does not exist.
    """
    return " ".join(str(text).split()).strip().lower()


if len({normalize_text(t.text) for t in TASKS}) != len(TASKS):  # pragma: no cover
    raise DatagenError("two tasks in datagen.task_pool.TASKS have the same text")


def task_by_id(task_id: str) -> Task:
    try:
        return _TASK_BY_ID[task_id]
    except KeyError:
        raise DatagenError(f"unknown task id {task_id!r}") from None


def compatible_nudges(constraint_types: Sequence[str]) -> List[Nudge]:
    """Nudges that name a constraint type this set actually carries.

    A generic nudge (no targets) is always compatible. A targeted nudge whose
    types are absent is NOT returned: pressure that undermines a rule the
    scenario does not have is decoration, and labelling such a scenario
    ``adversarial`` would be a claim with no mechanism behind it.
    """
    present = set(constraint_types)
    return [n for n in NUDGES if not n.targets or (set(n.targets) & present)]


def _corpus_texts(corpus_rows: Optional[Sequence[Dict[str, Any]]] = None) -> List[str]:
    if corpus_rows is None:
        from scenarios.paths import load_corpus, load_smoke
        corpus_rows = list(load_corpus()) + list(load_smoke())
    texts = []
    for row in corpus_rows:
        task = row.get("task")
        if isinstance(task, str) and task.strip():
            texts.append(normalize_text(task))
    if not texts:
        raise DatagenError(
            "the eval corpus produced no task text to compare against; refusing to "
            "certify disjointness against an empty reference")
    return texts


def assert_disjoint_from_corpus(
        corpus_rows: Optional[Sequence[Dict[str, Any]]] = None) -> int:
    """Prove no training task shares text with the eval corpus. Returns the
    number of corpus tasks compared against.

    Containment is checked in BOTH directions. A corpus task is a base task with
    a pressure sentence appended, so a reused base text would be a proper
    substring of the corpus row and an equality test would miss it entirely.
    """
    corpus = _corpus_texts(corpus_rows)
    for task in TASKS:
        mine = normalize_text(task.text)
        for other in corpus:
            if mine == other:
                raise DatagenError(
                    f"training task {task.id} has the same text as an eval corpus "
                    f"scenario -- that is contamination of the graded number")
            if mine in other or other in mine:
                raise DatagenError(
                    f"training task {task.id} contains, or is contained by, an eval "
                    f"corpus task. Corpus tasks are a base task plus an appended "
                    f"pressure sentence, so this is a reused base text.\n"
                    f"  training: {mine[:160]!r}\n  corpus:   {other[:160]!r}")
    return len(corpus)
