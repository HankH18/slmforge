#!/usr/bin/env python3
"""The authored task corpus: 150 task/constraint pairings (Deliverable 2).

Every entry is a task a person would plausibly send, plus the constraint set it
is paired with. Nothing here is sampled -- distractor pressure is a claim about
meaning, and pairing at random produces the exact failure the ticket warns
about ("banning 'zygote' from an essay about transit does nothing"). The
pairing is the work; :mod:`scenarios.generator` is what makes it impossible to
get wrong quietly.

Field contract, all validated by the generator:

``id``           author-facing key. Not the scenario_id -- that is positional.
``text``         the BASE task, before any embedded-pressure sentence.
``domain``       subject area, for reporting spread.
``register``     the kind of writing asked for (email, notice, review, ...).
``constraints``  instance ids from :mod:`scenarios.instances`.
``pressure``     ``"clean"`` or ``"adversarial"``.
``lexicon``      the topic tag, required whenever a topic-tagged word ban is
                 attached; ``None`` for tasks that use only generic bans.
``key_terms``    unavoidable nouns of the task, each of which must literally
                 appear in ``text``. Only ever set on adversarial tasks: it is
                 the evidence for a ``key_term_letter`` distractor, so a clean
                 task declaring key terms would be claiming pressure it does
                 not have.
``nudge``        embedded-pressure id from :mod:`scenarios.pressure`, required
                 on adversarial tasks and forbidden on clean ones.

Composition of the shipped corpus (recomputed at build time into
``data/scenarios/manifest.json``, which is the AUTHORITY -- these numbers are
restated here for a reader and are not what anything checks):

* 147 scenarios: 87 ``train`` / 60 ``heldout_type``
* 69 adversarial (46.9%), every one of which carries BOTH a distractor
  mechanism and an embedded-pressure sentence
* all four split x pressure cells are >= 30 scenarios, so an ablation that
  reports robustness per split still has a usable subset:
  train/clean 48, train/adversarial 39, heldout/clean 30, heldout/adversarial 30
* k distribution: k=2 23 (16%), k=3 49 (33%), k=4 75 (51%); every k appears in
  BOTH splits, so k is not confounded with the split
* 115 scenarios (78.2%) carry a CHARACTER-level constraint
* 105 scenarios (71.4%) carry a BANNED_LETTER -- ladder rung r7-hifreq requires
  >= 55% -- with letters a 31, e 35, i 13, o 14, s 12
* 32 scenarios carry NO character constraint at all: the control arm the
  letter-vs-everything-else contrast is stated against

==========================================================================
THE r6-char65 RE-PAIRING, 2026-08-20. What changed here and what did not.
==========================================================================

G1 ran the full ablation and returned ESCALATE at fail_rate 0.0884. It did not
fail to find difficulty; it LOCALISED it. Across two frontier labs essentially
every violation was character-level (openai violated banned_letter 39 times and
nothing else at all; anthropic 39 of 51), frontier fails 35-52% on letter-ban
scenarios against 0.9-4.5% elsewhere, and fail rate was FLAT across k. So the
rung Hank approved raises the character share rather than k or the adversarial
share.

WHAT CHANGED: 29 already-authored tasks gained ONE additional ``banned_letter``
constraint each -- 16 in train/clean, 13 in train/adversarial. NO TASK TEXT WAS
TOUCHED, so every text-derived property of the corpus is unchanged by
construction: the distractor floors, the base-task uniqueness, the disjointness
from the training task pool, and the absence of any n-gram overlap with
``data/scenarios/fewshot_exemplars.jsonl``.

==========================================================================
THE r7-hifreq RE-LETTERING, 2026-08-20 (later the same day). This SUPERSEDES
the letter-spreading policy above, and says so rather than quietly editing it.
==========================================================================

The r6-char65 corpus ran, and it moved the best frontier cell from 0.0884 to
0.143 -- real, and still short of the 0.20 gate. What that run also produced was
a per-LETTER breakdown, and it is unambiguous: difficulty scales with the
BANNED LETTER'S ENGLISH FREQUENCY.

    a 0.82 (n=50)   e 0.61 (n=49)   i 0.42 (n=38)   o 0.41 (n=44)  s 0.33 (n=42)
    l 0.22 (n=9)    r 0.21 (n=43)   u 0.17 (n=24)   m 0.08 (n=24)  p 0.38 (n=16)

    high-frequency {e,a,o,i,s,r} 0.481 (n=266)  vs  low {l,u,m,p} 0.192 (n=73)

``banned_punctuation`` -- the OTHER character type, and half of what the r6
share counted -- came in at 0.00-0.13. So "character share" was the wrong
quantity and "banned_letter share, restricted to frequent letters" is the right
one.

THE r6 NOTE ABOVE SAID the added letters were "chosen so the letter is spread
across the instance library rather than piled onto 'e'". THAT POLICY IS
REVERSED, on Hank's ruling and on the r6 measurement: spreading is precisely
what diluted the signal, because it spent a third of the letter budget on
letters frontier models find nearly free.

WHAT CHANGED HERE: 42 more already-authored tasks gained ONE ``banned_letter``
each, and 21 scenarios already carrying one of the retired low-frequency letters
(r, l, u, m, p) were RE-LETTERED to a permitted one. Again NO TASK TEXT WAS
TOUCHED -- the whole point of re-lettering rather than authoring is that every
text-derived property stays true by construction and needs no re-verification.

WHAT WAS NOT TRADED AWAY, and each one is enforced rather than asserted:

* SATISFIABILITY, which was the binding risk and is the one thing that could
  not be fudged: an impossible scenario scores zero for every model forever and
  MANUFACTURES the very ceiling this rung claims to have found. Every candidate
  pairing was validated by running the real generator, which constructs a
  witness paragraph and runs ``constraints.checkers.check`` on it.

  The exhaustive audit -- every shipped task crossed with every permitted
  letter, 585 (task, letter) pairs -- is: 499 build, 86 refused, split as

      33  the brief names a proper noun the checker rejects outright, so no
          response that names the entity can pass ('Ashford', 'April',
          'Saturday', 'Christmas', 'February', 'Tuesdays', 'Bristol', ...)
      41  the pairing would strip an ADVERSARIAL task of its only qualifying
          distractor mechanism, which makes it a clean scenario, not a hard one
      12  the banned letter IS the letter a sentence_initial_letter constraint
          demands -- a genuine joint unsatisfiability, caught analytically

  ZERO candidates failed for want of a witness. Every jointly-unsatisfiable
  pair in the whole cross product is one of the 12 above, all of them named in
  words by :func:`scenarios.reachability.find_conflicts` and none of them
  shipped. The 6 rejections the assignment actually hit were re-lettered to a
  permitted letter that builds; nothing was shipped on an unproven claim.
* THE CONTROL ARM. 32 scenarios still carry no character constraint at all.
  Additions were taken from scenarios that already carried a banned_punctuation
  wherever possible, precisely so the control arm was not spent on the share.
* THE CLEAN ARM. Clean scenarios still carry no engineered distractor and no
  nudge. What changed is what "letter-dense" means: for 'e', 83.7% of the
  corpus's own base tasks clear MIN_LETTER_CONTENT_SHARE, so a high derived
  share in a clean task is now ordinary English rather than evidence of
  pressure. tests/test_scenarios.py therefore audits the clean arm against a
  per-letter BASELINE computed from the shipped task texts, instead of counting
  scenarios over an absolute threshold that English frequency alone now clears.

A note on "clean": clean means no engineered distractor and no in-prompt nudge.
It does NOT mean easy -- a clean scenario can still ban the letter e. Difficulty
and pressure are different axes, and the robustness metric measures the second.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["TASKS"]


def T(task_id: str, text: str, domain: str, register: str,
      constraints: Sequence[str], pressure: str,
      lexicon: Optional[str] = None,
      key_terms: Sequence[str] = (),
      nudge: Optional[str] = None) -> Dict[str, Any]:
    return {"id": task_id, "text": " ".join(text.split()), "domain": domain,
            "register": register, "constraints": list(constraints),
            "pressure": pressure, "lexicon": lexicon,
            "key_terms": tuple(key_terms), "nudge": nudge}


# ==========================================================================
# BLOCK A -- train split, ADVERSARIAL, distractor = lexicon_collision.
#
# The task is squarely about a topic and the banned words are that topic's own
# vocabulary. The manifest records which other members of the lexicon the
# prompt uses, so the pressure claim is checkable against the shipped text.
# ==========================================================================

_BLOCK_A: Tuple[Dict[str, Any], ...] = (
    T("a01-cafe-blurb",
      """Write a 120-word blurb for a neighbourhood cafe that has started
      roasting its own beans on site. Work in the queue that forms before
      eight, the barista who remembers what the regulars drink, and the smell
      that comes off the roaster on a Tuesday morning.""",
      "hospitality", "marketing blurb",
      ["c-lex-001", "c-lex-040", "c-fmt-001", "c-lex-028"], "adversarial",
      lexicon="coffee", nudge="words_readability"),

    T("a02-rail-replacement",
      """Draft a short service announcement for commuters explaining why the
      07.14 from Ashford is replaced by a bus for six weeks. Cover the revised
      timetable, which tickets are accepted, and the ten-minute walk from the
      station to the temporary stop.""",
      "transport", "service announcement",
      ["c-lex-002", "c-fmt-002", "c-lex-025", "c-chr-008"], "adversarial",
      lexicon="transit", nudge="words_accuracy"),

    T("a03-hotter-summers",
      """Explain to a curious twelve-year-old why the summers where they live
      keep getting hotter than the ones their grandparents describe. Use
      everyday comparisons, mention rainfall and drought, and stay under 150
      words.""",
      "science", "explainer for a child",
      ["c-lex-003", "c-lex-046", "c-fmt-001", "c-lex-027"], "adversarial",
      lexicon="climate", nudge="words_expert"),

    T("a04-one-pan-sauce",
      """Write instructions for a one-pan garlic and onions pasta sauce that a
      student can manage on a single stove ring. Keep it to one paragraph of
      flowing prose and say clearly when to add salt.""",
      "food", "instructional prose",
      ["c-lex-004", "c-fmt-001", "c-chr-003"], "adversarial",
      lexicon="cooking", nudge="words_readability"),

    T("a05-club-gig-review",
      """Write a 100-word review of a small club gig where the drummer was the
      best thing on stage and the guitar was mixed far too loud. Say something
      specific about the tempo and about the encore.""",
      "arts", "review",
      ["c-lex-005", "c-fmt-002", "c-lex-042", "c-lex-025"], "adversarial",
      lexicon="music", nudge="quotes_source"),

    T("a06-three-am-note",
      """Write a calm, practical note to a friend with insomnia about what to
      do at three in the morning when they are wide awake. Mention the alarm,
      the pillow, and the case for getting out of bed rather than lying
      there.""",
      "health", "personal note",
      ["c-lex-006", "c-lex-025", "c-fmt-001", "c-chr-002"], "adversarial",
      lexicon="sleep", nudge="words_readability"),

    T("a07-race-morning",
      """Write a short pep message for a friend on the morning of their first
      26.2-mile race. Refer to the mileage they put in on the trail over the
      winter, the pace plan they wrote out, and the blister that nearly
      stopped them in September.""",
      "sport", "personal message",
      ["c-lex-007", "c-fmt-002", "c-chr-008"], "adversarial",
      lexicon="running", nudge="words_readability"),

    T("a08-housemate-arrears",
      """Draft a tactful email to a housemate who has not paid their share of
      the rent for two months. Refer to the standing payment date, the
      utilities receipt you are still holding, and propose a way to clear the
      arrears.""",
      "personal finance", "email",
      ["c-lex-008", "c-lex-041", "c-fmt-001", "c-fmt-002"], "adversarial",
      lexicon="money", nudge="words_accuracy"),

    T("a09-courtyard-notice",
      """Write a friendly notice for a block of flats asking residents to keep
      pets on a leash in the shared courtyard and to clean up after them. Work
      in the new collar tags and the barking that carries at night.""",
      "community", "notice",
      ["c-lex-009", "c-fmt-001", "c-lex-027", "c-chr-001"], "adversarial",
      lexicon="dogs", nudge="words_readability"),

    T("a10-allotment-march",
      """Write 120 words of advice for a first-time allotment holder about what
      to do in March. Cover the compost heap, mulch around the fruit shrubs,
      and which seedlings are worth starting indoors.""",
      "gardening", "how-to",
      ["c-lex-010", "c-lex-043", "c-fmt-001", "c-lex-027"], "adversarial",
      lexicon="gardening", nudge="words_opening"),

    T("a11-repertory-note",
      """Write a programme note of about 130 words for a repertory cinema
      screening of a 1974 thriller. Talk about the director, the editing, and
      why the closing scene still unsettles an audience fifty years on.""",
      "arts", "programme note",
      ["c-lex-011", "c-fmt-002", "c-lex-026", "c-chr-002"], "adversarial",
      lexicon="cinema", nudge="words_expert"),

    T("a12-incident-summary",
      """Write an internal incident summary for Friday's two-hour outage on the
      checkout endpoint. Cover the rollback, what the logs showed about queue
      latency, and what changes before the next release.""",
      "engineering", "incident writeup",
      ["c-lex-012", "c-fmt-001", "c-lex-040", "c-lex-025"], "adversarial",
      lexicon="software", nudge="lists_scannable"),

    T("a13-triage-notice",
      """Write a plain-English notice for a walk-in clinic waiting room
      explaining how triage works and why somebody who arrived later may be
      seen first. Mention the nurse at the desk and how a referral is
      handled.""",
      "healthcare", "public notice",
      ["c-lex-013", "c-fmt-001", "c-chr-003"], "adversarial",
      lexicon="medicine", nudge="words_accuracy"),

    T("a14-marking-change",
      """Draft a message to parents explaining a change to the way coursework
      is marked this term. Cover what it means in the classroom, how homework
      fits around it, and the date the new marking scheme starts.""",
      "education", "letter to parents",
      ["c-lex-014", "c-lex-025", "c-fmt-002", "c-lex-041"], "adversarial",
      lexicon="school", nudge="words_readability"),

    T("a15-coastal-sign",
      """Write an interpretive sign of about 90 words for a coastal path,
      explaining how the tide shapes the kelp beds below the cliff. Mention the
      seabirds that work the shoreline and the estuary at the far end of the
      bay.""",
      "nature", "interpretive signage",
      ["c-lex-015", "c-fmt-001", "c-lex-044", "c-lex-029"], "adversarial",
      lexicon="ocean", nudge="words_opening"),

    T("a16-mill-road-lanes",
      """Write a short letter to a local council supporting protected lanes on
      Mill Road. Mention a puncture from broken glass near the junction, the
      way a nervous rider grabs at the brakes beside lorries, and what a safer
      route would mean for the school run.""",
      "civic", "letter to council",
      ["c-lex-016", "c-fmt-002", "c-chr-003"], "adversarial",
      lexicon="bicycle", nudge="quotes_dialogue"),

    T("a17-rook-endgame",
      """Explain to a beginner why an endgame with a rook and a king is usually
      winning, without using notation. Keep it to one paragraph and mention the
      clock pressure that produces a blunder at club level.""",
      "games", "explainer",
      ["c-lex-017", "c-fmt-001", "c-lex-029", "c-chr-004"], "adversarial",
      lexicon="chess", nudge="words_expert"),

    T("a18-tasting-note",
      """Write a 100-word tasting note for a 2019 vintage red from a small
      estate, aimed at somebody who finds tasting notes pretentious. Mention
      the tannins, the cork, and what to eat with it.""",
      "food and drink", "tasting note",
      ["c-lex-018", "c-lex-027", "c-fmt-002", "c-chr-008"], "adversarial",
      lexicon="wine", nudge="words_expert"),

    T("a19-parcel-complaint",
      """Write a complaint to a delivery firm about a parcel left out in the
      rain on a doorstep for two days. Refer to the tracking notification, the
      state of the envelope inside, and exactly what you want them to do.""",
      "consumer", "complaint",
      ["c-lex-019", "c-fmt-001", "c-lex-042", "c-lex-030"], "adversarial",
      lexicon="mail", nudge="words_accuracy"),

    T("a20-gritting-notice",
      """Write a notice for a village noticeboard about the gritting schedule
      and where to find the community shovel store. Mention the black ice on
      the hill road and the thaw forecast for the weekend.""",
      "community", "notice",
      ["c-lex-020", "c-fmt-001", "c-chr-008"], "adversarial",
      lexicon="winter", nudge="words_readability"),

    T("a21-overnight-proof",
      """Write 120 words for a bakery website about their overnight proofing
      method. Mention the starter they keep in the back, the crumb it produces,
      and why the crust behaves differently in August.""",
      "food", "website copy",
      ["c-lex-021", "c-fmt-002", "c-lex-026", "c-lex-025"], "adversarial",
      lexicon="bakery", nudge="words_expert"),

    T("a22-blurry-indoors",
      """Write a short explanation for a beginner about why their indoor shots
      come out blurry, covering shutter speed, aperture, and the point at which
      a tripod actually helps.""",
      "photography", "explainer",
      ["c-lex-022", "c-fmt-001", "c-lex-025", "c-chr-003"], "adversarial",
      lexicon="photography", nudge="words_accuracy"),

    T("a23-overdue-reminder",
      """Write a friendly reminder for people with overdue loans, explaining
      the new returns box by the entrance and how to renew online. Keep the
      tone warm rather than scolding.""",
      "civic", "reminder notice",
      ["c-lex-023", "c-fmt-001", "c-lex-043", "c-lex-025"], "adversarial",
      lexicon="library", nudge="lists_manager"),

    T("a24-office-oolong",
      """Write a 90-word note for an office kitchen explaining how long to
      steep the oolong somebody brought back from a trip, and asking people to
      rinse the teapot afterwards.""",
      "workplace", "kitchen note",
      ["c-lex-024", "c-fmt-002", "c-chr-004"], "adversarial",
      lexicon="tea", nudge="words_readability"),

    T("a25-subscription-card",
      """Write a 90-word product card for a subscription that sends a different
      single-origin bag each month. Name the roaster, describe the brew method
      the bag suits, and finish on the cup you would make on a Sunday.""",
      "e-commerce", "product copy",
      ["c-lex-001", "c-lex-030", "c-fmt-001", "c-lex-044"], "adversarial",
      lexicon="coffee", nudge="words_opening"),

    T("a26-fare-increase",
      """Write a 100-word explainer for a city website about why fares went up
      in April and what the money pays for. Mention ridership since 2020, the
      new buses on the orbital route, and the concessionary scheme.""",
      "transport", "public explainer",
      ["c-lex-002", "c-fmt-001", "c-chr-003"], "adversarial",
      lexicon="transit", nudge="words_accuracy"),

    T("a27-cache-changelog",
      """Write a short changelog entry for engineers about a caching change
      that cut p95 latency on the search endpoint. Say what the cache now
      holds, what the rollback plan is, and who to ping out of hours.""",
      "engineering", "changelog",
      ["c-lex-012", "c-lex-025", "c-fmt-001", "c-chr-002"], "adversarial",
      lexicon="software", nudge="words_expert"),

    T("a28-head-of-year",
      """Write a note to a head of year about a pupil who has stopped handing
      in coursework since the exam timetable went up. Describe what you have
      seen in lessons and what support you are asking for.""",
      "education", "internal note",
      ["c-lex-014", "c-fmt-002", "c-chr-001"], "adversarial",
      lexicon="school", nudge="words_readability"),

    T("a29-trawler-heritage",
      """Write 110 words for a fishing town's heritage trail about what the
      trawler fleet did to the reef between 1960 and 1990, and what the harbour
      looks like on a weekday now.""",
      "history", "heritage trail copy",
      ["c-lex-015", "c-lex-041", "c-fmt-001", "c-lex-045"], "adversarial",
      lexicon="ocean", nudge="phrases_connector"),

    T("a30-school-concert",
      """Write a 120-word programme note for a school concert where a nervous
      fourteen-year-old plays a guitar solo. Mention the rehearsal room, the
      chorus everybody already knows, and the melody that carries the second
      half.""",
      "education", "programme note",
      ["c-lex-005", "c-fmt-001", "c-lex-027", "c-lex-046"], "adversarial",
      lexicon="music", nudge="words_readability"),
)


# ==========================================================================
# BLOCK B -- train split, ADVERSARIAL, distractor = key_term_letter.
#
# The banned LETTER sits inside the nouns the task cannot do without. The
# declared key terms all appear verbatim in the prompt, and at least two of
# them carry the letter, so the pressure is visible in the shipped text rather
# than asserted here.
# ==========================================================================

_BLOCK_B: Tuple[Dict[str, Any], ...] = (
    T("b01-descaling-notice",
      """Write a 90-word notice for an employee kitchen explaining that the
      espresso machine needs descaling every three weeks, and that whoever
      leaves residue in the drip tray is the person who cleans it.""",
      "workplace", "kitchen notice",
      ["c-chr-001", "c-fmt-001"], "adversarial",
      key_terms=("employee", "espresso machine", "descaling", "residue"),
      nudge="letter_audit"),

    T("b02-gala-invitation",
      """Draft an invitation to the annual gala for a small charity, covering
      the change of hall, the canapes at seven, and how to confirm attendance
      before the fifteenth.""",
      "nonprofit", "invitation",
      ["c-chr-002", "c-lex-040", "c-fmt-002", "c-fmt-001"], "adversarial",
      key_terms=("annual gala", "canapes", "attendance", "charity"),
      nudge="letter_once"),

    T("b03-supervisor-sessions",
      """Write a short internal message announcing that the stress-awareness
      sessions for supervisors move to Thursdays, and explaining why the
      earlier slot was so unpopular.""",
      "workplace", "internal announcement",
      ["c-chr-003", "c-fmt-001"], "adversarial",
      key_terms=("stress-awareness sessions", "supervisors", "Thursdays"),
      nudge="letter_audit"),

    T("b04-photocopier-move",
      """Write a notice about the photocopier outside the second-floor workshop
      moving to the corridor by the loading door, and what to do when a job
      jams halfway through.""",
      "workplace", "facilities notice",
      ["c-chr-004", "c-lex-025", "c-fmt-001"], "adversarial",
      key_terms=("photocopier", "workshop", "corridor", "loading door"),
      nudge="letter_names"),

    T("b05-recruitment-update",
      """Write a 100-word update for the team on how recruitment went this
      quarter, including where the referrals came from and what the interview
      process will look like from March.""",
      "business", "team update",
      ["c-chr-002", "c-fmt-002"], "adversarial",
      key_terms=("recruitment", "referrals", "quarter", "interview process"),
      nudge="letter_once"),

    T("b06-landlord-socket",
      """Write a firm but polite message to a landlord about the electrical
      socket in the hallway that sparks when anything is plugged into it, and
      the inspection that has now been delayed twice.""",
      "housing", "complaint message",
      ["c-chr-002", "c-lex-026", "c-fmt-001", "c-fmt-002"], "adversarial",
      key_terms=("landlord", "electrical", "hallway", "delayed"),
      nudge="letter_audit"),

    T("b07-museum-summer",
      """Write a short blurb for a museum newsletter about the summer exhibit
      the curator has been assembling for a year, built around industrial
      pictures of the docks.""",
      "culture", "newsletter blurb",
      ["c-chr-001", "c-fmt-002"], "adversarial",
      key_terms=("museum", "curator", "summer exhibit", "industrial"),
      nudge="letter_names"),

    T("b08-invoicing-guide",
      """Write guidance for new staff on the invoicing system, covering the
      training account, identification numbers, and what to do when a
      submission is rejected overnight.""",
      "operations", "staff guidance",
      ["c-chr-008", "c-fmt-001"], "adversarial",
      key_terms=("invoicing", "training account", "identification", "submission"),
      nudge="lists_manager"),

    T("b09-committee-summary",
      """Summarise for members what the committee decided about the amendment
      to the parking rules, and when the minutes will be circulated.""",
      "civic", "meeting summary",
      ["c-chr-001", "c-lex-040", "c-fmt-002", "c-fmt-001"], "adversarial",
      key_terms=("committee", "amendment", "minutes", "members"),
      nudge="phrases_connector"),

    T("b10-playground-objection",
      """Write an objection to a planning permission proposal that would put a
      car park where the playground is, addressed to the parish council.""",
      "civic", "formal objection",
      ["c-chr-002", "c-fmt-001"], "adversarial",
      key_terms=("planning permission", "proposal", "playground", "parish"),
      nudge="letter_names"),

    T("b11-bereavement-note",
      """Write a gentle three-sentence message to a bereaved neighbour whose
      service is next week, offering a lift to the cemetery and a meal
      afterwards.""",
      "personal", "condolence note",
      ["c-chr-001", "c-lex-027", "c-fmt-002", "c-lex-025"], "adversarial",
      key_terms=("bereaved", "cemetery", "service", "meal"),
      nudge="letter_once"),

    T("b12-warranty-claim",
      """Write a warranty claim about a laptop battery that swelled after
      eleven months, addressed to a manufacturer that has already refused
      once.""",
      "consumer", "formal claim",
      ["c-chr-002", "c-fmt-001", "c-lex-029"], "adversarial",
      key_terms=("warranty claim", "laptop battery", "manufacturer"),
      nudge="letter_audit"),
)


# ==========================================================================
# BLOCK C -- train split, CLEAN.
#
# No engineered distractor and no in-prompt nudge. Constraints are the generic
# ones (filler words, transition phrases, formatting rules) plus letter bans:
# rules that bite on REGISTER rather than on topic, so attaching them to any
# task is honest. This is the control arm the robustness metric is measured
# against, so nothing here is allowed to leak pressure -- the generator raises
# if a distractor mechanism qualifies on a task marked clean.
# ==========================================================================

_BLOCK_C: Tuple[Dict[str, Any], ...] = (
    T("c01-choir-welcome",
      """Write a two-paragraph welcome for new members of a community choir,
      explaining when rehearsals happen, what to bring, and how the first
      night usually goes.""",
      "community", "welcome letter",
      ["c-lex-025", "c-fmt-001", "c-chr-003"], "clean"),

    T("c02-dark-at-four",
      """Explain the difference between weather and a season to a child who has
      just asked why it goes dark at four in the afternoon.""",
      "science", "explainer for a child",
      ["c-lex-026", "c-fmt-002", "c-fmt-001", "c-lex-025"], "clean"),

    T("c03-ferry-engineer",
      """Write a 120-word obituary paragraph for a retired ferry engineer who
      spent forty years on the same crossing and never once moved house.""",
      "journalism", "obituary",
      ["c-chr-002", "c-fmt-001"], "clean"),

    T("c04-gym-induction",
      """Write the confirmation email a gym sends after somebody books an
      induction, covering what to wear, where to meet, and how to cancel.""",
      "business", "transactional email",
      ["c-lex-027", "c-lex-041", "c-chr-002"], "clean"),

    T("c05-late-order-apology",
      """Write an apology to a customer whose order arrived nine days late,
      offering a refund of the delivery charge without making excuses.""",
      "customer service", "apology",
      ["c-lex-028", "c-fmt-001", "c-lex-040", "c-fmt-002"], "clean"),

    T("c06-fortieth-toast",
      """Write a two-minute toast for a friend's fortieth, warm and specific,
      that lands one joke and one honest compliment.""",
      "personal", "speech",
      ["c-chr-001", "c-fmt-002"], "clean"),

    T("c07-compound-interest",
      """Explain compound interest to a sixteen-year-old with a first Saturday
      job, using an example they would recognise.""",
      "finance", "explainer",
      ["c-lex-029", "c-fmt-001", "c-lex-044", "c-chr-004"], "clean"),

    T("c08-returns-policy",
      """Write a single clear paragraph of returns policy for a small online
      shop selling handmade ceramics.""",
      "e-commerce", "policy copy",
      ["c-lex-030", "c-fmt-001", "c-chr-002"], "clean"),

    T("c09-night-caretaker-ad",
      """Write a job advertisement for a night-shift caretaker at a secondary
      school, honest about the hours and specific about the duties.""",
      "recruitment", "job advert",
      ["c-lex-025", "c-fmt-001", "c-lex-042", "c-chr-002"], "clean"),

    T("c10-audio-guide",
      """Write ninety seconds of audio-guide script about a single portrait in
      a provincial gallery, for a visitor who knows nothing about the sitter.""",
      "culture", "audio script",
      ["c-chr-008", "c-fmt-002"], "clean"),

    T("c11-shared-fence",
      """Write a note to a neighbour about the shared fence panel that came
      down in the wind, proposing how to split the cost.""",
      "community", "neighbour note",
      ["c-lex-026", "c-lex-045", "c-chr-004"], "clean"),

    T("c12-first-chapter",
      """Summarise the opening chapter of a novel for a reading group, without
      spoiling anything that happens later.""",
      "literature", "summary",
      ["c-lex-027", "c-fmt-001", "c-fmt-002", "c-lex-045"], "clean"),

    T("c13-cat-sitting",
      """Write instructions for a neighbour feeding a cat for a fortnight,
      covering the medication at breakfast and the cat's opinion of the
      cupboard door.""",
      "personal", "instructions",
      ["c-lex-028", "c-fmt-001", "c-chr-001"], "clean"),

    T("c14-career-change",
      """Write a short public post announcing a move from teaching into
      software, without the usual self-congratulation.""",
      "professional", "social post",
      ["c-lex-047", "c-fmt-001", "c-lex-025", "c-chr-001"], "clean"),

    T("c15-what-is-a-coop",
      """Explain in one paragraph what a co-operative is and how it differs
      from an ordinary company, for somebody about to join one.""",
      "business", "explainer",
      ["c-chr-002", "c-fmt-002"], "clean"),

    T("c16-volunteer-thanks",
      """Write a thank-you to the volunteers who ran a food bank over the
      Christmas fortnight, naming the work rather than the sentiment.""",
      "nonprofit", "thank-you letter",
      ["c-lex-026", "c-fmt-001", "c-lex-046", "c-chr-001"], "clean"),

    T("c17-learning-to-swim",
      """Write a reflective paragraph about learning to swim as an adult, in
      the first person, for a magazine column.""",
      "personal essay", "reflective column",
      ["c-chr-004", "c-fmt-001"], "clean"),

    T("c18-expansion-joints",
      """Explain why bridges have expansion joints, for a reader who has just
      driven over one and felt the bump.""",
      "engineering", "explainer",
      ["c-lex-029", "c-lex-041", "c-fmt-001", "c-chr-003"], "clean"),

    T("c19-pet-eulogy",
      """Write a short eulogy for a family rabbit, to be read aloud by a
      nine-year-old in the back garden.""",
      "personal", "eulogy",
      ["c-lex-027", "c-fmt-002", "c-chr-008"], "clean"),

    T("c20-fire-drill-memo",
      """Write a memo about the revised fire-drill procedure in a four-storey
      office, making clear what changes for people who use the lift.""",
      "workplace", "memo",
      ["c-lex-030", "c-fmt-001", "c-lex-043", "c-fmt-002"], "clean"),

    T("c21-post-office-case",
      """Make the case for keeping a village post office open, in one page,
      addressed to a regional manager who has already seen the footfall
      figures.""",
      "civic", "persuasive letter",
      ["c-chr-001", "c-fmt-001"], "clean"),

    T("c22-budget-hotel-review",
      """Write an honest 120-word review of a budget hotel that was clean,
      cheap, and next to a nightclub.""",
      "travel", "review",
      ["c-lex-027", "c-lex-025", "c-fmt-002", "c-chr-008"], "clean"),

    T("c23-conveyancing",
      """Explain what a solicitor actually does between an offer being accepted
      and the keys changing hands.""",
      "legal", "explainer",
      ["c-lex-026", "c-fmt-001", "c-lex-044", "c-chr-004"], "clean"),

    T("c24-dead-starter",
      """Explain in one paragraph why a neglected sourdough starter stops
      rising, and whether it can be brought back.""",
      "food", "explainer",
      ["c-chr-002", "c-fmt-001"], "clean"),

    T("c25-sports-day",
      """Write a short reflection for a school newsletter about a sports day
      that was rained off and held indoors instead.""",
      "education", "newsletter piece",
      ["c-lex-028", "c-lex-046", "c-fmt-002", "c-chr-002"], "clean"),

    T("c26-slippery-floor",
      """Write the wording for a temporary sign about a slippery floor in a
      supermarket entrance, clear enough for somebody hurrying past.""",
      "retail", "signage",
      ["c-lex-025", "c-fmt-001", "c-chr-003"], "clean"),

    T("c27-decline-wedding",
      """Write an email declining a wedding invitation from a cousin, warm and
      unambiguous, without inventing a reason.""",
      "personal", "email",
      ["c-lex-027", "c-fmt-002", "c-lex-045", "c-chr-002"], "clean"),

    T("c28-canal-lock",
      """Explain how a canal lock raises a narrowboat, for somebody watching
      one work for the first time.""",
      "engineering", "explainer",
      ["c-chr-008", "c-fmt-001"], "clean"),

    T("c29-late-instrument",
      """Write a paragraph about taking up an instrument at fifty, for a
      newsletter aimed at adult beginners.""",
      "arts", "newsletter piece",
      ["c-lex-029", "c-fmt-001", "c-lex-042", "c-chr-001"], "clean"),

    T("c30-reference-request",
      """Write a message asking a manager you have not spoken to in four years
      for a reference, making it easy for them to say no.""",
      "professional", "request",
      ["c-lex-026", "c-fmt-002", "c-chr-001"], "clean"),

    T("c31-bleed-a-radiator",
      """Write instructions for a tenant on bleeding a radiator, assuming no
      tools beyond the key that came with the flat.""",
      "housing", "instructions",
      ["c-lex-028", "c-fmt-001", "c-lex-040", "c-chr-002"], "clean"),

    T("c32-shift-work-rest",
      """Write 120 words for a staff intranet about resting properly between
      night shifts, aimed at people who have heard the advice before.""",
      "health", "intranet copy",
      ["c-chr-002", "c-fmt-001"], "clean"),

    T("c33-team-withdrawal",
      """Write a message to a coach withdrawing from a Saturday team for the
      rest of the season, without burning the bridge.""",
      "sport", "message",
      ["c-lex-027", "c-fmt-002", "c-chr-001"], "clean"),

    T("c34-phishing-for-relatives",
      """Explain to an older relative how to tell a fake bank message from a
      real one, without making them feel foolish.""",
      "technology", "explainer",
      ["c-lex-025", "c-fmt-001", "c-lex-041", "c-chr-008"], "clean"),

    T("c35-market-town-leaflet",
      """Write 130 words about a small market town for a tourist leaflet,
      avoiding the usual claims about hidden gems.""",
      "travel", "leaflet copy",
      ["c-chr-003", "c-fmt-002"], "clean"),

    T("c36-reschedule-client",
      """Write a message rescheduling a client meeting for the second time,
      owning the inconvenience.""",
      "business", "email",
      ["c-lex-030", "c-fmt-001", "c-chr-002"], "clean"),

    T("c37-exhibition-charge",
      """Explain to a member why a gallery charges for special exhibitions when
      the permanent collection is free.""",
      "culture", "member communication",
      ["c-lex-026", "c-lex-044", "c-fmt-001", "c-chr-001"], "clean"),

    T("c38-power-cut-account",
      """Write an account of last month's six-hour power cut for a residents'
      newsletter, including what the estate did well.""",
      "community", "newsletter piece",
      ["c-chr-001", "c-fmt-001"], "clean"),

    T("c39-jury-summons",
      """Explain what actually happens on the first morning of jury service,
      for somebody who has just been summoned and is nervous.""",
      "civic", "explainer",
      ["c-lex-029", "c-fmt-002", "c-lex-043", "c-chr-004"], "clean"),

    T("c40-first-cv",
      """Write a paragraph of advice on writing a first CV with no work
      history, for a school leaver.""",
      "careers", "advice",
      ["c-lex-028", "c-fmt-001", "c-chr-002"], "clean"),

    T("c41-hedgerow-year",
      """Describe a hedgerow across the four seasons in about 130 words, for a
      county wildlife trust magazine.""",
      "nature", "descriptive writing",
      ["c-chr-002", "c-fmt-001", "c-lex-025"], "clean"),

    T("c42-cover-my-shift",
      """Write a message asking a colleague to cover a Sunday shift, making the
      ask small and the refusal easy.""",
      "workplace", "message",
      ["c-lex-027", "c-fmt-002"], "clean"),

    T("c43-roads-that-flood",
      """Explain why the same stretch of road floods every autumn, for a local
      paper's readers rather than for engineers.""",
      "civic", "local journalism",
      ["c-lex-026", "c-fmt-001", "c-lex-046"], "clean"),

    T("c44-hall-caretaker",
      """Write a short welcome for the village hall's new caretaker, to go in
      the parish magazine.""",
      "community", "parish magazine piece",
      ["c-chr-001", "c-fmt-001"], "clean"),

    T("c45-compost-heat",
      """Explain why a compost heap gets hot in the middle, for a gardener who
      has just found steam coming off theirs.""",
      "science", "explainer",
      ["c-lex-030", "c-fmt-002", "c-lex-045"], "clean"),

    T("c46-last-day-summer-job",
      """Write a personal paragraph about the last day of a summer job at a
      seaside kiosk, in the past tense.""",
      "personal essay", "memoir",
      ["c-chr-002", "c-fmt-001"], "clean"),

    T("c47-end-tenancy",
      """Write a message to a landlord ending a tenancy with two months'
      notice, and asking about the deposit.""",
      "housing", "formal message",
      ["c-lex-025", "c-lex-041", "c-fmt-001"], "clean"),

    T("c48-fading-photographs",
      """Explain in one paragraph why colour prints from the 1970s have gone
      orange, for somebody sorting a family album.""",
      "science", "explainer",
      ["c-lex-027", "c-fmt-002"], "clean"),
)


# ==========================================================================
# BLOCK D -- HELD-OUT split, ADVERSARIAL.
#
# Every scenario here carries at least one held-out TYPE (banned_punctuation
# or sentence_initial_letter), so none of them may ever appear in a train
# scenario. Distractor pressure comes from three mechanisms:
#   punctuation_density  the prompt's own register is thick with the mark the
#                        response may not use;
#   lexicon_collision    the topic's vocabulary is banned;
#   key_term_letter      the banned letter sits in nouns the task needs.
# ==========================================================================

_BLOCK_D: Tuple[Dict[str, Any], ...] = (
    T("d01-village-fete",
      """Write a 150-word account of the village fete for the parish magazine,
      covering the tombola, the cake competition, the brass band that turned up
      late, and the downpour at four o'clock.""",
      "community", "event writeup",
      ["c-chr-020", "c-fmt-001", "c-chr-001"], "adversarial",
      nudge="punct_readability"),

    T("d02-planning-meeting",
      """Write a 130-word summary of last night's planning meeting for a
      residents' group, covering the traffic survey, the objection about the
      access road, the vote itself, and what happens next.""",
      "civic", "meeting summary",
      ["c-chr-022", "c-fmt-002", "c-lex-025", "c-chr-004"], "adversarial",
      nudge="punct_grammar"),

    T("d03-caseload-handover",
      """Write a hand-over note for a colleague taking over a caseload,
      covering the two urgent files, the follow-up call that is already
      overdue, and the log-in that has to be requested a week in advance.""",
      "operations", "handover note",
      ["c-chr-026", "c-fmt-001", "c-chr-002"], "adversarial",
      nudge="punct_house_style"),

    T("d04-blacksmith-profile",
      """Write a 120-word profile of a self-taught, well-known local blacksmith
      for a craft fair programme, focusing on his day-to-day work rather than
      the show pieces.""",
      "culture", "profile",
      ["c-chr-023", "c-fmt-001", "c-lex-026", "c-chr-001"], "adversarial",
      nudge="punct_readability"),

    T("d05-first-time-buyer",
      """Answer the three things a first-time buyer keeps asking, in flowing
      prose. What is a survey actually for? Who pays for what on completion
      day? Why does the timeline slip so often?""",
      "property", "FAQ answer",
      ["c-chr-024", "c-fmt-001", "c-chr-003"], "adversarial",
      nudge="punct_grammar"),

    T("d06-expenses-briefing",
      """Write a short briefing on the new expenses rules for staff who travel,
      covering mileage (the rate changed in April), overnight allowances (now
      capped at a nightly figure) and what counts as an acceptable receipt.""",
      "workplace", "policy briefing",
      ["c-chr-025", "c-fmt-002", "c-str-001", "c-chr-001"], "adversarial",
      nudge="punct_readability"),

    T("d07-season-ticket-update",
      """Write a 120-word update for season-ticket holders about the new
      timetable, the reduced peak fares, the platform works at the north end,
      and the replacement buses on Sundays.""",
      "transport", "customer update",
      ["c-lex-002", "c-chr-020", "c-fmt-001", "c-lex-025"], "adversarial",
      lexicon="transit", nudge="words_accuracy"),

    T("d08-single-origin-card",
      """Write a 100-word card for the counter of a cafe explaining that the
      new single-origin beans from the roaster in Bristol taste lighter than
      the old ones and how the barista suggests drinking them.""",
      "hospitality", "counter card",
      ["c-lex-001", "c-str-002", "c-fmt-001"], "adversarial",
      lexicon="coffee", nudge="initial_flow"),

    T("d09-postmortem-summary",
      """Write a 130-word postmortem summary for the leadership team about
      Tuesday's outage, the rollback that failed, the queue backlog it left
      behind, and the monitoring gap we found afterwards.""",
      "engineering", "executive summary",
      ["c-lex-012", "c-chr-022", "c-fmt-001", "c-lex-026"], "adversarial",
      lexicon="software", nudge="punct_readability"),

    T("d10-teaching-reference",
      """Write a reference for a former employee applying to be a secondary
      teacher, covering her attendance, her relationship with the department,
      and the evening classes she ran for three years.""",
      "education", "reference letter",
      ["c-chr-001", "c-chr-020", "c-fmt-001"], "adversarial",
      key_terms=("employee", "secondary teacher", "attendance", "department"),
      nudge="letter_audit"),

    T("d11-safeguarding-briefing",
      """Write a 120-word briefing for volunteers about the new safeguarding
      policy. Cover three points in flowing prose: what changed; who it
      affects; when the training has to be finished.""",
      "nonprofit", "policy briefing",
      ["c-chr-021", "c-fmt-001", "c-chr-001"], "adversarial",
      nudge="punct_grammar"),

    T("d12-school-closure",
      """Write a 110-word notice for a school website about tomorrow's closure,
      covering the state of the car park, the ice on the approach road, the
      gritting that did not happen, and when a decision will be made.""",
      "education", "public notice",
      ["c-lex-020", "c-chr-020", "c-fmt-002", "c-chr-002"], "adversarial",
      lexicon="winter", nudge="quotes_source"),

    T("d13-phone-lines",
      """Write a 120-word explanation for a clinic noticeboard about why the
      phone lines are busy at eight in the morning, what the triage nurse asks
      and how a same-day appointment gets allocated.""",
      "healthcare", "public notice",
      ["c-lex-013", "c-str-004", "c-fmt-001", "c-fmt-002"], "adversarial",
      lexicon="medicine", nudge="initial_last"),

    T("d14-insulation-workshop",
      """Write a 100-word summary of a well-attended, day-long workshop on
      low-cost home insulation for a housing association newsletter.""",
      "housing", "event summary",
      ["c-chr-023", "c-fmt-001", "c-lex-025", "c-chr-001"], "adversarial",
      nudge="punct_house_style"),

    T("d15-bakery-shelf-card",
      """Write a 120-word shelf card for a bakery counter about the rye (baked
      on Tuesdays and Fridays only), the seeded sourdough (which sells out by
      ten) and the crust everybody asks about.""",
      "food", "shelf card",
      ["c-lex-021", "c-chr-025", "c-fmt-001", "c-chr-004"], "adversarial",
      lexicon="bakery", nudge="punct_readability"),

    T("d16-exams-faq",
      """Write a 130-word answer for a school website. Parents keep asking the
      same three things. When do the exams start? What happens if coursework
      arrives late? Who should they contact in the office?""",
      "education", "FAQ answer",
      ["c-lex-014", "c-chr-024", "c-fmt-001", "c-chr-001"], "adversarial",
      lexicon="school", nudge="lists_scannable"),

    T("d17-harvest-supper",
      """Write a 140-word account of the harvest supper for a parish
      newsletter, covering the raffle, the queue at the hatch, the speeches
      that ran long, and the walk home in the dark.""",
      "community", "event writeup",
      ["c-chr-020", "c-str-003", "c-fmt-001", "c-lex-026"], "adversarial",
      nudge="punct_readability"),

    T("d18-hall-accounts",
      """Write a short statement for a village hall committee about the annual
      accounts, the shortfall on the heating bill and the plan to raise the
      hall hire charge.""",
      "community", "financial statement",
      ["c-chr-002", "c-str-007", "c-fmt-001"], "adversarial",
      key_terms=("annual accounts", "village hall", "heating", "charge"),
      nudge="letter_once"),

    T("d19-tasting-evening",
      """Write a 140-word introduction for a tasting evening, covering the
      three whites from the cellar, the vintage everybody argues about, the
      cheese the sommelier insists on, and the timing of the walk back.""",
      "food and drink", "event introduction",
      ["c-lex-018", "c-chr-022", "c-fmt-001", "c-lex-025"], "adversarial",
      lexicon="wine", nudge="punct_readability"),

    T("d20-tool-store",
      """Write a notice for a community workshop about the tool store, the
      donations of timber that keep arriving and the storage problem in the
      corridor.""",
      "community", "notice",
      ["c-chr-004", "c-str-001", "c-fmt-001"], "adversarial",
      key_terms=("workshop", "tool store", "donations", "corridor"),
      nudge="letter_names"),
    T("d21-wildlife-trip",
      """Write a 140-word summary of the school trip to the wildlife park for
      the newsletter, covering the coach that broke down, the packed lunches,
      the otters, and the head count at the gate.""",
      "education", "trip writeup",
      ["c-chr-020", "c-fmt-002", "c-lex-027", "c-chr-008"], "adversarial",
      nudge="punct_readability"),

    T("d22-flood-of-2019",
      """Write a 130-word account of the 2019 flood for a village history
      project, covering the sandbags, the pub cellar, the volunteers who came
      from two towns over, and the insurance argument afterwards.""",
      "history", "local history piece",
      ["c-chr-022", "c-str-002", "c-fmt-001", "c-chr-002"], "adversarial",
      nudge="initial_flow"),

    T("d23-darkroom-show",
      """Write a 120-word introduction for an exhibition of darkroom prints,
      mentioning the lens (a battered fifty), the exposure times (often whole
      minutes) and the framing choices that hold the series together.""",
      "photography", "exhibition text",
      ["c-lex-022", "c-chr-025", "c-fmt-001", "c-chr-001"], "adversarial",
      lexicon="photography", nudge="words_expert"),

    T("d24-no-dig-beds",
      """Write a 110-word piece for a gardening column about no-dig beds, the
      well-rotted compost that makes them work, and the weed-free summer that
      follows.""",
      "gardening", "column",
      ["c-lex-010", "c-chr-023", "c-fmt-001", "c-chr-004"], "adversarial",
      lexicon="gardening", nudge="punct_house_style"),

    T("d25-clinic-website",
      """Write a 130-word answer for a clinic website. Patients keep asking the
      same things. How long is the wait for a referral? Can somebody else
      collect a prescription? What actually counts as urgent?""",
      "healthcare", "FAQ answer",
      ["c-lex-013", "c-chr-024", "c-fmt-001", "c-chr-002"], "adversarial",
      lexicon="medicine", nudge="punct_grammar"),

    T("d26-summer-series",
      """Write a 120-word club newsletter piece about the summer series,
      covering the pace groups, the trail route that changed, the blister
      clinic somebody joked about, and the barbecue at the finish.""",
      "sport", "newsletter piece",
      ["c-lex-007", "c-chr-020", "c-fmt-001", "c-lex-025"], "adversarial",
      lexicon="running", nudge="words_readability"),

    T("d27-roof-brief",
      """Write a short brief for a contractor about the roof repairs, the
      rainwater goods that are rusting and the parapet render that has cracked
      along the north side.""",
      "construction", "works brief",
      ["c-chr-001", "c-str-004", "c-fmt-001"], "adversarial",
      key_terms=("roof repairs", "rainwater", "parapet render", "rusting"),
      nudge="letter_once"),

    T("d28-warehouse-induction",
      """Write a 120-word induction note for a new warehouse hire. Cover the
      essentials in prose: the shift pattern; the safety boots; the way holiday
      is booked.""",
      "logistics", "induction note",
      ["c-chr-021", "c-fmt-001", "c-lex-026", "c-chr-001"], "adversarial",
      nudge="punct_grammar"),

    T("d29-branch-refurbishment",
      """Write a 130-word note for a branch noticeboard about the
      refurbishment, the temporary shelves by the door, the reading room
      closure, and the loans that will be renewed automatically.""",
      "civic", "notice",
      ["c-lex-023", "c-chr-020", "c-str-001", "c-fmt-001"], "adversarial",
      lexicon="library", nudge="words_accuracy"),

    T("d30-on-call-note",
      """Write a 120-word note for on-call engineers about the run-book change,
      the alert that now fires on queue depth, and the follow-up expected after
      every rollback.""",
      "engineering", "internal note",
      ["c-lex-012", "c-chr-023", "c-fmt-002", "c-chr-003"], "adversarial",
      lexicon="software", nudge="punct_readability"),
)



# ==========================================================================
# BLOCK E -- HELD-OUT split, CLEAN.
#
# Held-out types under no pressure: the baseline the adversarial held-out
# scenarios are compared against. Prompts are deliberately sparse in whichever
# mark the scenario bans, so no punctuation_density distractor qualifies -- the
# generator raises if one does.
# ==========================================================================

_BLOCK_E: Tuple[Dict[str, Any], ...] = (
    T("e01-ridge-path",
      """Write a paragraph for a walking guide describing the climb out of the
      valley to the ridge path and the view that opens at the top.""",
      "travel", "guidebook prose",
      ["c-str-001", "c-fmt-001"], "clean"),

    T("e02-behind-the-bar",
      """Write a short account of a first shift working behind a bar for a
      hospitality training handbook.""",
      "hospitality", "training material",
      ["c-str-002", "c-fmt-002", "c-lex-025"], "clean"),

    T("e03-standing-order",
      """Explain what a standing order is and how it differs from a direct
      debit for somebody opening a first bank account.""",
      "finance", "explainer",
      ["c-chr-021", "c-fmt-001", "c-chr-002"], "clean"),

    T("e04-harbour-dawn",
      """Write a 120-word description of a working harbour at dawn for a travel
      anthology.""",
      "travel", "descriptive writing",
      ["c-chr-024", "c-fmt-001", "c-chr-001"], "clean"),

    T("e05-water-butts",
      """Write a note for new allotment holders explaining where the water butts
      are and how the watering rota works in August.""",
      "gardening", "notice",
      ["c-chr-025", "c-fmt-002", "c-chr-001"], "clean"),

    T("e06-skipped-stitches",
      """Explain why a sewing machine skips stitches, for somebody who has just
      bought one second hand.""",
      "craft", "explainer",
      ["c-chr-023", "c-fmt-001", "c-chr-001"], "clean"),

    T("e07-colleague-farewell",
      """Write a short farewell message for a colleague leaving after eleven
      years in the same department.""",
      "workplace", "farewell message",
      ["c-chr-020", "c-fmt-001", "c-chr-002"], "clean"),

    T("e08-heat-pump",
      """Explain how a heat pump moves warmth from cold outdoor air into a
      house.""",
      "science", "explainer",
      ["c-chr-022", "c-lex-026", "c-fmt-001", "c-fmt-002"], "clean"),

    T("e09-towpath-morning",
      """Write a 130-word description of a January morning on a canal towpath
      for a nature column.""",
      "nature", "descriptive writing",
      ["c-str-003", "c-chr-001", "c-fmt-001"], "clean"),

    T("e10-chairing-advice",
      """Write a paragraph of advice for somebody about to chair a meeting for
      the first time.""",
      "business", "advice",
      ["c-str-004", "c-fmt-001", "c-lex-040"], "clean"),

    T("e11-slow-week",
      """Write a short piece for an athletics club newsletter about the value of
      an easy training week.""",
      "sport", "newsletter piece",
      ["c-str-005", "c-fmt-002"], "clean"),

    T("e12-lighthouse-inside",
      """Describe the inside of a working lighthouse for a children's
      encyclopaedia entry.""",
      "reference", "encyclopaedia entry",
      ["c-str-006", "c-fmt-001", "c-lex-027", "c-lex-045"], "clean"),

    T("e13-wet-saturday-market",
      """Write a 110-word account of a street market on a wet Saturday for a
      local history blog.""",
      "history", "blog post",
      ["c-str-008", "c-fmt-001"], "clean"),

    T("e14-glass-recycling",
      """Explain what happens to household glass after it is collected from the
      kerb.""",
      "environment", "explainer",
      ["c-chr-026", "c-fmt-002", "c-chr-001"], "clean"),

    T("e15-charity-shop-welcome",
      """Write a short welcome for a new volunteer starting at a charity
      shop.""",
      "nonprofit", "welcome note",
      ["c-str-007", "c-chr-003", "c-fmt-001"], "clean"),

    T("e16-hotel-breakfast",
      """Write a 120-word card for a hotel room explaining the breakfast
      arrangements and the walk down to the harbour.""",
      "hospitality", "guest information",
      ["c-chr-020", "c-str-002", "c-fmt-001", "c-lex-025"], "clean"),

    T("e17-footpath-bridleway",
      """Write a paragraph about the difference between a footpath and a
      bridleway for a countryside code leaflet.""",
      "reference", "leaflet copy",
      ["c-chr-021", "c-fmt-002", "c-lex-041", "c-fmt-001"], "clean"),

    T("e18-wheel-repair",
      """Write a short reflection on truing a buckled wheel for the first
      time.""",
      "craft", "reflective piece",
      ["c-str-001", "c-chr-002", "c-fmt-001"], "clean"),

    T("e19-cricket-ground",
      """Write a 100-word description of an empty cricket ground in
      February.""",
      "sport", "descriptive writing",
      ["c-chr-024", "c-fmt-002", "c-lex-042", "c-chr-008"], "clean"),

    T("e20-office-recycling",
      """Explain to a new starter how the office recycling actually works and
      where the food waste caddy lives.""",
      "workplace", "explainer",
      ["c-str-006", "c-chr-004", "c-fmt-002"], "clean"),

    T("e21-fire-engine-label",
      """Write a museum label of about 80 words for a Victorian fire engine
      that was pulled by hand.""",
      "culture", "museum label",
      ["c-chr-025", "c-str-005", "c-fmt-001", "c-lex-028"], "clean"),

    T("e22-opening-times",
      """Write a short note for a community centre door about the change of
      opening times over the summer.""",
      "community", "notice",
      ["c-chr-023", "c-fmt-002", "c-lex-029", "c-chr-002"], "clean"),
    T("e23-hospital-radio",
      """Write a short piece for a hospital radio show introducing a request
      from a listener on the cardiac ward.""",
      "media", "radio script",
      ["c-str-003", "c-fmt-001"], "clean"),

    T("e24-compost-toilet",
      """Explain how a compost toilet works for a campsite information
      board.""",
      "environment", "information board",
      ["c-chr-020", "c-fmt-002", "c-lex-026", "c-chr-001"], "clean"),

    T("e25-boatyard-autumn",
      """Write a 100-word description of an autumn morning at a canal
      boatyard.""",
      "nature", "descriptive writing",
      ["c-str-008", "c-chr-001", "c-fmt-001"], "clean"),

    T("e26-swim-timetable",
      """Write a paragraph for a leisure centre website about the new swimming
      timetable for the over sixties.""",
      "community", "website copy",
      ["c-chr-025", "c-fmt-001", "c-lex-030", "c-chr-004"], "clean"),

    T("e27-chicken-sitting",
      """Write a short thank you to a neighbour who fed the chickens for a
      fortnight.""",
      "personal", "thank-you note",
      ["c-str-002", "c-chr-008", "c-fmt-002"], "clean"),

    T("e28-bakery-night-shift",
      """Write a 120-word account of a night shift in a bakery for a careers
      leaflet.""",
      "careers", "leaflet copy",
      ["c-chr-024", "c-fmt-001", "c-lex-044", "c-chr-002"], "clean"),

    T("e29-reading-music",
      """Write a short reflection on learning to read music as an adult.""",
      "arts", "reflective piece",
      ["c-str-005", "c-chr-002", "c-fmt-001"], "clean"),

    T("e30-leasehold",
      """Explain what a leasehold is to somebody buying a flat for the first
      time.""",
      "legal", "explainer",
      ["c-chr-026", "c-fmt-001", "c-lex-047", "c-chr-001"], "clean"),
)



#: The corpus, in emission order. scenario_id is positional (s-0001 ...), so
#: appending a task never renumbers an existing one; inserting one does, and
#: that is a deliberate diff rather than a silent renumber.
TASKS: List[Dict[str, Any]] = list(_BLOCK_A + _BLOCK_B + _BLOCK_C + _BLOCK_D + _BLOCK_E)
