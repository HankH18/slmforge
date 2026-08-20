"""Teacher-distillation pipeline for the Forbidden-Constraint Holder (T-5).

The shape of the thing, in one paragraph: a fresh task is paired with a
train-split constraint set, the pair is rendered through
:func:`scenarios.render.render_user_message` -- the ONE definition of the
prompt, shared with the eval harness -- a teacher drafts a response, and
:func:`constraints.checkers.check` (via :func:`harness.run_checkers`) decides
whether it ships. A draft that violates anything is retried with the violation
quoted back at it, and if it still violates, it is rejected and the reason is
recorded. Nothing unverified is written.

Four invariants this package enforces rather than intends, each with a test and
each re-checked by :mod:`datagen.audit` against the file on disk:

1. TRAIN-SPLIT TYPES ONLY. ``banned_punctuation`` and ``sentence_initial_letter``
   are the generalization measurement. A training example carrying one destroys
   the number the project exists to report, so the sampler, the writer and the
   auditor each refuse it independently.
2. WIDE SAMPLING. No constraint type may exceed 25% of the constraint slots.
3. TASK DISJOINTNESS. Training tasks are a FRESH pool (:mod:`datagen.task_pool`)
   with no text in common with ``data/scenarios/corpus.jsonl``. Disjointness is
   enforced at the task level, which is strictly stronger than the
   (task, constraint-set) pair level the ticket requires.
4. PROMPT CONTRACT. The stored user message is the shared renderer's output,
   byte for byte, and the row carries its own ``scenario`` so a third party can
   re-derive it.
"""
from __future__ import annotations

__all__ = ["DatagenError"]


class DatagenError(RuntimeError):
    """A structural problem in the pipeline -- never a measurement.

    Raised, never swallowed: every use of it names a condition under which a
    written dataset would silently mislead a downstream reader.
    """
