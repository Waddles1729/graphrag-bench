"""The labelled question set.

Each question carries three kinds of label, and they answer different things:

* `support` — the documents that together contain the facts needed. This is the
  only label every retrieval strategy can be scored against, so it is the one
  used for the head-to-head comparison.
* `needs` — the specific graph edges the answer rests on. Defined for graph
  strategies only, and the reason an extraction error is visible here at all: a
  relation the extractor missed shows up as a question that cannot be answered,
  rather than as a percentage in a table nobody connects to anything.
* `answer` — what a correct answer says, for the optional generation step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

KINDS = ("lookup", "multihop", "aggregation", "absence", "impact")


@dataclass(frozen=True)
class Question:
    id: str
    kind: str
    hops: int
    question: str
    answer_type: str
    answer: tuple[str, ...]
    support: tuple[str, ...]
    needs: tuple[tuple[str, str, str], ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, payload: dict) -> Question:
        return cls(
            id=payload["id"],
            kind=payload["kind"],
            hops=payload.get("hops", 1),
            question=payload["question"],
            answer_type=payload.get("answer_type", "set"),
            answer=tuple(payload.get("answer", ())),
            support=tuple(payload.get("support", ())),
            needs=tuple(tuple(edge) for edge in payload.get("needs", ())),
        )


def load_questions(path: str | Path) -> list[Question]:
    questions = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            questions.append(Question.from_dict(json.loads(line)))
    if not questions:
        raise ValueError(f"no questions in {path}")
    return questions
