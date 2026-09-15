"""What a retriever returns, and how it is measured."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..questions import Question


@dataclass
class Retrieved:
    #: Documents behind whatever was retrieved, best first.
    docs: list[str]
    #: The text a model would be given.
    context: str
    #: Graph edges, as (source name, type, target name). Empty for passage
    #: retrieval, which is exactly the difference being measured.
    facts: list[tuple[str, str, str]] = field(default_factory=list)
    #: The knob that produced this result — k for passages, hops for the graph.
    budget: int = 0

    @property
    def words(self) -> int:
        return len(self.context.split())

    def support_recall(self, question: Question) -> float:
        if not question.support:
            return 1.0
        found = sum(1 for doc in question.support if doc in self.docs)
        return found / len(question.support)

    def fact_recall(self, question: Question) -> float | None:
        """None when the question carries no edge labels, or nothing has facts."""
        if not question.needs or not self.facts:
            return None
        have = set(self.facts)
        found = sum(1 for edge in question.needs if edge in have)
        return found / len(question.needs)


class Retriever(Protocol):
    name: str

    def retrieve(self, question: Question, budget: int) -> Retrieved: ...

    @property
    def budgets(self) -> tuple[int, ...]:
        """The settings to sweep when plotting recall against context size."""
        ...
