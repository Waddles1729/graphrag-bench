"""Running the comparison, and the numbers it produces.

The measurement is deliberately narrow. This benchmark scores *retrieval*, not
answers: whether the facts a question needs reached the context, and what that
cost. Answer quality is a property of the model you put downstream, and mixing
the two is how a retrieval claim ends up resting on a prompt.

The metrics:

* **support recall** — of the documents that contain the needed facts, how many
  were retrieved. The only metric every strategy can be scored on, so it is the
  one used head to head.
* **context words** — the size of what would be sent to the model. Recall is
  cheap if you are allowed to retrieve everything, so recall without this number
  beside it means nothing.
* **fact recall** — of the specific edges an answer rests on, how many the graph
  actually contained. Defined for graph strategies only; it is where an
  extraction error becomes a wrong answer.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from .questions import Question


@dataclass
class Trial:
    retriever: str
    budget: int
    question_id: str
    kind: str
    support_recall: float
    fact_recall: float | None
    words: int
    milliseconds: float

    @property
    def complete(self) -> bool:
        return self.support_recall >= 1.0


def run(retrievers, questions: list[Question]) -> list[Trial]:
    trials: list[Trial] = []
    for retriever in retrievers:
        for budget in retriever.budgets:
            for question in questions:
                started = time.perf_counter()
                result = retriever.retrieve(question, budget)
                elapsed = (time.perf_counter() - started) * 1000
                trials.append(
                    Trial(
                        retriever=retriever.name,
                        budget=budget,
                        question_id=question.id,
                        kind=question.kind,
                        support_recall=result.support_recall(question),
                        fact_recall=result.fact_recall(question),
                        words=result.words,
                        milliseconds=elapsed,
                    )
                )
    return trials


@dataclass
class Summary:
    retriever: str
    budget: int
    kind: str
    questions: int
    support_recall: float
    complete: float
    median_words: int
    fact_recall: float | None


def summarise(trials: list[Trial], by_kind: bool = True) -> list[Summary]:
    buckets: dict[tuple[str, int, str], list[Trial]] = {}
    for trial in trials:
        key = (trial.retriever, trial.budget, trial.kind if by_kind else "all")
        buckets.setdefault(key, []).append(trial)

    summaries = []
    for (retriever, budget, kind), group in buckets.items():
        facts = [t.fact_recall for t in group if t.fact_recall is not None]
        summaries.append(
            Summary(
                retriever=retriever,
                budget=budget,
                kind=kind,
                questions=len(group),
                support_recall=statistics.fmean(t.support_recall for t in group),
                complete=statistics.fmean(1.0 if t.complete else 0.0 for t in group),
                median_words=int(statistics.median(t.words for t in group)),
                fact_recall=statistics.fmean(facts) if facts else None,
            )
        )
    return sorted(summaries, key=lambda s: (s.retriever, s.kind, s.budget))


def best_budget(trials: list[Trial], retriever: str) -> int:
    """The smallest budget that maximises complete-support rate.

    Chosen this way rather than by recall so that a strategy is credited for
    reaching the whole answer, not for getting most of the way there — half the
    facts for a multi-hop question is not half an answer.
    """
    candidates = [t for t in trials if t.retriever == retriever]
    if not candidates:
        raise ValueError(f"no trials for {retriever}")

    by_budget: dict[int, list[Trial]] = {}
    for trial in candidates:
        by_budget.setdefault(trial.budget, []).append(trial)

    scored = [
        (statistics.fmean(1.0 if t.complete else 0.0 for t in group), -budget, budget)
        for budget, group in by_budget.items()
    ]
    return max(scored)[2]


def cost_of_full_support(trials: list[Trial], retriever: str) -> dict[str, int | None]:
    """Per question: the context size at the smallest budget that got everything.

    None where no budget in the sweep was enough — which is the answer for most
    aggregation and absence questions under passage retrieval, and is more
    informative than any recall figure.
    """
    out: dict[str, int | None] = {}
    for trial in sorted(trials, key=lambda t: t.budget):
        if trial.retriever != retriever:
            continue
        out.setdefault(trial.question_id, None)
        if out[trial.question_id] is None and trial.complete:
            out[trial.question_id] = trial.words
    return out
