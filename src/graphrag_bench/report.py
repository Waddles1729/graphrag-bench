"""Rendering results as text tables."""

from __future__ import annotations

import statistics

from .benchmark import Trial, best_budget, cost_of_full_support
from .questions import KINDS


def table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    numeric = [
        all(_looks_numeric(row[i]) for row in rows) if rows else False
        for i in range(len(headers))
    ]

    def line(cells: list[str]) -> str:
        return "  ".join(
            cell.rjust(widths[i]) if numeric[i] else cell.ljust(widths[i])
            for i, cell in enumerate(cells)
        ).rstrip()

    out = [line(headers), "  ".join("-" * width for width in widths)]
    out.extend(line(row) for row in rows)
    return "\n".join(out)


def _looks_numeric(cell: str) -> bool:
    return bool(cell) and cell.replace(".", "").replace("%", "").replace("-", "").isdigit()


def headline(trials: list[Trial], retrievers: list[str]) -> str:
    """Complete-support rate and context size per question kind, best budget each."""
    rows = []
    for name in retrievers:
        budget = best_budget(trials, name)
        for kind in KINDS:
            group = [
                t for t in trials
                if t.retriever == name and t.budget == budget and t.kind == kind
            ]
            if not group:
                continue
            complete = statistics.fmean(1.0 if t.complete else 0.0 for t in group)
            rows.append([
                name,
                str(budget),
                kind,
                f"{complete * 100:.0f}%",
                str(int(statistics.median(t.words for t in group))),
            ])
    return table(["strategy", "budget", "kind", "full support", "words"], rows)


def curve(trials: list[Trial], retriever: str) -> str:
    """How complete-support rate and context size move together."""
    budgets = sorted({t.budget for t in trials if t.retriever == retriever})
    rows = []
    for budget in budgets:
        group = [t for t in trials if t.retriever == retriever and t.budget == budget]
        rows.append([
            str(budget),
            f"{statistics.fmean(1.0 if t.complete else 0.0 for t in group) * 100:.0f}%",
            f"{statistics.fmean(t.support_recall for t in group) * 100:.0f}%",
            str(int(statistics.median(t.words for t in group))),
        ])
    return table(["budget", "full support", "recall", "median words"], rows)


def unreachable(trials: list[Trial], retriever: str, questions) -> list[str]:
    """Question ids this strategy never fully answered, at any budget."""
    costs = cost_of_full_support(trials, retriever)
    order = {question.id: index for index, question in enumerate(questions)}
    return sorted((qid for qid, cost in costs.items() if cost is None), key=order.get)


def fact_recall(trials: list[Trial], retrievers: list[str], questions) -> str:
    """Of the edges each answer rests on, how many the graph actually held.

    Support recall cannot see an extraction error: a missing edge leaves the
    document it came from in the retrieved set, because some *other* edge points
    there. This is the metric where a missed relation becomes a missing answer.
    """
    labelled = {q.id for q in questions if q.needs}
    rows = []
    for name in retrievers:
        budget = best_budget(trials, name)
        for kind in KINDS:
            group = [
                t for t in trials
                if t.retriever == name
                and t.budget == budget
                and t.kind == kind
                and t.question_id in labelled
                and t.fact_recall is not None
            ]
            if not group:
                continue
            complete = [t for t in group if t.fact_recall >= 1.0]
            rows.append([
                name,
                kind,
                f"{statistics.fmean(t.fact_recall for t in group) * 100:.0f}%",
                f"{len(complete)}/{len(group)}",
            ])
    return table(["strategy", "kind", "fact recall", "questions with every edge"], rows)


def missing_edges(trials: list[Trial], retriever: str, questions) -> list[str]:
    budget = best_budget(trials, retriever)
    incomplete = {
        t.question_id
        for t in trials
        if t.retriever == retriever
        and t.budget == budget
        and t.fact_recall is not None
        and t.fact_recall < 1.0
    }
    return [q.id for q in questions if q.id in incomplete]


def cost_comparison(trials: list[Trial], left: str, right: str, questions) -> str:
    """Side by side: what full support costs each strategy, per question."""
    left_cost = cost_of_full_support(trials, left)
    right_cost = cost_of_full_support(trials, right)
    rows = []
    for question in questions:
        a, b = left_cost.get(question.id), right_cost.get(question.id)
        ratio = f"{a / b:.1f}x" if a and b else "—"
        rows.append([
            question.kind,
            question.id,
            str(a) if a is not None else "never",
            str(b) if b is not None else "never",
            ratio,
        ])
    return table(["kind", "question", f"{left} words", f"{right} words", "ratio"], rows)
