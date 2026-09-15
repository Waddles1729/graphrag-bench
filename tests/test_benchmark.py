import json

from graphrag_bench import benchmark, report
from graphrag_bench.cli import main
from graphrag_bench.embed import TfidfEmbedder
from graphrag_bench.graph.store import MemoryGraphStore
from graphrag_bench.retrieve import GraphRetriever, OracleRetriever, VectorRetriever


def _trials(gold, extracted, chunks, questions):
    return benchmark.run(
        [
            VectorRetriever(chunks, TfidfEmbedder()),
            OracleRetriever(chunks),
            GraphRetriever(gold, MemoryGraphStore(), label="graph(gold)"),
            GraphRetriever(extracted, MemoryGraphStore(), label="graph(extracted)"),
        ],
        questions,
    )


def test_every_retriever_is_scored_on_every_question(gold, extracted, chunks, questions):
    trials = _trials(gold, extracted, chunks, questions)
    for name in {trial.retriever for trial in trials}:
        for budget in {t.budget for t in trials if t.retriever == name}:
            scored = {
                t.question_id for t in trials
                if t.retriever == name and t.budget == budget
            }
            assert scored == {question.id for question in questions}


def test_recall_never_falls_as_the_passage_budget_grows(gold, extracted, chunks, questions):
    trials = _trials(gold, extracted, chunks, questions)
    name = next(t.retriever for t in trials if t.retriever.startswith("vector"))
    per_budget = {}
    for trial in trials:
        if trial.retriever == name:
            per_budget.setdefault(trial.budget, {})[trial.question_id] = trial.support_recall

    budgets = sorted(per_budget)
    for smaller, larger in zip(budgets, budgets[1:], strict=False):
        for question_id, recall in per_budget[smaller].items():
            assert per_budget[larger][question_id] >= recall, (question_id, smaller, larger)


def test_the_oracle_is_an_upper_bound_on_passage_recall(gold, extracted, chunks, questions):
    trials = _trials(gold, extracted, chunks, questions)
    oracle = {t.question_id: t.support_recall for t in trials if t.retriever == "oracle"}
    for trial in trials:
        if trial.retriever.startswith("vector"):
            assert trial.support_recall <= oracle[trial.question_id] + 1e-9


def test_graph_context_is_smaller_than_oracle_context(gold, extracted, chunks, questions):
    trials = _trials(gold, extracted, chunks, questions)
    oracle = {t.question_id: t.words for t in trials if t.retriever == "oracle"}
    graph = {
        t.question_id: t.words
        for t in trials
        if t.retriever == "graph(gold)" and t.budget == 2
    }
    cheaper = sum(1 for qid in graph if graph[qid] < oracle[qid])
    assert cheaper >= 0.8 * len(graph), f"only {cheaper}/{len(graph)} were cheaper"


def test_a_missing_edge_costs_answerable_questions(gold, extracted, chunks, questions):
    trials = _trials(gold, extracted, chunks, questions)
    lost = report.missing_edges(trials, "graph(extracted)", questions)
    intact = report.missing_edges(trials, "graph(gold)", questions)
    assert intact == []
    assert lost, "the extracted graph should lose at least one question"


def test_the_best_budget_is_the_smallest_one_that_wins(gold, extracted, chunks, questions):
    trials = _trials(gold, extracted, chunks, questions)
    chosen = benchmark.best_budget(trials, "graph(gold)")
    complete = {}
    for trial in trials:
        if trial.retriever == "graph(gold)":
            complete.setdefault(trial.budget, []).append(trial.complete)
    rates = {budget: sum(values) / len(values) for budget, values in complete.items()}
    best = max(rates.values())
    assert rates[chosen] == best
    assert all(budget >= chosen for budget, rate in rates.items() if rate == best)


def test_cost_of_full_support_reports_none_when_it_never_happens():
    trials = [
        benchmark.Trial("v", 1, "q1", "lookup", 0.5, None, 100, 1.0),
        benchmark.Trial("v", 2, "q1", "lookup", 0.5, None, 200, 1.0),
    ]
    assert benchmark.cost_of_full_support(trials, "v") == {"q1": None}


def test_summaries_cover_every_kind(gold, extracted, chunks, questions):
    trials = _trials(gold, extracted, chunks, questions)
    kinds = {summary.kind for summary in benchmark.summarise(trials)}
    assert kinds == {question.kind for question in questions}


# -- the command line -------------------------------------------------------


def test_extract_scores_against_gold(capsys):
    assert main(["extract", "--score"]) == 0
    printed = capsys.readouterr().out
    assert "precision" in printed
    assert "all relations" in printed


def test_extract_writes_a_graph(tmp_path, capsys):
    out = tmp_path / "graph.json"
    assert main(["extract", "--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["entities"] and payload["relations"]


def test_bench_runs_end_to_end(tmp_path, capsys):
    out = tmp_path / "trials.json"
    assert main(["bench", "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "full support" in printed
    assert "graph(gold)" in printed
    assert json.loads(out.read_text())


def test_ask_prints_the_facts_it_walked(capsys):
    assert main(["ask", "Which team owns the service checkout-api calls for fraud?"]) == 0
    printed = capsys.readouterr().out
    assert "fraud-scoring" in printed
    assert "document(s)" in printed


def test_a_missing_data_directory_is_an_error_not_a_traceback(capsys):
    assert main(["--data", "/nowhere", "extract"]) == 2
    assert "graphrag-bench:" in capsys.readouterr().err
