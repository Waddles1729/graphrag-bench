"""Command line: extract, bench, ask."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import benchmark, report
from .corpus import all_chunks, load_corpus
from .embed import build_embedder
from .extract import (
    RuleExtractor,
    relation_diff,
    score_entities,
    score_relations,
)
from .graph.store import MemoryGraphStore
from .questions import load_questions
from .retrieve import GraphRetriever, HybridRetriever, OracleRetriever, VectorRetriever
from .schema import RELATION_TYPES, Graph, load_graph


def _default_data() -> Path:
    """The repository's data directory when running from a checkout, else ./data."""
    beside_source = Path(__file__).resolve().parents[2] / "data"
    return beside_source if beside_source.is_dir() else Path("data")


DATA = _default_data()


def _paths(args) -> tuple[Path, Path, Path]:
    root = Path(args.data)
    return root / "corpus", root / "gold_graph.json", root / "questions.jsonl"


def _store(args):
    if args.graph_store == "neo4j":
        from .graph.neo4j_store import Neo4jGraphStore

        return Neo4jGraphStore(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            user=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "neo4j"),
        )
    return MemoryGraphStore()


# --------------------------------------------------------------------------


def cmd_extract(args) -> int:
    corpus_dir, gold_path, _ = _paths(args)
    documents = load_corpus(corpus_dir)
    extracted = RuleExtractor().extract(documents)

    if args.out:
        Path(args.out).write_text(
            json.dumps(extracted.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.out}")

    if not args.score:
        print(
            f"{len(extracted.entities)} entities, {len(extracted.relations)} relations"
        )
        return 0

    gold = load_graph(gold_path)
    rows = []
    entity_score = score_entities(extracted, gold)
    rows.append([
        "entities",
        f"{entity_score.precision:.2f}",
        f"{entity_score.recall:.2f}",
        f"{entity_score.f1:.2f}",
        str(entity_score.true_positives + entity_score.false_negatives),
    ])
    for relation_type in RELATION_TYPES:
        score = score_relations(extracted, gold, relation_type)
        rows.append([
            relation_type,
            f"{score.precision:.2f}",
            f"{score.recall:.2f}",
            f"{score.f1:.2f}",
            str(score.true_positives + score.false_negatives),
        ])
    overall = score_relations(extracted, gold)
    rows.append([
        "all relations",
        f"{overall.precision:.2f}",
        f"{overall.recall:.2f}",
        f"{overall.f1:.2f}",
        str(overall.true_positives + overall.false_negatives),
    ])
    print(report.table(["", "precision", "recall", "F1", "gold"], rows))

    spurious, missed = relation_diff(extracted, gold)
    names = {entity.id: entity.name for entity in gold.entities.values()}

    def show(relation) -> str:
        source = names.get(relation.source, relation.source)
        target = names.get(relation.target, relation.target)
        return f"{source} {relation.type} {target}  ({relation.doc})"

    if missed:
        print("\nmissed:")
        for relation in missed:
            print(f"  - {show(relation)}")
    if spurious:
        print("\nspurious:")
        for relation in spurious:
            print(f"  + {show(relation)}")
    if not missed and not spurious:
        print("\nthe extracted graph matches the gold graph exactly")
    return 0


def _retrievers(args, documents, gold: Graph, extracted: Graph):
    chunks = all_chunks(documents)
    vector = VectorRetriever(chunks, build_embedder(args.embeddings))
    built = [
        vector,
        OracleRetriever(chunks),
        GraphRetriever(gold, _store(args), label="graph(gold)"),
        GraphRetriever(extracted, MemoryGraphStore(), label="graph(extracted)"),
    ]
    if args.hybrid:
        built.append(HybridRetriever(built[2], vector))
    return built


def cmd_bench(args) -> int:
    corpus_dir, gold_path, questions_path = _paths(args)
    documents = load_corpus(corpus_dir)
    gold = load_graph(gold_path)
    extracted = RuleExtractor().extract(documents)
    questions = load_questions(questions_path)

    retrievers = _retrievers(args, documents, gold, extracted)
    trials = benchmark.run(retrievers, questions)
    names = [retriever.name for retriever in retrievers]

    print(f"{len(questions)} questions, {len(documents)} documents, "
          f"{len(all_chunks(documents))} chunks\n")

    print("Full support by question kind, each strategy at its best budget")
    print("(budget is k passages, or hops for the graph)\n")
    print(report.headline(trials, names))

    print("\n\nDid the graph actually hold the edges the answer rests on?")
    print("(support recall cannot see a missing edge; this can)\n")
    print(report.fact_recall(trials, ["graph(gold)", "graph(extracted)"], questions))

    for label in ("graph(gold)", "graph(extracted)"):
        incomplete = report.missing_edges(trials, label, questions)
        if incomplete:
            print(f"\nMissing at least one required edge under {label}:")
            for question_id in incomplete:
                print(f"  {question_id}")

    unanswerable = [q.id for q in questions if not q.needs]
    if unanswerable:
        print("\nNot expressible in this schema at all "
              "(no edge type carries the fact the question asks for):")
        for question_id in unanswerable:
            print(f"  {question_id}")

    vector_name = retrievers[0].name
    print(f"\n\nRecall against context size — {vector_name}\n")
    print(report.curve(trials, vector_name))
    print("\n\nRecall against context size — graph(gold)\n")
    print(report.curve(trials, "graph(gold)"))

    print("\n\nWhat full support costs, per question\n")
    print(report.cost_comparison(trials, vector_name, "graph(gold)", questions))

    never = report.unreachable(trials, vector_name, questions)
    if never:
        print(f"\nNever fully retrieved by {vector_name}, at any k in the sweep:")
        for question_id in never:
            print(f"  {question_id}")

    graph_never = report.unreachable(trials, "graph(gold)", questions)
    if graph_never:
        print("\nNever fully retrieved by graph(gold), at any depth:")
        for question_id in graph_never:
            print(f"  {question_id}")

    if args.json:
        Path(args.json).write_text(
            json.dumps([trial.__dict__ for trial in trials], indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")
    return 0


def cmd_ask(args) -> int:
    corpus_dir, gold_path, _ = _paths(args)
    documents = load_corpus(corpus_dir)
    graph = (
        load_graph(gold_path)
        if args.graph == "gold"
        else RuleExtractor().extract(documents)
    )
    from .questions import Question

    question = Question(
        id="ad-hoc", kind="lookup", hops=args.hops, question=args.question,
        answer_type="set", answer=(), support=(),
    )

    if args.retriever == "vector":
        retriever = VectorRetriever(all_chunks(documents), build_embedder(args.embeddings))
        budget = args.k
    else:
        retriever = GraphRetriever(graph, _store(args))
        budget = args.hops

    result = retriever.retrieve(question, budget)
    print(result.context or "(nothing retrieved)")
    print(f"\n{result.words} words · {len(result.docs)} document(s): "
          f"{', '.join(result.docs) or 'none'}")
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphrag-bench",
        description="Measure what a knowledge graph adds to retrieval, and what it costs.",
    )
    parser.add_argument("--data", default=str(DATA), help="directory holding corpus/ and labels")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="build the graph from the corpus")
    extract.add_argument("--score", action="store_true", help="compare it to the gold graph")
    extract.add_argument("--out", help="write the extracted graph to this path")
    extract.set_defaults(func=cmd_extract)

    bench = sub.add_parser("bench", help="run the retrieval comparison")
    bench.add_argument("--embeddings", default="tfidf", choices=["tfidf", "spacy", "openai"])
    bench.add_argument("--graph-store", default="memory", choices=["memory", "neo4j"])
    bench.add_argument("--hybrid", action="store_true", help="also score graph + passages")
    bench.add_argument("--json", help="write raw trials to this path")
    bench.set_defaults(func=cmd_bench)

    ask = sub.add_parser("ask", help="retrieve context for one question")
    ask.add_argument("question")
    ask.add_argument("--retriever", default="graph", choices=["graph", "vector"])
    ask.add_argument("--graph", default="gold", choices=["gold", "extracted"])
    ask.add_argument("--graph-store", default="memory", choices=["memory", "neo4j"])
    ask.add_argument("--embeddings", default="tfidf", choices=["tfidf", "spacy", "openai"])
    ask.add_argument("--hops", type=int, default=2)
    ask.add_argument("-k", type=int, default=5)
    ask.set_defaults(func=cmd_ask)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"graphrag-bench: {error}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        # `graphrag-bench bench | head` is a reasonable thing to type, and a
        # traceback is not a reasonable thing to get back for it.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
