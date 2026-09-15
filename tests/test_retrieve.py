import pytest

from graphrag_bench.embed import TfidfEmbedder, tokenize
from graphrag_bench.graph.store import MemoryGraphStore, serialise
from graphrag_bench.questions import Question
from graphrag_bench.retrieve import (
    GraphRetriever,
    HybridRetriever,
    OracleRetriever,
    VectorRetriever,
)
from graphrag_bench.schema import Entity, Graph, Relation


def _question(text: str, support=(), needs=(), kind="lookup") -> Question:
    return Question(
        id="q", kind=kind, hops=2, question=text, answer_type="set",
        answer=(), support=tuple(support), needs=tuple(needs),
    )


# -- embeddings -------------------------------------------------------------


def test_a_hyphenated_name_also_yields_its_parts():
    assert tokenize("checkout-api") == ["checkout-api", "checkout", "api"]


def test_tfidf_vectors_are_unit_length(chunks):
    embedder = TfidfEmbedder()
    texts = [chunk.text for chunk in chunks]
    embedder.fit(texts)
    matrix = embedder.encode(texts[:5])
    assert matrix.shape[0] == 5
    for row in matrix:
        assert row @ row == pytest.approx(1.0)


def test_encoding_before_fitting_is_an_error():
    with pytest.raises(RuntimeError):
        TfidfEmbedder().encode(["anything"])


def test_an_unseen_term_does_not_break_a_query(chunks):
    embedder = TfidfEmbedder()
    embedder.fit([chunk.text for chunk in chunks])
    vector = embedder.encode(["zzzzz qqqqq"])[0]
    assert vector.shape[0] > 0  # all zeros, but the right shape and no exception


# -- the graph store --------------------------------------------------------


def _toy() -> Graph:
    graph = Graph()
    for entity_id, entity_type, name in [
        ("service:a", "Service", "a"),
        ("service:b", "Service", "b"),
        ("service:c", "Service", "c"),
        ("team:x", "Team", "X"),
    ]:
        graph.add_entity(Entity(entity_id, entity_type, name, doc="catalogue"))
    graph.add_relation(Relation("service:a", "DEPENDS_ON", "service:b", "doc-a"))
    graph.add_relation(Relation("service:b", "DEPENDS_ON", "service:c", "doc-b"))
    graph.add_relation(Relation("team:x", "OWNS", "service:c", "doc-x"))
    return graph


def test_one_hop_reaches_the_neighbours_only():
    store = MemoryGraphStore()
    store.load(_toy())
    found = {r.key for r in store.expand(["service:a"], hops=1)}
    assert found == {("service:a", "DEPENDS_ON", "service:b")}


def test_two_hops_reach_the_far_side_of_an_edge_direction_change():
    # a -> b -> c <- X. Following edge direction alone never reaches X, which
    # is the answer to "who owns the service that a's dependency depends on".
    store = MemoryGraphStore()
    store.load(_toy())
    found = {r.key for r in store.expand(["service:a"], hops=3)}
    assert ("team:x", "OWNS", "service:c") in found


def test_expanding_from_an_unknown_seed_returns_nothing():
    store = MemoryGraphStore()
    store.load(_toy())
    assert store.expand(["service:nope"], hops=2) == []


def test_facts_render_one_per_line():
    graph = _toy()
    lines = serialise(graph.relations, graph)
    assert "X owns c" in lines
    assert len(lines) == 3


# -- retrievers -------------------------------------------------------------


def test_vector_retrieval_returns_at_most_k_chunks(chunks):
    retriever = VectorRetriever(chunks, TfidfEmbedder())
    result = retriever.retrieve(_question("who owns the ledger?"), 3)
    assert len(result.context.split("\n\n")) >= 1
    assert 0 < len(result.docs) <= 3


def test_a_bigger_k_never_loses_a_document(chunks):
    retriever = VectorRetriever(chunks, TfidfEmbedder())
    question = _question("which team owns fraud-scoring?")
    small = set(retriever.retrieve(question, 3).docs)
    large = set(retriever.retrieve(question, 8).docs)
    assert small <= large


def test_the_oracle_returns_exactly_the_support(chunks, questions):
    retriever = OracleRetriever(chunks)
    question = questions[0]
    result = retriever.retrieve(question)
    assert result.docs == list(question.support)
    assert result.support_recall(question) == 1.0


def test_graph_retrieval_follows_two_hops(gold, questions):
    retriever = GraphRetriever(gold, MemoryGraphStore())
    question = next(q for q in questions if q.id == "hop-owner-of-fraud-dep")
    result = retriever.retrieve(question, 2)
    assert result.support_recall(question) == 1.0
    assert result.fact_recall(question) == 1.0
    assert ("Risk", "OWNS", "fraud-scoring") in result.facts


def test_graph_retrieval_seeds_from_a_type_when_no_entity_is_named(gold):
    retriever = GraphRetriever(gold, MemoryGraphStore())
    named, types = retriever.seeds("Which services have no on-call primary?")
    assert named == []
    assert "Service" in types


def test_a_question_naming_nothing_retrieves_nothing(gold):
    retriever = GraphRetriever(gold, MemoryGraphStore())
    result = retriever.retrieve(_question("what happened last Tuesday?"), 2)
    assert result.docs == []
    assert result.context == ""


def test_graph_context_is_far_smaller_than_the_passages_it_replaces(gold, chunks, questions):
    question = next(q for q in questions if q.id == "hop-vendor-of-risk-service")
    graph_words = GraphRetriever(gold, MemoryGraphStore()).retrieve(question, 2).words
    oracle_words = OracleRetriever(chunks).retrieve(question).words
    assert graph_words * 3 < oracle_words


def test_a_missing_edge_shows_up_as_fact_recall_not_support_recall(
    gold, extracted, questions
):
    # The extractor misses exactly one relation. Both graphs still retrieve the
    # same documents, so only the edge-level metric can tell them apart — which
    # is the reason that metric exists.
    question = next(q for q in questions if q.id == "hop-vendor-of-risk-service")
    from_gold = GraphRetriever(gold, MemoryGraphStore()).retrieve(question, 2)
    from_extracted = GraphRetriever(extracted, MemoryGraphStore()).retrieve(question, 2)

    assert from_gold.support_recall(question) == from_extracted.support_recall(question) == 1.0
    assert from_gold.fact_recall(question) == 1.0
    assert from_extracted.fact_recall(question) < 1.0


def test_hybrid_retrieval_is_a_superset_of_both(gold, chunks, questions):
    graph = GraphRetriever(gold, MemoryGraphStore())
    vector = VectorRetriever(chunks, TfidfEmbedder())
    hybrid = HybridRetriever(graph, vector)
    question = next(q for q in questions if q.kind == "multihop")

    combined = set(hybrid.retrieve(question, 3).docs)
    assert set(graph.retrieve(question, 3).docs) <= combined
    assert set(vector.retrieve(question, 3).docs) <= combined
