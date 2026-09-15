"""The labels are the experiment. If they drift, every number below is fiction."""

from graphrag_bench.questions import KINDS
from graphrag_bench.schema import RELATION_DOMAINS, RELATION_TYPES


def test_every_gold_relation_is_well_typed(gold):
    for relation in gold.relations:
        assert relation.type in RELATION_TYPES, relation
        assert gold.is_well_typed(relation), relation


def test_every_gold_relation_names_a_real_document(gold, documents):
    known = {document.id for document in documents}
    for relation in gold.relations:
        assert relation.doc in known, relation


def test_every_gold_entity_names_a_real_document(gold, documents):
    known = {document.id for document in documents}
    for entity in gold.entities.values():
        assert entity.doc in known, entity


def test_support_documents_exist(questions, documents):
    known = {document.id for document in documents}
    for question in questions:
        assert question.support, question.id
        for doc in question.support:
            assert doc in known, (question.id, doc)


def test_required_edges_are_in_the_gold_graph(questions, gold):
    names = {entity.id: entity.name for entity in gold.entities.values()}
    edges = {
        (names[r.source], r.type, names[r.target]) for r in gold.relations
    }
    for question in questions:
        for edge in question.needs:
            assert edge in edges, (question.id, edge)


def test_question_kinds_are_known(questions):
    for question in questions:
        assert question.kind in KINDS, question.id


def test_every_kind_is_represented(questions):
    covered = {question.kind for question in questions}
    assert covered == set(KINDS)


def test_question_ids_are_unique(questions):
    ids = [question.id for question in questions]
    assert len(ids) == len(set(ids))


def test_relation_domains_cover_every_type():
    assert set(RELATION_DOMAINS) == set(RELATION_TYPES)
