"""The Neo4j store.

Two layers. The first fakes the driver and checks the Cypher we emit and the
rows we parse — that runs everywhere, and catches the errors that are actually
common: a parameter that cannot be a parameter, a relationship type built by
string concatenation, a row key that does not exist.

The second runs against a real Neo4j and is skipped unless NEO4J_URI is set. CI
sets it. A graph abstraction that has only ever been run against a mock is a
graph abstraction with a bug in it.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

from graphrag_bench.graph.store import MemoryGraphStore
from graphrag_bench.schema import Entity, Graph, Relation


def _toy() -> Graph:
    graph = Graph()
    graph.add_entity(Entity("service:a", "Service", "a", doc="cat"))
    graph.add_entity(Entity("service:b", "Service", "b", doc="cat"))
    graph.add_entity(Entity("team:x", "Team", "X", doc="team-x"))
    graph.add_relation(Relation("service:a", "DEPENDS_ON", "service:b", "doc-a"))
    graph.add_relation(Relation("team:x", "OWNS", "service:a", "doc-x"))
    return graph


# -- with a fake driver -----------------------------------------------------


class FakeSession:
    def __init__(self, log: list, rows: list[dict]) -> None:
        self._log = log
        self._rows = rows

    def run(self, query, **parameters):
        self._log.append((" ".join(query.split()), parameters))
        return list(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class FakeDriver:
    def __init__(self, rows: list[dict] | None = None) -> None:
        self.log: list = []
        self.rows = rows or []
        self.closed = False

    def session(self, database=None):
        return FakeSession(self.log, self.rows)

    def close(self):
        self.closed = True


@pytest.fixture
def store(monkeypatch):
    driver = FakeDriver()
    module = types.ModuleType("neo4j")
    module.GraphDatabase = types.SimpleNamespace(driver=lambda *a, **k: driver)
    monkeypatch.setitem(sys.modules, "neo4j", module)

    from graphrag_bench.graph.neo4j_store import Neo4jGraphStore

    instance = Neo4jGraphStore("bolt://x", "neo4j", "pw")
    return instance, driver


def test_loading_clears_then_writes_nodes_and_edges(store):
    instance, driver = store
    instance.load(_toy())
    queries = [query for query, _ in driver.log]

    assert any("DETACH DELETE" in query for query in queries)
    assert any("CREATE INDEX" in query for query in queries)
    assert any("MERGE (n:Node {id: row.id})" in query for query in queries)
    assert any("MERGE (a)-[rel:DEPENDS_ON]->(b)" in query for query in queries)
    assert any("MERGE (a)-[rel:OWNS]->(b)" in query for query in queries)


def test_a_relation_type_with_no_rows_writes_no_statement(store):
    instance, driver = store
    instance.load(_toy())
    queries = [query for query, _ in driver.log]
    assert not any("MERGE (a)-[rel:MEMBER_OF]->(b)" in query for query in queries)


def test_nodes_are_written_in_one_batched_statement(store):
    instance, driver = store
    instance.load(_toy())
    node_writes = [
        parameters for query, parameters in driver.log if "MERGE (n:Node" in query
    ]
    assert len(node_writes) == 1
    assert len(node_writes[0]["rows"]) == 3


def test_the_hop_bound_is_inlined_and_clamped(store):
    instance, driver = store
    instance.load(_toy())
    instance.expand(["service:a"], hops=99)
    query = [q for q, _ in driver.log if "MATCH path" in q][-1]
    # A variable-length pattern cannot take its bound as a parameter, so the
    # value is interpolated — and therefore has to be clamped where it is used.
    assert "[*1..6]" in query
    assert "$seeds" in query


def test_expanding_with_no_seeds_does_not_query_at_all(store):
    instance, driver = store
    instance.load(_toy())
    before = len(driver.log)
    assert instance.expand([], hops=2) == []
    assert len(driver.log) == before


def test_rows_are_parsed_into_relations(monkeypatch):
    driver = FakeDriver(
        rows=[{"source": "service:a", "type": "DEPENDS_ON", "target": "service:b", "doc": "doc-a"}]
    )
    module = types.ModuleType("neo4j")
    module.GraphDatabase = types.SimpleNamespace(driver=lambda *a, **k: driver)
    monkeypatch.setitem(sys.modules, "neo4j", module)

    from graphrag_bench.graph.neo4j_store import Neo4jGraphStore

    relations = Neo4jGraphStore("bolt://x", "neo4j", "pw").expand(["service:a"], 2)
    assert [r.key for r in relations] == [("service:a", "DEPENDS_ON", "service:b")]
    assert relations[0].doc == "doc-a"


def test_a_null_doc_becomes_an_empty_string(monkeypatch):
    driver = FakeDriver(
        rows=[{"source": "service:a", "type": "OWNS", "target": "service:b", "doc": None}]
    )
    module = types.ModuleType("neo4j")
    module.GraphDatabase = types.SimpleNamespace(driver=lambda *a, **k: driver)
    monkeypatch.setitem(sys.modules, "neo4j", module)

    from graphrag_bench.graph.neo4j_store import Neo4jGraphStore

    assert Neo4jGraphStore("bolt://x", "neo4j", "pw").expand(["service:a"], 1)[0].doc == ""


def test_closing_closes_the_driver(store):
    instance, driver = store
    instance.close()
    assert driver.closed


# -- against a real database ------------------------------------------------

integration = pytest.mark.skipif(
    not os.environ.get("NEO4J_URI"),
    reason="set NEO4J_URI to run the Neo4j integration tests",
)


@integration
def test_neo4j_and_memory_agree(gold, questions):
    from graphrag_bench.graph.neo4j_store import Neo4jGraphStore

    neo = Neo4jGraphStore(
        uri=os.environ["NEO4J_URI"],
        user=os.environ.get("NEO4J_USER", "neo4j"),
        password=os.environ.get("NEO4J_PASSWORD", "neo4j"),
    )
    memory = MemoryGraphStore()
    try:
        neo.load(gold)
        memory.load(gold)

        assert len(neo.entities()) == len(gold.entities)
        assert len(neo.entities("Service")) == len(gold.of_type("Service"))

        for seed in ["service:checkout-api", "vendor:stripe", "person:lena-fischer"]:
            for hops in (1, 2, 3):
                from_neo = {r.key for r in neo.expand([seed], hops)}
                from_memory = {r.key for r in memory.expand([seed], hops)}
                assert from_neo == from_memory, (seed, hops)
    finally:
        neo.close()


@integration
def test_the_whole_benchmark_runs_against_neo4j(capsys):
    from graphrag_bench.cli import main

    assert main(["bench", "--graph-store", "neo4j"]) == 0
    assert "full support" in capsys.readouterr().out
