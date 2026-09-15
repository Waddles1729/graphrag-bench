"""Neo4j-backed graph store.

The driver is imported lazily so that `pip install graphrag-bench` does not drag
a database client along for a benchmark that runs fine without one.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..schema import RELATION_TYPES, Entity, Graph, Relation


class Neo4jGraphStore:
    name = "neo4j"

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        try:
            from neo4j import GraphDatabase
        except ModuleNotFoundError as error:  # pragma: no cover - import guard
            raise RuntimeError(
                "the Neo4j store needs the driver: pip install 'graphrag-bench[neo4j]'"
            ) from error

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        self._graph = Graph()

    # -- writing ----------------------------------------------------------

    def load(self, graph: Graph) -> None:
        self._graph = graph
        with self._driver.session(database=self._database) as session:
            session.run("MATCH (n:Node) DETACH DELETE n")
            session.run("CREATE INDEX node_id IF NOT EXISTS FOR (n:Node) ON (n.id)")

            session.run(
                """
                UNWIND $rows AS row
                MERGE (n:Node {id: row.id})
                SET n.name = row.name, n.type = row.type
                """,
                rows=[
                    {"id": e.id, "name": e.name, "type": e.type}
                    for e in graph.entities.values()
                ],
            )

            # One statement per relation type: Neo4j will not take the type as a
            # parameter, and building it by string concatenation is only safe
            # because the set is closed and checked here.
            for relation_type in RELATION_TYPES:
                rows = [
                    {"source": r.source, "target": r.target, "doc": r.doc}
                    for r in graph.relations
                    if r.type == relation_type
                ]
                if not rows:
                    continue
                session.run(
                    f"""
                    UNWIND $rows AS row
                    MATCH (a:Node {{id: row.source}})
                    MATCH (b:Node {{id: row.target}})
                    MERGE (a)-[rel:{relation_type}]->(b)
                    SET rel.doc = row.doc
                    """,
                    rows=rows,
                )

    # -- reading ----------------------------------------------------------

    def entities(self, entity_type: str | None = None) -> list[Entity]:
        query = "MATCH (n:Node) RETURN n.id AS id, n.name AS name, n.type AS type"
        if entity_type:
            query = (
                "MATCH (n:Node {type: $type}) "
                "RETURN n.id AS id, n.name AS name, n.type AS type"
            )
        with self._driver.session(database=self._database) as session:
            return [
                Entity(id=row["id"], type=row["type"], name=row["name"])
                for row in session.run(query, type=entity_type)
            ]

    def expand(self, seeds: Sequence[str], hops: int) -> list[Relation]:
        if not seeds or hops < 1:
            return []
        # The bound has to be inlined: a variable-length pattern will not accept
        # its range as a parameter. It is an int from our own config, and it is
        # clamped here rather than trusted.
        bound = max(1, min(int(hops), 6))
        query = f"""
            MATCH path = (n:Node)-[*1..{bound}]-(:Node)
            WHERE n.id IN $seeds
            UNWIND relationships(path) AS rel
            RETURN DISTINCT
                startNode(rel).id AS source,
                type(rel) AS type,
                endNode(rel).id AS target,
                rel.doc AS doc
        """
        with self._driver.session(database=self._database) as session:
            return [
                Relation(
                    source=row["source"],
                    type=row["type"],
                    target=row["target"],
                    doc=row["doc"] or "",
                )
                for row in session.run(query, seeds=list(seeds))
            ]

    def close(self) -> None:
        self._driver.close()
