"""The graph interface, and the in-process implementation of it.

Two stores sit behind one protocol so that the benchmark numbers cannot quietly
depend on which one you ran. The in-memory store is the default because a
reader should be able to reproduce every figure in the README without Docker;
the Neo4j store exists because that is what this looks like in production, and
because a graph abstraction nobody has run against a real database is usually
wrong in a way nobody has noticed.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from typing import Protocol

from ..schema import Entity, Graph, Relation


class GraphStore(Protocol):
    name: str

    def load(self, graph: Graph) -> None: ...

    def entities(self, entity_type: str | None = None) -> list[Entity]: ...

    def expand(self, seeds: Sequence[str], hops: int) -> list[Relation]:
        """Every relation within `hops` undirected steps of any seed."""
        ...

    def close(self) -> None: ...


class MemoryGraphStore:
    name = "memory"

    def __init__(self) -> None:
        self._graph = Graph()
        self._adjacent: dict[str, list[Relation]] = {}

    def load(self, graph: Graph) -> None:
        self._graph = graph
        self._adjacent = {}
        for relation in graph.relations:
            self._adjacent.setdefault(relation.source, []).append(relation)
            self._adjacent.setdefault(relation.target, []).append(relation)

    def entities(self, entity_type: str | None = None) -> list[Entity]:
        values = self._graph.entities.values()
        if entity_type is None:
            return list(values)
        return [entity for entity in values if entity.type == entity_type]

    def expand(self, seeds: Sequence[str], hops: int) -> list[Relation]:
        # Undirected on purpose. "Who owns the service checkout-api calls?"
        # walks with the DEPENDS_ON edge and then against the OWNS edge, and a
        # store that only follows edge direction answers neither half.
        seen_edges: dict[tuple[str, str, str], Relation] = {}
        visited = {seed for seed in seeds if seed in self._graph.entities}
        frontier = deque((node, 0) for node in visited)

        while frontier:
            node, depth = frontier.popleft()
            if depth >= hops:
                continue
            for relation in self._adjacent.get(node, []):
                seen_edges.setdefault(relation.key, relation)
                other = relation.target if relation.source == node else relation.source
                if other not in visited:
                    visited.add(other)
                    frontier.append((other, depth + 1))

        return list(seen_edges.values())

    def close(self) -> None:  # nothing to release
        return None


def serialise(relations: Iterable[Relation], graph: Graph) -> list[str]:
    """Render relations as one readable fact per line.

    This is the context a graph retriever hands to a model, and its size is the
    whole point: a fact is a dozen words where the passage it came from is two
    hundred.
    """
    lines = []
    for relation in relations:
        source = graph.entities.get(relation.source)
        target = graph.entities.get(relation.target)
        if source is None or target is None:
            continue
        verb = relation.type.replace("_", " ").lower()
        lines.append(f"{source.name} {verb} {target.name}")
    return sorted(set(lines))
