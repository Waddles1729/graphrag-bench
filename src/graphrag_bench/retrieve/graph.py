"""Graph retrieval: link the question to entities, walk, return the facts.

There is no model in this path. Entity linking is a gazetteer, the traversal is
breadth-first, and the context is one line per edge. It is worth being explicit
about that, because "GraphRAG" is often used to mean an LLM writing Cypher — a
strictly more capable and strictly less reproducible thing. Everything measured
here is the floor that approach builds on.

When a question names no entity at all — "which services have no on-call
primary?" — the seeds are every entity of the type it does name. That is what a
generated Cypher query would do with `MATCH (s:Service)`, and it is the only way
a question about absence is answerable at all: you cannot prove a negative from
a top-k list.
"""

from __future__ import annotations

import re

from ..extract import Linker
from ..graph.store import GraphStore, serialise
from ..questions import Question
from ..schema import Graph
from .base import Retrieved

#: Words that name a type rather than an instance.
_TYPE_WORDS: tuple[tuple[str, str], ...] = (
    (r"services?\b", "Service"),
    (r"teams?\b", "Team"),
    (r"engineers?\b|people\b|person\b|who\b", "Person"),
    (r"vendors?\b|providers?\b|third.part(?:y|ies)\b", "Vendor"),
    (r"incidents?\b", "Incident"),
    (r"databases?\b|datastores?\b|stores?\b", "Datastore"),
)


class GraphRetriever:
    def __init__(self, graph: Graph, store: GraphStore, label: str = "graph") -> None:
        self.name = label
        self._graph = graph
        self._store = store
        self._linker = Linker(graph)
        store.load(graph)

    @property
    def budgets(self) -> tuple[int, ...]:
        return (1, 2, 3, 4)

    def seeds(self, question: str) -> tuple[list[str], list[str]]:
        """(entity seeds, type seeds) for one question."""
        named = [mention.entity_id for mention in self._linker.find(question)]
        types = [
            entity_type
            for pattern, entity_type in _TYPE_WORDS
            if re.search(pattern, question, re.IGNORECASE)
        ]
        return named, types

    def retrieve(self, question: Question, budget: int) -> Retrieved:
        named, types = self.seeds(question.question)

        if named:
            seeds, hops = named, budget
        elif types:
            # No instance to start from: start from all of them and take one
            # step, which is the whole neighbourhood of a type.
            seeds = [
                entity.id
                for entity_type in types
                for entity in self._graph.of_type(entity_type)
            ]
            hops = 1
        else:
            return Retrieved(docs=[], context="", budget=budget)

        relations = self._store.expand(seeds, hops)

        docs: list[str] = []
        for relation in relations:
            if relation.doc and relation.doc not in docs:
                docs.append(relation.doc)
        # Entities carry provenance too, and for a question about a whole type
        # the page that lists them is part of the answer's support.
        for seed in seeds:
            entity = self._graph.entities.get(seed)
            if entity and entity.doc and entity.doc not in docs:
                docs.append(entity.doc)

        facts = []
        for relation in relations:
            source = self._graph.entities.get(relation.source)
            target = self._graph.entities.get(relation.target)
            if source and target:
                facts.append((source.name, relation.type, target.name))

        return Retrieved(
            docs=docs,
            context="\n".join(serialise(relations, self._graph)),
            facts=sorted(set(facts)),
            budget=hops,
        )


class HybridRetriever:
    """Graph facts first, then the passages a passage retriever would have found.

    The obvious thing to build, and the one worth measuring honestly: it is
    strictly better at recall and strictly worse at context size, and whether
    that trade is worth making depends on what you are paying per token.
    """

    def __init__(self, graph_retriever: GraphRetriever, vector_retriever) -> None:
        self.name = f"hybrid({vector_retriever.name})"
        self._graph = graph_retriever
        self._vector = vector_retriever

    @property
    def budgets(self) -> tuple[int, ...]:
        return (1, 2, 3, 5, 8)

    def retrieve(self, question: Question, budget: int) -> Retrieved:
        from_graph = self._graph.retrieve(question, min(budget, 3))
        from_vector = self._vector.retrieve(question, budget)

        docs = list(from_graph.docs)
        for doc in from_vector.docs:
            if doc not in docs:
                docs.append(doc)

        context = from_graph.context
        if from_vector.context:
            context = f"{context}\n\n---\n\n{from_vector.context}".strip()

        return Retrieved(
            docs=docs, context=context, facts=from_graph.facts, budget=budget
        )
