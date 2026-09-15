"""Passage retrieval: embed the chunks, embed the question, take the top k."""

from __future__ import annotations

import numpy as np

from ..corpus import Chunk
from ..embed import Embedder
from ..questions import Question
from .base import Retrieved


class VectorRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        self.name = f"vector({embedder.name})"
        self._chunks = chunks
        self._embedder = embedder
        texts = [chunk.text for chunk in chunks]
        embedder.fit(texts)
        self._matrix = embedder.encode(texts)

    @property
    def budgets(self) -> tuple[int, ...]:
        return (1, 2, 3, 5, 8, 12, 20)

    def retrieve(self, question: Question, budget: int) -> Retrieved:
        query = self._embedder.encode([question.question])[0]
        scores = self._matrix @ query
        order = np.argsort(-scores)[:budget]
        chunks = [self._chunks[int(i)] for i in order]

        docs: list[str] = []
        for chunk in chunks:
            if chunk.doc_id not in docs:
                docs.append(chunk.doc_id)

        return Retrieved(
            docs=docs,
            context="\n\n".join(chunk.text for chunk in chunks),
            budget=budget,
        )


class OracleRetriever:
    """Passage retrieval with the ranking problem removed.

    It is handed the documents the question needs and returns their chunks. No
    embedding model can do better than this, so whatever it costs in context is
    the floor for the whole passage-retrieval approach — and on a multi-hop
    question that floor is still several times what a traversal costs.
    """

    name = "oracle"

    def __init__(self, chunks: list[Chunk]) -> None:
        self._by_doc: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            self._by_doc.setdefault(chunk.doc_id, []).append(chunk)

    @property
    def budgets(self) -> tuple[int, ...]:
        return (0,)

    def retrieve(self, question: Question, budget: int = 0) -> Retrieved:
        chunks = [
            chunk for doc in question.support for chunk in self._by_doc.get(doc, [])
        ]
        return Retrieved(
            docs=list(question.support),
            context="\n\n".join(chunk.text for chunk in chunks),
            budget=len(question.support),
        )
