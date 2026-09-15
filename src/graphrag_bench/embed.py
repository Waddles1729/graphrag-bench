"""Embedding backends for the passage-retrieval baseline.

The default is TF-IDF, implemented here rather than pulled in, so the benchmark
runs with numpy and nothing else. That is a lexical model and it is named as
one: it is not a stand-in for a good neural retriever.

Which is why `oracle` exists in `retrieve/` — a retriever that is handed the
right passages. If a question is still expensive to answer when retrieval is
perfect, no embedding model was ever going to fix it, and that is the finding
this benchmark is actually about.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9][a-z0-9-]*")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN.finditer(text.lower()):
        token = match.group(0)
        tokens.append(token)
        if "-" in token:
            # `checkout-api` should also match a question that says "checkout".
            tokens.extend(part for part in token.split("-") if part)
    return tokens


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class Embedder(Protocol):
    name: str

    def fit(self, corpus: Sequence[str]) -> None: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class TfidfEmbedder:
    """Sublinear TF, smoothed IDF, cosine. About forty lines and no dependency."""

    name = "tfidf"

    def __init__(self) -> None:
        self._vocabulary: dict[str, int] = {}
        self._idf = np.zeros(0)

    def fit(self, corpus: Sequence[str]) -> None:
        document_frequency: Counter[str] = Counter()
        tokenized = [tokenize(text) for text in corpus]
        for tokens in tokenized:
            document_frequency.update(set(tokens))

        self._vocabulary = {term: i for i, term in enumerate(sorted(document_frequency))}
        total = len(corpus)
        self._idf = np.zeros(len(self._vocabulary))
        for term, index in self._vocabulary.items():
            self._idf[index] = math.log((1 + total) / (1 + document_frequency[term])) + 1.0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self._vocabulary:
            raise RuntimeError("fit() the embedder before encoding")
        matrix = np.zeros((len(texts), len(self._vocabulary)), dtype=np.float64)
        for row, text in enumerate(texts):
            counts = Counter(tokenize(text))
            for term, count in counts.items():
                index = self._vocabulary.get(term)
                if index is not None:
                    matrix[row, index] = (1.0 + math.log(count)) * self._idf[index]
        return _normalise(matrix)


class SpacyEmbedder:
    """300-dimensional static word vectors, averaged over the passage.

    Weaker than a modern sentence encoder and stronger than nothing; it is here
    so the benchmark can be re-run with a dense model and no API key.
    """

    name = "spacy"

    def __init__(self, model: str = "en_core_web_md") -> None:
        try:
            import spacy
        except ModuleNotFoundError as error:  # pragma: no cover - import guard
            raise RuntimeError("pip install 'graphrag-bench[spacy]'") from error
        self._nlp = spacy.load(model, disable=["parser", "ner", "tagger", "lemmatizer"])

    def fit(self, corpus: Sequence[str]) -> None:  # nothing to learn
        return None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.array([self._nlp(text).vector for text in texts], dtype=np.float64)
        return _normalise(vectors)


class OpenAIEmbedder:
    """text-embedding-3-small over the HTTP API, batched, no SDK."""

    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 64) -> None:
        self._model = model
        self._batch_size = batch_size
        self._key = os.environ.get("OPENAI_API_KEY")
        if not self._key:
            raise RuntimeError("OPENAI_API_KEY is not set")

    def fit(self, corpus: Sequence[str]) -> None:
        return None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import json
        import urllib.request

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            request = urllib.request.Request(
                "https://api.openai.com/v1/embeddings",
                data=json.dumps({"model": self._model, "input": batch}).encode(),
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read())
            vectors.extend(item["embedding"] for item in payload["data"])
        return _normalise(np.array(vectors, dtype=np.float64))


def build_embedder(name: str) -> Embedder:
    if name == "tfidf":
        return TfidfEmbedder()
    if name == "spacy":
        return SpacyEmbedder()
    if name == "openai":
        return OpenAIEmbedder()
    raise ValueError(f"unknown embedder {name!r}; try tfidf, spacy or openai")
