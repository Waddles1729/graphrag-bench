from pathlib import Path

import pytest

from graphrag_bench.corpus import all_chunks, load_corpus
from graphrag_bench.extract import RuleExtractor
from graphrag_bench.questions import load_questions
from graphrag_bench.schema import load_graph

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="session")
def documents():
    return load_corpus(DATA / "corpus")


@pytest.fixture(scope="session")
def chunks(documents):
    return all_chunks(documents)


@pytest.fixture(scope="session")
def gold():
    return load_graph(DATA / "gold_graph.json")


@pytest.fixture(scope="session")
def extracted(documents):
    return RuleExtractor().extract(documents)


@pytest.fixture(scope="session")
def questions():
    return load_questions(DATA / "questions.jsonl")
