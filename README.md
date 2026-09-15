# graphrag-bench

**What does a knowledge graph actually add to retrieval, and what does it cost?**
A labelled corpus, a closed schema, five kinds of question, and four retrieval
strategies measured against each other — including one that cheats, so you can
tell a retrieval problem from a representation problem.

[![CI](https://github.com/Waddles1729/graphrag-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/Waddles1729/graphrag-bench/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

---

## The short version

```bash
git clone https://github.com/Waddles1729/graphrag-bench
cd graphrag-bench
pip install -e .
graphrag-bench bench
```

No API key, no database, well under a second.

```
Full support by question kind, each strategy at its best budget
(budget is k passages, or hops for the graph)

strategy          budget  kind         full support  words
----------------  ------  -----------  ------------  -----
vector(tfidf)         20  lookup               100%   1265
vector(tfidf)         20  multihop              90%   1277
vector(tfidf)         20  aggregation           50%   1303
vector(tfidf)         20  absence               50%   1192
vector(tfidf)         20  impact                50%   1462
oracle                 0  lookup               100%    154
oracle                 0  multihop             100%    252
oracle                 0  aggregation          100%    455
oracle                 0  absence              100%    543
oracle                 0  impact               100%    475
graph(gold)            2  lookup               100%     80
graph(gold)            2  multihop             100%     78
graph(gold)            2  aggregation          100%    211
graph(gold)            2  absence              100%    164
graph(gold)            2  impact               100%    103
```

*Full support* is the share of questions where every document containing a
needed fact reached the context. *Words* is the size of that context.

## What it measures, and what it does not

This scores **retrieval**, not answers. Whether a model then reasons correctly
over the context is a property of the model, and folding the two together is how
a retrieval claim ends up resting on a prompt. Three numbers:

| Metric | Question it answers |
| --- | --- |
| **support recall** | Did the documents holding the needed facts reach the context? |
| **context words** | How much text did that take? Recall is free if you retrieve everything. |
| **fact recall** | Did the graph actually contain the specific edges the answer rests on? |

The corpus is a fictional company's internal wiki: 24 documents, 46 entities, 68
relations, 32 questions in five kinds — `lookup`, `multihop`, `aggregation`,
`absence` and `impact`. Every relation is hand-labelled with the document it is
stated in, and every question with the documents and the graph edges its answer
needs.

## Four findings

### 1. The gap is structural, not an embedding problem

`oracle` is a retriever that is handed the right documents. No embedding model
can beat it. It is in the table for one reason: to separate "the retriever
ranked badly" from "passages are the wrong unit".

On single-hop lookups, TF-IDF reaches the oracle — ranking is the only problem
and a better model would close it. On `aggregation`, `absence` and `impact`,
the oracle still needs 455–543 words where the graph needs 103–211, because the
facts are scattered across six to ten documents by the corpus's own structure. A
better embedding model does not make ten documents into three.

Swapping TF-IDF for 300-dimensional static word vectors makes the baseline
*worse* here (62% full support at k=20 against 75%), because the questions name
entities exactly and averaging word vectors throws that away. The shape of the
result does not change either way.

### 2. Two hops answers everything, for about a tenth of the context

```
Recall against context size — vector(tfidf)

budget  full support  recall  median words
------  ------------  ------  ------------
     1           12%     30%            61
     2           22%     42%           120
     3           31%     50%           196
     5           38%     55%           313
     8           53%     69%           526
    12           56%     75%           757
    20           75%     89%          1277


Recall against context size — graph(gold)

budget  full support  recall  median words
------  ------------  ------  ------------
     1           59%     78%            29
     2          100%    100%           109
     3          100%    100%           206
     4          100%    100%           239
```

Twenty passages and 1,277 words gets three quarters of the way. Two hops and 109
words gets all of it.

Eight of the 32 questions are never fully retrieved by passage search at any k
in the sweep. They are the ones where the answer is a count, a complete list, or
a transitive closure:

```
hop-owner-of-reporting-metadata-source   agg-most-dependencies
agg-teams-multiple-services              abs-services-without-deps
agg-count-vendors-in-deps                abs-engineers-not-on-rota
imp-warehouse-blast-radius               imp-vendors-behind-checkout
```

"Which services have no on-call primary?" cannot be answered from a top-k list
at all, at any k, because a ranked list is not evidence of absence.

### 3. 98.5% extraction recall is not 98.5% of questions answered

The rule-based extractor recovers 67 of 68 relations with no false positives:

```
               precision  recall    F1  gold
-------------  ---------  ------  ----  ----
entities            1.00    1.00  1.00    46
OWNS                1.00    1.00  1.00    10
DEPENDS_ON          1.00    0.94  0.97    17
MEMBER_OF           1.00    1.00  1.00    14
ON_CALL_FOR         1.00    1.00  1.00     8
STORES_IN           1.00    1.00  1.00     9
AFFECTED            1.00    1.00  1.00     7
CAUSED_BY           1.00    1.00  1.00     3
all relations       1.00    0.99  0.99    68

missed:
  - fraud-scoring DEPENDS_ON Sift  (svc-fraud-scoring)
```

One edge. It is stated as an appositive with no verb between the two entities —
*"The second is Sift, a third-party risk provider"* — which every lexical cue
in the pattern set walks straight past.

That single miss costs **five of the 29 edge-labelled questions**: 17% of the
benchmark, from 1.5% of the graph.

```
strategy          kind         fact recall  questions with every edge
----------------  -----------  -----------  -------------------------
graph(gold)       multihop            100%  10/10
graph(gold)       aggregation         100%  5/5
graph(extracted)  multihop             95%  9/10
graph(extracted)  aggregation          94%  3/5
graph(extracted)  absence              98%  2/3
graph(extracted)  impact               94%  3/4
```

Note what support recall says about the same two graphs: **100% for both, on
every kind**. The document the missing edge came from is still retrieved,
because a different edge points at it. Document-level metrics cannot see this
failure. If you are evaluating a GraphRAG pipeline on retrieval hit rate, you
are not measuring the thing that breaks.

### 4. Where the graph loses

Three questions the graph cannot answer at all, at any depth, with a perfect
extractor:

- **"When does the Sift contract renew?"** — a vendor attribute, not an edge.
- **"Which teams have no engineering manager?"** — there is no `MANAGES`
  relation in the schema. The fact is in the prose on two team pages, and
  passage retrieval finds it in 303 words.
- **"How many services does the company run?"** — answerable by counting nodes,
  but not by traversing edges, which is what the retriever does.

A closed schema is what makes traversal reliable and is also exactly what makes
it brittle: the question you did not anticipate is not a slow query, it is an
impossible one. Passage retrieval has no schema and therefore no such wall.
`--hybrid` scores both together, and is strictly better at recall and strictly
worse at context size.

One more, in the other direction: on the simplest lookup in the set —
*"which database does checkout-api store orders in?"* — passage retrieval wins
outright, 13 words against 32. The passage that answers it is short and says
every word the question said.

## Running it

```bash
graphrag-bench extract --score       # build the graph, compare it to the gold one
graphrag-bench bench                 # the full comparison
graphrag-bench bench --hybrid        # add graph + passages
graphrag-bench bench --embeddings openai     # needs OPENAI_API_KEY

# dense static vectors instead of TF-IDF
pip install 'graphrag-bench[spacy]' && python -m spacy download en_core_web_md
graphrag-bench bench --embeddings spacy
graphrag-bench ask "Which team owns the service checkout-api calls for fraud?"
```

```
$ graphrag-bench ask "If Stripe is unavailable, which services are affected?"

Diego Ramos on call for ledger
Payments owns ledger
checkout-api depends on ledger
invoicing depends on ledger
ledger depends on Stripe
ledger depends on identity
ledger stores in ledger-postgres
reporting depends on ledger

33 words · 7 document(s): svc-ledger, team-payments, svc-checkout-api, svc-invoicing, svc-reporting, oncall-rota, vendors
```

Everything needed to answer — and, being two undirected hops, a little that is
not. Passage retrieval needed 1,490 words to cover the same question.

### Against a real Neo4j

```bash
docker run -d -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5-community
pip install -e '.[neo4j]'
NEO4J_URI=bolt://localhost:7687 NEO4J_PASSWORD=password \
  graphrag-bench bench --graph-store neo4j
```

The two stores are checked against each other — same seeds, same depths, same
edges — in CI, on every push.

## How it works

**Extraction** is a gazetteer plus about thirty lexical cues, with a type check
on every edge. Entity discovery is structural: services come from the catalogue
page, datastores from backticked identifiers, people from capitalised bigrams on
team pages. An `LLMExtractor` sits behind the same interface and asks a model for
the same closed schema; its provider is injected rather than imported, so the
prompt, the parsing and the type check are all tested without a key.

Rules do well here because the entity list is known and the schema is closed.
That is a real setting — most companies have a service catalogue and an HR
directory — and it is also the setting where rules stop working the moment
either assumption breaks.

**Graph retrieval** links the question to entities and walks. There is no model
in the path: no generated Cypher, no LLM reranking. When a question names no
entity — *"which services have no on-call primary?"* — the seeds are every
entity of the type it does name, which is what `MATCH (s:Service)` would do.
Everything here is the floor that a Cypher-generating agent builds on, not a
competitor to one.

**Chunking** is by markdown section, which favours the baseline: sections are
topically coherent and the document title is prepended to each so no section
loses its subject. If the baseline still cannot answer a question, it is not
because the chunking was unkind.

**A note on the graph's document list.** Graph retrieval "touches" far more
documents than it charges for, because its context is facts, not passages — the
model sees `invoicing depends on notify`, not the page that sentence came from.
The document list is provenance for citation. The comparable quantity is what
enters the context window, which is what *words* measures.

## Limits

- One corpus, 24 documents, and it was written for this benchmark. The
  proportions here are not a forecast for yours.
- The sentence splitter is a regex. It handles hard-wrapped markdown and
  lowercase sentence starts, and it would mishandle "e.g." — there is none in
  the corpus.
- Answer generation is not scored. Retrieval is measured; what a model does with
  it is a separate experiment, and pairing this with a regression gate such as
  [evalgate](https://github.com/Waddles1729/evalgate) is the sensible way to run
  it.

## Development

```bash
pip install -e '.[dev]'
pytest        # 68 tests; 2 of them need a Neo4j and skip without one
ruff check src tests
```

The Neo4j store is unit-tested against a fake driver — the Cypher it emits, the
rows it parses, the hop bound it inlines — and integration-tested against a real
Neo4j in CI. A graph abstraction that has only ever run against a mock is a
graph abstraction with a bug in it.

## License

MIT.
