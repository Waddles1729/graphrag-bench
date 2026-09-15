"""Prose to graph.

Two extractors share one interface. The rule-based one is the default because
it is free, deterministic, and — on a corpus like this, where the entity list is
known and the schema is closed — surprisingly hard to beat. The LLM one exists
because the moment either of those assumptions breaks, patterns stop scaling and
you need a model.

Whichever you use, the number that matters is not how clever the extractor looks
but how much of the gold graph it actually recovers, and how much of what it
recovers is wrong. `graphrag-bench extract --score` reports both, because every
error here becomes an unanswerable question later.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from .corpus import Document
from .schema import Entity, Graph, Relation

# --------------------------------------------------------------------------
# entity discovery
# --------------------------------------------------------------------------

_DATASTORE = re.compile(r"`([a-z][a-z0-9-]*-(?:postgres|redis|kafka|opensearch|s3))`")
_INCIDENT = re.compile(r"\b(INC-\d+)\b")
_BOLD_LEAD = re.compile(r"^\*\*([A-Z][A-Za-z0-9]+)\*\*", re.MULTILINE)
_BULLET = re.compile(r"^[-*]\s+(.*)$", re.MULTILINE)
_PERSON = re.compile(r"\b([A-Z][a-z]+) ([A-Z][a-z]+)\b")

#: Capitalised bigrams that are not people. Kept short on purpose: a long
#: stoplist is a sign the rule is wrong, not that the list is incomplete.
_NOT_A_PERSON = {
    "Engineering", "Operations", "Finance", "Slack", "Both", "Search", "There",
    "Please", "Every", "Anything", "Services", "Data", "Platform", "Payments",
    "Risk", "Growth", "Fulfilment", "Monday", "Mondays",
}


def _is_team_page(document: Document) -> bool:
    """A team page is titled with the team's name and nothing else."""
    title = document.title.strip()
    return bool(re.fullmatch(r"[A-Z][a-z]+", title))


def discover_entities(documents: list[Document]) -> Graph:
    graph = Graph()

    # Services come from the catalogue page: the one document that claims to
    # list them all. Guessing service names out of prose instead would be a
    # worse rule and an unnecessary one — every company has this page.
    for document in documents:
        if "catalogue" not in document.title.lower():
            continue
        for bullet in _BULLET.findall(document.text):
            name = bullet.strip().strip("`")
            if re.fullmatch(r"[a-z][a-z0-9-]*", name):
                graph.add_entity(
                    Entity(id=f"service:{name}", type="Service", name=name, doc=document.id)
                )

    for document in documents:
        if _is_team_page(document):
            name = document.title.strip()
            graph.add_entity(
                Entity(id=f"team:{name.lower()}", type="Team", name=name, doc=document.id)
            )

    for document in documents:
        text = document.text
        for name in _DATASTORE.findall(text):
            _first_seen(graph, Entity(f"datastore:{name}", "Datastore", name, document.id))
        for name in _INCIDENT.findall(text):
            _first_seen(graph, Entity(f"incident:{name}", "Incident", name, document.id))

        lowered = document.title.lower()
        if "provider" in lowered or "vendor" in lowered:
            for name in _BOLD_LEAD.findall(text):
                _first_seen(
                    graph, Entity(f"vendor:{name.lower()}", "Vendor", name, document.id)
                )

        if _is_team_page(document):
            for first, last in _PERSON.findall(text):
                if first in _NOT_A_PERSON or last in _NOT_A_PERSON:
                    continue
                full = f"{first} {last}"
                _first_seen(
                    graph,
                    Entity(
                        "person:" + full.lower().replace(" ", "-"), "Person", full, document.id
                    ),
                )

    return graph


def _first_seen(graph: Graph, entity: Entity) -> None:
    """Keep the first document an entity was seen in, not the last."""
    if entity.id not in graph.entities:
        graph.add_entity(entity)


# --------------------------------------------------------------------------
# mentions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Mention:
    entity_id: str
    start: int
    end: int


#: Types whose names are distinctive enough to match without regard to case.
#: Team and Person names are ordinary English words ("Data", "Risk", "Growth"),
#: so matching those case-insensitively turns "a risk decision" into the Risk
#: team and quietly poisons everything downstream.
_CASE_INSENSITIVE = {"Service", "Datastore", "Incident", "Vendor"}


def _mention_pattern(name: str, entity_type: str) -> re.Pattern[str]:
    # A plain \b boundary would let `identity` match inside `identity-postgres`,
    # because a hyphen is a word boundary. Excluding hyphens on both sides is
    # the whole fix, and getting it wrong quietly inflates every number here.
    flags = re.IGNORECASE if entity_type in _CASE_INSENSITIVE else 0
    return re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])", flags)


class Linker:
    """Finds entity mentions in text, longest name first."""

    def __init__(self, graph: Graph) -> None:
        self._entities = sorted(graph.entities.values(), key=lambda e: -len(e.name))
        self._patterns = {e.id: _mention_pattern(e.name, e.type) for e in self._entities}
        self._graph = graph

    def find(self, text: str) -> list[Mention]:
        taken: list[tuple[int, int]] = []
        mentions: list[Mention] = []
        for entity in self._entities:
            for match in self._patterns[entity.id].finditer(text):
                span = (match.start(), match.end())
                if any(span[0] < end and start < span[1] for start, end in taken):
                    continue
                taken.append(span)
                mentions.append(Mention(entity.id, *span))
        return sorted(mentions, key=lambda m: m.start)


# --------------------------------------------------------------------------
# relation patterns
# --------------------------------------------------------------------------

#: (cue, relation type, reversed). A reversed cue means the entity on the left
#: is the target: "X is owned by Y" is OWNS(Y, X).
_CUES: tuple[tuple[str, str, bool], ...] = (
    (r"depends? on", "DEPENDS_ON", False),
    (r"depending on", "DEPENDS_ON", False),
    (r"calls?", "DEPENDS_ON", False),
    (r"uses?", "DEPENDS_ON", False),
    (r"talks? to", "DEPENDS_ON", False),
    (r"reads? (?:\w+ ){0,3}from", "DEPENDS_ON", False),
    (r"pulls? (?:\w+ ){0,3}from", "DEPENDS_ON", False),
    (r"sends? (?:\w+ ){0,4}through", "DEPENDS_ON", False),
    (r"goes? out through", "DEPENDS_ON", False),
    (r"is called (?:\w+ ){0,2}by", "DEPENDS_ON", True),
    (r"owns?", "OWNS", False),
    (r"runs?", "OWNS", False),
    (r"is owned by", "OWNS", True),
    (r"are owned by", "OWNS", True),
    (r"belongs? to", "OWNS", True),
    (r"sits? with", "OWNS", True),
    (r"stored in", "STORES_IN", False),
    (r"stores? in", "STORES_IN", False),
    (r"lives? in", "STORES_IN", False),
    (r"sits? in", "STORES_IN", False),
    (r"held in", "STORES_IN", False),
    (r"kept in", "STORES_IN", False),
    (r"cached in", "STORES_IN", False),
    (r"written to", "STORES_IN", False),
    (r"writes? to", "STORES_IN", False),
    (r"was affected", "AFFECTED", True),
    (r"were affected", "AFFECTED", True),
    (r"was the cause", "CAUSED_BY", True),
    (r"was also the cause", "CAUSED_BY", True),
    (r"caused by", "CAUSED_BY", False),
    (r"the cause was in", "CAUSED_BY", False),
    (r"is ours", "OWNS", True),
    (r"are ours", "OWNS", True),
)

_COMPILED_CUES = tuple(
    (re.compile(rf"(?<![\w-])(?:{cue})(?![\w-])", re.IGNORECASE), rtype, flipped)
    for cue, rtype, flipped in _CUES
)

_NEGATION = re.compile(r"\b(?:no|not|never|nothing|n't)\b", re.IGNORECASE)

# Sentences here start with a service name as often as with a capital letter
# ("...the 400ms budget. checkout-api did what it is designed to do"), so the
# usual `(?=[A-Z])` lookahead silently glues two sentences together and every
# cue in the second one picks up arguments from the first. Digits are excluded
# so "Severity 2. 2026-03-04" and version numbers survive.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Za-z`\"'])|\n")


def _units(document: Document) -> Iterable[tuple[str, str]]:
    """Yield (heading, sentence) for every sentence, bullet and table row.

    Markdown is hard-wrapped, so a line is not a sentence. Consecutive prose
    lines are joined back into a paragraph before splitting, otherwise a
    relation stated across a line break — "it depends on\\nTwilio" — is invisible
    to every pattern below.
    """
    heading = document.title
    paragraph: list[str] = []

    def flush() -> Iterable[tuple[str, str]]:
        if not paragraph:
            return
        joined = " ".join(paragraph)
        paragraph.clear()
        for sentence in _SENTENCE.split(joined):
            sentence = sentence.strip()
            if sentence:
                yield heading, sentence

    for line in document.text.splitlines():
        stripped = line.strip()
        is_structural = (
            not stripped
            or stripped.startswith("#")
            or stripped.startswith("|")
            or bool(_BULLET.match(stripped))
        )
        if is_structural:
            yield from flush()
            if stripped.startswith("## "):
                heading = stripped[3:].strip()
            elif stripped.startswith("|") and not set(stripped) <= set("|- :"):
                yield heading, stripped
            elif _BULLET.match(stripped):
                yield heading, stripped
            continue
        paragraph.append(stripped)

    yield from flush()


def _subject(document: Document, graph: Graph) -> str | None:
    """The entity a document is about, if it is about one."""
    title = document.title.strip()
    for entity in graph.entities.values():
        if entity.name.lower() == title.lower():
            return entity.id
    match = _INCIDENT.match(title)
    if match and f"incident:{match.group(1)}" in graph.entities:
        return f"incident:{match.group(1)}"
    return None


def _table_rows(text: str) -> list[tuple[list[str], list[list[str]]]]:
    """Return (header cells, data rows) for every markdown table in `text`."""
    tables: list[tuple[list[str], list[list[str]]]] = []
    header: list[str] | None = None
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        is_row = stripped.startswith("|") and stripped.endswith("|")
        if is_row and set(stripped) <= set("|- :"):
            continue
        if is_row:
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if header is None:
                header = cells
            else:
                rows.append(cells)
            continue
        if header is not None:
            tables.append((header, rows))
            header, rows = None, []
    if header is not None:
        tables.append((header, rows))
    return tables


class Extractor(Protocol):
    name: str

    def extract(self, documents: list[Document]) -> Graph: ...


class RuleExtractor:
    """Gazetteer linking plus lexical cues, with a type check on every edge."""

    name = "rules"

    def extract(self, documents: list[Document]) -> Graph:
        graph = discover_entities(documents)
        linker = Linker(graph)

        for document in documents:
            subject = _subject(document, graph)
            self._from_tables(document, graph, linker, subject)
            self._from_lists(document, graph, linker, subject)
            self._from_sentences(document, graph, linker, subject)
            self._people_on_team_pages(document, graph, linker, subject)

        return graph

    # -- structure --------------------------------------------------------

    def _from_tables(self, document, graph, linker, subject) -> None:
        for header, rows in _table_rows(document.text):
            lowered = [cell.lower() for cell in header]
            for row in rows:
                mentions = [linker.find(cell) for cell in row]
                first = mentions[0][0].entity_id if mentions and mentions[0] else None
                if first is None:
                    continue

                # An on-call table: service in one column, a person in another.
                if any("on-call" in h or "primary" in h for h in lowered):
                    for cell in mentions[1:]:
                        for mention in cell:
                            self._add(graph, mention.entity_id, "ON_CALL_FOR", first, document.id)

                # A table on a team page listing that team's services.
                if subject and graph.type_of(subject) == "Team":
                    self._add(graph, subject, "OWNS", first, document.id)

    def _from_lists(self, document, graph, linker, subject) -> None:
        heading = document.title
        lead_in = ""
        for line in document.text.splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                heading = stripped[3:].strip()
                lead_in = ""
                continue
            bullet = _BULLET.match(stripped)
            if not bullet:
                # A list is usually introduced by the line above it, and that
                # line is where the relation type is stated: "Services under our
                # ownership:" tells you what the bullets mean, the heading often
                # does not.
                if stripped and not stripped.startswith("#"):
                    lead_in = stripped
                continue
            mentions = linker.find(bullet.group(1))
            if not mentions or subject is None:
                continue
            target = mentions[0].entity_id
            context = f"{heading} {lead_in}".lower()
            subject_type = graph.type_of(subject)

            if subject_type == "Service" and re.search(r"call|depend|uses?\b", context):
                self._add(graph, subject, "DEPENDS_ON", target, document.id)
            elif subject_type == "Team" and re.search(r"own|service", context):
                self._add(graph, subject, "OWNS", target, document.id)

    def _people_on_team_pages(self, document, graph, linker, subject) -> None:
        if subject is None or graph.type_of(subject) != "Team":
            return
        for mention in linker.find(document.text):
            if graph.type_of(mention.entity_id) == "Person":
                self._add(graph, mention.entity_id, "MEMBER_OF", subject, document.id)

    # -- prose ------------------------------------------------------------

    def _from_sentences(self, document, graph, linker, subject) -> None:
        for _heading, sentence in _units(document):
            self._one_sentence(sentence, document, graph, linker, subject)

    def _one_sentence(self, sentence, document, graph, linker, subject) -> None:
        mentions = linker.find(sentence)
        if not mentions:
            return

        hits = sorted(
            (
                (cue.start(), cue.end(), rtype, flipped)
                for pattern, rtype, flipped in _COMPILED_CUES
                for cue in pattern.finditer(sentence)
            ),
            key=lambda hit: hit[0],
        )

        # A mention that has already served as the object of a cue cannot be the
        # subject of the next one: in "it reads balances from ledger and sends
        # them through notify", `ledger` is what was read, not what sends.
        consumed: set[int] = set()

        for index, (start, end, rtype, flipped) in enumerate(hits):
            if _NEGATION.search(sentence[max(0, start - 45) : start]):
                continue

            # Arguments belong to this cue's clause: everything up to the next
            # cue, not to the end of the sentence.
            clause_start = hits[index - 1][1] if index else 0
            clause_end = hits[index + 1][0] if index + 1 < len(hits) else len(sentence)
            left = [m for m in mentions if m.end <= start]
            right = [m for m in mentions if m.start >= end and m.end <= clause_end]

            if flipped:
                # "X was affected and was also the cause" shares one subject
                # across two cues, so when this clause has no subject of its own
                # the nearest one to the left is it.
                in_clause = [m for m in left if m.start >= clause_start]
                targets, sources = (in_clause or left[-1:]), right
            else:
                targets, sources = right, left

            if not targets:
                continue

            # All of the targets ("X, Y and Z"), but only the nearest source,
            # falling back to whatever the document is about when the sentence
            # says "it".
            candidates = [
                m.entity_id for m in reversed(sources) if m.start not in consumed
            ]
            if subject is not None:
                candidates.append(subject)

            for target in targets:
                for source in candidates:
                    if self._add(graph, source, rtype, target.entity_id, document.id):
                        break

            if not flipped:
                consumed.update(m.start for m in targets)

    # -- shared -----------------------------------------------------------

    @staticmethod
    def _add(graph: Graph, source: str, rtype: str, target: str, doc: str) -> bool:
        if source == target:
            return False
        relation = Relation(source=source, type=rtype, target=target, doc=doc)
        if not graph.is_well_typed(relation):
            return False
        graph.add_relation(relation)
        return True


class LLMExtractor:
    """Asks a model for the same closed schema, one document at a time.

    The provider is injected rather than imported so this is testable without a
    key: anything with a `complete(prompt) -> str` method will do.
    """

    name = "llm"

    def __init__(self, provider, entities: Graph | None = None) -> None:
        self._provider = provider
        self._entities = entities

    def extract(self, documents: list[Document]) -> Graph:
        graph = self._entities or discover_entities(documents)
        names = {entity.name.lower(): entity.id for entity in graph.entities.values()}

        for document in documents:
            raw = self._provider.complete(_prompt(document, graph))
            for item in _parse(raw):
                source = names.get(str(item.get("source", "")).lower())
                target = names.get(str(item.get("target", "")).lower())
                rtype = str(item.get("type", "")).upper()
                if not source or not target or source == target:
                    continue
                relation = Relation(source=source, type=rtype, target=target, doc=document.id)
                if graph.is_well_typed(relation):
                    graph.add_relation(relation)
        return graph


def _prompt(document: Document, graph: Graph) -> str:
    from .schema import RELATION_DOMAINS

    allowed = "\n".join(
        f"  {rtype}: {'|'.join(src)} -> {'|'.join(dst)}"
        for rtype, (src, dst) in RELATION_DOMAINS.items()
    )
    known = ", ".join(sorted(entity.name for entity in graph.entities.values()))
    return (
        "Read the document and list every relation it states, using only these "
        "types:\n"
        f"{allowed}\n\n"
        "Use only these entity names, exactly as written:\n"
        f"{known}\n\n"
        "Report only what the document actually states. A relation the document "
        "denies ('X does not call Y') is not a relation. Answer with a JSON "
        'array of {"source": ..., "type": ..., "target": ...} and nothing else.'
        f"\n\n---\n{document.text}"
    )


def _parse(raw: str) -> list[dict]:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [item for item in payload if isinstance(item, dict)]


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


@dataclass
class Score:
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total else 1.0

    @property
    def recall(self) -> float:
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def score_entities(extracted: Graph, gold: Graph) -> Score:
    got = {(e.type, e.name.lower()) for e in extracted.entities.values()}
    want = {(e.type, e.name.lower()) for e in gold.entities.values()}
    return Score(len(got & want), len(got - want), len(want - got))


def score_relations(extracted: Graph, gold: Graph, relation_type: str | None = None) -> Score:
    def keys(graph: Graph) -> set[tuple[str, str, str]]:
        return {
            r.key for r in graph.relations
            if relation_type is None or r.type == relation_type
        }

    got, want = keys(extracted), keys(gold)
    return Score(len(got & want), len(got - want), len(want - got))


def relation_diff(extracted: Graph, gold: Graph) -> tuple[list[Relation], list[Relation]]:
    """(spurious, missed) — the edges worth reading one by one."""
    got = {r.key: r for r in extracted.relations}
    want = {r.key: r for r in gold.relations}
    spurious = [got[key] for key in got.keys() - want.keys()]
    missed = [want[key] for key in want.keys() - got.keys()]
    return sorted(spurious, key=lambda r: r.key), sorted(missed, key=lambda r: r.key)
