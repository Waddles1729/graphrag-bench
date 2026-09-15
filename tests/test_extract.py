from graphrag_bench.corpus import Document, split_sections
from graphrag_bench.extract import (
    Linker,
    LLMExtractor,
    RuleExtractor,
    _units,
    discover_entities,
    relation_diff,
    score_entities,
    score_relations,
)
from graphrag_bench.schema import Entity, Graph


def _document(body: str, title: str = "checkout-api", doc_id: str = "svc-checkout-api"):
    return Document(
        id=doc_id, title=title, text=f"# {title}\n\n{body}",
        chunks=tuple(split_sections(doc_id, title, body)),
    )


# -- mentions ---------------------------------------------------------------


def test_a_name_does_not_match_inside_a_hyphenated_token():
    graph = Graph()
    graph.add_entity(Entity("service:identity", "Service", "identity"))
    graph.add_entity(Entity("datastore:identity-postgres", "Datastore", "identity-postgres"))
    found = Linker(graph).find("Data lives in identity-postgres.")
    assert [m.entity_id for m in found] == ["datastore:identity-postgres"]


def test_team_names_are_matched_case_sensitively():
    # "a risk decision" is not the Risk team, and treating it as one poisons
    # every relation extracted from that sentence.
    graph = Graph()
    graph.add_entity(Entity("team:risk", "Team", "Risk"))
    assert Linker(graph).find("a returning customer looks new to a risk decision") == []
    assert len(Linker(graph).find("Risk owns fraud-scoring")) == 1


def test_service_names_are_matched_whatever_the_case():
    graph = Graph()
    graph.add_entity(Entity("service:ledger", "Service", "ledger"))
    assert len(Linker(graph).find("Ledger is append-only.")) == 1


# -- sentence units ---------------------------------------------------------


def test_a_wrapped_line_is_still_one_sentence():
    document = _document("For SMS it depends on\nTwilio.")
    sentences = [text for _heading, text in _units(document)]
    assert any("depends on Twilio" in sentence for sentence in sentences)


def test_a_sentence_starting_with_a_lowercase_name_is_split_off():
    document = _document("It missed its budget. checkout-api queued the order.")
    sentences = [text for _heading, text in _units(document)]
    assert "It missed its budget." in sentences
    assert "checkout-api queued the order." in sentences


def test_digits_after_a_full_stop_do_not_start_a_sentence():
    document = _document("**Severity 2. 2026-03-04, 14:12 UTC.**")
    sentences = [text for _heading, text in _units(document)]
    assert len(sentences) == 1


# -- relations --------------------------------------------------------------


def _extract(
    body: str,
    entities: list[Entity],
    subject: str = "service:checkout-api",
    **kwargs,
) -> Graph:
    graph = Graph()
    for entity in entities:
        graph.add_entity(entity)
    document = _document(body, **kwargs)

    RuleExtractor()._from_sentences(document, graph, Linker(graph), subject)
    return graph


SERVICES = [
    Entity("service:checkout-api", "Service", "checkout-api"),
    Entity("service:ledger", "Service", "ledger"),
    Entity("service:notify", "Service", "notify"),
    Entity("vendor:stripe", "Vendor", "Stripe"),
]


def test_a_denied_relation_is_not_a_relation():
    graph = _extract("It does not talk to Stripe directly.", SERVICES)
    assert graph.relations == []


def test_an_object_of_one_clause_is_not_the_subject_of_the_next():
    graph = _extract(
        "It reads balances from ledger and sends the documents through notify.", SERVICES
    )
    keys = {r.key for r in graph.relations}
    assert ("service:checkout-api", "DEPENDS_ON", "service:ledger") in keys
    assert ("service:checkout-api", "DEPENDS_ON", "service:notify") in keys
    # The trap: `ledger` sits between the two cues and looks like a subject.
    assert ("service:ledger", "DEPENDS_ON", "service:notify") not in keys


def test_a_type_impossible_relation_is_dropped():
    entities = [
        Entity("team:platform", "Team", "Platform"),
        Entity("service:identity", "Service", "identity"),
    ]
    # Platform is a Team, and a Team cannot DEPENDS_ON anything.
    graph = _extract(
        "Platform depends on identity.", entities, title="Platform", doc_id="team-platform",
        subject="team:platform",
    )
    assert graph.relations == []


def test_two_cues_can_share_one_subject():
    entities = [
        Entity("incident:INC-2041", "Incident", "INC-2041"),
        Entity("service:fraud-scoring", "Service", "fraud-scoring"),
    ]
    graph = _extract(
        "fraud-scoring was affected and was also the cause.", entities,
        title="INC-2041 — checkout conversion drop", doc_id="inc-2041",
        subject="incident:INC-2041",
    )
    keys = {r.key for r in graph.relations}
    assert ("incident:INC-2041", "AFFECTED", "service:fraud-scoring") in keys
    assert ("incident:INC-2041", "CAUSED_BY", "service:fraud-scoring") in keys


def test_the_document_subject_stands_in_for_a_pronoun():
    entities = [
        Entity("service:identity", "Service", "identity"),
        Entity("datastore:identity-postgres", "Datastore", "identity-postgres"),
        Entity("team:data", "Team", "Data"),
    ]
    # "Data" is the nearest mention to the left and is the wrong type; the
    # fallback to the document's own subject is what saves this.
    graph = _extract(
        "Data lives in identity-postgres.", entities, title="identity", doc_id="svc-identity",
        subject="service:identity",
    )
    assert [r.key for r in graph.relations] == [
        ("service:identity", "STORES_IN", "datastore:identity-postgres")
    ]


# -- against the gold graph -------------------------------------------------


def test_entity_discovery_is_exact(extracted, gold):
    score = score_entities(extracted, gold)
    assert score.precision == 1.0
    assert score.recall == 1.0


def test_relation_extraction_makes_no_mistakes(extracted, gold):
    spurious, _missed = relation_diff(extracted, gold)
    assert spurious == [], spurious


def test_relation_extraction_recall_is_at_least_ninety_five_percent(extracted, gold):
    # Pinned so a change to the pattern set cannot quietly cost recall. The one
    # relation it misses is stated as an appositive, with no verb between the
    # two entities: "The second is Sift, a third-party risk provider".
    score = score_relations(extracted, gold)
    assert score.recall >= 0.95, score


def test_every_extracted_entity_carries_provenance(extracted):
    for entity in extracted.entities.values():
        assert entity.doc, entity


def test_discovery_alone_finds_no_relations(documents):
    graph = discover_entities(documents)
    assert graph.entities
    assert graph.relations == []


# -- the LLM path -----------------------------------------------------------


class FakeProvider:
    """Answers with a fixed payload, so the parsing path is testable with no key."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_the_llm_extractor_reads_a_fenced_json_reply(documents):
    reply = (
        "Here you go:\n```json\n"
        '[{"source": "Payments", "type": "OWNS", "target": "ledger"}]\n```'
    )
    provider = FakeProvider(reply)
    graph = LLMExtractor(provider, discover_entities(documents)).extract(documents[:1])
    assert ("team:payments", "OWNS", "service:ledger") in {r.key for r in graph.relations}
    assert provider.prompts and "OWNS" in provider.prompts[0]


def test_the_llm_extractor_drops_a_type_impossible_relation(documents):
    reply = '[{"source": "ledger", "type": "OWNS", "target": "Payments"}]'
    graph = LLMExtractor(FakeProvider(reply), discover_entities(documents)).extract(documents[:1])
    assert graph.relations == []


def test_the_llm_extractor_survives_an_unparsable_reply(documents):
    provider = FakeProvider("I could not find any relations.")
    graph = LLMExtractor(provider, discover_entities(documents)).extract(documents[:1])
    assert graph.relations == []
