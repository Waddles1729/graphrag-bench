"""Entities, relations, and the graph they form.

The schema is closed on purpose. Open-ended extraction produces a graph that
looks impressive and answers nothing reliably, because two documents that say
the same thing end up with two different relation types. A fixed set of types —
chosen from the questions people actually ask — is what makes traversal a
substitute for search rather than a decoration on it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ENTITY_TYPES = ("Service", "Team", "Person", "Vendor", "Datastore", "Incident")

RELATION_TYPES = (
    "OWNS",          # Team    -> Service
    "DEPENDS_ON",    # Service -> Service | Vendor
    "MEMBER_OF",     # Person  -> Team
    "ON_CALL_FOR",   # Person  -> Service
    "STORES_IN",     # Service -> Datastore
    "AFFECTED",      # Incident-> Service
    "CAUSED_BY",     # Incident-> Service | Vendor
)

#: Which entity types each relation is allowed to connect. Used to reject
#: extractions that are locally plausible but type-impossible.
RELATION_DOMAINS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "OWNS": (("Team",), ("Service",)),
    "DEPENDS_ON": (("Service",), ("Service", "Vendor")),
    "MEMBER_OF": (("Person",), ("Team",)),
    "ON_CALL_FOR": (("Person",), ("Service",)),
    "STORES_IN": (("Service",), ("Datastore",)),
    "AFFECTED": (("Incident",), ("Service",)),
    "CAUSED_BY": (("Incident",), ("Service", "Vendor")),
}


@dataclass(frozen=True)
class Entity:
    id: str
    type: str
    name: str
    #: The document this entity was first read out of. A question about a whole
    #: type ("which services…") is answered partly by the page that lists them,
    #: so entities need provenance for the same reason relations do.
    doc: str = ""
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Relation:
    source: str
    type: str
    target: str
    #: The document this relation was read out of. Carrying provenance on every
    #: edge is what lets a graph answer cite its sources.
    doc: str

    @property
    def key(self) -> tuple[str, str, str]:
        """Identity for comparison — provenance deliberately excluded."""
        return (self.source, self.type, self.target)


@dataclass
class Graph:
    entities: dict[str, Entity] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)

    def add_entity(self, entity: Entity) -> None:
        self.entities[entity.id] = entity

    def add_relation(self, relation: Relation) -> None:
        if relation.key not in {existing.key for existing in self.relations}:
            self.relations.append(relation)

    def by_name(self) -> dict[str, Entity]:
        return {entity.name: entity for entity in self.entities.values()}

    def of_type(self, entity_type: str) -> list[Entity]:
        return [e for e in self.entities.values() if e.type == entity_type]

    def type_of(self, entity_id: str) -> str | None:
        entity = self.entities.get(entity_id)
        return entity.type if entity else None

    def is_well_typed(self, relation: Relation) -> bool:
        domains = RELATION_DOMAINS.get(relation.type)
        if domains is None:
            return False
        source_types, target_types = domains
        return (
            self.type_of(relation.source) in source_types
            and self.type_of(relation.target) in target_types
        )

    def to_dict(self) -> dict:
        return {
            "entities": [
                {
                    "id": e.id,
                    "type": e.type,
                    "name": e.name,
                    **({"doc": e.doc} if e.doc else {}),
                    **({"attributes": e.attributes} if e.attributes else {}),
                }
                for e in self.entities.values()
            ],
            "relations": [
                {"source": r.source, "type": r.type, "target": r.target, "doc": r.doc}
                for r in self.relations
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Graph:
        graph = cls()
        for item in payload.get("entities", []):
            graph.add_entity(
                Entity(
                    id=item["id"],
                    type=item["type"],
                    name=item["name"],
                    doc=item.get("doc", ""),
                    attributes=item.get("attributes", {}),
                )
            )
        for item in payload.get("relations", []):
            graph.add_relation(
                Relation(
                    source=item["source"],
                    type=item["type"],
                    target=item["target"],
                    doc=item.get("doc", ""),
                )
            )
        return graph


def load_graph(path: str | Path) -> Graph:
    return Graph.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
