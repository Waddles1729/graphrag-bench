"""The document corpus, and how it is cut up for retrieval.

Chunking is by markdown section rather than by a fixed window. That is a
deliberate choice in the baseline's favour: sections are topically coherent, so
a passage retriever gets the cleanest possible units to work with. If the
baseline still cannot answer a question, it is not because the chunking was
unkind to it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage."""

    id: str
    doc_id: str
    heading: str
    text: str

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    chunks: tuple[Chunk, ...]


_SECTION = re.compile(r"^##\s+(.*)$", re.MULTILINE)
_TITLE = re.compile(r"^#\s+(.*)$", re.MULTILINE)


def split_sections(doc_id: str, title: str, body: str) -> list[Chunk]:
    """Split a document body into one chunk per `##` section, plus the preamble."""
    parts: list[Chunk] = []
    matches = list(_SECTION.finditer(body))

    preamble = body[: matches[0].start()] if matches else body
    preamble = preamble.strip()
    if preamble:
        parts.append(
            Chunk(id=f"{doc_id}#0", doc_id=doc_id, heading=title, text=f"{title}\n\n{preamble}")
        )

    for index, match in enumerate(matches, start=1):
        end = matches[index].start() if index < len(matches) else len(body)
        heading = match.group(1).strip()
        section = body[match.end() : end].strip()
        if not section:
            continue
        parts.append(
            Chunk(
                id=f"{doc_id}#{index}",
                doc_id=doc_id,
                heading=heading,
                # The document title is prepended so a section does not lose its
                # subject: "## Failure modes" on its own retrieves badly, and
                # that would be an unforced handicap rather than a real finding.
                text=f"{title} — {heading}\n\n{section}",
            )
        )
    return parts


def load_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    title_match = _TITLE.search(raw)
    title = title_match.group(1).strip() if title_match else path.stem
    body = raw[title_match.end() :] if title_match else raw
    doc_id = path.stem
    return Document(
        id=doc_id,
        title=title,
        text=raw,
        chunks=tuple(split_sections(doc_id, title, body)),
    )


def load_corpus(directory: str | Path) -> list[Document]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"no corpus directory at {directory}")
    documents = [load_document(path) for path in sorted(directory.glob("*.md"))]
    if not documents:
        raise ValueError(f"no markdown documents in {directory}")
    return documents


def all_chunks(documents: list[Document]) -> list[Chunk]:
    return [chunk for document in documents for chunk in document.chunks]
