from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.document_loader import Page

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 80


@dataclass(frozen=True)
class Chunk:
    text: str
    metadata: dict


def chunk_pages(
    pages: list[Page],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split each page into overlapping chunks, preserving provenance.

    A chunk inherits ``source`` and ``page`` from its parent page plus a
    per-page ``chunk_index`` so that every result can point back to the
    original document and page.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    for page in pages:
        for index, part in enumerate(splitter.split_text(page.text)):
            chunks.append(
                Chunk(
                    text=part,
                    metadata={**page.metadata, "chunk_index": index},
                )
            )
    return chunks
