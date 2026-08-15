import math
import sys
import zlib
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TokenEmbedding(Embeddings):
    """Deterministic bag-of-tokens embedding so tests can assert semantics."""

    def __init__(self, size: int = 32):
        self.size = size

    def _vec(self, text: str) -> list:
        vector = [0.0] * self.size
        for token in text.lower().split():
            vector[zlib.crc32(token.encode()) % self.size] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture()
def make_pdf(tmp_path):
    def _make(name: str, *lines: str) -> Path:
        path = tmp_path / name
        writer = PdfWriter()
        page = writer.add_blank_page(width=612, height=792)
        content = DecodedStreamObject()
        content.set_data(f"BT /F1 24 Tf 72 720 Td ({lines[0]}) Tj ET".encode())
        page[NameObject("/Contents")] = content
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        with path.open("wb") as fh:
            writer.write(fh)
        return path

    return _make
