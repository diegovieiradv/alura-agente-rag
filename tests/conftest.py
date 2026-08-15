import sys
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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