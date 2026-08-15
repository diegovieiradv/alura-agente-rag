from src.chunking import chunk_pages
from src.document_loader import Page


def _page(text: str, source: str = "manual.pdf", page: int = 1) -> Page:
    return Page(text=text, source=source, page=page, total_pages=3)


def test_pagina_curta_vira_um_unico_chunk():
    chunks = chunk_pages([_page("texto curto")])

    assert len(chunks) == 1
    assert chunks[0].text == "texto curto"
    assert chunks[0].metadata == {
        "source": "manual.pdf",
        "page": 1,
        "total_pages": 3,
        "chunk_index": 0,
    }


def test_pagina_longa_gera_multiplos_chunks_com_metadados():
    texto = "palavra repetida. " * 200
    chunks = chunk_pages([_page(texto)], chunk_size=800, overlap=0)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["source"] == "manual.pdf"
        assert chunk.metadata["page"] == 1
        assert chunk.metadata["chunk_index"] == chunks.index(chunk)
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[-1].metadata["chunk_index"] == len(chunks) - 1


def test_overlap_preserva_trecho_entre_chunks_vizinhos():
    texto = "palavra repetida. " * 300
    chunks = chunk_pages([_page(texto)], chunk_size=400, overlap=100)

    assert len(chunks) >= 2
    cabeca = chunks[1].text[:60]
    assert cabeca in chunks[0].text or any(cabeca in c.text for c in chunks[:-1])


def test_paginas_diferentes_nao_se_misturam():
    p1 = _page("apenas pagina um " * 300, page=1)
    p2 = _page("somente pagina dois " * 300, page=2)
    chunks = chunk_pages([p1, p2], chunk_size=600, overlap=0)

    paginas_usadas = {c.metadata["page"] for c in chunks}
    assert paginas_usadas == {1, 2}

    sem_mistura = all(("apenas pagina" in c.text) == (c.metadata["page"] == 1) for c in chunks)
    assert sem_mistura
