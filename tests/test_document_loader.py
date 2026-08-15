from pathlib import Path

import pytest

from src.document_loader import DocumentLoaderError, load_documents


def test_extrai_texto_e_metadados_de_pdf(make_pdf, tmp_path):
    make_pdf("manual.pdf", "Bem vindo ao agente")
    pages, report = load_documents(tmp_path)

    assert report.ok
    assert report.documents == 1
    assert report.pages == 1
    assert len(pages) == 1
    assert pages[0].text == "Bem vindo ao agente"
    assert pages[0].metadata == {
        "source": "manual.pdf",
        "page": 1,
        "total_pages": 1,
    }


def test_txt_isolado_como_uma_pagina(tmp_path):
    (tmp_path / "notas.txt").write_text("  Conteudo espalhado   em\n  varias linhas  ", encoding="utf-8")
    pages, report = load_documents(tmp_path)

    assert report.ok
    assert pages[0].text == "Conteudo espalhado em varias linhas"
    assert pages[0].metadata == {"source": "notas.txt", "page": 1, "total_pages": 1}


def test_pdf_corrompido_nao_aborta_carga(tmp_path):
    (tmp_path / "quebrado.pdf").write_bytes(b"nao-e-um-pdf-valido")
    (tmp_path / "ok.txt").write_text("conteudo valido", encoding="utf-8")

    pages, report = load_documents(tmp_path)

    assert report.pages == 1
    assert pages[0].source == "ok.txt"
    assert report.errors
    assert "quebrado.pdf" in report.errors[0]


def test_extensoes_nao_suportadas_ignoradas(tmp_path):
    (tmp_path / "imagem.png").write_bytes(b"\x89PNG\r\n")
    pages, report = load_documents(tmp_path)

    assert report.pages == 0
    assert report.skipped == 1
    assert report.documents == 0


def test_diretorio_inexistente_levanta_erro(tmp_path):
    with pytest.raises(DocumentLoaderError):
        load_documents(tmp_path / "nao-existe")


def test_base_vazia_reporta_problema(tmp_path):
    pages, report = load_documents(tmp_path)
    assert pages == []
    assert not report.ok
    assert "nenhum documento carregado" in report.errors[0]