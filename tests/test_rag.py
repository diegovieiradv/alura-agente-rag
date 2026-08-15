import pytest
from conftest import TokenEmbedding

from src.chunking import Chunk
from src.rag import (
    NO_EVIDENCE_RESPONSE,
    SYSTEM_PROMPT,
    RAGEngine,
    RAGError,
)
from src.vector_store import build_vector_store


class FakeLLM:
    def __init__(self, resposta="Resposta gerada a partir do contexto.", erro=None):
        self.resposta = resposta
        self.erro = erro
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        if self.erro:
            raise self.erro
        return self.resposta


CHUNKS = [
    Chunk(
        "O cafe expresso custa dez reais na cafeteria da empresa.",
        {"source": "cardapio.pdf", "page": 1, "total_pages": 2, "chunk_index": 0},
    ),
    Chunk(
        "A cafeteria abre de segunda a sexta das oito as dezoito horas.",
        {"source": "cardapio.pdf", "page": 2, "total_pages": 2, "chunk_index": 0},
    ),
]


def _engine(tmp_path, llm, min_relevance=0.1):
    store = build_vector_store(CHUNKS, tmp_path, embedding_function=TokenEmbedding(size=512))
    from src.retriever import Retriever

    retriever = Retriever(store, k=4, min_relevance=min_relevance)
    return RAGEngine(retriever, llm)


def test_responde_usando_contexto_e_fontes(tmp_path):
    llm = FakeLLM("O cafe custa dez reais.")
    engine = _engine(tmp_path, llm)

    resposta = engine.answer("quanto custa o cafe?")

    assert resposta.found
    assert resposta.response == "O cafe custa dez reais."
    assert len(resposta.sources) == 1
    assert resposta.sources[0].document == "cardapio.pdf"
    assert resposta.sources[0].page == 1
    assert resposta.sources[0].excerpt.startswith("O cafe expresso custa dez reais")
    system, user = llm.calls[0]
    assert "cafe expresso custa dez reais" in user
    assert SYSTEM_PROMPT in system


def test_sem_evidencia_nao_chama_llm(tmp_path):
    llm = FakeLLM()
    engine = _engine(tmp_path, llm, min_relevance=0.9)

    resposta = engine.answer("receita de bolo de chocolate")

    assert not resposta.found
    assert resposta.response == NO_EVIDENCE_RESPONSE
    assert resposta.sources == []
    assert llm.calls == []


def test_fontes_deduplicadas_por_documento_e_pagina(tmp_path):
    llm = FakeLLM()
    engine = _engine(tmp_path, llm, min_relevance=0.05)

    resposta = engine.answer("o cafe e o horario da cafeteria")

    paginas = sorted(s.page for s in resposta.sources)
    assert paginas == [1, 2]
    assert len(resposta.sources) == 2


def test_falha_do_llm_vira_ragerror(tmp_path):
    llm = FakeLLM(erro=RuntimeError("quota excedida"))
    engine = _engine(tmp_path, llm)

    with pytest.raises(RAGError, match="quota excedida"):
        engine.answer("quanto custa o cafe?")


def test_resposta_vazia_vira_ragerror(tmp_path):
    llm = FakeLLM(resposta="")
    engine = _engine(tmp_path, llm)

    with pytest.raises(RAGError, match="vazia"):
        engine.answer("quanto custa o cafe?")


def test_bloco_de_fontes_aparece_na_resposta(tmp_path):
    llm = FakeLLM("O cafe custa dez reais.")
    engine = _engine(tmp_path, llm)

    resposta = engine.answer("quanto custa o cafe?")

    assert "cardapio.pdf (página 1)" in resposta.display_response
    assert "Fontes:" in resposta.display_response


def test_fonte_inclui_excerto_real(tmp_path):
    llm = FakeLLM("Resposta.")
    engine = _engine(tmp_path, llm)

    resposta = engine.answer("quanto custa o cafe?")

    assert resposta.sources[0].excerpt.startswith("O cafe expresso custa dez reais")
    assert len(resposta.sources[0].excerpt) <= 160


def test_sem_evidencia_bloco_de_fontes_vazio(tmp_path):
    llm = FakeLLM()
    engine = _engine(tmp_path, llm, min_relevance=0.9)

    resposta = engine.answer("receita de bolo de chocolate")

    assert resposta.display_response == resposta.response
    assert "Fontes:" not in resposta.display_response
