import pytest
from conftest import TokenEmbedding

from src.agent import Agent, AgentResult
from src.chunking import Chunk
from src.rag import NO_EVIDENCE_RESPONSE, RAGEngine, RAGError
from src.retriever import Retriever
from src.vector_store import build_vector_store


class ScriptedLLM:
    """Returns one scripted response per call, in order."""

    def __init__(self, respostas):
        self.respostas = list(respostas)
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        if not self.respostas:
            return ""
        return self.respostas.pop(0)


CHUNKS = [
    Chunk(
        "O cafe expresso custa dez reais na cafeteria da empresa.",
        {"source": "cardapio.pdf", "page": 1, "total_pages": 2, "chunk_index": 0},
    ),
]


def _agent(tmp_path, llm):
    store = build_vector_store(CHUNKS, tmp_path, embedding_function=TokenEmbedding(size=512))
    retriever = Retriever(store, k=4, min_relevance=0.1)
    rag = RAGEngine(retriever, llm)
    return Agent(rag, llm)


def test_cumprimento_responde_diretamente_sem_consultar_base(tmp_path):
    llm = ScriptedLLM(['{"tool": "none", "reply": "Olá! Como posso ajudar?"}'])
    agente = _agent(tmp_path, llm)

    resultado = agente.respond("ola bom dia")

    assert isinstance(resultado, AgentResult)
    assert resultado.response == "Olá! Como posso ajudar?"
    assert resultado.conversational
    assert resultado.sources == []
    assert llm.calls == 1


def test_pergunta_de_dominio_usa_ferramenta_consultar_base(tmp_path):
    llm = ScriptedLLM(
        [
            '{"tool": "consultar_base"}',
            '{"query": "cafe expresso preco"}',
            "O cafe expresso custa dez reais.",
        ]
    )
    agente = _agent(tmp_path, llm)

    resultado = agente.respond("quanto custa o cafe?")

    assert not resultado.conversational
    assert resultado.found
    assert resultado.response == "O cafe expresso custa dez reais."
    assert len(resultado.sources) == 1
    assert resultado.sources[0].document == "cardapio.pdf"
    assert resultado.sources[0].page == 1
    assert llm.calls == 3


def test_consulta_sem_query_usa_pergunta_original(tmp_path):
    llm = ScriptedLLM(
        [
            '{"tool": "consultar_base"}',
            "resposta nao-json na expansao",
            "O cafe expresso custa dez reais.",
        ]
    )
    agente = _agent(tmp_path, llm)

    resultado = agente.respond("quanto custa o cafe?")

    assert resultado.found
    assert resultado.response == "O cafe expresso custa dez reais."
    assert llm.calls == 3


def test_pergunta_fora_da_base_via_agente(tmp_path):
    llm = ScriptedLLM(
        [
            '{"tool": "consultar_base"}',
            '{"query": "bolo morango receita"}',
        ]
    )
    agente = _agent(tmp_path, llm)

    resultado = agente.respond("receita de bolo de morango")

    assert not resultado.found
    assert "base de conhecimento" in resultado.response
    assert resultado.response == NO_EVIDENCE_RESPONSE


def test_decisao_sem_json_assume_consulta_a_base(tmp_path):
    llm = ScriptedLLM(
        [
            "resposta sem json nenhum",
            '{"query": "cafe expresso"}',
            "Resposta baseada na base.",
        ]
    )
    agente = _agent(tmp_path, llm)

    resultado = agente.respond("quanto custa o cafe?")

    assert resultado.found
    assert "baseada na base" in resultado.response


def test_pergunta_vazia_responde_fallback(tmp_path):
    llm = ScriptedLLM([])
    agente = _agent(tmp_path, llm)

    resultado = agente.respond("   ")

    assert resultado.conversational
    assert llm.calls == 0


def test_erro_do_llm_propaga_ragerror(tmp_path):
    class LlmQueimado(ScriptedLLM):
        def complete(self, system, user):
            raise RuntimeError("api fora do ar")

    agente = _agent(tmp_path, LlmQueimado([]))

    with pytest.raises(RAGError, match="api fora do ar"):
        agente.respond("quanto custa o cafe?")
