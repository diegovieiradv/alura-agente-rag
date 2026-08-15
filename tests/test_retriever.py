from src.chunking import Chunk
from src.retriever import Retriever
from src.vector_store import build_vector_store
from conftest import TokenEmbedding


def _store(tmp_path, chunks):
    return build_vector_store(chunks, tmp_path, embedding_function=TokenEmbedding(size=512))


CHUNKS = [
    Chunk(
        "O cafe expresso custa dez reais na cafeteria da empresa.",
        {"source": "cardapio.pdf", "page": 1, "total_pages": 2, "chunk_index": 0},
    ),
    Chunk(
        "A cafeteria abre de segunda a sexta das oito as dezoito horas.",
        {"source": "cardapio.pdf", "page": 2, "total_pages": 2, "chunk_index": 0},
    ),
    Chunk(
        "A reuniao de planejamento acontece toda sexta feira na sala tres.",
        {"source": "agenda.pdf", "page": 1, "total_pages": 1, "chunk_index": 0},
    ),
]


def test_retrieval_encontra_conteudo_relevante(tmp_path):
    store = _store(tmp_path, CHUNKS)
    retriever = Retriever(store, k=4, min_relevance=0.1)

    resultado = retriever.retrieve("quanto custa um cafe")

    assert resultado
    assert resultado[0].metadata["source"] == "cardapio.pdf"
    assert "dez reais" in resultado[0].text


def test_resultados_ordenados_por_relevancia(tmp_path):
    store = _store(tmp_path, CHUNKS)
    retriever = Retriever(store, k=4, min_relevance=0.0)

    resultado = retriever.retrieve("cafe expresso das oito")

    relevancias = [c.relevance for c in resultado]
    assert relevancias == sorted(relevancias, reverse=True)


def test_limite_k_respeitado(tmp_path):
    store = _store(tmp_path, CHUNKS)
    retriever = Retriever(store, k=2, min_relevance=0.0)

    assert len(retriever.retrieve("cafe")) == 2


def test_sem_evidencia_retorna_vazio(tmp_path):
    store = _store(tmp_path, CHUNKS)
    retriever = Retriever(store, k=4, min_relevance=0.5)

    resultado = retriever.retrieve("receita de bolo de chocolate com morango")

    assert len(resultado) == 0
    assert not retriever.has_evidence("receita de bolo de chocolate com morango")


def test_query_vazia_retorna_vazio(tmp_path):
    store = _store(tmp_path, CHUNKS)
    retriever = Retriever(store)

    assert retriever.retrieve("   ") == []


def test_load_from_persisted(tmp_path):
    _store(tmp_path, CHUNKS)
    retriever = Retriever.from_persisted(
        persist_dir=str(tmp_path),
        collection_name="alura_rag",
        k=2,
        min_relevance=0.1,
        embedding_function=TokenEmbedding(size=512),
    )

    resultado = retriever.retrieve("cafe")
    assert resultado
    assert "cafeteria" in resultado[0].text
