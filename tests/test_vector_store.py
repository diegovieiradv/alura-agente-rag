from langchain_core.embeddings import FakeEmbeddings

from src.chunking import Chunk
from src.vector_store import (
    VectorStoreError,
    build_vector_store,
    load_vector_store,
)


def _chunks(nomes: tuple[str, ...] = ("manual.pdf",)) -> list[Chunk]:
    return [
        Chunk(
            text=f"conteudo do documento {nome} {i}",
            metadata={"source": nome, "page": 1, "total_pages": 1, "chunk_index": i},
        )
        for nome in nomes
        for i in range(3)
    ]


def test_indexacao_persiste_e_nao_duplica(tmp_path):
    store1 = build_vector_store(_chunks(), tmp_path, embedding_function=FakeEmbeddings(size=8))
    count1 = store1._collection.count()
    assert count1 == 3

    store2 = build_vector_store(_chunks(), tmp_path, embedding_function=FakeEmbeddings(size=8))
    assert store2._collection.count() == count1


def test_mudanca_de_documentos_reindexa(tmp_path):
    build_vector_store(_chunks(("manual.pdf",)), tmp_path, embedding_function=FakeEmbeddings(size=8))
    store = build_vector_store(
        _chunks(("manual.pdf", "apostila.pdf")), tmp_path, embedding_function=FakeEmbeddings(size=8)
    )
    assert store._collection.count() == 6


def test_reutiliza_base_persistida(tmp_path):
    build_vector_store(_chunks(), tmp_path, embedding_function=FakeEmbeddings(size=8))
    store = load_vector_store(tmp_path, embedding_function=FakeEmbeddings(size=8))
    assert store._collection.count() == 3


def test_metadados_preservados(tmp_path):
    store = build_vector_store(_chunks(), tmp_path, embedding_function=FakeEmbeddings(size=8))
    dados = store.get()
    metadados = dados.get("metadatas", [])
    fontes = {m.get("source") for m in metadados}
    assert fontes == {"manual.pdf"}


def test_sem_chunks_levanta_erro(tmp_path):
    try:
        build_vector_store([], tmp_path, embedding_function=FakeEmbeddings(size=8))
    except VectorStoreError as exc:
        assert "nenhum chunk" in str(exc)
    else:
        raise AssertionError("deveria levantar VectorStoreError")


def test_carregar_base_inexistente_levanta_erro(tmp_path):
    try:
        load_vector_store(tmp_path / "inexistente", embedding_function=FakeEmbeddings(size=8))
    except VectorStoreError as exc:
        assert "inexistente" in str(exc)
    else:
        raise AssertionError("deveria levantar VectorStoreError")