import logging
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.agent import Agent
from src.chunking import chunk_pages
from src.config import ConfigError
from src.document_loader import load_documents
from src.llm import GroqClient
from src.rag import RAGEngine
from src.retriever import Retriever
from src.vector_store import build_vector_store

logging.basicConfig(level=logging.INFO)

DOCUMENTOS_DIR = "documentos"

st.set_page_config(page_title="Agente RAG", page_icon=":books:")


@st.cache_resource(show_spinner="Carregando a base de conhecimento...")
def initialize(cfg: config.Config):
    pages, report = load_documents(DOCUMENTOS_DIR)
    if not pages:
        raise ValueError(
            "A pasta documentos/ está vazia ou não contém arquivos"
            " suportados (PDF, TXT, MD). Adicione os documentos e reinicie."
        )
    chunks = chunk_pages(pages)
    store = build_vector_store(chunks, persist_dir=cfg.chroma_dir)
    retriever = Retriever(store)
    rag = RAGEngine(retriever, GroqClient(cfg))
    return Agent(rag, GroqClient(cfg)), len(chunks), report


def bootstrap() -> tuple[Agent, int, "object", config.Config]:
    cfg = config.Config.from_env()
    require_groq_api_key(cfg)
    agente, n_chunks, report = initialize(cfg)
    return agente, n_chunks, report, cfg


def require_groq_api_key(cfg: config.Config) -> str:
    return config.require_groq_api_key(cfg)


def render_fontes(result) -> None:
    if not result.sources:
        return
    with st.expander("Fontes utilizadas"):
        for fonte in result.sources:
            st.markdown(f"**{fonte.document}** — página {fonte.page}")
            if fonte.excerpt:
                st.caption(f'"{fonte.excerpt}"')


def main() -> None:
    try:
        agente, n_chunks, report, cfg = bootstrap()
    except ConfigError as exc:
        st.error(str(exc))
        st.info("Configure a variável GROQ_API_KEY no arquivo .env e reinicie a aplicação.")
        st.stop()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.title("Agente RAG")
    st.caption(
        "Assistente que responde perguntas em linguagem natural usando"
        " exclusivamente a base de conhecimento."
    )

    with st.sidebar:
        st.subheader("Sobre")
        st.markdown(
            f"- Modelo de linguagem: `{cfg.llm_model}`\n"
            f"- Embeddings: `{cfg.embedding_model}`\n"
            f"- Chunks indexados: `{n_chunks}`\n"
            f"- Documentos carregados: `{report.documents}`\n"
            f"- Páginas: `{report.pages}`"
        )
        if report.ok is False and report.errors:
            st.warning("; ".join(report.errors))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for mensagem in st.session_state.messages:
        with st.chat_message(mensagem["role"]):
            st.markdown(mensagem["content"])

    pergunta = st.chat_input("Digite sua pergunta sobre os documentos...")
    if not pergunta:
        return

    st.session_state.messages.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with placeholder.container():
            with st.spinner("Consultando a base de conhecimento..."):
                try:
                    resultado = agente.respond(pergunta)
                except Exception as exc:
                    st.error(f"Ocorreu um erro ao responder. Detalhes: {exc}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Erro: {exc}"}
                    )
                    st.stop()
                st.markdown(resultado.response)
                render_fontes(resultado)

    st.session_state.messages.append(
        {"role": "assistant", "content": resultado.response}
    )


if __name__ == "__main__":
    main()