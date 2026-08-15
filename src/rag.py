from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from src.retriever import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Minimal interface so the RAG engine can run without a fixed vendor."""

    model: str

    def complete(self, system: str, user: str) -> str: ...


class RAGError(Exception):
    """Raised when the LLM call fails."""


SYSTEM_PROMPT = (
    "Você é um assistente que responde exclusivamente com base na base de "
    "conhecimento fornecida no contexto abaixo. Responda em português. Use "
    "apenas as informações do contexto; nunca invente dados. Se o contexto não "
    "contiver informação suficiente para responder, diga isso claramente ao "
    "usuário em vez de adivinhar."
)

NO_EVIDENCE_RESPONSE = (
    "Não encontrei informações suficientes na base de conhecimento para "
    "responder a essa pergunta. Tente reformular ou perguntar sobre um tema "
    "contido nos documentos disponíveis."
)


@dataclass(frozen=True)
class Source:
    document: str
    page: int


@dataclass
class Answer:
    question: str
    response: str
    sources: List[Source] = field(default_factory=list)
    evidence: List[RetrievedChunk] = field(default_factory=list)
    found: bool = False


class RAGEngine:
    """Retrieval-Augmented Generation: evidence retrieval + grounded answer."""

    def __init__(self, retriever: Retriever, llm: LLMClient):
        self._retriever = retriever
        self._llm = llm

    def _build_user_prompt(self, question: str, evidence: List[RetrievedChunk]) -> str:
        fragmentos = []
        for chunk in evidence:
            origem = chunk.metadata.get("source", "desconhecido")
            pagina = chunk.metadata.get("page", "?")
            fragmentos.append(f"Trecho (fonte: {origem}, página {pagina}):\n{chunk.text}")
        contexto = "\n\n".join(fragmentos)
        return (
            f"CONTEXTO (base de conhecimento):\n{contexto}\n\n"
            f"PERGUNTA: {question}\n\n"
            "Responda usando somente o CONTEXTO acima."
        )

    def _extract_sources(self, evidence: List[RetrievedChunk]) -> List[Source]:
        vistos = set()
        fontes: List[Source] = []
        for chunk in evidence:
            chave = (chunk.metadata.get("source"), chunk.metadata.get("page"))
            if chave in vistos:
                continue
            vistos.add(chave)
            fontes.append(Source(document=chave[0], page=int(chave[1])))
        return fontes

    def answer(self, question: str) -> Answer:
        question = question.strip()
        evidence = self._retriever.retrieve(question)
        if not evidence:
            return Answer(question=question, response=NO_EVIDENCE_RESPONSE)

        prompt = self._build_user_prompt(question, evidence)
        try:
            response = (self._llm.complete(SYSTEM_PROMPT, prompt) or "").strip()
        except Exception as exc:
            logger.exception("falha na geracao da resposta")
            raise RAGError(f"erro ao gerar resposta: {exc}") from exc

        if not response:
            raise RAGError("modelo retornou resposta vazia")

        return Answer(
            question=question,
            response=response,
            sources=self._extract_sources(evidence),
            evidence=evidence,
            found=True,
        )