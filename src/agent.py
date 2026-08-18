from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from src.rag import LLMClient, RAGEngine, RAGError, Source

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = (
    "Você é um agente inteligente. Sua tarefa é decidir a melhor ação para "
    "responder ao usuário. Você tem uma ferramenta disponível:\n"
    "- consultar_base(pergunta): consulta a base de conhecimento para\n"
    "  responder perguntas sobre os documentos disponíveis.\n"
    "Regras de decisao:\n"
    "1. Responda diretamente, de forma breve e educada, SOMENTE se a pergunta\n"
    "   for um cumprimento, despedida ou agradecimento (ex.: 'oi', 'bom dia',\n"
    "   'obrigado'). Nunca invente fatos, previsoes ou dados em respostas\n"
    "   diretas.\n"
    "2. Para qualquer outra pergunta (mesmo que pareca ser de assunto geral),\n"
    "   escolha a ferramenta consultar_base.\n"
    "Responda SOMENTE em JSON, com um destes formatos:\n"
    '{"tool": "consultar_base"}\n'
    '{"tool": "none", "reply": "sua resposta direta aqui"}'
)

QUERY_EXPANSION_PROMPT = (
    "Você gera termos de busca para um retriever de documentos técnicos. "
    "Dada a pergunta do usuário, responda SOMENTE com JSON no formato "
    '{"query": "termos de busca com 3 a 6 termos técnicos concretos, '
    'separados por virgula"}. Use nomes reais de tecnologias, frameworks, '
    "praticas ou processos relacionados ao tema (ex.: react, typescript, "
    "pull request, core web vitals, spring boot). NAO inclua o nome da "
    "empresa nem palavras genéricas."
)

DIRECT_FALLBACK_REPLY = "Desculpe, não entendi a pergunta. Pode reformular?"


@dataclass
class AgentResult:
    """Unified output of the agent (base answer or conversational reply)."""

    question: str
    response: str
    sources: list[Source] = field(default_factory=list)
    found: bool = False
    conversational: bool = False


class Agent:
    """Orchestrates tools: choose the right tool for each user question."""

    def __init__(self, rag_engine: RAGEngine, llm: LLMClient):
        self._rag = rag_engine
        self._llm = llm

    @property
    def tools(self) -> dict:
        return {
            "consultar_base": "consulta a base de conhecimento para responder "
            "perguntas sobre os documentos disponíveis"
        }

    def _decide_action(self, question: str) -> dict:
        prompt = f'Usuario perguntou: "{question}"\n\nDecida a acao correspondente e retorne o JSON pedido.'
        try:
            raw = self._llm.complete(AGENT_SYSTEM_PROMPT, prompt)
        except Exception as exc:
            logger.exception("falha na decisao do agente")
            raise RAGError(f"erro ao decidir acao: {exc}") from exc
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if not match:
            logger.warning("decisao do agente sem JSON; assumindo consulta a base")
            return {"tool": "consultar_base"}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("decisao do agente invalida; assumindo consulta a base")
            return {"tool": "consultar_base"}
        return data if isinstance(data, dict) else {"tool": "consultar_base"}

    def _expand_query(self, question: str) -> str | None:
        """Gera termos-chave para a busca semantica, com fallback seguro."""
        try:
            raw = self._llm.complete(QUERY_EXPANSION_PROMPT, f'Pergunta: "{question}"')
        except Exception as exc:
            logger.warning("falha na expansao de query (%s); usando pergunta original", exc)
            return None
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        query = data.get("query")
        return query.strip() if isinstance(query, str) and query.strip() else None

    def respond(self, question: str) -> AgentResult:
        question = question.strip()
        if not question:
            return AgentResult(question=question, response=DIRECT_FALLBACK_REPLY, conversational=True)

        decision = self._decide_action(question)

        if decision.get("tool") == "none" and decision.get("reply"):
            return AgentResult(
                question=question,
                response=decision["reply"].strip(),
                conversational=True,
            )

        expanded = self._expand_query(question)
        answer = self._rag.answer(question, query=expanded)
        return AgentResult(
            question=question,
            response=answer.response,
            sources=answer.sources,
            found=answer.found,
            conversational=False,
        )
