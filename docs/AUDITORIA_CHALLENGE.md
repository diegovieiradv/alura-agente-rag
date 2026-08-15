# Auditoria do Challenge — Alura Agente RAG (G10)

Estado verificado em **14/08/2026** contra os requisitos conhecidos do
Challenge. Esta auditoria reflete o que está **implementado e testado**, não
declarações de intenção.

| Requisito                    | Implementado | Evidência                                                    |
| ---------------------------- | ------------ | ------------------------------------------------------------ |
| Agente inteligente           | Sim          | `src/agent.py` — decide a ação por pergunta (conversa ou ferramenta `consultar_base`) e orquestra o uso da base |
| Respostas em linguagem natural | Sim        | `src/rag.py` (prompt de sistema) + `src/agent.py` + interface de chat `app.py` |
| Base de conhecimento própria | Sim          | `documentos/` (PDF/TXT/MD) + `src/document_loader.py` (extração, validação, metadados) + `src/chunking.py` |
| RAG                          | Sim          | `src/vector_store.py` (ChromaDB + embeddings Hugging Face), `src/retriever.py` (busca semântica c/ limiar), `src/rag.py` (pipeline) |
| Regra anti-alucinação        | Sim          | Sem evidência recuperada → resposta clara de ausência (`src/rag.py`); fontes apenas derivadas das evidências |
| Fontes nas respostas         | Sim          | Documento, página e trecho real recuperado (`Answer.display_response`) |
| GitHub / documentação        | Sim          | Repositório Git com 12 commits evolutivos; `README.md` completo |
| Deploy                       | **Parcial**  | Preparado (`.streamlit/config.toml`, secrets no `app.py`, runbook no README). **Não publicado** — sem URL |
| OCI, se obrigatório          | **Não**      | Não implementado. Avaliação separada documentada no README    |

## Testes automatizados

- **43 testes passando** (`pytest`).
- Cobertura por camada: loader, chunking, vetores/índice, retriever, RAG,
  agente, config, e leitura de secrets para deploy.
- Lint/formação: `ruff check` e `ruff format --check` limpos.

## Pendências detectadas

1. **Base de conhecimento real**: a pasta `documentos/` está pronta, mas os
   documentos definitivos do tema escolhido precisam ser adicionados.
2. **Deploy publicado**: a aplicação está preparada, porém a publicação
   (Streamlit Community Cloud e/ou OCI) e a URL pública ainda precisam ser
   executadas e verificadas.
3. **Validação com chave real**: a resposta do LLM (rota Groq) exige uma
   `GROQ_API_KEY` real para validação ponta a ponta.

## Conclusão

Todos os requisitos **de implementação** estão atendidos. Os itens
**Deploy** e **OCI** dependem de decisão/execução externa (conta no serviço,
chave real e documentos do tema) e ficam como ação pendente para a entrega
final.