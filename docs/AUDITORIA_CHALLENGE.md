# Auditoria do Challenge — Alura Agente RAG (G10)

Estado verificado em **17/08/2026** contra os requisitos conhecidos do
Challenge. Esta auditoria reflete o que está **implementado, testado e
validado**, não declarações de intenção.

| Requisito                    | Implementado | Evidência                                                    |
| ---------------------------- | ------------ | ------------------------------------------------------------ |
| Agente inteligente           | Sim          | `src/agent.py` — decide a ação por pergunta (conversa ou ferramenta `consultar_base`) e orquestra o uso da base |
| Respostas em linguagem natural | Sim        | `src/rag.py` (prompt de sistema) + `src/agent.py` + interface de chat `app.py` |
| Base de conhecimento própria | Sim          | `documentos/classificador-conteudo-tecnico.txt` (10 exemplos reais) + `src/document_loader.py` (extração, validação, metadados) + `src/chunking.py` |
| RAG                          | Sim          | `src/vector_store.py` (ChromaDB + embeddings Hugging Face), `src/retriever.py` (busca semântica c/ limiar), `src/rag.py` (pipeline) |
| Regra anti-alucinação        | Sim          | Sem evidência recuperada → resposta clara de ausência (`src/rag.py`); fontes apenas derivadas das evidências; decisão do agente só responde direto para cumprimentos/despedidas |
| Fontes nas respostas         | Sim          | Documento, página e trecho real recuperado (`Answer.display_response`) |
| GitHub / documentação        | Sim          | Repositório público `diegovieiradv/alura-agente-rag` com commits evolutivos; `README.md` completo |
| Deploy                       | **Parcial**  | Preparado (`.streamlit/config.toml`, secrets no `app.py`, runbook no README). **Não publicado** — sem URL |
| OCI, se obrigatório          | **Não**      | Não implementado. Avaliação separada documentada no README    |

## Testes automatizados

- **43 testes passando** (`pytest`).
- Cobertura por camada: loader, chunking, vetores/índice, retriever, RAG,
  agente, config, e leitura de secrets para deploy.
- Lint/formação: `ruff check` e `ruff format --check` limpos.

## Validação ponta a ponta (chave Groq real — 17/08/2026)

Fluxo completo executado localmente com a API Groq real:

| Pergunta                                        | Resultado                          |
| ----------------------------------------------- | ---------------------------------- |
| APIs REST com Spring Boot                       | ✅ Backend + fonte                  |
| Deploys com Docker e Kubernetes                 | ✅ DevOps + fonte                   |
| Análise de dados com pandas                     | ✅ Dados + fonte                    |
| Interfaces com React e TypeScript               | ✅ Frontend + fonte                 |
| Política de reembolso (fora da base)            | ✅ Recusa sem inventar              |
| Previsão do tempo (fora da base)                | ✅ Recusa sem inventar              |
| "Oi, tudo bem?"                                 | ✅ Resposta conversacional direta   |

> Durante a validação, o modelo padrão `llama-3.3-70b-versatile` foi
> descontinuado pela Groq (HTTP 404). O padrão foi atualizado para
> `openai/gpt-oss-120b` (disponível na conta usada) e a regra de decisão do
> agente foi ajustada para garantir que perguntas gerais também passem pela
> base (evitando alucinação em respostas diretas).

## Pendências detectadas

1. **Deploy publicado**: a aplicação está preparada e o repositório está
   público no GitHub, porém a publicação no Streamlit Community Cloud
   (e/ou OCI) e a URL pública ainda precisam ser executadas e verificadas
   no painel do Streamlit.

## Conclusão

Todos os requisitos **de implementação** estão atendidos e o fluxo foi
validado ponta a ponta com chave real. O item **Deploy** depende de
execução externa (deploy no painel do Streamlit Community Cloud) e fica
como ação pendente para a entrega final.