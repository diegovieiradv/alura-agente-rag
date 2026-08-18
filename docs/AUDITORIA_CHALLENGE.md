# Auditoria do Challenge — Alura Agente RAG (G10)

Estado verificado em **17/08/2026** contra os requisitos conhecidos do
Challenge. Esta auditoria reflete o que está **implementado, testado e
validado**, não declarações de intenção.

| Requisito                    | Implementado | Evidência                                                    |
| ---------------------------- | ------------ | ------------------------------------------------------------ |
| Agente inteligente           | Sim          | `src/agent.py` — decide a ação por pergunta (conversa ou ferramenta `consultar_base`), expande a pergunta em termos-chave e orquestra o uso da base |
| Respostas em linguagem natural | Sim        | `src/rag.py` (prompt de sistema) + `src/agent.py` + interface de chat `app.py` |
| Base de conhecimento própria | Sim          | `documentos/Santo_Pegasus_Base_Conhecimento_Completa.pdf` (76 páginas reais: guias Front-end, Back-end, Microsserviços, Onboarding, Incidentes) + `src/document_loader.py` (extração, validação, metadados) + `src/chunking.py` |
| RAG                          | Sim          | `src/vector_store.py` (ChromaDB + embeddings Hugging Face), `src/retriever.py` (busca semântica + multi-query com Reciprocal Rank Fusion), `src/rag.py` (pipeline) |
| Regra anti-alucinação        | Sim          | Sem evidência recuperada → resposta clara de ausência (`src/rag.py`); fontes apenas derivadas das evidências; decisão do agente só responde direto para cumprimentos/despedidas |
| Fontes nas respostas         | Sim          | Documento, página e trecho real recuperado (`Answer.display_response`) |
| GitHub / documentação        | Sim          | Repositório público `diegovieiradv/alura-agente-rag` com commits evolutivos; `README.md` completo |
| Deploy                       | **Sim**      | Publicado no Streamlit Community Cloud: `alura-agente-rag-x6dyastn7mtrhufbz85quw.streamlit.app` — secrets configurados, app no ar |
| OCI, se obrigatório          | **Não**      | Não implementado. Avaliação separada documentada no README    |

## Testes automatizados

- **49 testes passando** (`pytest`).
- Cobertura por camada: loader, chunking, vetores/índice, retriever, RAG,
  agente (incluindo expansão de query), config, e leitura de secrets para deploy.
- Lint/formação: `ruff check` e `ruff format --check` limpos.

## Validação ponta a ponta (chave Groq real — 17/08/2026)

Fluxo completo executado com a API Groq real sobre a base
`Santo_Pegasus_Base_Conhecimento_Completa.pdf` (76 páginas):

| Pergunta                                        | Resultado                          |
| ----------------------------------------------- | ---------------------------------- |
| Qual o stack de front-end?                      | ✅ React 18, TypeScript 5, Next.js 14, Vite, Node 20 + fonte |
| Qual framework de backend?                      | ✅ Spring Boot 3+ + fonte           |
| Como funciona o Pull Request?                   | ✅ Processo completo (2 aprovações, CI verde) + fonte |
| Política de férias CLT                          | ✅ 30 dias, fracionamento, 1/3, Portal do Colaborador + fonte |
| Boas práticas de Core Web Vitals                | ✅ LCP/FID/CLS + fonte              |
| Política de home office/híbrido                 | ✅ Híbrido flexível + fonte         |
| Previsão do tempo (fora da base)                | ✅ Recusa sem inventar              |
| "Olá, tudo bem?"                                | ✅ Resposta conversacional direta   |

> Para documentos técnicos densos, o chunking foi recalibrado (280 → 800
> caracteres, overlap 80) e a recuperação passou a combinar a pergunta original
> com termos-chave expandidos via multi-query + Reciprocal Rank Fusion
> (`retriever.retrieve_multi`), elevando a precisão da recuperação.

## Pendências detectadas

Nenhuma pendência de implementação. Pontos opcionais documentados no README
(OCI, métricas de avaliação da recuperação, streaming).