# Alura Agente RAG

Agente inteligente que responde perguntas em linguagem natural com base em
uma base de conhecimento própria, usando **RAG (Retrieval-Augmented
Generation)**. As respostas são geradas **exclusivamente** a partir do que é
recuperado nos documentos fornecidos e, sempre que possível, indicam a origem
do conteúdo (documento, página e trecho).

Projeto desenvolvido para o **Challenge obrigatório Alura Agente — G10**.

---

## Problema

Responder perguntas de forma objetiva e confiável sobre um conjunto de
documentos exige localizar o conteúdo exato e usá-lo sem inventar
informações. Consultas manuais são lentas e respostas geradas por um LLM
"puro" podem alucinar — isto é, afirmar fatos que não existem na fonte.

## Objetivo

Construir um assistente que:

- responda perguntas em linguagem natural;
- realize busca semântica sobre uma base de conhecimento própria;
- use apenas as evidências recuperadas para gerar respostas;
- informe quando a base não contiver informação suficiente (sem inventar);
- indique a origem do conteúdo utilizado;
- seja fácil de executar localmente e esteja preparado para deploy.

## Solução

Uma aplicação Python que segue o pipeline clássico de RAG:

```text
Documentos/PDFs (pasta documentos/)
      ↓ extração de texto (pypdf)
Textos com metadados (fonte, página)
      ↓ chunking
Trechos com sobreposição e metadados
      ↓ embeddings Hugging Face (sentence-transformers)
Vetores
      ↓ indexação
Banco vetorial ChromaDB (persistente)
      ↓ busca semântica
Evidências com pontuação de relevância
      ↓ RAG (LLM Groq / Llama)
Resposta com fontes + regra anti-alucinação
      ↓
Interface Streamlit (chat)
```

Um **agente** decide a ação adequada para cada pergunta: cumprimentos e
assuntos gerais são respondidos diretamente; perguntas sobre os documentos
acionam a ferramenta `consultar_base`, que usa o pipeline RAG.

## Arquitetura

Por camadas, de forma a manter cada responsabilidade isolada e testável:

```text
app.py                  Interface Streamlit (chat web)
src/agent.py            Agente: escolha de ação e ferramenta consultar_base
src/rag.py              Pipeline RAG + regra anti-alucinação + fontes
src/retriever.py        Busca semântica com limiar de relevância
src/vector_store.py     Indexação e persistência no ChromaDB (sem duplicação)
src/chunking.py         Divisão dos textos em trechos com metadados
src/document_loader.py  Carregamento de PDF/TXT/MD com validação
src/llm.py              Cliente Groq (Llama)
src/config.py           Configuração segura por variáveis de ambiente
```

## Tecnologias

- Python 3.13
- LangChain (core, text-splitters, chroma, huggingface)
- ChromaDB (banco vetorial)
- sentence-transformers (embeddings Hugging Face — `paraphrase-multilingual-MiniLM-L12-v2`, suporte a português)
- Groq API (modelo Llama)
- pypdf (extração de PDF)
- Streamlit (interface web)
- python-dotenv (variáveis de ambiente)
- pytest (testes automatizados)
- Git/GitHub (versionamento)

## Estrutura de diretórios

```text
alura-agente-rag/
├── app.py                  # Interface Streamlit
├── README.md
├── requirements.txt
├── pyproject.toml          # Lint (ruff) e configuração do pytest
├── .env.example            # Exemplo de variáveis de ambiente
├── .gitignore
├── documentos/             # Base de conhecimento (PDF, TXT, MD)
├── notebooks/              # Experimentação (em desenvolvimento)
├── src/                    # Módulos Python da aplicação
│   ├── config.py
│   ├── document_loader.py
│   ├── chunking.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── rag.py
│   ├── llm.py
│   └── agent.py
└── tests/                  # testes (pytest)
```

## Instalação

Pré-requisitos: Python 3.13+ e acesso à internet (para dependências e
download do modelo de embeddings na primeira execução).

```bash
git clone <url-do-repositorio>
cd alura-agente-rag

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

## Configuração

1. Copie o arquivo de exemplo:

```bash
copy .env.example .env          # Windows
# cp .env.example .env          # Linux/macOS
```

2. Preencha a chave da API Groq no arquivo `.env` (obtenha em
   https://console.groq.com/keys):

```
GROQ_API_KEY=sua_chave_aqui
```

3. Coloque seus documentos na pasta `documentos/` (PDF, TXT ou MD).

## Variáveis de ambiente

| Variável        | Obrigatória | Padrão                                | Descrição                          |
| --------------- | ----------- | -------------------------------------- | ---------------------------------- |
| `GROQ_API_KEY`  | Sim         | —                                      | Chave da API Groq (sem segredos no código) |
| `EMBEDDING_MODEL` | Não      | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Modelo de embeddings Hugging Face (suporte a português) |
| `LLM_MODEL`     | Não         | `llama-3.3-70b-versatile`              | Modelo de linguagem da Groq        |
| `CHROMA_DIR`    | Não         | `data/chromadb`                        | Diretório persistente do banco vetorial |

O arquivo `.env` nunca é versionado (protegido pelo `.gitignore`).

## Execução

```bash
streamlit run app.py
```

A aplicação abre no navegador (por padrão em `http://localhost:8501`). Na
primeira execução, a base de conhecimento é carregada e indexada
automaticamente (pode levar alguns minutos por causa do download e carregamento
do modelo de embeddings; o resultado fica em cache e não é refeito se os
documentos não mudarem).

Para executar os testes:

```bash
pytest
```

Para verificar lint e formatação:

```bash
ruff check src tests app.py
ruff format --check src tests app.py
```

## Utilização

- Digite a pergunta no campo de chat e envie.
- Se a resposta vier da base de conhecimento, os **documentos e páginas**
  utilizados aparecem na seção "Fontes utilizadas", com o trecho recuperado.
- Se a base não contiver informação suficiente, o agente informa isso
  claramente em vez de inventar uma resposta.

## Exemplos

Tipo de pergunta recomendada para validar o RAG:

1. **Pergunta cuja resposta existe claramente nos documentos** — deve retornar
   resposta fundamentada e fontes.
2. **Pergunta cuja resposta está espalhada em mais de um trecho/página** — deve
   combinar informações de mais de uma fonte.
3. **Pergunta completamente fora da base** — deve responder que a informação
   não está disponível na base de conhecimento.

Exemplo:

```text
Usuário: Qual a política de reembolso descrita no manual?
Agente: Segundo o documento manual.pdf (página 3), reembolsos são solicitados
em até 30 dias após a compra...

Fontes: manual.pdf (página 3)
```

## Limitações

- As respostas dependem da qualidade e da cobertura dos documentos inseridos
  em `documentos/`.
- A busca é limitada ao modelo de embeddings escolhido e ao limiar de
  relevância configurado no retriever.
- O LLM pode se recusar quando a evidência for insuficiente, mas a recusa só
  é acionada sobre a ausência de conteúdo recuperado.
- A aplicação é protegida pela key da Groq; os limites de uso da conta
  (rate limits) se aplicam.
- A primeira inicialização é mais lenta por causa do carregamento do modelo de
  embeddings.

## Deploy

A aplicação está preparada para execução na nuvem com as configurações via
variáveis de ambiente.

### Streamlit Community Cloud (opção prevista para esta entrega)

1. Publique o repositório no GitHub (todos os arquivos necessários já estão
   versionados: `app.py`, `requirements.txt`, `.streamlit/config.toml` e a
   pasta `documentos/` com a base).
2. No painel do [Streamlit Community Cloud](https://streamlit.io/cloud),
   conecte o repositório, informe `app.py` como arquivo de entrada e defina o
   Python version (3.13).
3. Em **Settings → Secrets**, configure as variáveis:

```ini
GROQ_API_KEY=sua_chave
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
LLM_MODEL=llama-3.3-70b-versatile
CHROMA_DIR=/tmp/chromadb
```

   O app lê Secrets do Streamlit como variáveis de ambiente
   (`carregar_segredos_do_streamlit()`), então não é preciso nenhum arquivo
   `.env` no deploy. Em plataformas que expõem variáveis de ambiente
   diretamente, basta configurá-las no painel.

> **Nenhum deploy foi realizado até o momento.** A URL pública será
> documentada aqui somente depois de o deploy ser executado e verificado.

### Oracle Cloud Infrastructure (OCI)

A descrição original do Challenge menciona OCI. Isso **ainda não foi
implementado** — avaliar separadamente. Em uma possível execução OCI, a
aplicação poderia rodar em um Compute Instance com o mesmo `requirements.txt`,
usando Gunicorn/`streamlit run` atrás de um proxy reverso; essa integração será
documentada se for adotada.

## Possíveis melhorias

- Respostas com streaming de tokens na interface.
- Avaliação de qualidade da recuperação (ex.: `hits@k`, `MRR`) usando um
  conjunto fixo de perguntas de teste.
- Seleção de modelo de embeddings por idioma/domínio do conteúdo.
- Histórico persistido entre sessões.
- Múltiplas coleções por tema.
- Upload de documentos pela própria interface.

## Autor

Diego Vieira — desafio Alura Agente (G10).