import pytest

from src.config import Config, ConfigError, require_groq_api_key


def test_valores_padrao_quando_env_ausente(monkeypatch):
    for var in ("GROQ_API_KEY", "EMBEDDING_MODEL", "LLM_MODEL", "CHROMA_DIR"):
        monkeypatch.delenv(var, raising=False)

    cfg = Config.from_env()

    assert cfg.embedding_model == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    assert cfg.llm_model == "openai/gpt-oss-120b"
    assert cfg.chroma_dir == "data/chromadb"
    assert cfg.groq_api_key == ""


def test_variaveis_de_ambiente_prevalecem(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "modelo/custom")
    monkeypatch.setenv("LLM_MODEL", "llama-custom")
    monkeypatch.setenv("CHROMA_DIR", "dados/teste")

    cfg = Config.from_env()

    assert cfg.embedding_model == "modelo/custom"
    assert cfg.llm_model == "llama-custom"
    assert cfg.chroma_dir == "dados/teste"


def test_chave_vazia_levanta_configerror(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "   ")

    cfg = Config.from_env()

    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        require_groq_api_key(cfg)


def test_chave_presente_retorna_valor(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "sk-teste")

    cfg = Config.from_env()

    assert require_groq_api_key(cfg) == "sk-teste"


@pytest.mark.parametrize("chave_colada", ['"sk-teste"', "'sk-teste'", "  sk-teste  ", '"sk-teste"  '])
def test_chave_limpa_aspas_e_espacos(monkeypatch, chave_colada):
    monkeypatch.setenv("GROQ_API_KEY", chave_colada)

    cfg = Config.from_env()

    assert require_groq_api_key(cfg) == "sk-teste"
