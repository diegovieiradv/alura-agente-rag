from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env")


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Config:
    groq_api_key: str
    embedding_model: str
    llm_model: str
    chroma_dir: str

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ),
            llm_model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
            chroma_dir=os.getenv("CHROMA_DIR", "data/chromadb"),
        )


def require_groq_api_key(config: Config) -> str:
    key = config.groq_api_key.strip().strip('"').strip("'")
    if not key:
        raise ConfigError("GROQ_API_KEY nao configurada. Copie .env.example para .env e preencha a chave.")
    if len(key) < 40:
        raise ConfigError(
            f"GROQ_API_KEY invalida: valor tem apenas {len(key)} caracteres. "
            "Chaves da Groq possuem ~57 caracteres. Confira o valor salvo no Streamlit Secrets."
        )
    return key
