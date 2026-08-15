import os
from types import SimpleNamespace

import app


def test_segredos_viram_variaveis_de_ambiente(monkeypatch):
    class SecretsFake:
        def __init__(self):
            self._dados = {"GROQ_API_KEY": "sk-secreto", "LLM_MODEL": "llama-x"}

        def get(self, chave):
            return self._dados.get(chave)

    monkeypatch.setattr(app, "st", SimpleNamespace(secrets=SecretsFake()))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    app.carregar_segredos_do_streamlit()

    assert os.environ["GROQ_API_KEY"] == "sk-secreto"
    assert os.environ["LLM_MODEL"] == "llama-x"


def test_segredos_nao_sobrescrevem_variaveis_existentes(monkeypatch):
    class SecretsFake:
        def get(self, chave):
            return "sk-do-secret"

    monkeypatch.setattr(app, "st", SimpleNamespace(secrets=SecretsFake()))
    monkeypatch.setenv("GROQ_API_KEY", "sk-do-ambiente")

    app.carregar_segredos_do_streamlit()

    assert os.environ["GROQ_API_KEY"] == "sk-do-ambiente"


def test_ausencia_de_secrets_nao_falha(monkeypatch):
    class SecretsFakeRaising:
        def __getattr__(self, item):
            raise RuntimeError("secrets nao configurado")

    monkeypatch.setattr(app, "st", SimpleNamespace(secrets=SecretsFakeRaising()))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    app.carregar_segredos_do_streamlit()

    assert os.environ.get("GROQ_API_KEY") is None