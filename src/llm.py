from __future__ import annotations

import logging

from groq import Groq

from src.config import Config, require_groq_api_key

logger = logging.getLogger(__name__)


class GroqClient:
    """LLM client backed by the Groq API (Llama models)."""

    def __init__(self, config: Config):
        self._client = Groq(api_key=require_groq_api_key(config))
        self.model = config.llm_model

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""