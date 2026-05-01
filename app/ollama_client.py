"""Small Ollama HTTP client used by the RAG pipeline."""

from __future__ import annotations

import requests

from app.config import OLLAMA_MODEL_NAME, OLLAMA_URL


class OllamaClient:
    """Client for local Ollama text generation."""

    def __init__(self, model_name: str = OLLAMA_MODEL_NAME, url: str = OLLAMA_URL, timeout: int = 120):
        self.model_name = model_name
        self.url = url
        self.timeout = timeout

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate text using Ollama's /api/generate endpoint."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        response = requests.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
