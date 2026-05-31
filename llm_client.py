"""Client LLM : interroge Ollama via l'API compatible OpenAI."""

from __future__ import annotations

from openai import OpenAI

from config import OLLAMA_BASE_URL


def build_client() -> OpenAI:
    """Construit un client OpenAI pointant vers Ollama local."""
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def query_model(client: OpenAI, model: str, question: str):
    """Interroge le LLM en demandant les logprobs (top 3 alternatives par token)."""
    return client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un assistant factuel. Réponds de manière concise et "
                    "directe à la question posée."
                ),
            },
            {"role": "user", "content": question},
        ],
        logprobs=True,
        top_logprobs=3,
    )
