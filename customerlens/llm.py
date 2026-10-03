"""LLM answer backend: OpenAI-compatible chat completions.

Uses the `openai` package, which also works against Azure OpenAI or Ollama by
pointing LLM_BASE_URL at the right endpoint. The extractive backend stays the
default; this one is opt-in via LLM_BACKEND=openai (or an explicit backend
name) and falls back to extractive on any failure: missing key, missing
package, network error, or an empty model response.
"""

from __future__ import annotations

import os
from typing import Any

from .rag import AnswerBackend, AzureOpenAIBackend, ExtractiveBackend, SearchHit

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIChatBackend:
    """Generative backend via any OpenAI-compatible chat completions API.

    Env vars: LLM_API_KEY (required), LLM_BASE_URL (default api.openai.com),
    LLM_MODEL (default gpt-4o-mini). A prebuilt client can be injected for
    tests via the `client` argument.
    """

    name = "openai"

    def __init__(self, client: Any = None, api_key: str | None = None,
                 base_url: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("LLM_API_KEY", "")
        self.base_url = ((base_url or os.environ.get("LLM_BASE_URL", "")
                          or DEFAULT_BASE_URL).rstrip("/"))
        self.model = model or os.environ.get("LLM_MODEL", "") or DEFAULT_MODEL
        self._client = client
        self.fallback = ExtractiveBackend()

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        from openai import OpenAI  # local import: openai stays an optional dep
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _prompt(self, query: str, hits: list[SearchHit]) -> tuple[str, str]:
        context = "\n\n".join(f"[{h.chunk.doc_id}] {h.chunk.text}" for h in hits[:4])
        system = ("You are a customer-support analyst. Answer the question using "
                  "ONLY the ticket excerpts below. Cite every claim with the ticket "
                  "ID in brackets, like [T-2001]. If the excerpts do not contain "
                  "the answer, say so plainly.")
        user = f"Question: {query}\n\nTicket excerpts:\n{context}"
        return system, user

    def answer(self, query: str, hits: list[SearchHit]) -> str:
        if not self.configured or not hits:
            return self.fallback.answer(query, hits)
        try:
            client = self._get_client()
            system, user = self._prompt(query, hits)
            resp = client.chat.completions.create(
                model=self.model, temperature=0.1, max_tokens=400,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("empty model response")
            return text
        except Exception as exc:  # key/network/bad response -> graceful fallback
            return (self.fallback.answer(query, hits)
                    + f"\n\n(note: LLM backend unavailable: {type(exc).__name__})")


def select_backend(name: str | None = None) -> AnswerBackend:
    """Pick an answer backend by name, honoring the LLM_BACKEND env var.

    Names: "extractive", "openai", "azure". An unrecognized backend falls back
    to extractive, and a recognized-but-unconfigured one does too. When no
    name is given and LLM_BACKEND is unset, the legacy default applies:
    Azure OpenAI when configured, else extractive.
    """
    name = (name or os.environ.get("LLM_BACKEND", "")).strip().lower()
    if name == "openai":
        backend = OpenAIChatBackend()
        return backend if backend.configured else ExtractiveBackend()
    if name == "extractive":
        return ExtractiveBackend()
    if name == "azure":
        backend = AzureOpenAIBackend()
        return backend if backend.configured else ExtractiveBackend()
    azure = AzureOpenAIBackend()
    return azure if azure.configured else ExtractiveBackend()
