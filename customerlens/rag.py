"""RAG layer: chunking, TF-IDF retrieval, and pluggable answer backends.

No API keys are required to run: the default extractive backend answers from
retrieved chunks. Set the AZURE_OPENAI_* env vars to route synthesis through
Azure OpenAI instead.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_documents(documents: list[dict], words_per_chunk: int = 120,
                    overlap: int = 20) -> list[Chunk]:
    """Sliding-window chunking over word tokens with overlap."""
    chunks: list[Chunk] = []
    for doc in documents:
        words = doc["text"].split()
        step = max(words_per_chunk - overlap, 1)
        idx = 0
        for start in range(0, len(words), step):
            window = words[start:start + words_per_chunk]
            if not window:
                break
            chunks.append(Chunk(doc_id=doc["id"], chunk_index=idx,
                               text=" ".join(window),
                               metadata=dict(doc.get("metadata", {}))))
            idx += 1
            if start + words_per_chunk >= len(words):
                break
    return chunks


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


class Retriever:
    """TF-IDF + cosine similarity retriever over chunks."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english",
                                         min_df=1, max_df=0.9, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query: str, k: int = 3) -> list[SearchHit]:
        if not query.strip():
            return []
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        top = np.argsort(sims)[::-1][:k]
        return [SearchHit(chunk=self.chunks[i], score=float(sims[i]))
                for i in top if sims[i] > 0]

    def stats(self) -> dict:
        return {"chunks": len(self.chunks),
                "vocab_size": len(self.vectorizer.vocabulary_)}


class AnswerBackend(Protocol):
    name: str

    def answer(self, query: str, hits: list[SearchHit]) -> str: ...


class ExtractiveBackend:
    """Zero-dependency backend: answers are the best retrieved passages."""

    name = "extractive"

    def answer(self, query: str, hits: list[SearchHit]) -> str:
        if not hits:
            return "I couldn't find anything relevant in the ingested tickets."
        best = hits[0]
        cited = " ".join(f"[{h.chunk.doc_id}]" for h in hits[:3])
        return (f"Based on the ingested tickets {cited}:\n\n{best.chunk.text}\n\n"
                f"(extractive answer from ticket {best.chunk.doc_id}, "
                f"relevance {best.score:.2f})")


class AzureOpenAIBackend:
    """Optional generative backend via Azure OpenAI chat completions.

    Requires env vars: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT. Falls back to extractive on any failure.
    """

    name = "azure-openai"

    def __init__(self) -> None:
        self.endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
        self.api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        self.deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
        self.fallback = ExtractiveBackend()

    @property
    def configured(self) -> bool:
        return bool(self.endpoint and self.api_key and self.deployment)

    def answer(self, query: str, hits: list[SearchHit]) -> str:
        if not self.configured or not hits:
            return self.fallback.answer(query, hits)
        context = "\n\n".join(f"[{h.chunk.doc_id}] {h.chunk.text}" for h in hits[:4])
        payload = {
            "messages": [
                {"role": "system",
                 "content": ("You are a customer-support analyst. Answer the question "
                             "using ONLY the ticket excerpts below. Cite ticket IDs "
                             "like [T-2001]. If the excerpts lack the answer, say so.")},
                {"role": "user", "content": f"Question: {query}\n\nTickets:\n{context}"},
            ],
            "temperature": 0.1,
            "max_tokens": 400,
        }
        url = (f"{self.endpoint}/openai/deployments/{self.deployment}"
               f"/chat/completions?api-version=2024-06-01")
        try:
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json",
                         "api-key": self.api_key}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode())
            return body["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # network/auth failure -> graceful fallback
            return (self.fallback.answer(query, hits)
                    + f"\n\n(note: Azure OpenAI unavailable: {type(exc).__name__})")


@dataclass
class Answer:
    text: str
    citations: list[dict[str, Any]]
    backend: str


def answer_question(query: str, retriever: Retriever,
                    backend: AnswerBackend | None = None,
                    k: int = 3) -> Answer:
    backend = backend or ExtractiveBackend()
    hits = retriever.search(query, k=k)
    text = backend.answer(query, hits)
    citations = [{"ticket_id": h.chunk.doc_id, "score": round(h.score, 4),
                  "snippet": h.chunk.text[:160] + ("..." if len(h.chunk.text) > 160 else ""),
                  "metadata": h.chunk.metadata}
                 for h in hits]
    return Answer(text=text, citations=citations, backend=backend.name)
