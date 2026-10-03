"""Tests for the OpenAI-compatible LLM answer backend.

The fake client stands in for the `openai` package: no network, no keys.
"""

from customerlens.llm import OpenAIChatBackend, select_backend
from customerlens.rag import ExtractiveBackend, answer_question


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, text="model answer", exc=None):
        self.text = text
        self.exc = exc
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return _FakeResponse(self.text)


class _FakeClient:
    def __init__(self, text="model answer", exc=None):
        completions = _FakeCompletions(text=text, exc=exc)
        self.chat = type("Chat", (), {"completions": completions})()
        self.calls = completions.calls


def _backend(text="The outage is tracked in [T-1].", exc=None, **kwargs):
    return OpenAIChatBackend(client=_FakeClient(text=text, exc=exc),
                             api_key="test-key", **kwargs)


def test_openai_returns_model_text(engine):
    backend = _backend()
    hits = engine.retriever.search("Atlas CRM outage", k=2)
    out = backend.answer("What happened?", hits)
    assert out == "The outage is tracked in [T-1]."


def test_openai_prompt_includes_ticket_context(engine):
    backend = _backend()
    hits = engine.retriever.search("Atlas CRM outage", k=2)
    backend.answer("What happened?", hits)
    messages = backend._client.calls[0]["messages"]
    user_msg = next(m["content"] for m in messages if m["role"] == "user")
    assert "[T-3]" in user_msg  # top hit id is in the prompt
    assert "cite" in messages[0]["content"].lower() or "Citation" in messages[0]["content"]


def test_openai_uses_configured_model(engine):
    backend = _backend(model="my-model")
    hits = engine.retriever.search("outage", k=1)
    backend.answer("q", hits)
    assert backend._client.calls[0]["model"] == "my-model"


def test_openai_falls_back_on_client_error(engine):
    backend = _backend(exc=RuntimeError("boom"))
    hits = engine.retriever.search("outage", k=1)
    out = backend.answer("q", hits)
    assert "extractive answer" in out
    assert "RuntimeError" in out


def test_openai_falls_back_on_empty_response(engine):
    backend = _backend(text="   ")
    hits = engine.retriever.search("outage", k=1)
    out = backend.answer("q", hits)
    assert "extractive answer" in out
    assert "ValueError" in out


def test_openai_falls_back_without_api_key(engine):
    backend = OpenAIChatBackend(api_key="")
    assert not backend.configured
    hits = engine.retriever.search("outage", k=1)
    out = backend.answer("q", hits)
    assert "extractive answer" in out


def test_openai_falls_back_without_hits():
    backend = _backend()
    out = backend.answer("q", [])
    assert "couldn't find anything" in out


def test_answer_shape_matches_extractive(engine):
    backend = _backend()
    ans = answer_question("Atlas CRM outage ERR-9001", engine.retriever, backend, k=2)
    assert ans.backend == "openai"
    assert isinstance(ans.text, str) and ans.text
    for c in ans.citations:
        assert {"ticket_id", "score", "snippet", "metadata"} <= set(c)
        assert isinstance(c["score"], float)


def test_select_backend_names(monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    assert isinstance(select_backend("extractive"), ExtractiveBackend)
    assert isinstance(select_backend("openai"), ExtractiveBackend)  # no key -> fallback
    assert isinstance(select_backend("azure"), ExtractiveBackend)  # not configured
    assert isinstance(select_backend("nonsense"), ExtractiveBackend)

    monkeypatch.setenv("LLM_API_KEY", "k")
    b = select_backend("openai")
    assert isinstance(b, OpenAIChatBackend)
    monkeypatch.setenv("LLM_BACKEND", "openai")
    assert isinstance(select_backend(None), OpenAIChatBackend)
    monkeypatch.setenv("LLM_BACKEND", "extractive")
    assert isinstance(select_backend(None), ExtractiveBackend)


def test_dashboard_imports_cleanly():
    import dashboard
    assert callable(dashboard.main)
    assert callable(dashboard.load_engine)
