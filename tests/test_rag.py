"""Tests for chunking, retrieval, and answer backends."""

from customerlens.rag import (ExtractiveBackend, answer_question,
                              chunk_documents)


def test_chunking_windows_with_overlap():
    docs = [{"id": "D-1", "text": " ".join(f"word{i}" for i in range(300)),
             "metadata": {}}]
    chunks = chunk_documents(docs, words_per_chunk=120, overlap=20)
    assert len(chunks) == 3
    assert chunks[0].doc_id == "D-1"
    assert chunks[0].chunk_index == 0
    # overlap: last words of chunk 0 appear at the start of chunk 1
    assert "word119" in chunks[0].text.split()[-1]
    assert chunks[1].text.split()[0] == "word100"


def test_chunking_empty_text():
    assert chunk_documents([{"id": "D-1", "text": "", "metadata": {}}]) == []


def test_retrieval_returns_right_ticket(engine):
    hits = engine.retriever.search("Atlas CRM dashboard error ERR-5124 date filter", k=3)
    assert hits, "expected at least one hit"
    assert hits[0].chunk.doc_id == "T-1"
    # scores descending
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_retrieval_no_match_returns_empty(engine):
    hits = engine.retriever.search("zxqv wobble flibbertigibbet", k=3)
    assert hits == []


def test_retrieval_empty_query(engine):
    assert engine.retriever.search("   ") == []


def test_extractive_answer_cites_ticket(engine):
    ans = answer_question("What happened with Atlas CRM outage ERR-9001?",
                          engine.retriever, ExtractiveBackend(), k=2)
    assert ans.backend == "extractive"
    assert ans.citations[0]["ticket_id"] == "T-3"
    assert "ERR-9001" in ans.text
