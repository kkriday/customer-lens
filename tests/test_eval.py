"""Tests for the eval harness."""

from customerlens.eval import run_eval
from customerlens.rag import ExtractiveBackend


def test_eval_reports_metrics(engine, eval_pairs_file):
    report = run_eval(engine.retriever, eval_pairs_file, ExtractiveBackend(), k=3)
    assert report["n"] == 2
    assert report["hit_rate"] == 1.0
    assert report["mrr"] == 1.0
    assert report["citation_precision"] == 1.0
    assert 0.0 <= report["keyword_coverage"] <= 1.0
    assert report["failures"] == []


def test_eval_handles_misses(engine, eval_pairs_file, tmp_path):
    import json
    pairs = [{"question": "quantum entanglement in ticket systems",
              "expected_ticket_id": "T-1", "keywords": ["quantum"]}]
    p = tmp_path / "miss.json"
    p.write_text(json.dumps(pairs))
    report = run_eval(engine.retriever, str(p), ExtractiveBackend(), k=3)
    assert report["hit_rate"] == 0.0
    assert len(report["failures"]) == 1
