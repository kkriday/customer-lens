"""Answer-quality eval harness.

Runs a fixed set of question/answer pairs against the retriever and reports:
  - hit_rate@k: fraction of questions where the expected ticket is in top-k
  - MRR: mean reciprocal rank of the expected ticket
  - keyword_coverage: fraction of expected keywords present in the answer text
  - citation_precision: fraction of citations that reference ingested tickets
"""

from __future__ import annotations

import json
from pathlib import Path

from .rag import AnswerBackend, ExtractiveBackend, Retriever, answer_question


def eval_details(retriever: Retriever, pairs_path: str | Path,
                 backend: AnswerBackend | None = None, k: int = 3) -> list[dict]:
    """Per-question eval results, for dashboards and debugging."""
    pairs = json.loads(Path(pairs_path).read_text(encoding="utf-8"))
    backend = backend or ExtractiveBackend()
    known_ids = {c.doc_id for c in retriever.chunks}

    details: list[dict] = []
    for p in pairs:
        ans = answer_question(p["question"], retriever, backend, k=k)
        ranks = [i + 1 for i, c in enumerate(ans.citations)
                 if c["ticket_id"] == p["expected_ticket_id"]]
        hit = bool(ranks)
        kw = p.get("keywords", [])
        coverage = (sum(1 for w in kw if w.lower() in ans.text.lower()) / len(kw)
                    if kw else 1.0)
        details.append({
            "question": p["question"],
            "expected_ticket_id": p["expected_ticket_id"],
            "hit": hit,
            "rank": ranks[0] if hit else None,
            "keyword_coverage": round(coverage, 3),
            "citations_ok": all(c["ticket_id"] in known_ids for c in ans.citations),
            "backend": ans.backend,
        })
    return details


def run_eval(retriever: Retriever, pairs_path: str | Path,
             backend: AnswerBackend | None = None, k: int = 3) -> dict:
    backend = backend or ExtractiveBackend()
    details = eval_details(retriever, pairs_path, backend, k=k)
    n = len(details)
    hit_rate = sum(d["hit"] for d in details) / n if n else 0.0
    mrr = sum(1 / d["rank"] for d in details if d["rank"]) / n if n else 0.0
    avg_cov = sum(d["keyword_coverage"] for d in details) / n if n else 0.0
    cit_ok = sum(d["citations_ok"] for d in details) / n if n else 0.0
    return {
        "n": n, "k": k, "backend": backend.name,
        "hit_rate": round(hit_rate, 3),
        "mrr": round(mrr, 3),
        "keyword_coverage": round(avg_cov, 3),
        "citation_precision": round(cit_ok, 3),
        "failures": [{"question": d["question"], "expected": d["expected_ticket_id"]}
                     for d in details if not d["hit"]],
    }


def print_report(report: dict) -> None:
    print(f"eval: n={report['n']} k={report['k']} backend={report['backend']}")
    print(f"  hit_rate@k        {report['hit_rate']}")
    print(f"  mrr               {report['mrr']}")
    print(f"  keyword_coverage  {report['keyword_coverage']}")
    print(f"  citation_prec     {report['citation_precision']}")
    if report["failures"]:
        print("  failures:")
        for f in report["failures"]:
            print(f"    - expected {f['expected']}: {f['question'][:80]}...")
