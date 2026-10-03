"""Tests for ingestion, profiling, schema inference, and cleaning."""

import pytest

from customerlens import ingest as ing


def test_profile_reports_nulls_and_duplicates(tickets_df):
    report = ing.profile(tickets_df)
    assert report.rows == 5
    assert report.duplicate_rows == 1
    by_name = {c.name: c for c in report.columns_detail}
    assert by_name["customer_name"].null_count == 1
    assert any("duplicate" in n for n in report.notes)


def test_infer_schema_detects_types(tickets_df):
    schema = ing.infer_schema(tickets_df)
    assert schema["created_at"] == "datetime"
    assert schema["resolved"] == "boolean"
    assert schema["text"] == "text"


def test_clean_tickets_dedupes_and_types(tickets_df):
    clean = ing.clean_tickets(tickets_df)
    assert len(clean) == 4  # one duplicate ticket_id dropped
    assert clean["ticket_id"].is_unique
    assert clean["resolved"].dtype == bool
    assert clean["created_at"].dtype.kind == "M"
    # whitespace stripped
    assert clean.loc[clean.ticket_id == "T-4", "text"].iloc[0].startswith("How do")


def test_clean_tickets_rejects_missing_columns(tickets_df):
    with pytest.raises(ValueError, match="missing columns"):
        ing.clean_tickets(tickets_df.drop(columns=["text"]))


def test_build_documents_enriches_with_customer(engine):
    docs = engine.documents
    assert len(docs) == 4
    t1 = next(d for d in docs if d["id"] == "T-1")
    assert "Northwind Traders" in t1["text"]
    assert "Enterprise" in t1["text"]
    assert t1["metadata"]["ticket_id"] == "T-1"
