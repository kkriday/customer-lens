"""Shared fixtures for CustomerLens tests."""

import json

import pandas as pd
import pytest

from customerlens import ingest as ing
from customerlens.api import Engine
from customerlens.rag import chunk_documents, Retriever


@pytest.fixture()
def tickets_df():
    return pd.DataFrame([
        {"ticket_id": "T-1", "customer_id": "C-1", "customer_name": "Ava Sharma",
         "product": "Atlas CRM", "category": "Bug", "created_at": "2026-02-01 10:00",
         "priority": "High", "resolved": "True",
         "text": "The Atlas CRM dashboard throws error ERR-5124 whenever we filter by date range."},
        {"ticket_id": "T-2", "customer_id": "C-2", "customer_name": "Liam Garcia",
         "product": "Beacon Analytics", "category": "Refund", "created_at": "2026-02-02 11:00",
         "priority": "Medium", "resolved": "False",
         "text": "Charged $214.50 for Beacon Analytics after downgrading. Refund invoice INV-88213."},
        {"ticket_id": "T-3", "customer_id": "C-1", "customer_name": "Ava Sharma",
         "product": "Atlas CRM", "category": "Outage", "created_at": "2026-02-03 09:30",
         "priority": "Critical", "resolved": "True",
         "text": "Complete outage of Atlas CRM this morning, error ERR-9001 in all workspaces."},
        # messy rows: duplicate + whitespace + missing name
        {"ticket_id": "T-3", "customer_id": "C-1", "customer_name": "Ava Sharma",
         "product": "Atlas CRM", "category": "Outage", "created_at": "2026-02-03 09:30",
         "priority": "Critical", "resolved": "True",
         "text": "Complete outage of Atlas CRM this morning, error ERR-9001 in all workspaces."},
        {"ticket_id": "T-4", "customer_id": "C-3", "customer_name": None,
         "product": "Cobalt Storage", "category": "How-to", "created_at": "2026-02-04 14:00",
         "priority": "Low", "resolved": "True",
         "text": "  How do we configure data retention in Cobalt Storage?  "},
    ])


@pytest.fixture()
def customers_df():
    return pd.DataFrame([
        {"customer_id": "C-1", "name": "Ava Sharma", "email": "ava@example.com",
         "company": "Northwind Traders", "tier": "Enterprise", "country": "USA",
         "joined_at": "2024-05-01"},
        {"customer_id": "C-2", "name": "Liam Garcia", "email": "liam@example.com",
         "company": "Contoso Labs", "tier": "Premium", "country": "Canada",
         "joined_at": "2025-01-15"},
        {"customer_id": "C-3", "name": "Maya Kim", "email": "maya@example.com",
         "company": "Fabrikam Inc", "tier": "Standard", "country": "UK",
         "joined_at": "2025-06-20"},
    ])


@pytest.fixture()
def engine(tickets_df, customers_df):
    eng = Engine()
    eng.tickets = ing.clean_tickets(tickets_df)
    eng.customers = ing.clean_customers(customers_df)
    eng.documents = ing.build_documents(eng.tickets, eng.customers)
    eng.retriever = Retriever(chunk_documents(eng.documents))
    eng.profile = {"tickets": ing.profile(tickets_df).to_dict(),
                   "customers": ing.profile(customers_df).to_dict()}
    return eng


@pytest.fixture()
def eval_pairs_file(tmp_path):
    pairs = [
        {"question": "What did Ava Sharma report about Atlas CRM Bug? Details: ERR-5124",
         "expected_ticket_id": "T-1", "keywords": ["Atlas", "ERR-5124"]},
        {"question": "What did Liam Garcia report about Beacon Analytics Refund? Details: INV-88213 $214.50",
         "expected_ticket_id": "T-2", "keywords": ["Beacon", "INV-88213"]},
    ]
    p = tmp_path / "pairs.json"
    p.write_text(json.dumps(pairs), encoding="utf-8")
    return str(p)
