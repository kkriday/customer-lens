"""CSV ingestion, schema inference, profiling, and cleaning."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

REQUIRED_TICKET_COLS = {"ticket_id", "customer_id", "product", "category",
                        "created_at", "priority", "resolved", "text"}


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    null_count: int
    null_pct: float
    unique_count: int
    sample_values: list[Any] = field(default_factory=list)


@dataclass
class ProfileReport:
    rows: int
    columns: list[str]
    duplicate_rows: int
    columns_detail: list[ColumnProfile]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "columns": self.columns,
            "duplicate_rows": self.duplicate_rows,
            "notes": self.notes,
            "columns_detail": [
                {"name": c.name, "dtype": c.dtype, "null_count": c.null_count,
                 "null_pct": round(c.null_pct, 2), "unique_count": c.unique_count,
                 "sample_values": c.sample_values}
                for c in self.columns_detail
            ],
        }


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV, keeping everything as text so cleaning is explicit."""
    return pd.read_csv(path, dtype=str, keep_default_na=True)


def infer_schema(df: pd.DataFrame) -> dict[str, str]:
    """Best-effort semantic schema: datetime / boolean / numeric / text."""
    schema: dict[str, str] = {}
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            schema[col] = "text"
            continue
        lowered = s.astype(str).str.strip().str.lower()
        if lowered.isin({"true", "false", "yes", "no", "1", "0"}).all():
            schema[col] = "boolean"
        elif re.search(r"date|_at$|time", col, re.IGNORECASE) and \
                pd.to_datetime(s, errors="coerce", format="mixed").notna().mean() > 0.8:
            schema[col] = "datetime"
        elif pd.to_numeric(s, errors="coerce").notna().mean() > 0.9:
            schema[col] = "numeric"
        else:
            schema[col] = "text"
    return schema


def profile(df: pd.DataFrame) -> ProfileReport:
    n = len(df)
    detail = []
    for col in df.columns:
        s = df[col]
        detail.append(ColumnProfile(
            name=col,
            dtype=str(s.dtype),
            null_count=int(s.isna().sum()),
            null_pct=round(float(s.isna().mean() * 100), 2),
            unique_count=int(s.nunique(dropna=True)),
            sample_values=[str(v) for v in s.dropna().unique()[:3]],
        ))
    notes: list[str] = []
    dupes = int(df.duplicated().sum())
    if dupes:
        notes.append(f"{dupes} fully-duplicate rows detected")
    for c in detail:
        if c.null_pct > 0:
            notes.append(f"column '{c.name}' is {c.null_pct}% null")
    return ProfileReport(rows=n, columns=list(df.columns),
                         duplicate_rows=dupes, columns_detail=detail, notes=notes)


def _to_bool(v: Any) -> bool:
    return str(v).strip().lower() in {"true", "yes", "1"}


def clean_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize messy ticket data into a clean, typed frame."""
    df = df.copy()
    missing = REQUIRED_TICKET_COLS - set(df.columns)
    if missing:
        raise ValueError(f"tickets CSV is missing columns: {sorted(missing)}")

    text_cols = ["ticket_id", "customer_id", "customer_name", "product",
                 "category", "priority", "text"]
    for col in text_cols:
        if col in df.columns:
            df[col] = (df[col].astype("string").fillna("")
                       .str.strip().str.replace(r"\s+", " ", regex=True))

    # Fix common typo artifacts and stray shouting.
    df["text"] = df["text"].str.replace(r"\bteh\b", "the", regex=True)

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", format="mixed")
    df["resolved"] = df["resolved"].map(_to_bool)
    df["priority"] = df["priority"].str.title()

    before = len(df)
    df = df.drop_duplicates(subset=["ticket_id"], keep="first")
    df = df.dropna(subset=["ticket_id", "text"])
    df = df.reset_index(drop=True)
    df.attrs["dropped_rows"] = before - len(df)
    return df


def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    df["joined_at"] = pd.to_datetime(df["joined_at"], errors="coerce", format="mixed")
    return df.drop_duplicates(subset=["customer_id"]).reset_index(drop=True)


def build_documents(tickets: pd.DataFrame, customers: pd.DataFrame) -> list[dict]:
    """One searchable document per ticket, enriched with customer context."""
    cust = customers.set_index("customer_id").to_dict(orient="index")
    docs = []
    for row in tickets.itertuples():
        c = cust.get(row.customer_id, {})
        docs.append({
            "id": row.ticket_id,
            "text": (f"Ticket {row.ticket_id}: {row.customer_name or 'unknown customer'} "
                     f"({c.get('company', 'unknown company')}, {c.get('tier', 'unknown')} tier) "
                     f"reported a {row.priority} priority {row.category} issue with "
                     f"{row.product} on {row.created_at.date() if pd.notna(row.created_at) else 'unknown date'}. "
                     f"Resolved: {row.resolved}. Details: {row.text}"),
            "metadata": {
                "ticket_id": row.ticket_id,
                "customer_id": row.customer_id,
                "product": row.product,
                "category": row.category,
                "priority": row.priority,
                "resolved": bool(row.resolved),
            },
        })
    return docs
