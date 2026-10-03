"""FastAPI service: ingest customer CSVs, profile them, ask questions."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import ingest as ing
from .eval import print_report, run_eval
from .llm import select_backend
from .rag import (Retriever, answer_question, chunk_documents)

VERSION = "0.1.0"
DEFAULT_ARTIFACTS = Path("data/artifacts")


class Engine:
    """Holds the ingested data, profile, and retriever in memory."""

    def __init__(self) -> None:
        self.tickets = None
        self.customers = None
        self.documents: list[dict] = []
        self.retriever: Retriever | None = None
        self.profile: dict | None = None
        self.schema: dict | None = None

    def is_ready(self) -> bool:
        return self.retriever is not None

    def load(self, tickets_path: str, customers_path: str) -> dict:
        raw_tickets = ing.load_csv(tickets_path)
        raw_customers = ing.load_csv(customers_path)
        self.profile = {
            "tickets": ing.profile(raw_tickets).to_dict(),
            "customers": ing.profile(raw_customers).to_dict(),
        }
        self.schema = ing.infer_schema(raw_tickets)
        self.tickets = ing.clean_tickets(raw_tickets)
        self.customers = ing.clean_customers(raw_customers)
        self.documents = ing.build_documents(self.tickets, self.customers)
        chunks = chunk_documents(self.documents)
        self.retriever = Retriever(chunks)
        return {
            "tickets": len(self.tickets),
            "customers": len(self.customers),
            "documents": len(self.documents),
            "chunks": len(chunks),
            "dropped_ticket_rows": int(self.tickets.attrs.get("dropped_rows", 0)),
            "profile_notes": self.profile["tickets"]["notes"],
        }

    def save(self, artifact_dir: str | Path) -> Path:
        d = Path(artifact_dir)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "retriever.pkl", "wb") as f:
            pickle.dump(self.retriever, f)
        with open(d / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)
        self.tickets.to_csv(d / "tickets_clean.csv", index=False)
        with open(d / "profile.json", "w", encoding="utf-8") as f:
            json.dump(self.profile, f, indent=2, default=str)
        return d

    def load_artifacts(self, artifact_dir: str | Path) -> bool:
        d = Path(artifact_dir)
        if not (d / "retriever.pkl").exists():
            return False
        with open(d / "retriever.pkl", "rb") as f:
            self.retriever = pickle.load(f)
        with open(d / "documents.pkl", "rb") as f:
            self.documents = pickle.load(f)
        with open(d / "profile.json", encoding="utf-8") as f:
            self.profile = json.load(f)
        return True


engine = Engine()
app = FastAPI(title="CustomerLens", version=VERSION,
              description="Customer data copilot: ingest, profile, and ask.")


class IngestRequest(BaseModel):
    tickets_path: str = Field(..., description="Path to tickets CSV")
    customers_path: str = Field(..., description="Path to customers CSV")
    artifact_dir: str = Field(default=str(DEFAULT_ARTIFACTS))


class AskRequest(BaseModel):
    question: str
    k: int = Field(default=3, ge=1, le=10)
    backend: str | None = Field(
        default=None,
        description="Answer backend: extractive, openai, or azure (default: LLM_BACKEND env or auto)")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": VERSION, "ready": engine.is_ready()}


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    try:
        summary = engine.load(req.tickets_path, req.customers_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=f"file not found: {exc.filename}")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    engine.save(req.artifact_dir)
    return {"status": "ingested", **summary}


@app.get("/profile")
def profile() -> dict:
    if not engine.profile:
        raise HTTPException(status_code=409, detail="no data ingested yet; POST /ingest first")
    return engine.profile


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    if not engine.is_ready():
        raise HTTPException(status_code=409, detail="no data ingested yet; POST /ingest first")
    ans = answer_question(req.question, engine.retriever,
                        select_backend(req.backend), k=req.k)
    return {"answer": ans.text, "citations": ans.citations, "backend": ans.backend}


@app.post("/eval")
def eval_endpoint(pairs_path: str = "data/eval_pairs.json", k: int = 3,
                  backend: str | None = None) -> dict:
    if not engine.is_ready():
        raise HTTPException(status_code=409, detail="no data ingested yet; POST /ingest first")
    report = run_eval(engine.retriever, pairs_path, select_backend(backend), k=k)
    print_report(report)
    return report


def create_app() -> FastAPI:
    return app
