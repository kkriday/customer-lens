"""CLI for CustomerLens: ingest, ask, eval, serve."""

from __future__ import annotations

from pathlib import Path

import typer

from .api import DEFAULT_ARTIFACTS, Engine
from .eval import print_report, run_eval
from .llm import select_backend
from .rag import answer_question

app = typer.Typer(help="CustomerLens: customer data copilot.")


@app.command()
def ingest(tickets: str = typer.Option("data/tickets.csv", help="Path to tickets CSV"),
           customers: str = typer.Option("data/customers.csv", help="Path to customers CSV"),
           out: str = typer.Option(str(DEFAULT_ARTIFACTS), help="Artifact directory")):
    """Ingest CSVs, profile them, and build the retrieval index."""
    engine = Engine()
    summary = engine.load(tickets, customers)
    engine.save(out)
    typer.echo(f"ingested {summary['tickets']} tickets, {summary['customers']} customers "
               f"-> {summary['chunks']} chunks (dropped {summary['dropped_ticket_rows']} bad rows)")
    for note in summary["profile_notes"]:
        typer.echo(f"  note: {note}")


@app.command()
def ask(question: str = typer.Argument(..., help="Natural-language question"),
        artifacts: str = typer.Option(str(DEFAULT_ARTIFACTS), help="Artifact directory"),
        k: int = typer.Option(3, help="Top-k chunks to retrieve"),
        backend: str | None = typer.Option(
            None, help="Answer backend: extractive, openai, azure (default: LLM_BACKEND env or auto)")):
    """Ask a question over ingested tickets."""
    engine = Engine()
    if not engine.load_artifacts(artifacts):
        typer.echo(f"no artifacts in {artifacts}; run `customerlens ingest` first", err=True)
        raise typer.Exit(1)
    ans = answer_question(question, engine.retriever, select_backend(backend), k=k)
    typer.echo(f"[{ans.backend}]\n{ans.text}")
    typer.echo("\ncitations:")
    for c in ans.citations:
        typer.echo(f"  - {c['ticket_id']} (score {c['score']}): {c['snippet']}")


@app.command()
def evaluate(pairs: str = typer.Option("data/eval_pairs.json", help="Eval pairs JSON"),
             artifacts: str = typer.Option(str(DEFAULT_ARTIFACTS), help="Artifact directory"),
             k: int = typer.Option(3, help="Top-k for hit-rate"),
             backend: str | None = typer.Option(
                 None, help="Answer backend: extractive, openai, azure (default: LLM_BACKEND env or auto)")):
    """Run the answer-quality eval harness."""
    engine = Engine()
    if not engine.load_artifacts(artifacts):
        typer.echo(f"no artifacts in {artifacts}; run `customerlens ingest` first", err=True)
        raise typer.Exit(1)
    print_report(run_eval(engine.retriever, pairs, select_backend(backend), k=k))


@app.command()
def serve(host: str = typer.Option("0.0.0.0"), port: int = typer.Option(8000),
          artifacts: str = typer.Option(str(DEFAULT_ARTIFACTS), help="Artifact directory")):
    """Start the FastAPI server, loading artifacts if present."""
    import uvicorn
    from .api import app as fastapi_app, engine
    if Path(artifacts, "retriever.pkl").exists():
        engine.load_artifacts(artifacts)
        typer.echo(f"loaded artifacts from {artifacts}")
    uvicorn.run(fastapi_app, host=host, port=port)


if __name__ == "__main__":
    app()
