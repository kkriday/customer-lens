# CustomerLens

A customer data copilot. Feed it messy customer CSVs, it profiles and cleans them,
then answers questions about the data through a REST API. Ships with tests, linting,
CI, and a Dockerfile so it runs anywhere.

I built it as a portfolio project for a Forward Deployed Engineer role. The job is
basically: get embedded with a customer, understand their data, and ship something
useful fast. This is a small, working version of that loop.

## What it does

1. **Ingest:** load raw customer CSVs, infer schema, produce a data-quality profile
   (nulls, duplicates, per-column stats), and clean them (dedup, normalization, typing).
2. **Index:** turn cleaned tickets into enriched documents and build a TF-IDF
   retrieval index. No API keys needed; works fully offline.
3. **Ask:** natural-language Q&A over the tickets via a pluggable answer backend
   (extractive by default; optional Azure OpenAI synthesis via env vars), with
   citations and relevance scores on every answer.
4. **Evaluate:** a fixed Q&A eval harness reporting hit-rate, MRR, keyword
   coverage, and citation precision.

## Architecture

```
 ┌──────────┐   ┌─────────────────────────────────────────────┐
 │ tickets  │   │  ingest.py                                   │
 │ customers│──▶│  load → schema inference → profiling → clean │
 │  (CSV)   │   └──────────────────────┬──────────────────────┘
 └──────────┘                          ▼ documents
                  ┌─────────────────────────────────────────────┐
                  │  rag.py                                     │
                  │  chunk → TF-IDF index → retrieve → answer   │
                  │  backends: extractive │ openai │ azure      │
                  └──────────────────────┬──────────────────────┘
                                         ▼
                  ┌─────────────────────────────────────────────┐
                  │  api.py (FastAPI)        cli.py (typer)       │
                  │  /ingest /ask /profile   ingest ask          │
                  │  /health /eval           eval serve          │
                  └─────────────────────────────────────────────┘
                                         │
                              eval.py (quality harness)
```

## Quickstart

```bash
# 1. Create the sample data (seeded, reproducible)
python data/make_data.py

# 2. Install deps
pip install -r requirements.txt

# 3. Ingest + build the index
python -m customerlens.cli ingest

# 4. Ask a question
python -m customerlens.cli ask "Which tickets mention error ERR-9280?"

# 5. Run the eval harness
python -m customerlens.cli evaluate

# 6. Serve the API
python -m customerlens.cli serve --port 8000
```

Or with Docker:

```bash
docker compose up --build
```

The image bakes the retrieval artifacts at build time, so the API starts ready.

## Dashboard

A Streamlit UI over the whole thing: ask questions with citations, upload CSVs
and inspect the data-quality profile, and run the eval harness.

```bash
streamlit run dashboard.py
```

It opens at http://localhost:8501. The sidebar selects the answer backend
(extractive or openai), the retrieval depth, and shows whether the index is
loaded. With docker compose, the dashboard runs next to the API:

```bash
docker compose up --build
```

That starts the API on :8000 and the dashboard on :8501.

## API reference

| Method | Path      | Body                                              | Description                          |
|--------|-----------|---------------------------------------------------|--------------------------------------|
| GET    | /health   | (none)                                            | liveness + readiness                 |
| POST   | /ingest   | `{"tickets_path": ..., "customers_path": ...}`    | ingest CSVs, profile, build index    |
| GET    | /profile  | (none)                                            | data-quality profile report          |
| POST   | /ask      | `{"question": "...", "k": 3}`                     | answer with citations + scores       |
| POST   | /eval     | (none)                                            | run the eval harness                 |

curl examples:

```bash
curl -X POST localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"tickets_path": "data/tickets.csv", "customers_path": "data/customers.csv"}'

curl -X POST localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What refund issues were reported for Beacon Analytics?", "k": 3}'

curl localhost:8000/profile | python -m json.tool
```

## Evaluating answer quality

```bash
python -m customerlens.cli evaluate
```

Runs 12 fixed question/answer pairs (generated deterministically in `data/eval_pairs.json`)
and prints `hit_rate@3`, `mrr`, `keyword_coverage`, and `citation_precision`. The same
report is available at `POST /eval`. Every answer carries citations with ticket IDs and
relevance scores, so failures are inspectable, not vibes.

## Optional: LLM answer backend

The default backend answers extractively (it quotes the best ticket). For
generative answers, point it at any OpenAI-compatible chat API:

```bash
export LLM_BACKEND=openai
export LLM_API_KEY=<your-key>
# optional: LLM_BASE_URL (default https://api.openai.com/v1)
# optional: LLM_MODEL (default gpt-4o-mini)

python -m customerlens.cli ask "Which tickets mention ERR-9280?" --backend openai
```

The backend builds a prompt from the retrieved chunks, asks the model to answer
with citations like [T-2001], and falls back to extractive on any failure
(missing key, network error, empty response). Pointing LLM_BASE_URL at Azure
OpenAI or a local Ollama server works too. Per-call override: `POST /ask`
accepts a `backend` field, and `evaluate --backend openai` runs the harness
against the LLM.

## Optional: Azure OpenAI backend

Copy `.env.example` to `.env` and fill in `AZURE_OPENAI_ENDPOINT`,
`AZURE_OPENAI_API_KEY`, and `AZURE_OPENAI_DEPLOYMENT`. The API and CLI pick it up
automatically and fall back to extractive answers if the call fails.

## Project layout

```
customerlens/        # the service package
  ingest.py          # CSV loading, schema inference, profiling, cleaning
  rag.py             # chunking, TF-IDF retrieval, answer backends
  llm.py             # OpenAI-compatible LLM backend + backend selection
  api.py             # FastAPI: /ingest /ask /profile /health /eval
  eval.py            # answer-quality eval harness
  cli.py             # typer CLI: ingest, ask, evaluate, serve
dashboard.py         # Streamlit UI: Ask / Data / Eval tabs
data/
  make_data.py       # seeded generator for tickets.csv + customers.csv
  eval_pairs.json    # fixed eval questions (generated)
tests/               # pytest suite (ingest, rag, eval, api)
Dockerfile           # builds API with baked-in artifacts
docker-compose.yml   # one-command local deploy
.github/workflows/ci.yml  # ruff + pytest + docker build
```

## Roadmap

- Dense embeddings + hybrid retrieval (BM25 + vectors)
- Incremental re-ingest and index versioning
- Auth + per-tenant data isolation
- Streaming answers (SSE) and a lightweight web UI
- Eval-driven regression tracking in CI (fail the build on metric drops)
