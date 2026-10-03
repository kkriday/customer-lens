"""Streamlit dashboard for CustomerLens.

Run from the repo root:

    streamlit run dashboard.py

LLM backend config comes from env vars (see .env.example): export
LLM_BACKEND=openai and LLM_API_KEY before launching to use it in the
dashboard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from customerlens.api import DEFAULT_ARTIFACTS, Engine  # noqa: E402
from customerlens.eval import eval_details  # noqa: E402
from customerlens.llm import OpenAIChatBackend, select_backend  # noqa: E402
from customerlens.rag import answer_question  # noqa: E402

DATA_DIR = ROOT / "data"


@st.cache_resource
def load_engine(tickets_path: str, customers_path: str) -> Engine:
    engine = Engine()
    engine.load(tickets_path, customers_path)
    return engine


def current_paths() -> tuple[str, str] | None:
    custom = st.session_state.get("data_paths")
    if custom and Path(custom[0]).exists() and Path(custom[1]).exists():
        return custom
    default = (str(DATA_DIR / "tickets.csv"), str(DATA_DIR / "customers.csv"))
    if Path(default[0]).exists() and Path(default[1]).exists():
        return default
    return None


def render_ask(engine: Engine, backend_name: str, k: int) -> None:
    st.subheader("Ask")
    question = st.text_input("Question",
                             placeholder="Which tickets mention error ERR-9280?")
    if st.button("Ask", type="primary"):
        if not engine.is_ready():
            st.warning("No data ingested yet. Go to the Data tab and ingest first.")
            return
        if not question.strip():
            st.warning("Type a question first.")
            return
        with st.spinner("Searching tickets..."):
            ans = answer_question(question, engine.retriever,
                                  select_backend(backend_name), k=k)
        st.markdown("**Answer**")
        st.write(ans.text)
        st.caption(f"backend: {ans.backend}")
        st.markdown("**Citations**")
        for c in ans.citations:
            with st.expander(f"{c['ticket_id']} (relevance {c['score']})"):
                st.write(c["snippet"])
                st.json(c["metadata"])


def render_data(engine: Engine) -> None:
    st.subheader("Data")
    tickets_file = st.file_uploader("tickets.csv", type="csv", key="tickets_csv")
    customers_file = st.file_uploader("customers.csv", type="csv", key="customers_csv")
    if st.button("Ingest"):
        if not (tickets_file and customers_file):
            st.error("Upload both CSVs first.")
        else:
            updir = DATA_DIR / "uploads"
            updir.mkdir(exist_ok=True)
            tp, cp = updir / "tickets.csv", updir / "customers.csv"
            tp.write_bytes(tickets_file.getvalue())
            cp.write_bytes(customers_file.getvalue())
            try:
                fresh = Engine()
                summary = fresh.load(str(tp), str(cp))
                fresh.save(DEFAULT_ARTIFACTS)
            except (ValueError, FileNotFoundError) as exc:
                st.error(f"Ingest failed: {exc}")
            else:
                st.session_state["data_paths"] = (str(tp), str(cp))
                load_engine.clear()
                st.success(f"Ingested {summary['tickets']} tickets and "
                           f"{summary['customers']} customers.")
                st.rerun()

    if not (engine.is_ready() and engine.profile):
        st.info("No profile to show yet. Ingest data above, or make sure "
                "data/tickets.csv and data/customers.csv exist.")
        return

    tprof, cprof = engine.profile["tickets"], engine.profile["customers"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Tickets", tprof["rows"])
    m2.metric("Customers", cprof["rows"])
    m3.metric("Duplicate rows", tprof["duplicate_rows"])

    st.markdown("**Null percentage by column (tickets)**")
    cols_df = pd.DataFrame(tprof["columns_detail"])
    st.bar_chart(cols_df.set_index("name")["null_pct"])

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Tickets by category**")
        st.bar_chart(engine.tickets["category"].value_counts())
    with d2:
        st.markdown("**Tickets by priority**")
        st.bar_chart(engine.tickets["priority"].value_counts())

    st.markdown("**Sample rows**")
    st.dataframe(engine.tickets.head(10))
    if tprof["notes"]:
        st.markdown("**Notes**")
        for note in tprof["notes"]:
            st.write("- " + note)


def render_eval(engine: Engine, backend_name: str, k: int) -> None:
    st.subheader("Eval")
    pairs_path = str(DATA_DIR / "eval_pairs.json")
    if not engine.is_ready():
        st.warning("No data ingested yet. Go to the Data tab and ingest first.")
        return
    if st.button("Run eval", type="primary"):
        with st.spinner("Running eval..."):
            details = eval_details(engine.retriever, pairs_path,
                                   select_backend(backend_name), k=k)
        n = len(details)
        hit_rate = sum(d["hit"] for d in details) / n if n else 0.0
        mrr = sum(1 / d["rank"] for d in details if d["rank"]) / n if n else 0.0
        cov = sum(d["keyword_coverage"] for d in details) / n if n else 0.0
        cit = sum(d["citations_ok"] for d in details) / n if n else 0.0
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("hit_rate@k", f"{hit_rate:.3f}")
        m2.metric("mrr", f"{mrr:.3f}")
        m3.metric("keyword_coverage", f"{cov:.3f}")
        m4.metric("citation_precision", f"{cit:.3f}")
        st.markdown("**Per-question results**")
        st.dataframe(pd.DataFrame(details)[
            ["question", "expected_ticket_id", "hit", "rank",
             "keyword_coverage", "citations_ok", "backend"]])


def main() -> None:
    st.set_page_config(page_title="CustomerLens", layout="wide")
    st.title("CustomerLens")
    st.caption("Customer data copilot: ingest messy CSVs, ask questions, check quality.")

    paths = current_paths()
    engine = load_engine(*paths) if paths else Engine()

    st.sidebar.header("Settings")
    backend_name = st.sidebar.selectbox("Answer backend", ["extractive", "openai"])
    if backend_name == "openai" and not OpenAIChatBackend().configured:
        st.sidebar.warning("LLM_API_KEY is not set. Answers will fall back to extractive.")
    k = st.sidebar.slider("Retrieval depth (k)", min_value=1, max_value=10, value=3)

    st.sidebar.header("Service health")
    if engine.is_ready():
        st.sidebar.success("Index ready")
        st.sidebar.write(f"{len(engine.documents)} documents indexed")
    else:
        st.sidebar.error("No data ingested")

    ask_tab, data_tab, eval_tab = st.tabs(["Ask", "Data", "Eval"])
    with ask_tab:
        render_ask(engine, backend_name, k)
    with data_tab:
        render_data(engine)
    with eval_tab:
        render_eval(engine, backend_name, k)


if __name__ == "__main__":
    main()
