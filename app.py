import json
import time
import io
import pandas as pd
import streamlit as st
import plotly.express as px

from metrics import exact_match, token_f1, embedding_similarity
from judge import llm_judge_score, hallucination_check
from runner import run_predictions, load_app_backend

st.set_page_config(page_title="LLM Output Evaluator", page_icon="🧪", layout="wide")

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "golden_df" not in st.session_state:
    st.session_state.golden_df = None
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "run_history" not in st.session_state:
    st.session_state.run_history = []  # list of {label, df, timestamp} for regression tracking

st.title("🧪 LLM Output Evaluator")
st.caption(
    "Golden dataset → run predictions → deterministic metrics + LLM-as-judge → "
    "faithfulness / hallucination check → aggregated report. Built to evaluate the "
    "Project-1 RAG app, but works with any Q&A style LLM app."
)

# ---------------------------------------------------------------------------
# Sidebar: configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("1. App under test")
    backend_choice = st.selectbox(
        "How should we get answers to the golden questions?",
        [
            "Upload pre-generated predictions (CSV)",
            "Call my RAG app via HTTP endpoint",
            "Call an LLM API directly (no retrieval, for smoke-testing prompts)",
        ],
    )

    app_endpoint = None
    app_headers_raw = "{}"
    direct_model_provider = None
    if backend_choice == "Call my RAG app via HTTP endpoint":
        app_endpoint = st.text_input(
            "POST endpoint (expects JSON {question, context?} -> {answer, contexts?})",
            placeholder="http://localhost:8000/query",
        )
        app_headers_raw = st.text_area("Extra headers (JSON)", value="{}")
    elif backend_choice == "Call an LLM API directly (no retrieval, for smoke-testing prompts)":
        direct_model_provider = st.selectbox("Provider", ["Anthropic", "OpenAI", "Groq"])
        st.text_input("Model name", key="direct_model_name", placeholder="e.g. claude-sonnet-4-5")
        st.text_input("API key", key="direct_api_key", type="password")

    st.subheader("2. Judge model")
    judge_provider = st.selectbox("Judge provider", ["Anthropic", "OpenAI", "Groq"], key="judge_provider")
    judge_model = st.text_input(
        "Judge model name",
        value={"Anthropic": "claude-sonnet-4-5", "OpenAI": "gpt-4o-mini", "Groq": "llama-3.3-70b-versatile"}[
            judge_provider
        ],
        key="judge_model",
    )
    judge_api_key = st.text_input("Judge API key", type="password", key="judge_api_key")

    st.subheader("3. Metrics to compute")
    do_exact_match = st.checkbox("Exact match", value=True)
    do_token_f1 = st.checkbox("Token overlap F1", value=True)
    do_embed_sim = st.checkbox("Embedding cosine similarity", value=True)
    do_judge_faithful = st.checkbox("LLM-judge: faithfulness (1-5)", value=True)
    do_judge_relevance = st.checkbox("LLM-judge: relevance (1-5)", value=True)
    do_hallucination = st.checkbox("Hallucination check (claim-support)", value=True)

    st.subheader("4. Run label (for regression tracking)")
    run_label = st.text_input("Label this run", value=f"run-{len(st.session_state.run_history) + 1}")

# ---------------------------------------------------------------------------
# Step 1: Golden dataset
# ---------------------------------------------------------------------------
st.header("1️⃣ Golden Dataset")
st.write(
    "Upload a CSV with columns: `question`, `expected_answer`, `context` (optional, for RAG "
    "faithfulness checks — can contain the retrieved chunks used to generate the answer)."
)

col1, col2 = st.columns([2, 1])
with col1:
    uploaded_golden = st.file_uploader("Upload golden dataset (CSV)", type=["csv"])
with col2:
    st.write("")
    st.write("")
    if st.button("Use bundled sample dataset (20 Q&A pairs)"):
        st.session_state.golden_df = pd.read_csv("sample_golden_dataset.csv")

if uploaded_golden is not None:
    st.session_state.golden_df = pd.read_csv(uploaded_golden)

if st.session_state.golden_df is not None:
    required_cols = {"question", "expected_answer"}
    missing = required_cols - set(st.session_state.golden_df.columns)
    if missing:
        st.error(f"Missing required columns: {missing}")
    else:
        if "context" not in st.session_state.golden_df.columns:
            st.session_state.golden_df["context"] = ""
        st.success(f"Loaded {len(st.session_state.golden_df)} Q&A pairs.")
        st.dataframe(st.session_state.golden_df, use_container_width=True, height=220)

        with st.expander("➕ Add a row manually"):
            with st.form("add_row_form", clear_on_submit=True):
                q = st.text_input("Question")
                a = st.text_input("Expected answer")
                c = st.text_area("Context (source chunks, optional)")
                submitted = st.form_submit_button("Add row")
                if submitted and q and a:
                    new_row = pd.DataFrame([{"question": q, "expected_answer": a, "context": c}])
                    st.session_state.golden_df = pd.concat(
                        [st.session_state.golden_df, new_row], ignore_index=True
                    )
                    st.rerun()

# ---------------------------------------------------------------------------
# Step 2: Get predictions (optionally already provided in a CSV)
# ---------------------------------------------------------------------------
st.header("2️⃣ Run Predictions")

predictions_df = None
if backend_choice == "Upload pre-generated predictions (CSV)":
    st.write("Upload a CSV with columns `question`, `actual_answer` (and optionally `context`).")
    uploaded_preds = st.file_uploader("Upload predictions CSV", type=["csv"], key="preds_upload")
    if uploaded_preds is not None:
        predictions_df = pd.read_csv(uploaded_preds)

run_col1, run_col2 = st.columns([1, 3])
with run_col1:
    run_button = st.button("🚀 Run / Load Predictions", type="primary", use_container_width=True)

if run_button:
    if st.session_state.golden_df is None:
        st.error("Load a golden dataset first.")
    else:
        golden = st.session_state.golden_df.copy()
        try:
            if backend_choice == "Upload pre-generated predictions (CSV)":
                if predictions_df is None:
                    st.error("Upload a predictions CSV, or choose a live backend instead.")
                    st.stop()
                # Avoid duplicate 'context' columns clobbering each other: prefer the
                # predictions file's context (what the app actually retrieved) and only
                # fall back to the golden dataset's context if the predictions file has none.
                preds = predictions_df.copy()
                if "context" in preds.columns and "context" in golden.columns:
                    golden = golden.rename(columns={"context": "context_golden"})
                    merged = golden.merge(preds, on="question", how="left")
                    merged["context"] = merged["context"].fillna(merged["context_golden"])
                    merged = merged.drop(columns=["context_golden"])
                else:
                    merged = golden.merge(preds, on="question", how="left")
                merged["actual_answer"] = merged["actual_answer"].fillna("")
                if "context" not in merged.columns:
                    merged["context"] = ""
                merged["context"] = merged["context"].fillna("")
                st.session_state.results_df = merged
            else:
                headers = json.loads(app_headers_raw or "{}")
                backend_fn = load_app_backend(
                    backend_choice,
                    endpoint=app_endpoint,
                    headers=headers,
                    provider=direct_model_provider,
                    model_name=st.session_state.get("direct_model_name"),
                    api_key=st.session_state.get("direct_api_key"),
                )
                progress = st.progress(0.0, text="Running predictions...")

                def on_progress(i, n):
                    progress.progress(i / n, text=f"Running predictions... {i}/{n}")

                st.session_state.results_df = run_predictions(golden, backend_fn, on_progress)
                progress.empty()
            st.success(f"Got predictions for {len(st.session_state.results_df)} items.")
        except Exception as e:
            st.error(f"Failed to get predictions: {e}")

if st.session_state.results_df is not None:
    st.dataframe(
        st.session_state.results_df[["question", "expected_answer", "actual_answer"]],
        use_container_width=True,
        height=220,
    )

# ---------------------------------------------------------------------------
# Step 3: Score
# ---------------------------------------------------------------------------
st.header("3️⃣ Score Outputs")

score_button = st.button("📊 Compute Metrics", type="primary")

if score_button:
    if st.session_state.results_df is None:
        st.error("Run predictions first.")
    else:
        df = st.session_state.results_df.copy()
        n = len(df)
        progress = st.progress(0.0, text="Scoring...")

        rows = []
        for i, row in df.iterrows():
            question = str(row.get("question", ""))
            expected = str(row.get("expected_answer", ""))
            actual = str(row.get("actual_answer", ""))
            context = str(row.get("context", "") or "")

            record = {
                "question": question,
                "expected_answer": expected,
                "actual_answer": actual,
                "context": context,
            }

            if do_exact_match:
                record["exact_match"] = exact_match(expected, actual)
            if do_token_f1:
                record["token_f1"] = token_f1(expected, actual)
            if do_embed_sim:
                try:
                    record["embedding_similarity"] = embedding_similarity(expected, actual)
                except Exception as e:
                    record["embedding_similarity"] = None
                    record["embed_error"] = str(e)

            if (do_judge_faithful or do_judge_relevance) and judge_api_key:
                try:
                    judge_result = llm_judge_score(
                        provider=judge_provider,
                        model=judge_model,
                        api_key=judge_api_key,
                        question=question,
                        expected_answer=expected,
                        actual_answer=actual,
                        context=context,
                        want_faithfulness=do_judge_faithful,
                        want_relevance=do_judge_relevance,
                    )
                    record.update(judge_result)
                except Exception as e:
                    record["judge_error"] = str(e)

            if do_hallucination and judge_api_key and context.strip():
                try:
                    hall_result = hallucination_check(
                        provider=judge_provider,
                        model=judge_model,
                        api_key=judge_api_key,
                        context=context,
                        actual_answer=actual,
                    )
                    record.update(hall_result)
                except Exception as e:
                    record["hallucination_error"] = str(e)

            rows.append(record)
            progress.progress((i + 1) / n, text=f"Scoring... {i+1}/{n}")

        scored_df = pd.DataFrame(rows)
        st.session_state.results_df = scored_df
        st.session_state.run_history.append(
            {"label": run_label, "df": scored_df, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        )
        progress.empty()
        st.success("Scoring complete.")

# ---------------------------------------------------------------------------
# Step 4: Report / Dashboard
# ---------------------------------------------------------------------------
st.header("4️⃣ Report Dashboard")

if st.session_state.results_df is not None and any(
    c in st.session_state.results_df.columns
    for c in ["exact_match", "token_f1", "embedding_similarity", "faithfulness_score", "relevance_score"]
):
    df = st.session_state.results_df

    metric_cols = [
        c
        for c in [
            "exact_match",
            "token_f1",
            "embedding_similarity",
            "faithfulness_score",
            "relevance_score",
        ]
        if c in df.columns
    ]

    st.subheader("Aggregate scores")
    agg_cols = st.columns(len(metric_cols) + 1)
    for i, col in enumerate(metric_cols):
        mean_val = pd.to_numeric(df[col], errors="coerce").mean()
        agg_cols[i].metric(col.replace("_", " ").title(), f"{mean_val:.2f}" if pd.notna(mean_val) else "N/A")

    if "hallucinated" in df.columns:
        hall_rate = (df["hallucinated"] == True).mean()  # noqa: E712
        agg_cols[-1].metric("Hallucination rate", f"{hall_rate:.0%}")

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        if metric_cols:
            melted = df[metric_cols].apply(pd.to_numeric, errors="coerce").melt(var_name="metric", value_name="score")
            fig = px.box(melted, x="metric", y="score", points="all", title="Score distribution by metric")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "faithfulness_score" in df.columns and "relevance_score" in df.columns:
            fig2 = px.scatter(
                df,
                x="faithfulness_score",
                y="relevance_score",
                hover_data=["question"],
                title="Faithfulness vs Relevance per item",
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Per-item drill-down")
    show_cols = ["question", "expected_answer", "actual_answer"] + metric_cols
    if "hallucinated" in df.columns:
        show_cols += ["hallucinated", "hallucination_reason"]
    if "faithfulness_reason" in df.columns:
        show_cols += ["faithfulness_reason"]
    if "relevance_reason" in df.columns:
        show_cols += ["relevance_reason"]
    show_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(df[show_cols], use_container_width=True, height=400)

    # Regression tracking across runs
    if len(st.session_state.run_history) > 1:
        st.subheader("📈 Regression tracking across runs")
        history_rows = []
        for run in st.session_state.run_history:
            rdf = run["df"]
            row = {"run": run["label"], "timestamp": run["timestamp"]}
            for col in metric_cols:
                if col in rdf.columns:
                    row[col] = pd.to_numeric(rdf[col], errors="coerce").mean()
            history_rows.append(row)
        hist_df = pd.DataFrame(history_rows)
        st.dataframe(hist_df, use_container_width=True)
        if metric_cols:
            melted_hist = hist_df.melt(id_vars=["run"], value_vars=metric_cols, var_name="metric", value_name="score")
            fig3 = px.line(melted_hist, x="run", y="score", color="metric", markers=True, title="Metrics over runs")
            st.plotly_chart(fig3, use_container_width=True)

    st.subheader("⬇️ Export report")
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(
        "Download results CSV", csv_buf.getvalue(), file_name="eval_results.csv", mime="text/csv"
    )
else:
    st.info("Run predictions and click 'Compute Metrics' to see the report.")

st.divider()
st.caption(
    "Tip: change your RAG app's prompt or chunk size, re-run with a new run label, and watch the "
    "regression tracking chart to catch quality drops before they ship."
)