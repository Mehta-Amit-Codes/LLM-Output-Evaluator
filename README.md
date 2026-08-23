# LLM Output Evaluator (Eval Harness)

A Streamlit app for evaluating an LLM/RAG app against a golden Q&A dataset.

## Setup

```bash
cd LLM-Output-Evaluator
pip install -r requirements.txt
streamlit run app.py
```

## What it does

1. **Golden dataset** — upload a CSV (`question`, `expected_answer`, `context`) or use the
   bundled 20-row sample (`sample_golden_dataset.csv`).
2. **Run predictions** — three ways to get answers:
   - Upload a pre-generated `predictions.csv` (`question`, `actual_answer`).
   - Call your own RAG app's HTTP endpoint (`POST {question, context} -> {answer, contexts}`).
   - Call an LLM directly (Anthropic / OpenAI / Groq) with no retrieval, useful for smoke-testing
     prompts before wiring up retrieval.
3. **Score**:
   - **Deterministic**: exact match, token-overlap F1, embedding cosine similarity
     (local `sentence-transformers` model, no API needed).
   - **LLM-as-judge**: faithfulness (1-5) and relevance (1-5), each with a rubric and a
     short reason, scored by a second LLM call (Anthropic/OpenAI/Groq).
   - **Hallucination check**: binary "is every claim in the answer supported by the context?"
     with a reason, run whenever context is present.
4. **Report** — aggregate metrics, score distributions, faithfulness-vs-relevance scatter,
   per-item drill-down table, CSV export, and a regression-tracking chart across multiple
   runs in the same session (label each run, e.g. `baseline-prompt`, `shorter-chunks`, etc.).

## Reducing judge bias

- Rubrics are anchored (each 1-5 score has a written description) rather than open-ended,
  which reduces judge drift/variance.
- Judge temperature is fixed at 0 for reproducibility.
- The judge only ever sees `question` / `context` / `answer` — never which model produced
  the answer — to avoid self-preference bias.
- Consider running each judge call twice and averaging, or using two different judge models
  and reporting disagreement, if you need tighter bias control (not included by default to
  keep API costs down).

## Extending

- Swap `sentence-transformers` for a hosted embeddings API in `metrics.py` if you'd rather
  not download a local model.
- Add cost/latency tracking: time each call in `runner.py` / `judge.py`, store
  `latency_ms` and `tokens`/`cost_usd` per row, then plot with `plotly.express.scatter`
  (quality metric vs. cost) — the "stretch goal" from the project brief.
- Swap in RAGAS or DeepEval by writing an adapter that maps their scorers into the
  same row-level `record` dict used in `app.py`'s scoring loop.
