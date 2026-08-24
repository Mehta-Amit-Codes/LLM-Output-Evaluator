import time
import requests
import pandas as pd

from judge import _call_llm  # reuse the same lightweight multi-provider caller


def load_app_backend(backend_choice: str, endpoint=None, headers=None, provider=None, model_name=None, api_key=None):
    """Returns a function question -> (answer, context) for the chosen backend."""

    if backend_choice == "Call my RAG app via HTTP endpoint":
        if not endpoint:
            raise ValueError("No endpoint provided.")

        def backend_fn(question: str, context: str):
            resp = requests.post(endpoint, json={"question": question, "context": context}, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("answer", "")
            returned_context = data.get("contexts") or data.get("context") or context
            if isinstance(returned_context, list):
                returned_context = "\n".join(str(c) for c in returned_context)
            return answer, returned_context

        return backend_fn

    elif backend_choice == "Call an LLM API directly (no retrieval, for smoke-testing prompts)":
        if not api_key or not model_name:
            raise ValueError("Provide a model name and API key for the direct LLM backend.")

        def backend_fn(question: str, context: str):
            system = "You are a helpful assistant. Answer the question using only the provided context if given."
            user = f"Context:\n{context}\n\nQuestion: {question}" if context else question
            answer = _call_llm(provider, model_name, api_key, system, user)
            return answer, context

        return backend_fn

    else:
        raise ValueError(f"Unsupported backend choice: {backend_choice}")


def run_predictions(golden_df: pd.DataFrame, backend_fn, on_progress=None) -> pd.DataFrame:
    rows = []
    n = len(golden_df)
    for i, row in golden_df.iterrows():
        question = str(row["question"])
        context = str(row.get("context", "") or "")
        try:
            answer, returned_context = backend_fn(question, context)
        except Exception as e:
            answer, returned_context = f"[ERROR: {e}]", context
        rows.append(
            {
                "question": question,
                "expected_answer": row.get("expected_answer", ""),
                "actual_answer": answer,
                "context": returned_context,
            }
        )
        if on_progress:
            on_progress(i + 1, n)
        time.sleep(0.05)  # gentle pacing to avoid hammering rate limits
    return pd.DataFrame(rows)
