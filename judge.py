import json
import re
import requests

FAITHFULNESS_RUBRIC = """You are a strict, impartial evaluator grading an AI assistant's answer
for FAITHFULNESS to the provided context (i.e. is it grounded / not hallucinated?).

Score 1-5 using this rubric:
5 = Every claim in the answer is directly and fully supported by the context.
4 = Answer is mostly supported; at most one minor unsupported detail.
3 = Answer is partially supported; some claims not traceable to the context.
2 = Answer contradicts the context or contains significant unsupported claims.
1 = Answer is entirely unsupported by or contradicts the context.

If context is empty, judge faithfulness to general world knowledge instead and note that
in your reason.

Respond ONLY with strict JSON: {"score": <int 1-5>, "reason": "<one or two sentences>"}"""

RELEVANCE_RUBRIC = """You are a strict, impartial evaluator grading how RELEVANT an AI
assistant's answer is to the user's question, and how well it matches the expected answer's
intent (not necessarily exact wording).

Score 1-5 using this rubric:
5 = Directly and completely answers the question; matches the intent of the expected answer.
4 = Answers the question well with minor gaps or extra irrelevant info.
3 = Partially answers the question; missing key parts.
2 = Mostly off-topic or misses the point of the question.
1 = Does not answer the question at all.

Respond ONLY with strict JSON: {"score": <int 1-5>, "reason": "<one or two sentences>"}"""

HALLUCINATION_PROMPT = """You are a fact-checking evaluator. Given a CONTEXT (source material)
and an ANSWER produced by an AI assistant, determine whether every factual claim in the ANSWER
is supported by the CONTEXT.

Rules:
- If the answer includes any claim, number, name, or detail not present in or inferable from
  the context, mark it as hallucinated.
- General filler language ("Based on the context...") does not count as a claim.
- If the answer correctly says it doesn't know / the context doesn't cover it, that is NOT
  a hallucination.

Respond ONLY with strict JSON:
{"hallucinated": <true or false>, "reason": "<one or two sentences explaining which claims are/aren't supported>"}"""


def _extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from a model response."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Judge did not return JSON. Raw response: {text[:300]}")
    return json.loads(match.group(0))


def _call_llm(provider: str, model: str, api_key: str, system: str, user: str) -> str:
    provider = provider.lower()
    if provider == "anthropic":
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 500,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return "".join(block.get("text", "") for block in data.get("content", []))

    elif provider in ("openai", "groq"):
        base_url = (
            "https://api.openai.com/v1/chat/completions"
            if provider == "openai"
            else "https://api.groq.com/openai/v1/chat/completions"
        )
        resp = requests.post(
            base_url,
            headers={"Authorization": f"Bearer {api_key}", "content-type": "application/json"},
            json={
                "model": model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    else:
        raise ValueError(f"Unsupported provider: {provider}")


def llm_judge_score(
    provider: str,
    model: str,
    api_key: str,
    question: str,
    expected_answer: str,
    actual_answer: str,
    context: str,
    want_faithfulness: bool = True,
    want_relevance: bool = True,
) -> dict:
    result = {}

    if want_faithfulness:
        user_prompt = (
            f"CONTEXT:\n{context or '(no context provided)'}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"ANSWER TO EVALUATE:\n{actual_answer}"
        )
        raw = _call_llm(provider, model, api_key, FAITHFULNESS_RUBRIC, user_prompt)
        parsed = _extract_json(raw)
        result["faithfulness_score"] = parsed.get("score")
        result["faithfulness_reason"] = parsed.get("reason")

    if want_relevance:
        user_prompt = (
            f"QUESTION:\n{question}\n\n"
            f"EXPECTED ANSWER (for reference intent):\n{expected_answer}\n\n"
            f"ANSWER TO EVALUATE:\n{actual_answer}"
        )
        raw = _call_llm(provider, model, api_key, RELEVANCE_RUBRIC, user_prompt)
        parsed = _extract_json(raw)
        result["relevance_score"] = parsed.get("score")
        result["relevance_reason"] = parsed.get("reason")

    return result


def hallucination_check(provider: str, model: str, api_key: str, context: str, actual_answer: str) -> dict:
    user_prompt = f"CONTEXT:\n{context}\n\nANSWER:\n{actual_answer}"
    raw = _call_llm(provider, model, api_key, HALLUCINATION_PROMPT, user_prompt)
    parsed = _extract_json(raw)
    return {
        "hallucinated": bool(parsed.get("hallucinated")),
        "hallucination_reason": parsed.get("reason"),
    }
