import re
import functools
import numpy as np


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(expected: str, actual: str) -> int:
    return int(_normalize(expected) == _normalize(actual))


def token_f1(expected: str, actual: str) -> float:
    """Token-level overlap F1, similar to SQuAD-style scoring."""
    exp_tokens = _normalize(expected).split()
    act_tokens = _normalize(actual).split()
    if not exp_tokens or not act_tokens:
        return 0.0

    common = {}
    for t in act_tokens:
        common[t] = common.get(t, 0) + 1

    overlap = 0
    exp_counts = {}
    for t in exp_tokens:
        exp_counts[t] = exp_counts.get(t, 0) + 1
    for t, c in exp_counts.items():
        overlap += min(c, common.get(t, 0))

    if overlap == 0:
        return 0.0
    precision = overlap / len(act_tokens)
    recall = overlap / len(exp_tokens)
    return round(2 * precision * recall / (precision + recall), 4)


@functools.lru_cache(maxsize=1)
def _get_embedder():
    """Lazily load a local sentence-transformer model so the app works offline
    without requiring an external embeddings API."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def embedding_similarity(expected: str, actual: str) -> float:
    """Cosine similarity between sentence embeddings of expected vs actual answer."""
    if not expected.strip() or not actual.strip():
        return 0.0
    model = _get_embedder()
    vecs = model.encode([expected, actual])
    a, b = vecs[0], vecs[1]
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    sim = float(np.dot(a, b) / denom)
    return round(sim, 4)
