"""Stage 2: retrieval over the brand's historical (message -> reply) pairs.

Embeds every grounding customer message once, caches the matrix, and at query
time returns the top-k most similar past cases. Max similarity also serves as an
uncertainty signal for the escalation gate.
"""
import json
import hashlib

import numpy as np

from src.common import config, gemini_client, ROOT, retry

_CACHE = {}


def _grounding_rows():
    cfg = config()
    path = ROOT / cfg["data"]["sample_dir"] / "grounding.jsonl"
    return [json.loads(l) for l in open(path)]


def _embed(client, model, texts, task_type):
    from google.genai import types
    out = []
    for i in range(0, len(texts), 100):
        chunk = texts[i:i + 100]
        resp = retry(lambda: client.models.embed_content(
            model=model, contents=chunk,
            config=types.EmbedContentConfig(task_type=task_type)))
        out.extend([e.values for e in resp.embeddings])
    arr = np.array(out, dtype=np.float32)
    arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)  # normalize for cosine
    return arr


def _index(client=None, model=None):
    cfg = config()
    client = client or gemini_client()
    model = model or cfg["models"]["embed"]
    rows = _grounding_rows()

    cache_dir = ROOT / cfg["data"]["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Key on corpus content + model so a stale cache never loads.
    key = hashlib.md5((model + str(len(rows)) +
                       rows[0]["cust_tweet_id"]).encode()).hexdigest()[:10]
    npy = cache_dir / f"emb_{key}.npy"

    if npy.exists():
        mat = np.load(npy)
    else:
        mat = _embed(client, model, [r["customer_text"] for r in rows],
                     "RETRIEVAL_DOCUMENT")
        np.save(npy, mat)
    return rows, mat, client, model


def retrieve(query: str, top_k=None, client=None, model=None):
    cfg = config()
    top_k = top_k or cfg["retrieval"]["top_k"]
    if "idx" not in _CACHE:
        _CACHE["idx"] = _index(client, model)
    rows, mat, client, model = _CACHE["idx"]

    qv = _embed(client, model, [query], "RETRIEVAL_QUERY")[0]
    sims = mat @ qv
    order = np.argsort(-sims)[:top_k]
    return [{
        "customer_text": rows[i]["customer_text"],
        "brand_reply": rows[i]["brand_reply"],
        "similarity": float(sims[i]),
    } for i in order]


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "my package never arrived"
    for r in retrieve(q):
        print(f"[{r['similarity']:.3f}] {r['customer_text'][:60]}")
        print(f"        -> {r['brand_reply'][:80]}")
