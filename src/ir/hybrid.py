import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ir.bm25 import BM25Index, load_documents  # noqa: E402
from src.ir.vector_index import VectorIndex, get_model  # noqa: E402

DEFAULT_ALPHA = 0.5


def _min_max_normalize(scores):
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def hybrid_search(query, bm25_index, vector_index, top_k=5, alpha=DEFAULT_ALPHA, candidate_k=30):
    model = get_model()
    q_emb = np.array(model.encode([query], normalize_embeddings=True)[0])

    bm25_results = bm25_index.search(query, top_k=candidate_k)
    bm25_scores = {r["doc_id"]: r["score"] for r in bm25_results}

    sims = vector_index.embeddings @ q_emb
    top_vector_idx = np.argsort(-sims)[:candidate_k]
    vector_scores = {vector_index.doc_ids[i]: float(sims[i]) for i in top_vector_idx}

    doc_id_to_idx = {d: i for i, d in enumerate(vector_index.doc_ids)}
    query_tokens = bm25_index.preprocess(query)
    all_doc_ids = set(bm25_scores) | set(vector_scores)
    for doc_id in all_doc_ids:
        if doc_id not in bm25_scores:
            bm25_scores[doc_id] = bm25_index.score(query_tokens, doc_id)
        if doc_id not in vector_scores:
            idx = doc_id_to_idx.get(doc_id)
            vector_scores[doc_id] = float(sims[idx]) if idx is not None else 0.0

    norm_bm25 = _min_max_normalize(bm25_scores)
    norm_vector = _min_max_normalize(vector_scores)

    combined = {
        doc_id: alpha * norm_bm25.get(doc_id, 0.0) + (1 - alpha) * norm_vector.get(doc_id, 0.0)
        for doc_id in all_doc_ids
    }
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for doc_id, score in ranked:
        meta = bm25_index.doc_metadata.get(doc_id, {})
        content = str(meta.get("content", ""))
        results.append({
            "doc_id": doc_id,
            "score": score,
            "bm25_score": bm25_scores.get(doc_id, 0.0),
            "vector_score": vector_scores.get(doc_id, 0.0),
            "title": meta.get("title", ""),
            "content_snippet": content[:200],
        })
    return results


def _run_test_queries(bm25_index, vector_index):
    test_queries = [
        "benefits of vitamin c",
        "veg diet plan for diabetes",
        "how does zinc help immunity",
        "iron deficiency symptoms",
        "turmeric health benefits",
    ]
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        for rank, r in enumerate(hybrid_search(query, bm25_index, vector_index, top_k=3), start=1):
            print(f"  {rank}. [{r['doc_id']}] {r['title']} "
                  f"(combined={r['score']:.4f}, bm25={r['bm25_score']:.4f}, vec={r['vector_score']:.4f})")


if __name__ == "__main__":
    documents = load_documents()
    bm25_index = BM25Index()
    bm25_index.build(documents)

    vector_index = VectorIndex()
    vector_index.load_or_build(documents)

    _run_test_queries(bm25_index, vector_index)
