import os
import re
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ir.bm25 import BM25Index, load_documents  # noqa: E402

FEATURE_NAMES = [
    "bm25_score",
    "vector_score",
    "tfidf_cosine",
    "term_overlap",
    "query_coverage",
    "doc_length_norm",
    "keyword_match",
    "title_match",
    "category_relevance",
]


def _doc_token_set(document, bm25_index):
    doc_id = document.get("id")
    if doc_id in bm25_index.doc_term_freqs:
        return set(bm25_index.doc_term_freqs[doc_id].keys())
    text = f"{document.get('title', '')} {document.get('content', '')} {document.get('keywords', '')}"
    return set(bm25_index.preprocess(text))


def extract_features(query, document, bm25_score, bm25_index, vector_score=0.0):
    doc_id = document.get("id")
    query_tokens = set(bm25_index.preprocess(query))
    doc_tokens = _doc_token_set(document, bm25_index)

    intersection = query_tokens & doc_tokens
    union = query_tokens | doc_tokens
    term_overlap = len(intersection) / len(union) if union else 0.0
    query_coverage = len(intersection) / len(query_tokens) if query_tokens else 0.0

    doc_length = bm25_index.doc_lengths.get(doc_id, len(doc_tokens))
    max_doc_length = max(bm25_index.doc_lengths.values()) if bm25_index.doc_lengths else 1
    doc_length_norm = doc_length / max_doc_length if max_doc_length else 0.0

    doc_keywords = [k.strip().lower() for k in re.split(r"[;,]", str(document.get("keywords", ""))) if k.strip()]
    query_lower = query.lower()
    keyword_match = (
        sum(1 for kw in doc_keywords if kw in query_lower) / len(doc_keywords)
        if doc_keywords else 0.0
    )

    title_tokens = set(bm25_index.preprocess(document.get("title", "")))
    title_match = (
        len(query_tokens & title_tokens) / len(query_tokens) if query_tokens else 0.0
    )

    category_tokens = set(bm25_index.preprocess(document.get("category", "")))
    category_relevance = 1.0 if (query_tokens & category_tokens) else 0.5

    tfidf_cosine = bm25_index.get_tfidf_similarity(query, doc_id)

    return {
        "bm25_score": bm25_score,
        "vector_score": vector_score,
        "tfidf_cosine": tfidf_cosine,
        "term_overlap": term_overlap,
        "query_coverage": query_coverage,
        "doc_length_norm": doc_length_norm,
        "keyword_match": keyword_match,
        "title_match": title_match,
        "category_relevance": category_relevance,
    }


def extract_features_batch(query, search_results, bm25_index):
    rows = []
    doc_ids = []
    for result in search_results:
        doc_id = result["doc_id"]
        meta = bm25_index.doc_metadata.get(doc_id, {})
        document = {"id": doc_id, **meta}
        bm25_score = result.get("bm25_score", result["score"])
        vector_score = result.get("vector_score", 0.0)
        features = extract_features(query, document, bm25_score, bm25_index, vector_score=vector_score)
        rows.append(features)
        doc_ids.append(doc_id)

    return pd.DataFrame(rows, columns=FEATURE_NAMES, index=doc_ids)


if __name__ == "__main__":
    documents = load_documents()
    index = BM25Index()
    index.build(documents)

    query = "benefits of zinc"
    results = index.search(query, top_k=5)

    features_df = extract_features_batch(query, results, index)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 10)
    print(f"Query: '{query}'\n")
    print(features_df)
