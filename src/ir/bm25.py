import math
import os
import re
from collections import Counter

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MERGED_CORPUS_PATH = os.path.join(BASE_DIR, "data", "processed", "ir_corpus_merged.csv")

STOPWORDS = {
    "the", "is", "at", "which", "on", "a", "an", "and", "or", "but", "in", "to",
    "of", "for", "with", "by", "from", "as", "it", "its", "this", "that", "are",
    "was", "were", "be", "been", "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might", "not",
    "no", "so", "if", "than", "too", "very", "just", "about", "also",
}

SUFFIXES = ["ing", "ed", "ly", "tion", "ness", "ment", "s"]

K1 = 1.5
B = 0.75


class BM25Index:
    def __init__(self):
        self.inverted_index = {}
        self.doc_term_freqs = {}
        self.doc_lengths = {}
        self.avg_doc_length = 0.0
        self.total_docs = 0
        self.doc_metadata = {}

    @staticmethod
    def preprocess(text):
        tokens = re.findall(r"[a-zA-Z0-9]+", str(text).lower())
        tokens = [t for t in tokens if t not in STOPWORDS]

        stemmed = []
        for token in tokens:
            for suffix in SUFFIXES:
                if token.endswith(suffix) and len(token) - len(suffix) > 3:
                    token = token[: -len(suffix)]
                    break
            stemmed.append(token)
        return stemmed

    def build(self, documents):
        self.inverted_index = {}
        self.doc_term_freqs = {}
        self.doc_lengths = {}
        self.doc_metadata = {}

        for doc in documents:
            doc_id = doc["id"]
            text = f"{doc.get('title', '')} {doc.get('content', '')} {doc.get('keywords', '')}"
            tokens = self.preprocess(text)

            term_freqs = Counter(tokens)
            self.doc_term_freqs[doc_id] = term_freqs
            self.doc_lengths[doc_id] = len(tokens)
            self.doc_metadata[doc_id] = {
                "title": doc.get("title", ""),
                "content": doc.get("content", ""),
                "keywords": doc.get("keywords", ""),
                "category": doc.get("category", ""),
            }

            for term, freq in term_freqs.items():
                self.inverted_index.setdefault(term, {})[doc_id] = freq

        self.total_docs = len(documents)
        self.avg_doc_length = (
            sum(self.doc_lengths.values()) / self.total_docs if self.total_docs else 0.0
        )

    def _idf(self, term):
        df = len(self.inverted_index.get(term, {}))
        return math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_tokens, doc_id):
        doc_length = self.doc_lengths.get(doc_id, 0)
        term_freqs = self.doc_term_freqs.get(doc_id, {})
        total = 0.0
        for term in query_tokens:
            f = term_freqs.get(term, 0)
            if f == 0:
                continue
            idf = self._idf(term)
            numerator = f * (K1 + 1)
            denominator = f + K1 * (1 - B + B * doc_length / self.avg_doc_length)
            total += idf * (numerator / denominator)
        return total

    def search(self, query, top_k=5):
        query_tokens = self.preprocess(query)

        candidate_doc_ids = set()
        for term in query_tokens:
            candidate_doc_ids.update(self.inverted_index.get(term, {}).keys())

        scored = [
            (doc_id, self.score(query_tokens, doc_id)) for doc_id in candidate_doc_ids
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in scored[:top_k]:
            meta = self.doc_metadata.get(doc_id, {})
            content = str(meta.get("content", ""))
            results.append({
                "doc_id": doc_id,
                "score": score,
                "title": meta.get("title", ""),
                "content_snippet": content[:200],
            })
        return results

    def get_tfidf_similarity(self, query, doc_id):
        query_tokens = self.preprocess(query)
        query_tf = Counter(query_tokens)
        doc_tf = self.doc_term_freqs.get(doc_id, {})

        vocab = set(query_tf.keys()) | set(doc_tf.keys())
        if not vocab:
            return 0.0

        query_vec = []
        doc_vec = []
        for term in vocab:
            idf = self._idf(term)
            query_vec.append(query_tf.get(term, 0) * idf)
            doc_vec.append(doc_tf.get(term, 0) * idf)

        dot = sum(q * d for q, d in zip(query_vec, doc_vec))
        norm_q = math.sqrt(sum(q * q for q in query_vec))
        norm_d = math.sqrt(sum(d * d for d in doc_vec))

        if norm_q == 0 or norm_d == 0:
            return 0.0
        return dot / (norm_q * norm_d)


def load_documents(path=MERGED_CORPUS_PATH):
    df = pd.read_csv(path)
    df = df.fillna("")
    return df.to_dict(orient="records")


def _run_test_queries(index):
    test_queries = [
        "benefits of vitamin c",
        "veg diet plan for diabetes",
        "how does zinc help immunity",
        "what is anemia",
        "turmeric health benefits",
        "creatine supplement evidence",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = index.search(query, top_k=3)
        if not results:
            print("  No results found.")
            continue
        for rank, result in enumerate(results, start=1):
            print(f"  {rank}. [{result['doc_id']}] {result['title']} (score={result['score']:.4f})")
            print(f"     {result['content_snippet']}")


if __name__ == "__main__":
    documents = load_documents()
    index = BM25Index()
    index.build(documents)

    unique_terms = len(index.inverted_index)
    print(f"Total docs indexed: {index.total_docs}")
    print(f"Unique terms: {unique_terms}")
    print(f"Avg doc length: {index.avg_doc_length:.2f}")

    _run_test_queries(index)
