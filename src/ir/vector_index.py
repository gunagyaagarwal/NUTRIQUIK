import json
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ir.bm25 import load_documents  # noqa: E402

MODEL_NAME = "all-MiniLM-L6-v2"
MODELS_DIR = os.path.join(BASE_DIR, "models")
EMBEDDINGS_PATH = os.path.join(MODELS_DIR, "doc_embeddings.npy")
DOC_IDS_PATH = os.path.join(MODELS_DIR, "doc_ids.json")

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


class VectorIndex:
    def __init__(self):
        self.doc_ids = []
        self.embeddings = None
        self.doc_metadata = {}

    def build(self, documents):
        model = get_model()
        texts = [f"{d.get('title', '')} {d.get('content', '')} {d.get('keywords', '')}" for d in documents]
        self.doc_ids = [d["id"] for d in documents]
        self.doc_metadata = {
            d["id"]: {
                "title": d.get("title", ""), "content": d.get("content", ""),
                "keywords": d.get("keywords", ""), "category": d.get("category", ""),
            }
            for d in documents
        }
        self.embeddings = np.array(model.encode(texts, show_progress_bar=False, normalize_embeddings=True))

    def save(self, embeddings_path=EMBEDDINGS_PATH, doc_ids_path=DOC_IDS_PATH):
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        np.save(embeddings_path, self.embeddings)
        with open(doc_ids_path, "w") as f:
            json.dump({"doc_ids": self.doc_ids, "doc_metadata": self.doc_metadata}, f)

    def load(self, embeddings_path=EMBEDDINGS_PATH, doc_ids_path=DOC_IDS_PATH):
        self.embeddings = np.load(embeddings_path)
        with open(doc_ids_path) as f:
            data = json.load(f)
        self.doc_ids = data["doc_ids"]
        self.doc_metadata = data["doc_metadata"]

    def load_or_build(self, documents, embeddings_path=EMBEDDINGS_PATH, doc_ids_path=DOC_IDS_PATH):
        if os.path.exists(embeddings_path) and os.path.exists(doc_ids_path):
            self.load(embeddings_path, doc_ids_path)
        else:
            self.build(documents)
            self.save(embeddings_path, doc_ids_path)

    def search(self, query, top_k=5):
        model = get_model()
        q_emb = np.array(model.encode([query], normalize_embeddings=True)[0])
        sims = self.embeddings @ q_emb
        ranked = np.argsort(-sims)[:top_k]

        results = []
        for idx in ranked:
            doc_id = self.doc_ids[idx]
            meta = self.doc_metadata.get(doc_id, {})
            content = str(meta.get("content", ""))
            results.append({
                "doc_id": doc_id,
                "score": float(sims[idx]),
                "title": meta.get("title", ""),
                "content_snippet": content[:200],
            })
        return results

    def get_similarity(self, query_emb, doc_id):
        if doc_id not in self.doc_ids:
            return 0.0
        idx = self.doc_ids.index(doc_id)
        return float(self.embeddings[idx] @ query_emb)


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
        for rank, result in enumerate(index.search(query, top_k=3), start=1):
            print(f"  {rank}. [{result['doc_id']}] {result['title']} (score={result['score']:.4f})")


if __name__ == "__main__":
    documents = load_documents()
    index = VectorIndex()
    index.load_or_build(documents)
    print(f"Total docs embedded: {len(index.doc_ids)}, dim: {index.embeddings.shape[1]}")
    _run_test_queries(index)
