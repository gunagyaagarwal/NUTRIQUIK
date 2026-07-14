import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ir.bm25 import BM25Index, load_documents  # noqa: E402
from src.ml.features import extract_features_batch  # noqa: E402

MODELS_DIR = os.path.join(BASE_DIR, "models")
REGISTRY_PATH = os.path.join(MODELS_DIR, "model_registry.json")

FEATURE_WEIGHTS = {
    "bm25_score": 0.24, "vector_score": 0.20, "tfidf_cosine": 0.12, "term_overlap": 0.08,
    "query_coverage": 0.12, "doc_length_norm": 0.04, "keyword_match": 0.08, "title_match": 0.08,
    "category_relevance": 0.04,
}
UNBOUNDED_COLS = ["bm25_score", "vector_score", "tfidf_cosine", "term_overlap"]


def load_registry():
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def load_model_and_metadata(model_name):
    model = joblib.load(os.path.join(MODELS_DIR, f"{model_name}_model.pkl"))
    metadata = joblib.load(os.path.join(MODELS_DIR, f"{model_name}_metadata.pkl"))
    return model, metadata


def score_documents(query, search_results, bm25_index):
    features_df = extract_features_batch(query, search_results, bm25_index)
    norm = features_df.copy()
    for col in UNBOUNDED_COLS:
        lo, hi = norm[col].min(), norm[col].max()
        norm[col] = (norm[col] - lo) / (hi - lo) if hi > lo else 0.5

    contributions_df = pd.DataFrame({col: norm[col] * w for col, w in FEATURE_WEIGHTS.items()})
    trust_scores = contributions_df.sum(axis=1).clip(0, 1)

    results = []
    for doc_id, trust_score in zip(features_df.index, trust_scores):
        meta = bm25_index.doc_metadata.get(doc_id, {})
        results.append({
            "doc_id": doc_id,
            "title": meta.get("title", ""),
            "content": meta.get("content", ""),
            "trust_score": float(trust_score),
            "features": features_df.loc[doc_id].to_dict(),
            "contributions": contributions_df.loc[doc_id].to_dict(),
        })
    return results


def rank_by_trust(scored_docs):
    return sorted(scored_docs, key=lambda d: d["trust_score"], reverse=True)


def apply_guardrail(scored_docs, threshold=0.5):
    passed = [d for d in scored_docs if d["trust_score"] >= threshold]
    rejected = [
        {**d, "reason": f"trust_score {d['trust_score']:.2f} below threshold {threshold}"}
        for d in scored_docs if d["trust_score"] < threshold
    ]
    return passed, rejected


def _encode_inputs(user_inputs, feature_cols, encoders):
    row = {}
    for col in feature_cols:
        value = user_inputs.get(col)
        if col in encoders:
            try:
                row[col] = encoders[col].transform([str(value)])[0]
            except ValueError:
                row[col] = 0
        else:
            row[col] = float(value) if value is not None else 0.0
    return pd.DataFrame([row], columns=feature_cols)


def predict_health(model_name, user_inputs):
    model, metadata = load_model_and_metadata(model_name)
    feature_cols = metadata["feature_cols"]
    label_mapping = metadata["label_mapping"]
    reverse_mapping = {v: k for k, v in label_mapping.items()}

    X = _encode_inputs(user_inputs, feature_cols, metadata["encoders"])
    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))

    return {
        "prediction": pred_idx,
        "prediction_label": reverse_mapping.get(pred_idx, str(pred_idx)),
        "confidence": float(proba[pred_idx]),
        "all_probabilities": {reverse_mapping.get(i, str(i)): float(p) for i, p in enumerate(proba)},
        "feature_names": feature_cols,
        "feature_values": X.iloc[0].to_dict(),
    }


def recommend_diet(user_inputs):
    result = predict_health("diet_recommendation", user_inputs)

    documents = load_documents()
    index = BM25Index()
    index.build(documents)
    query = f"{result['prediction_label']} diet {user_inputs.get('Disease_Type', '')}"
    matching_plans = index.search(query, top_k=3)

    return {
        "predicted_condition": result["prediction_label"],
        "confidence": result["confidence"],
        "matching_plans": matching_plans,
    }


def get_shap_explanation(model, feature_values, feature_names, class_index=None):
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(np.array([feature_values]))
        if isinstance(raw, list):
            idx = class_index if class_index is not None else 0
            values = np.array(raw[idx])[0]
        else:
            arr = np.array(raw)
            if arr.ndim == 3:
                idx = class_index if class_index is not None else 0
                values = arr[0, :, idx]
            else:
                values = arr[0]
        return dict(zip(feature_names, values.tolist()))
    except Exception:
        return None


if __name__ == "__main__":
    documents = load_documents()
    index = BM25Index()
    index.build(documents)
    results = index.search("benefits of zinc", top_k=5)
    scored = rank_by_trust(score_documents("benefits of zinc", results, index))
    passed, rejected = apply_guardrail(scored)
    print(f"MODE1 top1={scored[0]['doc_id']} trust={scored[0]['trust_score']:.3f} passed={len(passed)} rejected={len(rejected)}")

    anemia_input = {"Age": 45, "Sex": 0, "RBC": 3.8, "PCV": 30.0, "MCV": 78.0, "MCH": 24.0,
                     "MCHC": 30.0, "RDW": 16.0, "TLC": 7.0, "PLT/mm3": 250.0, "HGB": 9.5}
    r2 = predict_health("anemia", anemia_input)
    print(f"MODE2 label={r2['prediction_label']} confidence={r2['confidence']:.3f}")
