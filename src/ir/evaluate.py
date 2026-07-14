import json
import math
import os
import sys

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ir.bm25 import BM25Index, load_documents  # noqa: E402
from src.ir.vector_index import VectorIndex  # noqa: E402
from src.ir.hybrid import hybrid_search  # noqa: E402
from src.ml.train_models import (  # noqa: E402
    MODEL_CONFIGS, RANDOM_STATE, clean_missing_values, encode_categorical_features, encode_target,
)

GROUND_TRUTH_PATH = os.path.join(BASE_DIR, "data", "processed", "ground_truth.json")
EVAL_RESULTS_PATH = os.path.join(BASE_DIR, "models", "eval_results.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def _precision_recall_at_k(ranked_ids, relevant, k=5):
    top_k = ranked_ids[:k]
    hits = len(set(top_k) & relevant)
    precision = hits / k
    recall = hits / len(relevant) if relevant else 0.0
    return precision, recall


def _average_precision(ranked_ids, relevant):
    if not relevant:
        return 0.0
    hits = 0
    precisions = []
    for i, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            hits += 1
            precisions.append(hits / i)
    return sum(precisions) / len(relevant) if precisions else 0.0


def _ndcg_at_k(ranked_ids, relevant, k=5):
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked_ids[:k], start=1) if d in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ir(bm25_index, ground_truth_path=GROUND_TRUTH_PATH, top_k=10, search_fn=None):
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)

    if search_fn is None:
        search_fn = bm25_index.search

    ap_scores, ndcg_scores, p_scores, r_scores = [], [], [], []
    for query, relevant_ids in ground_truth.items():
        relevant = set(relevant_ids)
        results = search_fn(query, top_k=top_k)
        ranked_ids = [r["doc_id"] for r in results]

        ap_scores.append(_average_precision(ranked_ids, relevant))
        ndcg_scores.append(_ndcg_at_k(ranked_ids, relevant, k=5))
        p, r = _precision_recall_at_k(ranked_ids, relevant, k=5)
        p_scores.append(p)
        r_scores.append(r)

    n = len(ground_truth)
    return {
        "MAP": sum(ap_scores) / n,
        "NDCG@5": sum(ndcg_scores) / n,
        "Precision@5": sum(p_scores) / n,
        "Recall@5": sum(r_scores) / n,
        "num_queries": n,
    }


def _rebuild_test_split(config):
    full_path = os.path.join(BASE_DIR, config["path"])
    if not os.path.exists(full_path):
        return None

    df = pd.read_csv(full_path)

    for col in config.get("percent_cols", []):
        if col in df.columns:
            df[col] = df[col].astype(str).str.rstrip("%").astype(float)
    for col in config.get("fillna_none_cols", []):
        if col in df.columns:
            df[col] = df[col].fillna("None")
    if "subsample" in config:
        target_col = config["target_col"]
        neg_n = config["subsample"]["neg_n"]
        neg_df = df[df[target_col] == 0]
        pos_df = df[df[target_col] == 1]
        neg_sample = neg_df.sample(n=min(neg_n, len(neg_df)), random_state=RANDOM_STATE)
        df = pd.concat([neg_sample, pos_df], ignore_index=True)

    drop_cols = list(config.get("id_cols", [])) + list(config.get("drop_cols", []))
    if "derive_target" in config:
        source_col = config["derive_target"]["source_col"]
        df["_derived_target"] = df[source_col].apply(config["derive_target"]["fn"])
        target_col = "_derived_target"
        if source_col not in drop_cols:
            drop_cols.append(source_col)
    else:
        target_col = config["target_col"]

    if "feature_cols" in config:
        feature_cols = [c for c in config["feature_cols"] if c in df.columns]
        X = df[feature_cols].copy()
    else:
        cols_to_drop = set(drop_cols) | {target_col}
        X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()

    y_raw = df[target_col]
    X = clean_missing_values(X)
    X, _ = encode_categorical_features(X)
    y_encoded, _ = encode_target(y_raw, explicit_map=config.get("target_map"))
    num_classes = len(set(y_encoded))

    _, X_test, _, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=RANDOM_STATE
    )
    return X_test, y_test, num_classes


def evaluate_all_models():
    rows = []
    for config in MODEL_CONFIGS:
        name = config["name"]
        model_path = os.path.join(MODELS_DIR, f"{name}_model.pkl")
        if not os.path.exists(model_path):
            continue

        split = _rebuild_test_split(config)
        if split is None:
            continue
        X_test, y_test, num_classes = split

        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)

        avg = "binary" if num_classes == 2 else "weighted"
        try:
            auc = (
                roc_auc_score(y_test, proba[:, 1])
                if num_classes == 2
                else roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
            )
        except ValueError:
            auc = None

        rows.append({
            "model": name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, average=avg, zero_division=0)),
            "precision": float(precision_score(y_test, y_pred, average=avg, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average=avg, zero_division=0)),
            "auc_roc": float(auc) if auc is not None else None,
        })

    return pd.DataFrame(rows)


def run_full_evaluation():
    documents = load_documents()
    index = BM25Index()
    index.build(documents)
    vector_index = VectorIndex()
    vector_index.load_or_build(documents)

    bm25_results = evaluate_ir(index)
    hybrid_results = evaluate_ir(
        index, search_fn=lambda q, top_k: hybrid_search(q, index, vector_index, top_k=top_k)
    )

    ml_df = evaluate_all_models()
    ml_results = ml_df.to_dict(orient="records")

    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump({
            "ir_evaluation_bm25": bm25_results,
            "ir_evaluation_hybrid": hybrid_results,
            "ml_evaluation": ml_results,
        }, f, indent=2)

    print("=== IR Evaluation: BM25 (keyword only) ===")
    for k, v in bm25_results.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n=== IR Evaluation: Hybrid (BM25 + MiniLM) ===")
    for k, v in hybrid_results.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n=== ML Evaluation ===")
    print(ml_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved to: {EVAL_RESULTS_PATH}")
    return bm25_results, hybrid_results, ml_df


if __name__ == "__main__":
    run_full_evaluation()
