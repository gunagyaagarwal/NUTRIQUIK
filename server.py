"""FastAPI wrapper around the existing NutriQuik pipeline (src/guard, src/ir,
src/ml) — presentation-layer swap only. All retrieval scoring, trust weights,
BM25 cap, embeddings, query guard, and intent routing are reused unchanged from
src/pipeline.py and src/ml/*; nothing here re-implements or re-tunes any of it.
"""
import os
import sys

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.pipeline import (  # noqa: E402
    run_full_pipeline, FACTUAL_RELEVANCE_THRESHOLD, extract_symptom_focused_answer,
    get_bm25_index, get_vector_index,
)
from src.ml.predict import (  # noqa: E402
    predict_health, get_shap_explanation, load_model_and_metadata, FEATURE_WEIGHTS,
)
from src.ml.form_spec import (  # noqa: E402
    build_field_spec, encode_prediction_inputs, DISEASE_TO_MODEL,
    GENERIC_FORM_EXCLUDED_MODELS, DIET_LABEL_TO_DOC_ID, HIGH_RISK_THRESHOLD,
    FIELD_LABELS, get_prediction_risk_probability, get_registry_safe,
)
from src.utils.llm_refine import refine_answer  # noqa: E402

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = FastAPI(title="NutriQuik API")


@app.on_event("startup")
def _warm_up():
    # Building the BM25/vector index involves loading the corpus + the fastembed
    # ONNX model — do it once at startup instead of blocking the first real query.
    try:
        get_bm25_index()
        get_vector_index()
    except Exception:
        pass  # surfaced per-query instead, via run_full_pipeline's own try/except


class QueryIn(BaseModel):
    query: str


class MetaIn(BaseModel):
    disease: str


class PredictIn(BaseModel):
    disease: str
    values: dict = {}


def _refined_body(result, query):
    text = result.get("content", "")
    try:
        refined = refine_answer(text, query)
        if refined and refined.strip():
            return refined
    except Exception:
        pass
    return text


@app.post("/api/query")
def api_query(payload: QueryIn):
    query = (payload.query or "").strip()
    if not query:
        return {"intent": "blocked", "message": "Please enter a question."}

    try:
        result = run_full_pipeline(query)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"{type(e).__name__}: {e}"})

    if result["status"] == "REJECTED_BY_GUARD":
        return {"intent": "blocked", "message": result["reason"]}

    intent = result["intent"]

    if intent in ("prediction", "diet_form"):
        disease = "diet_recommendation" if intent == "diet_form" else result.get("disease")
        return {"intent": "prediction", "disease": disease}

    results = result.get("results", [])
    rejected_results = result.get("rejected_results", [])

    if intent == "factual":
        if not results or results[0]["vector_score"] < FACTUAL_RELEVANCE_THRESHOLD:
            return {"intent": "factual", "notfound": True, "query": query}

        top = results[0]
        top_body = _refined_body(top, query)
        if len(results) > 1:
            extra = results[1]
            title = f"{top['title']} & {extra['title']}"
            body = (
                f"**{top['title']}**\n\n{top_body}\n\n---\n\n"
                f"**{extra['title']}**\n\n{extra.get('content', '')}"
            )
        else:
            title = top["title"]
            body = top_body

        return {
            "intent": "factual",
            "title": title,
            "body": body,
            "sources": [r["doc_id"] for r in results],
            "score": float(top["vector_score"]),
        }

    # advisory
    if not results:
        return {
            "intent": "advisory", "title": "", "body": "No results passed the trust threshold for this query.",
            "trust": 0.0, "features": [], "sources": [], "ranked": [],
        }

    top = results[0]
    focused_answer = extract_symptom_focused_answer(query, top["content"])
    prefix = ""
    if focused_answer is None:
        prefix = f"We don't have deficiency symptoms specifically listed for {top['title']} yet.\n\n"
        focused_answer = top["content"]

    features = [
        [name, float(top["contributions"][name] / weight)]
        for name, weight in FEATURE_WEIGHTS.items()
    ]

    ranked = [[r["title"], float(r["trust_score"]), False] for r in results[1:]]
    ranked += [[r["title"], float(r["trust_score"]), True] for r in rejected_results]
    ranked.sort(key=lambda row: row[1], reverse=True)

    return {
        "intent": "advisory",
        "title": top["title"],
        "body": prefix + focused_answer,
        "trust": float(top["trust_score"]),
        "features": features,
        "sources": [top["doc_id"]],
        "ranked": ranked,
    }


@app.get("/api/diseases")
def api_diseases():
    registry = get_registry_safe()
    return {"diseases": sorted(set(registry.keys()) - GENERIC_FORM_EXCLUDED_MODELS)}


@app.post("/api/predict/meta")
def api_predict_meta(payload: MetaIn):
    model_name = DISEASE_TO_MODEL.get(payload.disease, payload.disease)
    registry = get_registry_safe()
    if model_name not in registry:
        return JSONResponse(
            status_code=404,
            content={"error": f"No trained model available for '{model_name}' yet."},
        )
    try:
        return build_field_spec(model_name)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"{type(e).__name__}: {e}"})


@app.post("/api/predict")
def api_predict(payload: PredictIn):
    model_name = DISEASE_TO_MODEL.get(payload.disease, payload.disease)
    registry = get_registry_safe()
    if model_name not in registry:
        return JSONResponse(
            status_code=404,
            content={"error": f"No trained model available for '{model_name}' yet."},
        )

    try:
        inputs, missing = encode_prediction_inputs(model_name, payload.values or {})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"{type(e).__name__}: {e}"})

    # A blank numeric field silently defaulting to 0 is NOT a neutral "unknown" value
    # for these models — 0 is often the single most extreme possible reading (e.g. 0%
    # of RDA vitamin C intake, 0 g/dL hemoglobin). Refuse to predict rather than
    # silently guessing, matching the original app's behavior exactly.
    if missing:
        friendly = [FIELD_LABELS.get(c, c.replace("_", " ")) for c in missing]
        return JSONResponse(
            status_code=422,
            content={
                "error": "missing_fields",
                "message": (
                    f"Please fill in: {', '.join(friendly)}. These are lab/measurement values, "
                    "and leaving them blank would default to 0, which is an extreme (not neutral) "
                    "value for this model and would produce an unreliable, misleading result."
                ),
                "fields": missing,
            },
        )

    try:
        prediction = predict_health(model_name, inputs)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Prediction failed: {e}"})

    risk_pct = get_prediction_risk_probability(model_name, prediction)
    positive = risk_pct >= 0.5

    try:
        model_obj, _ = load_model_and_metadata(model_name)
    except Exception:
        model_obj = None

    factors = []
    try:
        if model_obj is not None:
            feature_values = [prediction["feature_values"][c] for c in prediction["feature_names"]]
            shap_dict = get_shap_explanation(
                model_obj, feature_values, prediction["feature_names"],
                class_index=prediction["prediction"],
            )
            if shap_dict:
                top_factors = sorted(shap_dict.items(), key=lambda kv: -abs(kv[1]))[:5]
                factors = [
                    {"name": name, "direction": "raises" if val > 0 else "lowers"}
                    for name, val in top_factors
                ]
    except Exception:
        factors = []

    display_name = model_name.replace("_", " ").title()
    message = (
        f"For the {display_name} assessment, the model predicts {prediction['prediction_label']} "
        f"with a risk/confidence score of {risk_pct * 100:.1f}% "
        f"(model confidence: {prediction['confidence'] * 100:.1f}%). "
    )
    if risk_pct >= HIGH_RISK_THRESHOLD:
        message += "High risk detected — please take advice from a doctor for further evaluation."
    else:
        message += (
            "Be careful and take care of your health — maintain a balanced diet, "
            "regular exercise, and routine checkups."
        )

    if model_name == "diet_recommendation":
        doc_id = DIET_LABEL_TO_DOC_ID.get(prediction["prediction_label"])
        if doc_id:
            index, _ = get_bm25_index()
            doc_meta = index.doc_metadata.get(doc_id)
            if doc_meta:
                message += f"\n\nMatching diet plan — {doc_meta['title']}: {doc_meta['content']}"

    # "confidence" is this model's predict_proba for THIS specific input — how sure the
    # model is about this one case, not how often the model is right overall. That's a
    # separate, fixed number from held-out test evaluation (model_registry.json) — surfaced
    # here too so the two aren't conflated in the UI.
    model_accuracy = registry.get(model_name, {}).get("accuracy")

    return {
        "positive": bool(positive),
        "label": prediction["prediction_label"],
        "confidence": float(prediction["confidence"]),
        "modelAccuracy": float(model_accuracy) if model_accuracy is not None else None,
        "message": message,
        "hasLabs": True,
        "factors": factors,
    }


@app.get("/")
def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")
