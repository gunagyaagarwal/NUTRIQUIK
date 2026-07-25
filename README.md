# NutriQuik

An intelligent question-answering system for nutrition and immunology, built around three query tracks:

- **Factual** — BM25 (from scratch) + MiniLM semantic hybrid retrieval returns the single best-matching reference, with optional Gemini LLM refinement.
- **Advisory** — the same hybrid retrieval returns top-5 candidates, scored by a 9-feature heuristic trust model and filtered by a 0.30 guardrail threshold.
- **Prediction** — 7 trained XGBoost models (anemia, diabetes, heart, kidney, vitamin deficiency, supplement benefit, diet recommendation) with real SHAP explanations.

A regex + TF-IDF/LogisticRegression query guard blocks harmful or off-domain questions before any retrieval runs.

The UI is a static `frontend/index.html` (no build step) served by a thin FastAPI wrapper (`server.py`) around the existing pipeline — no ML/IR/guard logic lives in the web layer.

## Setup

```
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8501
```

Then open `http://localhost:8501`. First launch builds and caches the MiniLM document embeddings (`models/doc_embeddings.npy`); subsequent launches load them instantly. Gemini answer-polish is optional — set `GEMINI_API_KEY` in the environment to enable it; the app runs fully without it.

## API

- `POST /api/query` `{query}` → `{intent: blocked|factual|advisory|prediction, ...}` — runs the query guard, then the intent router, then the matching retrieval/scoring track.
- `GET /api/diseases` → the trained models available for the general "pick a condition" picker.
- `POST /api/predict/meta` `{disease}` → the form field spec for that model (tiers, options, help text), derived from its real trained `feature_cols`.
- `POST /api/predict` `{disease, values}` → runs the saved XGBoost model + SHAP, refusing to predict if any required lab/measurement field is blank (a blank numeric field would default to 0, which is an extreme, not neutral, value for these models).

## Structure

- `src/guard/` — query guard (blocklist + domain classifier) and intent router
- `src/ir/` — BM25 (pure Python, no IR libraries), MiniLM vector index, hybrid retrieval, evaluation
- `src/ml/` — feature extraction, XGBoost training, prediction + SHAP, and `form_spec.py` (prediction-form field tiers/labels/encoders, ported from the former Streamlit widget code)
- `src/pipeline.py` — the query pipeline (query guard → intent router → factual/advisory/prediction/diet_form), ported verbatim from the former Streamlit app's `run_full_pipeline`
- `src/utils/` — data loading, Wikipedia image lookup, Gemini LLM refine/summarize
- `server.py` — FastAPI app: serves `frontend/index.html` + the API above
- `frontend/index.html` — the static UI (HTML/CSS/JS, no build step)
- `models/` — trained models, registry, cached embeddings, eval results
- `data/` — IR corpus and ML training datasets

## Re-running evaluation

```
python src/ir/evaluate.py
```

Regenerates `models/eval_results.json` (IR metrics on the 20-query ground truth set, plus held-out test metrics for all 9 models).
