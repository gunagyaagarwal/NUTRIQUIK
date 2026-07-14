# NutriQuik

An intelligent question-answering system for nutrition and immunology, built around three query tracks:

- **Factual** — BM25 (from scratch) + MiniLM semantic hybrid retrieval returns the single best-matching reference, with optional Gemini LLM refinement.
- **Advisory** — the same hybrid retrieval returns top-5 candidates, scored by a 9-feature heuristic trust model and filtered by a 0.50 guardrail threshold.
- **Prediction** — 9 trained XGBoost models (anemia, diabetes, heart, liver, kidney, hypothyroid, vitamin deficiency, supplement benefit, diet recommendation) with real SHAP explanations.

A regex + TF-IDF/LogisticRegression query guard blocks harmful or off-domain questions before any retrieval runs.

## Setup

```
pip install -r requirements.txt
streamlit run app.py
```

First launch builds and caches the MiniLM document embeddings (`models/doc_embeddings.npy`); subsequent launches load them instantly.

## App layout

`app.py` is a single Streamlit app with a sidebar-navigated control panel:

1. **QA Pipeline & Query Interface** — the live query flow (factual/advisory/prediction)
2. **Evaluation & Trust Analytics** — real IR metrics (BM25 vs hybrid) and XGBoost model metrics, from `models/eval_results.json`
3. **Guardrail & Rejected Results Panel** — live guardrail tester + sample rejected low-trust results
4. **Profile & Interaction Matrix** — user profile + curated nutrient-drug interaction reference
5. **Corpus & Dataset Inventory** — the 378-document IR corpus and the 9-model registry
6. **System Blueprint & WBS Timeline** — architecture diagram and project plan

## Structure

- `src/guard/` — query guard (blocklist + domain classifier) and intent router
- `src/ir/` — BM25 (pure Python, no IR libraries), MiniLM vector index, hybrid retrieval, evaluation
- `src/ml/` — feature extraction, XGBoost training, prediction + SHAP
- `src/utils/` — data loading, Wikipedia image lookup, Gemini LLM refine/summarize
- `src/ui/charts.py` + `ui_components.py` — Plotly chart functions
- `models/` — trained models, registry, cached embeddings, eval results
- `data/` — IR corpus and ML training datasets

## Re-running evaluation

```
python src/ir/evaluate.py
```

Regenerates `models/eval_results.json` (IR metrics on the 20-query ground truth set, plus held-out test metrics for all 9 models) shown in the Evaluation view.
