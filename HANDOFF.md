# NutriQuik — Frontend Swap Handoff (for Claude Code)

Goal: replace the old Streamlit UI with the new static frontend, **keeping every
existing feature identical**. Only the presentation layer changes — no ML/IR/guard
behavior, accuracies, corpus, thresholds, or scoring logic may change.

---

## 1. Put these files in the repo

```
NUTRIQUIK/
├── frontend/
│   └── index.html        ← use the provided index.html
├── src/                  (UNCHANGED — ir, ml, guard)
├── models/               (UNCHANGED — 7 registered models + registry)
├── data/                 (UNCHANGED — corpus, ground truth, eval)
├── app.py                (will be REMOVED)
├── ui_components.py       (will be REMOVED)
├── server.py             (Claude Code will CREATE this)
└── requirements.txt
```

Copy `index.html` into a new `frontend/` folder. Then run `claude` in the repo root
and paste the prompt in section 2.

---

## 2. Prompt to paste into Claude Code

```
Replace the Streamlit UI with the new static frontend in frontend/index.html, WITHOUT
changing any of the ML/IR/guard behavior. Keep every existing feature identical — only
the presentation layer changes.

WHAT TO REMOVE
- Delete app.py and ui_components.py (the Streamlit app). Remove streamlit/plotly from
  requirements.txt.

WHAT TO KEEP UNTOUCHED
- Everything in src/ (ir, ml, guard), models/ (all 7 registered models + registry), and
  data/. Do not retrain, re-tune, or change thresholds, trust weights, the BM25 cap, the
  fastembed/ONNX embeddings, the query guard, the intent router, or the Gemini optional-
  polish logic. Reuse these modules as a library.

WHAT TO BUILD
- Create server.py: a FastAPI app (add fastapi + uvicorn to requirements.txt) that:
  1. Serves frontend/index.html at "/" and any static assets.
  2. Exposes the JSON API below, calling the EXISTING pipeline functions (query guard →
     intent router → factual/advisory/prediction tracks, trust scoring, SHAP, relevance/
     trust thresholds, optional Gemini polish). Import from src/*, don't reimplement.
- Update frontend/index.html's JavaScript: replace the mock `answer()` function and the
  hardcoded client-side prediction form with real fetch() calls to the API below. Preserve
  ALL current UI behavior exactly:
    * intent badges
    * factual answers WITH NO trust ring
    * advisory answers WITH the trust ring + expandable "see why" (9-feature breakdown +
      ranked list with below-cutoff items dimmed)
    * the friendly tiered prediction form (easy fields shown, lab fields in an optional
      expander, live BMI calculator, tooltips, honest lower confidence when labs are blank,
      never fabricating clinical values)
    * the "no verified information" state
  Add a "blocked" state for guardrail rejections.

API CONTRACT (the frontend already expects these shapes — match them exactly)

POST /api/query   body: { "query": string }
  Run the guard first, then the router. Return ONE of:
  - Guard blocked:  { "intent":"blocked", "message": string }
  - Factual hit:    { "intent":"factual", "title": str, "body": str, "sources":[str], "score": float }
  - Factual miss:   { "intent":"factual", "notfound": true, "query": str }   (below relevance floor)
  - Advisory:       { "intent":"advisory", "title": str, "body": str,
                      "trust": float(0..1),
                      "features": [[name:str, value:float(0..1)], ...],   (the 9 trust features)
                      "sources": [str],
                      "ranked": [[name:str, score:float(0..1), belowCutoff:bool], ...] }
  - Prediction:     { "intent":"prediction", "disease": str }   (which model the router detected)

POST /api/predict/meta   body: { "disease": str }
  Build the form from that model's REAL feature spec (use nutriquik_model_input_spec.md +
  nutriquik_form_tooltips.md if present). Return:
  { "disease": str, "title": str,
    "fields": [ { "id":str, "label":str, "type":"number|select|segment|multiselect",
                  "options":[str]?, "tier":"easy|lab|derived", "help":str,
                  "min":num?, "max":num?, "step":num?, "unit":str? }, ... ] }

POST /api/predict   body: { "disease": str, "values": { field_id: value, ... } }
  Encode inputs the way the model was trained (respect one-hot vs label-encoded vs numeric),
  run the saved XGBoost model + encoders, compute SHAP top factors. Never fabricate a missing
  clinical value; if key lab fields are blank, lower the reported confidence honestly. Return:
  { "positive": bool, "label": str, "confidence": float(0..1),
    "message": str, "hasLabs": bool,
    "factors": [ { "name": str, "direction": "raises|lowers"? }, ... ] }   (from SHAP)

REQUIREMENTS
- Core app must run with NO API key (Gemini optional; read GEMINI_API_KEY via os.environ;
  degrade silently if absent).
- Same-origin, so CORS not needed. Wrap pipeline calls in try/except and return a friendly
  error JSON.
- Add a run command to README: `uvicorn server:app --host 0.0.0.0 --port 8501`.
- Verify end-to-end:
    "what is vitamin c"              → factual (no ring)
    "best foods for iron deficiency" → advisory with trust breakdown
    "check my anemia risk"           → prediction form → real prediction + SHAP
    off-topic/harmful query          → blocked state

Do not change model accuracies, corpus, or any scoring logic. This is a frontend swap +
a thin API wrapper only.
```

---

## 3. Feature parity checklist (verify after Claude Code finishes)

- [ ] app.py / ui_components.py removed; streamlit + plotly gone from requirements.txt
- [ ] All 7 models still load from model_registry.json; accuracies unchanged
- [ ] Query guard blocks harmful + off-topic → UI shows "blocked" state
- [ ] Intent router: factual / advisory / prediction routing unchanged
      ("suggest diabetic recipes" → advisory; "foods rich in protein" → diet flow)
- [ ] Factual answer shows source + scores, NO trust ring
- [ ] Advisory shows trust ring + "see why" (9 features) + ranked list, below-cutoff dimmed
- [ ] Relevance floors still return "no verified information" when nothing good exists
- [ ] Prediction: tiered form (easy default, labs in expander), live BMI, tooltips
- [ ] Prediction never fabricates clinical values; confidence drops honestly without labs
- [ ] SHAP factors shown on every prediction
- [ ] Gemini polish optional; app runs with no key
- [ ] Runs via `uvicorn server:app --port 8501`; deploys same as before on AWS

---

## 4. Notes

- The provided index.html currently ships a **client-side demo** (mock answers + an
  anemia-only prediction form). The prompt tells Claude Code to make it dynamic:
  fetch the field spec per disease from /api/predict/meta and POST to /api/predict —
  that's what restores the full tiered-form + SHAP + all-7-models behavior.
- A "blocked" UI state must be added; the static file doesn't render guardrail
  rejections yet.
- Keep the file self-contained (all CSS/JS inline) unless you prefer to split assets.
