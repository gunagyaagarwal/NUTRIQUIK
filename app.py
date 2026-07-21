import base64
import json
import html
import os
import re
import sys
import urllib.parse

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.guard.query_guard import run_query_guard  # noqa: E402
from src.guard.intent_router import classify_intent, detect_disease_context  # noqa: E402
from src.ir.bm25 import BM25Index, load_documents  # noqa: E402
from src.ir.vector_index import VectorIndex  # noqa: E402
from src.ir.hybrid import hybrid_search  # noqa: E402
from src.ml.predict import (  # noqa: E402
    score_documents, rank_by_trust, apply_guardrail, predict_health,
    get_shap_explanation, load_model_and_metadata, load_registry,
)
from src.utils.llm_refine import refine_answer, summarize_results  # noqa: E402
from src.ui.charts import (  # noqa: E402
    shap_waterfall_chart, prediction_donut_chart, feature_importance_chart, risk_gauge,
)
from ui_components import (  # noqa: E402
    create_trust_gauge, create_evidence_pyramid, create_retrieval_scatter,
    create_shap_chart, create_ir_metrics_chart, create_ml_metrics_chart,
    create_trust_comparison_chart,
)

MODELS_DIR = os.path.join(BASE_DIR, "models")
REGISTRY_PATH = os.path.join(MODELS_DIR, "model_registry.json")

# Below this raw semantic-similarity score, the "best" retrieval match is judged too
# weak to be a real answer (e.g. corpus has no doc for the topic and BM25/vector
# search just returned the least-irrelevant thing) — show a "not available" message
# instead of presenting a misleading result.
FACTUAL_RELEVANCE_THRESHOLD = 0.35

# Composite trust-score cutoff for the advisory guardrail. Recalibrated alongside the
# bm25_score normalization fix in src/ml/predict.py: the old 0.5 cutoff was tuned
# against a scoring formula that batch-relative-normalized already-bounded [0, 1]
# features, artificially inflating scores. Under the corrected (more honest) scale,
# an adversarial/off-domain query now tops out around 0.23, while genuinely relevant
# advisory results typically run 0.3-0.6+ — 0.30 keeps clear separation from junk
# while no longer rejecting most real, on-topic content.
ADVISORY_TRUST_THRESHOLD = 0.30

DISEASE_TO_MODEL = {
    "diabetes": "diabetes", "anemia": "anemia", "heart": "heart",
    "kidney": "kidney", "vitamin_deficiency": "vitamin_deficiency",
    "supplement": "supplement", "weight": "weight", "diet_recommendation": "diet_recommendation",
}

# One-hot column groups per model: {model_name: {group_prefix: [full_column_names]}}.
# The dropdown label for each column is the column name with "<prefix>_" stripped,
# so option lists are derived directly from the model's real feature_cols.
ONE_HOT_GROUPS = {
    "diabetes": {
        "gender": ["gender_Female", "gender_Male", "gender_Other"],
        "smoking": ["smoking_No Info", "smoking_current", "smoking_ever", "smoking_former",
                    "smoking_never", "smoking_not current"],
    },
    "vitamin_deficiency": {
        "gender": ["gender_Female", "gender_Male"],
        "smoking_status": ["smoking_status_Current", "smoking_status_Former", "smoking_status_Never"],
        "alcohol_consumption": ["alcohol_consumption_Heavy", "alcohol_consumption_Moderate",
                                 "alcohol_consumption_None"],
        "exercise_level": ["exercise_level_Active", "exercise_level_Light", "exercise_level_Moderate",
                            "exercise_level_Sedentary"],
        "diet_type": ["diet_type_Omnivore", "diet_type_Pescatarian", "diet_type_Vegan", "diet_type_Vegetarian"],
        "sun_exposure": ["sun_exposure_High", "sun_exposure_Low", "sun_exposure_Moderate"],
        "income_level": ["income_level_High", "income_level_Low", "income_level_Middle"],
        "latitude_region": ["latitude_region_High", "latitude_region_Low", "latitude_region_Mid"],
    },
}

# Plain 0/1 columns (not one-hot, not label-encoded) rendered as a Yes/No select.
# Safe to default to "No" (0) since absence of a condition is a legitimate value,
# unlike numeric lab measurements where 0 would be clinically meaningless.
BINARY_FIELDS = {
    "anemia": ["Sex"],
    "kidney": ["Htn"],
    "heart": ["Sex", "FBS over 120", "Exercise angina"],
    "diabetes": ["hypertension", "heart_disease"],
    "vitamin_deficiency": [
        "has_night_blindness", "has_fatigue", "has_bleeding_gums", "has_bone_pain",
        "has_muscle_weakness", "has_numbness_tingling", "has_memory_problems",
        "has_pale_skin", "has_multiple_deficiencies",
    ],
}

# Small-integer categoricals with a fixed valid-value set (not full label encoders,
# but still must be constrained so users can't submit an out-of-range code).
CATEGORICAL_RANGES = {
    "heart": {
        "Chest pain type": [1, 2, 3, 4],
        "EKG results": [0, 1, 2],
        "Slope of ST": [1, 2, 3],
        "Number of vessels fluro": [0, 1, 2, 3],
        "Thallium": [3, 6, 7],
    },
}

# Display labels (with clinical units, derived from each model's real training data
# ranges) for plain numeric feature_cols, so users know what scale/unit to enter a
# value in rather than guessing from a bare column name.
FIELD_LABELS = {
    # anemia
    "RBC": "RBC (million/µL)", "PCV": "PCV / Hematocrit (%)", "MCV": "MCV (fL)",
    "MCH": "MCH (pg)", "MCHC": "MCHC (g/dL)", "RDW": "RDW (%)",
    "TLC": "TLC / WBC Count (×10³/µL)", "PLT/mm3": "Platelets (×10³/mm³)", "HGB": "Hemoglobin (g/dL)",
    # diabetes
    "age": "Age (years)", "HbA1c_level": "HbA1c Level (%)", "blood_glucose_level": "Blood Glucose Level (mg/dL)",
    # kidney
    "Bp": "Blood Pressure (mmHg)", "Sg": "Urine Specific Gravity (1.005-1.025)",
    "Al": "Albumin (grade 0-5)", "Su": "Sugar (grade 0-5)", "Rbc": "Red Blood Cells (0=abnormal, 1=normal)",
    "Bu": "Blood Urea (mg/dL)", "Sc": "Serum Creatinine (mg/dL)", "Sod": "Sodium (mEq/L)",
    "Pot": "Potassium (mEq/L)", "Hemo": "Hemoglobin (g/dL)", "Wbcc": "WBC Count (cells/µL)",
    "Rbcc": "RBC Count (million/µL)",
    # heart
    "Age": "Age (years)", "BP": "Resting Blood Pressure (mmHg)", "Cholesterol": "Cholesterol (mg/dL)",
    "Max HR": "Max Heart Rate (bpm)", "ST depression": "ST Depression (mm)",
    # vitamin_deficiency
    "vitamin_a_percent_rda": "Vitamin A Intake (% of RDA)", "vitamin_c_percent_rda": "Vitamin C Intake (% of RDA)",
    "vitamin_d_percent_rda": "Vitamin D Intake (% of RDA)", "vitamin_e_percent_rda": "Vitamin E Intake (% of RDA)",
    "vitamin_b12_percent_rda": "Vitamin B12 Intake (% of RDA)", "folate_percent_rda": "Folate Intake (% of RDA)",
    "calcium_percent_rda": "Calcium Intake (% of RDA)", "iron_percent_rda": "Iron Intake (% of RDA)",
    "hemoglobin_g_dl": "Hemoglobin (g/dL)", "serum_vitamin_d_ng_ml": "Serum Vitamin D (ng/mL)",
    "serum_vitamin_b12_pg_ml": "Serum Vitamin B12 (pg/mL)", "serum_folate_ng_ml": "Serum Folate (ng/mL)",
    "symptoms_count": "Number of Symptoms (count)",
    # supplement
    "Weeks": "Duration (weeks)", "Initial_WT": "Initial Weight (kg)", "Final_WT": "Final Weight (kg)",
    "Strength_Gain": "Strength Gain (fraction, e.g. 0.15 = 15%)",
    # diet_recommendation
    "Daily_Caloric_Intake": "Daily Caloric Intake (kcal)", "Cholesterol_mg/dL": "Cholesterol (mg/dL)",
    "Blood_Pressure_mmHg": "Blood Pressure (mmHg)", "Glucose_mg/dL": "Glucose (mg/dL)",
    "Weekly_Exercise_Hours": "Weekly Exercise (hours)",
}

# Per-model lab "measured" flag derivation (derived from whether the user filled
# in that lab value, rather than asked as a separate question). No models use this
# currently.
MEASURED_FLAG_PAIRS = {}

# Models whose feature_cols include a BMI feature. weight_col/height_col are only
# set when the model ALSO has raw Weight_kg/Height_cm as real feature_cols (so the
# same two calculator inputs feed all three, with no duplicate fields).
BMI_FEATURE_CONFIG = {
    "diabetes": {"bmi_col": "bmi"},
    "vitamin_deficiency": {"bmi_col": "bmi"},
    "diet_recommendation": {"bmi_col": "BMI", "weight_col": "Weight_kg", "height_col": "Height_cm"},
}

# Feature columns that are computed/historical measurements (e.g. how closely someone
# has adhered to a past diet plan, a nutrient-imbalance score derived from tracked
# intake) rather than anything a first-time user filling out the form could actually
# know. Hidden from the form entirely and fed a fixed, documented default instead of
# asking the user to guess a number that has no real meaning to them yet.
HIDDEN_DEFAULT_FIELDS = {
    "diet_recommendation": {
        "Adherence_to_Diet_Plan": 75.0,  # dataset median ~74.9 (0-100 scale)
        "Dietary_Nutrient_Imbalance_Score": 2.5,  # dataset median ~2.4 (0-5 scale)
    },
}

# diet_recommendation's predicted Diet_Label -> the matching curated plan doc in the IR corpus.
DIET_LABEL_TO_DOC_ID = {
    "High_Protein": "diet_high_protein",
    "Low_Fat": "diet_low_fat",
    "Low_Carb": "diet_low_carb",
    "Low_Sodium": "diet_low_sodium",
    "High_Fiber": "diet_high_fiber",
    "Balanced": "diet_balanced",
}

# Ordered longest-first so "vitamin b12"/"vitamin b6" are matched specifically
# rather than only ever matching the generic "vitamin b" prefix.
NUTRIENT_QUERY_TERMS = [
    "vitamin b12", "vitamin b6", "vitamin a", "vitamin b", "vitamin c", "vitamin d",
    "vitamin e", "vitamin k", "calcium", "iron", "magnesium", "zinc", "potassium",
    "protein", "fiber", "fibre",
]

# Requires actual planning/recommendation intent, not just a nutrient word appearing
# anywhere in the query — "foods rich in protein" or "recipes containing low sodium"
# are plain advisory/factual lookups, not a request to build a personalized diet plan.
PERSONALIZED_DIET_PHRASES = [
    "diet plan", "meal plan", "what should i eat", "suggest meals", "suggest a meal plan",
    "suggest my meals", "personalized diet", "custom diet", "custom meal plan",
    "plan my diet", "plan my meals", "design my diet", "create a diet plan",
    "create my diet", "recommend a diet plan", "recommend my diet", "build me a diet",
]
# "more protein"/"less sodium" only counts as a diet-planning signal when it's also
# talking about diet/meals/food, not just any comparison question.
_MORE_LESS_PATTERN = re.compile(r"\b(more|less)\s+\w+\b.*\b(diet|meal|food|foods)\b")


def is_personalized_diet_request(query):
    query_lower = query.lower()
    if any(phrase in query_lower for phrase in PERSONALIZED_DIET_PHRASES):
        return True
    return bool(_MORE_LESS_PATTERN.search(query_lower))


# BM25/category-relevance/term-overlap all rely on exact token matches, so a
# misspelling like "recipies" scores near-zero on several trust features at once
# even against a genuinely relevant document — normalizing common misspellings
# before retrieval fixes the root cause instead of patching each symptom.
_QUERY_SPELLING_FIXES = [
    (re.compile(r"\brecipies\b", re.IGNORECASE), "recipes"),
    (re.compile(r"\brecipie\b", re.IGNORECASE), "recipe"),
    (re.compile(r"\brecepies\b", re.IGNORECASE), "recipes"),
    (re.compile(r"\brecepie\b", re.IGNORECASE), "recipe"),
    (re.compile(r"\brecipy\b", re.IGNORECASE), "recipe"),
    (re.compile(r"\bchickengunia\b", re.IGNORECASE), "chikungunya"),
    (re.compile(r"\bchickungunya\b", re.IGNORECASE), "chikungunya"),
    (re.compile(r"\bchickungunia\b", re.IGNORECASE), "chikungunya"),
]


def normalize_query_spelling(query):
    normalized = query
    for pattern, replacement in _QUERY_SPELLING_FIXES:
        normalized = pattern.sub(replacement, normalized)
    return normalized


# Every Vitamins & Minerals / Deficiency Diseases doc is written with this
# consistent "Deficiency symptoms: ..." sentence — when the user specifically
# asks about symptoms, extract just that sentence instead of showing the whole
# document (benefits, food sources, RDA, etc. they didn't ask for).
_SYMPTOM_SENTENCE_PATTERN = re.compile(r"Deficiency symptoms:[^.]*\.", re.IGNORECASE)


def extract_symptom_focused_answer(query, content):
    if "symptom" not in query.lower():
        return content
    match = _SYMPTOM_SENTENCE_PATTERN.search(content)
    return match.group(0).strip() if match else None


def find_named_vitamin_mineral_doc(query, index):
    """Direct name match for a Vitamins & Minerals doc mentioned in the query.
    Needed because semantic ranking alone is unreliable here: for an ambiguous
    query like "vitamin a deficiency" or "vitamin b deficiency", the generic
    Vitamin Deficiency Overview (or even an unrelated doc like Anemia) can score
    higher on vector similarity than the specific vitamin's own page, purely
    from sharing more "deficiency"-flavored vocabulary — not because it's a
    better answer to "the user asked about vitamin A specifically"."""
    query_lower = query.lower()
    candidates = [
        (doc_id, meta.get("title", "")) for doc_id, meta in index.doc_metadata.items()
        if meta.get("category") == "Vitamins & Minerals"
    ]
    candidates.sort(key=lambda x: len(x[1]), reverse=True)
    for doc_id, title in candidates:
        if title and re.search(rf"\b{re.escape(title.lower())}\b", query_lower):
            return doc_id
    return None


# A query naming two things ("egg or chicken", "milk vs paneer", "which is better ...")
# wants BOTH sides addressed — the factual track otherwise always truncates to a single
# top-ranked document, which silently drops the second thing being asked about and
# answers as if only one item was ever mentioned.
_COMPARISON_PATTERN = re.compile(
    r"\b(vs\.?|versus|compare[d]?(?:\s+to)?|which is (?:better|healthier)|better than|"
    r"worse than|difference between)\b",
    re.IGNORECASE,
)
_OR_COMPARISON_PATTERN = re.compile(
    r"\b(better|healthier|healthy|good for (?:you|health)|worse)\b.*\bor\b|"
    r"\bor\b.*\b(better|healthier)\b",
    re.IGNORECASE,
)


def is_comparison_query(query):
    query_lower = query.lower()
    return bool(_COMPARISON_PATTERN.search(query_lower) or _OR_COMPARISON_PATTERN.search(query_lower))


# Direct overrides for topics where "veg"/"non-veg" style phrasing reliably ranks the
# wrong Diet Plans doc via vector similarity (both docs share almost all the same
# "diet"/"nutrition" vocabulary, so the ranking signal is weak) — checked non-veg
# first since "non-vegetarian" also contains the word "vegetarian".
_NAMED_DIET_OVERRIDES = [
    (re.compile(r"\bnon[- ]?veg(?:etarian)?\b", re.IGNORECASE), "diet_002"),
    (re.compile(r"\bveg(?:etarian)?\b", re.IGNORECASE), "diet_001"),
]


def find_named_diet_doc(query):
    for pattern, doc_id in _NAMED_DIET_OVERRIDES:
        if pattern.search(query):
            return doc_id
    return None


def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"], [data-testid="stAppViewContainer"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        .stApp {
            background: #0B0F17;
            color: #F8FAFC;
        }
        [data-testid="stHeader"] {
            background: transparent;
            height: 1.4rem;
            min-height: 1.4rem;
        }
        [data-testid="stToolbar"] {
            right: 1rem;
        }
        .main .block-container,
        [data-testid="stMainBlockContainer"],
        [data-testid="stAppViewBlockContainer"] {
            max-width: 1400px;
            width: 100%;
            padding-top: 0rem !important;
            padding-bottom: 2.5rem;
            transition: max-width 0.25s ease;
        }
        div[data-testid="stVerticalBlock"] > div:first-child {
            margin-top: 0 !important;
        }
        [data-testid="stSidebar"] {
            background: #272833;
            border-right: 1px solid #3F414D;
            min-width: 275px !important;
            max-width: 310px !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: #F8FAFC;
        }
        [data-testid="stSidebar"] hr {
            border-color: #474957;
        }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"] > div {
            background: #080C14 !important;
            border-color: #080C14 !important;
            color: #F8FAFC !important;
            border-radius: 8px !important;
        }
        .nq-brand-row {
            display: flex;
            align-items: center;
            gap: 0.9rem;
            margin: 1rem 0 1.6rem;
        }
        .nq-brand-logo {
            width: 48px;
            height: 48px;
            flex-shrink: 0;
            border-radius: 10px;
            object-fit: contain;
            background: #0B0F17;
            filter: drop-shadow(0 8px 18px rgba(20, 184, 166, 0.28));
        }
        .nq-sidebar-title {
            font-size: 1.05rem;
            font-weight: 800;
            line-height: 1.3;
            color: #FFFFFF;
            margin: 0;
        }
        .nq-profile-note {
            background: #214664;
            color: #38BDF8;
            border-radius: 8px;
            padding: 1rem;
            line-height: 1.55;
            text-align: center;
            margin-top: 1rem;
            border: 1px solid rgba(56, 189, 248, 0.12);
        }
        .nq-credits {
            border: 1px solid #555866;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            color: #F8FAFC;
            line-height: 1.65;
            margin-top: 0.8rem;
        }
        .nq-credits strong {
            color: #FFFFFF;
        }
        .nq-hero {
            background: linear-gradient(135deg, #111827 0%, #112A3B 48%, #00533F 100%);
            color: #FFFFFF;
            padding: 1.8rem 2.2rem;
            border-radius: 18px;
            border: 1px solid rgba(34, 211, 238, 0.13);
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.26);
            margin: -0.8rem 0 1.4rem;
        }
        .nq-hero-title {
            color: #00BDF2;
            font-size: 2.45rem;
            font-weight: 800;
            letter-spacing: 0;
            margin-bottom: 0.9rem;
        }
        .nq-hero-subtitle {
            color: #A9B4C4;
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 1.35rem;
        }
        .nq-hero-meta {
            color: #D6DEE9;
            font-size: 0.92rem;
            font-weight: 600;
        }
        .nq-section-header {
            display: flex;
            align-items: center;
            gap: 1.1rem;
            padding: 1.3rem 1.6rem;
            border-radius: 14px;
            margin: 0.4rem 0 1.5rem;
            box-shadow: 0 14px 30px rgba(0, 0, 0, 0.22);
        }
        .nq-section-badge {
            flex-shrink: 0;
            width: 52px;
            height: 52px;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.16);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.6rem;
        }
        .nq-section-heading {
            color: #FFFFFF;
            font-size: 1.55rem;
            font-weight: 800;
            line-height: 1.25;
            margin: 0 0 0.3rem;
        }
        .nq-section-subtitle {
            color: rgba(255, 255, 255, 0.85);
            font-size: 0.92rem;
            font-weight: 500;
            line-height: 1.45;
            margin: 0;
        }
        .nq-section-teal {
            background: linear-gradient(120deg, #0F766E 0%, #0891B2 100%);
        }
        .nq-section-purple {
            background: linear-gradient(120deg, #6D28D9 0%, #4C1D95 100%);
        }
        .nq-section-amber {
            background: linear-gradient(120deg, #B45309 0%, #92400E 100%);
        }
        .nq-section-blue {
            background: linear-gradient(120deg, #1D4ED8 0%, #1E3A8A 100%);
        }
        h1, h2, h3, h4,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4 {
            color: #FFFFFF;
            letter-spacing: 0;
        }
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span {
            color: inherit;
        }
        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-baseweb="select"] > div {
            background: #272833 !important;
            border: 1px solid #383A46 !important;
            border-radius: 8px !important;
            color: #F8FAFC !important;
        }
        div[data-testid="stTextInput"] label,
        div[data-testid="stNumberInput"] label,
        div[data-testid="stSelectbox"] label {
            color: #F8FAFC !important;
            font-weight: 700 !important;
        }
        div.stButton > button {
            background: #111620;
            color: #F8FAFC;
            border: 1px solid #343A46;
            border-radius: 8px;
            padding: 0.62rem 1.05rem;
            font-weight: 800;
            box-shadow: none;
            transition: background 0.18s ease, border-color 0.18s ease, transform 0.12s ease;
        }
        div.stButton > button:hover {
            color: #FFFFFF;
            border-color: #16B8D9;
            background: #152034;
            transform: translateY(-1px);
        }
        div.stButton > button:active {
            transform: translateY(0);
        }
        div.stButton > button:focus-visible {
            outline: 2px solid #22D3EE;
            outline-offset: 2px;
        }
        div.stButton > button[kind="primary"] {
            background: #FF4B4B;
            border-color: #FF4B4B;
            color: #FFFFFF;
        }
        div.stButton > button[kind="primary"]:hover {
            background: #F43F5E;
            border-color: #F43F5E;
        }
        div[data-testid="stAlert"] {
            border-radius: 10px;
        }
        div[data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
        }
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #06B6D4, #22C55E);
        }
        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-baseweb="select"] > div {
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }
        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stNumberInput"] input:focus,
        div[data-baseweb="select"]:focus-within > div {
            border-color: #22D3EE !important;
            box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.15) !important;
        }
        .st-key-query_bar_container div[data-testid="stTextInput"] input {
            background: #0B2318 !important;
            border: 1.5px solid #22C55E !important;
            color: #F0FDF4 !important;
            font-weight: 600;
        }
        .st-key-query_bar_container div[data-testid="stTextInput"] input:focus {
            border-color: #4ADE80 !important;
            box-shadow: 0 0 0 3px rgba(74, 222, 128, 0.2) !important;
        }
        .st-key-query_bar_container div[data-testid="stTextInput"] label {
            color: #4ADE80 !important;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label {
            border-radius: 8px;
            padding: 0.45rem 0.6rem;
            transition: background 0.15s ease;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(56, 189, 248, 0.08);
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
            background: rgba(34, 211, 238, 0.12);
        }
        [data-testid="stExpander"] {
            border: 1px solid #2A2E3A;
            border-radius: 10px;
            overflow: hidden;
        }
        [data-testid="stExpander"] summary {
            font-weight: 700;
        }
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        ::-webkit-scrollbar-track {
            background: #12161F;
        }
        ::-webkit-scrollbar-thumb {
            background: #343A46;
            border-radius: 8px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #454B5C;
        }
        .nq-card {
            border: 1px solid #155A4E;
            border-radius: 12px;
            padding: 1.1rem 1.25rem;
            margin: 0.75rem 0 1.15rem;
            background: #073F2E;
            box-shadow: 0 12px 26px rgba(0, 0, 0, 0.18);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }
        .nq-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.28);
        }
        .nq-card-title {
            color: #4ADE80;
            font-size: 1.18rem;
            font-weight: 750;
            line-height: 1.25;
            margin-bottom: 0.45rem;
        }
        .nq-card-body {
            color: #D1FAE5;
            font-size: 1rem;
            line-height: 1.55;
        }
        .nq-card-meta {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-bottom: 0.55rem;
            color: #A7F3D0;
            font-size: 0.88rem;
        }
        .nq-trust-badge {
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            font-size: 0.82rem;
            font-weight: 700;
        }
        .nq-trust-high {
            background: #064E3B;
            color: #86EFAC;
            border: 1px solid #10B981;
        }
        .nq-trust-medium {
            background: #45350B;
            color: #FCD34D;
            border: 1px solid #F59E0B;
        }
        .nq-trust-low {
            background: #4C1010;
            color: #FCA5A5;
            border: 1px solid #EF4444;
        }
        .nq-metric-box {
            background: #12161F;
            border: 1px solid #2A2E3A;
            padding: 1rem 1.2rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.16);
            transition: transform 0.18s ease, border-color 0.18s ease;
        }
        .nq-metric-box:hover {
            transform: translateY(-2px);
            border-color: #3F414D;
        }
        .nq-metric-value {
            font-size: 1.7rem;
            font-weight: 800;
            color: #4ADE80;
        }
        .nq-metric-label {
            font-size: 0.82rem;
            color: #9CA3AF;
            font-weight: 600;
        }
        .nq-footer {
            border-top: 1px solid #343A46;
            margin-top: 3rem;
            padding: 2rem 0 1rem;
            text-align: center;
            color: #64748B;
            line-height: 1.8;
        }
        .nq-footer-brand {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.4rem;
            margin-bottom: 0.3rem;
            color: #F8FAFC;
        }
        .nq-footer-leaf {
            width: 20px;
            height: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _trust_class(score):
    if score >= 0.55:
        return "nq-trust-high"
    if score >= ADVISORY_TRUST_THRESHOLD:
        return "nq-trust-medium"
    return "nq-trust-low"


def render_section_header(icon, title, subtitle, theme="teal"):
    st.markdown(
        f'<div class="nq-section-header nq-section-{theme}">'
        f'<div class="nq-section-badge">{icon}</div>'
        '<div>'
        f'<div class="nq-section-heading">{html.escape(title)}</div>'
        f'<div class="nq-section-subtitle">{html.escape(subtitle)}</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_result_card(title, body, trust_score=None, meta=None):
    title_html = html.escape(str(title))
    body_html = html.escape(str(body)).replace("\n", "<br>")
    meta_parts = []
    if meta:
        meta_parts.append(f"<span>{html.escape(str(meta))}</span>")
    if trust_score is not None:
        trust_pct = trust_score * 100
        meta_parts.append(
            f'<span class="nq-trust-badge {_trust_class(trust_score)}">'
            f"Trust {trust_pct:.1f}%</span>"
        )
    meta_html = f'<div class="nq-card-meta">{"".join(meta_parts)}</div>' if meta_parts else ""
    st.markdown(
        f"""
        <div class="nq-card">
            <div class="nq-card-title">{title_html}</div>
            {meta_html}
            <div class="nq-card-body">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_prediction_risk_probability(model_name, prediction):
    probabilities = prediction.get("all_probabilities", {})
    for positive_label in ("1", "Presence", "Positive", "Disease", "Yes"):
        if positive_label in probabilities:
            return probabilities[positive_label]
    if model_name in {"heart"} and "Absence" in probabilities:
        return 1.0 - probabilities["Absence"]
    return prediction["confidence"]


# Above this risk/confidence level, the closing statement urges a doctor visit
# instead of the standard "be careful" advisory.
HIGH_RISK_THRESHOLD = 0.95


def render_prediction_verdict(model_name, prediction, risk_pct):
    display_name = model_name.replace("_", " ").title()
    label = prediction["prediction_label"]
    st.markdown(
        f"📝 **Summary:** For the **{display_name}** assessment, the model predicts **{label}** "
        f"with a risk/confidence score of **{risk_pct * 100:.1f}%** "
        f"(model confidence: {prediction['confidence'] * 100:.1f}%)."
    )
    if risk_pct >= HIGH_RISK_THRESHOLD:
        st.error("🚨 **High risk detected — please take advice from a doctor for further evaluation.**")
    else:
        st.info(
            "💡 **Be careful and take care of your health** — maintain a balanced diet, "
            "regular exercise, and routine checkups."
        )


def reveal_more_results():
    st.session_state.show_more_clicked = True


def set_query_prompt(prompt):
    st.session_state.query_input = prompt
    st.session_state.show_more_clicked = False


def render_hero():
    st.markdown(
        """
        <div class="nq-hero">
            <div class="nq-hero-title">NUTRIQUIK</div>
            <div class="nq-hero-subtitle">
                An Intelligent Question-Answering System for Nutrition and Immunology
            </div>
            <div class="nq-hero-meta">
                Hybrid IR Pipeline (BM25 + MiniLM) &bull; Machine Learning Trust Scoring (XGBoost + SHAP) &bull; Guardrail Filter Safety
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _leaf_svg(gradient_id, css_class=""):
    class_attr = f' class="{css_class}"' if css_class else ""
    return (
        f'<svg{class_attr} viewBox="0 0 64 64" role="img" aria-label="NutriQuik leaf">'
        f'<defs><linearGradient id="{gradient_id}" x1="0" x2="1" y1="0" y2="1">'
        '<stop offset="0%" stop-color="#2DD4BF"/><stop offset="100%" stop-color="#0891B2"/>'
        '</linearGradient></defs>'
        f'<path fill="url(#{gradient_id})" d="M52 7C31 9 14 19 9 35c-2 8 2 16 9 19 9 4 20-1 25-11 6-12 7-24 9-36Z"/>'
        '<path fill="#22D3EE" opacity=".9" d="M11 55c11-11 21-21 35-34" stroke="#22D3EE" stroke-width="5" stroke-linecap="round"/>'
        '</svg>'
    )


def _sidebar_logo_html():
    return _leaf_svg("leafGradientSidebar", "nq-brand-logo")


def render_sidebar_brand():
    row_html = (
        '<div class="nq-brand-row">'
        + _sidebar_logo_html()
        + '<div class="nq-sidebar-title">NUTRIQUIK<br>Control Panel</div>'
        + "</div>"
    )
    st.sidebar.markdown(row_html, unsafe_allow_html=True)


def render_footer():
    footer_html = (
        '<div class="nq-footer">'
        '<div class="nq-footer-brand">'
        + _leaf_svg("leafGradientFooter", "nq-footer-leaf")
        + '<strong>NUTRIQUIK</strong></div>'
        '&copy; 2025-2026 | Jaypee Institute of Information Technology (JIIT)<br>'
        'Team: Gunagya Agarwal &bull; Mridul Rai &bull; Suryansh Singh &bull; Pranav S Nair &bull; Pulkit Sukhija'
        '</div>'
    )
    st.markdown(footer_html, unsafe_allow_html=True)


@st.cache_resource
def get_bm25_index():
    documents = load_documents()
    index = BM25Index()
    index.build(documents)
    return index, documents


@st.cache_resource
def get_vector_index():
    documents = load_documents()
    index = VectorIndex()
    index.load_or_build(documents)
    return index


def get_registry_safe():
    try:
        return load_registry()
    except Exception:
        return {}


def _onehot_option_label(prefix, col):
    return col[len(prefix) + 1:]


def render_prediction_form(model_name):
    try:
        _, metadata = load_model_and_metadata(model_name)
    except Exception:
        st.warning(f"Missing model/metadata for '{model_name}'.")
        return None

    if model_name == "anemia":
        st.caption(
            "ℹ️ The anemia model's original training data was small (264 rows) and its labels were "
            "noisy — some clinically normal blood counts were marked anemic. It has since been "
            "retrained on the original data plus 550 clinically-realistic synthetic rows (added to "
            "correct the class balance and label consistency); treat predictions as illustrative "
            "rather than clinically validated."
        )

    feature_cols = metadata["feature_cols"]
    encoders = metadata["encoders"]
    onehot_groups = ONE_HOT_GROUPS.get(model_name, {})
    binary_fields = set(BINARY_FIELDS.get(model_name, []))
    categorical_ranges = CATEGORICAL_RANGES.get(model_name, {})
    measured_pairs = MEASURED_FLAG_PAIRS.get(model_name, {})
    derived_cols = set(measured_pairs.values())
    grouped_cols = {c for cols in onehot_groups.values() for c in cols}
    bmi_config = BMI_FEATURE_CONFIG.get(model_name)
    bmi_related_cols = set(bmi_config.values()) if bmi_config else set()
    hidden_default_fields = HIDDEN_DEFAULT_FIELDS.get(model_name, {})
    skip_cols = grouped_cols | derived_cols | bmi_related_cols | set(hidden_default_fields)

    profile = st.session_state.user_profile
    prefilled_fields = []
    group_choice = {}
    field_values = {}

    height_cm = weight_kg = bmi_value = None
    if bmi_config:
        height_default = profile.get("Height_cm")
        weight_default = profile.get("Weight_kg")
        st.markdown("**⚖️ BMI Calculator**")
        bmi_widget_cols = st.columns(3)
        height_cm = bmi_widget_cols[0].number_input(
            "Height (cm)", min_value=0.0,
            value=float(height_default) if height_default is not None else None,
            key=f"{model_name}_height_cm",
        )
        weight_kg = bmi_widget_cols[1].number_input(
            "Weight (kg)", min_value=0.0,
            value=float(weight_default) if weight_default is not None else None,
            key=f"{model_name}_weight_kg",
        )
        if height_cm and height_cm > 0 and weight_kg:
            bmi_value = weight_kg / ((height_cm / 100) ** 2)
        bmi_widget_cols[2].metric("Your BMI", f"{bmi_value:.1f}" if bmi_value is not None else "—")

    with st.form(key=f"form_{model_name}"):
        st.subheader(f"🧾 {model_name.replace('_', ' ').title()} Assessment Form")
        if hidden_default_fields:
            st.caption(
                f"ℹ️ {', '.join(c.replace('_', ' ') for c in hidden_default_fields)} "
                "use a typical baseline value since there's no way for a first-time user to know these — "
                "not asked below."
            )
        widget_cols = st.columns(3)
        idx = 0

        for group_name, group_cols in onehot_groups.items():
            target = widget_cols[idx % 3]
            idx += 1
            options = [_onehot_option_label(group_name, c) for c in group_cols]
            selected_col = next((c for c in group_cols if profile.get(c) == 1), None)
            if selected_col is not None:
                prefilled_fields.append(group_name)
            default_index = group_cols.index(selected_col) if selected_col in group_cols else 0
            chosen = target.selectbox(
                group_name.replace("_", " ").title(), options,
                index=default_index, key=f"{model_name}_group_{group_name}",
            )
            group_choice[group_name] = chosen

        for col in feature_cols:
            if col in skip_cols:
                continue
            target = widget_cols[idx % 3]
            idx += 1
            saved = profile.get(col)

            if col in encoders:
                classes = list(encoders[col].classes_)
                index = classes.index(saved) if saved in classes else 0
                if saved in classes:
                    prefilled_fields.append(col)
                field_values[col] = target.selectbox(col, classes, index=index, key=f"{model_name}_{col}")
            elif col in binary_fields:
                options = ["No", "Yes"]
                default_index = 1 if saved == 1 else 0
                if saved is not None:
                    prefilled_fields.append(col)
                choice = target.selectbox(col, options, index=default_index, key=f"{model_name}_{col}")
                field_values[col] = 1 if choice == "Yes" else 0
            elif col in categorical_ranges:
                options = categorical_ranges[col]
                index = options.index(saved) if saved in options else 0
                if saved in options:
                    prefilled_fields.append(col)
                field_values[col] = target.selectbox(col, options, index=index, key=f"{model_name}_{col}")
            else:
                if saved is not None:
                    prefilled_fields.append(col)
                field_values[col] = target.number_input(
                    FIELD_LABELS.get(col, col), value=float(saved) if saved is not None else None,
                    key=f"{model_name}_{col}",
                )

        if prefilled_fields:
            st.info(f"📋 Using your saved info: {', '.join(prefilled_fields)}")
        submitted = st.form_submit_button("🔮 Predict")

    if not submitted:
        return None

    inputs = {}
    missing_fields = []
    for col in feature_cols:
        if col in grouped_cols or col in derived_cols or col in bmi_related_cols or col in hidden_default_fields:
            continue
        val = field_values.get(col)
        if val is None:
            missing_fields.append(col)
            continue
        inputs[col] = val

    if bmi_config and bmi_value is None:
        missing_fields.append(bmi_config["bmi_col"])

    # A blank numeric field silently defaulting to 0 is NOT a neutral "unknown" value for
    # these models — 0 is often the single most extreme possible reading (e.g. 0% of RDA
    # vitamin C intake, 0 g/dL hemoglobin, 0 mg/dL blood glucose), so it previously produced
    # wildly wrong, overconfident predictions (e.g. 98% anemia risk from only entering age;
    # near-100% "Scurvy" from otherwise-blank intake fields). Refuse to predict rather than
    # silently guessing when any of these lab/measurement fields are left blank.
    if missing_fields:
        friendly = [FIELD_LABELS.get(c, c.replace("_", " ")) for c in missing_fields]
        st.error(
            f"🚫 Prediction blocked — please fill in: {', '.join(friendly)}. These are lab/measurement "
            "values, and leaving them blank would default to 0, which is an extreme (not neutral) value "
            "for this model and would produce an unreliable, misleading result."
        )
        return None

    for group_name, group_cols in onehot_groups.items():
        chosen_col = f"{group_name}_{group_choice[group_name]}"
        for c in group_cols:
            inputs[c] = 1 if c == chosen_col else 0

    for lab_col, measured_col in measured_pairs.items():
        inputs[measured_col] = 1 if lab_col in inputs else 0
        inputs.setdefault(lab_col, 0.0)

    if bmi_config:
        inputs[bmi_config["bmi_col"]] = bmi_value
        if "weight_col" in bmi_config:
            inputs[bmi_config["weight_col"]] = weight_kg
        if "height_col" in bmi_config:
            inputs[bmi_config["height_col"]] = height_cm
        st.session_state.user_profile["Height_cm"] = height_cm
        st.session_state.user_profile["Weight_kg"] = weight_kg

    for col, default_value in hidden_default_fields.items():
        inputs[col] = default_value

    return model_name, inputs


# Not disease-risk assessments — they're recommendation models with their own
# dedicated entry points (personalized diet request flow / advisory track), so they
# don't belong in a "select a condition to assess" disease picker.
GENERIC_FORM_EXCLUDED_MODELS = {"supplement", "diet_recommendation"}


def render_generic_form():
    registry = get_registry_safe()
    condition_options = sorted(set(registry.keys()) - GENERIC_FORM_EXCLUDED_MODELS)
    if not condition_options:
        st.warning("No trained models available yet.")
        return None

    st.subheader("🧾 General Health Assessment")
    st.caption("Select a condition below to open its dedicated assessment form.")
    selected_model = st.selectbox(
        "Select a condition to assess:", condition_options,
        index=None, placeholder="Choose a condition...", key="generic_model_select",
    )
    if selected_model is None:
        return None
    return render_prediction_form(selected_model)


def run_prediction_flow(disease):
    if disease is None:
        result = render_generic_form()
    else:
        model_name = DISEASE_TO_MODEL.get(disease, disease)
        result = render_prediction_form(model_name)

    if result is None:
        return

    model_name, inputs = result
    st.session_state.user_profile.update(inputs)
    registry = get_registry_safe()
    if model_name not in registry:
        st.warning(f"No trained model available for '{model_name}' yet.")
        return

    try:
        prediction = predict_health(model_name, inputs)
    except Exception as e:
        st.warning(f"Prediction failed for '{model_name}': {e}")
        return

    st.success(f"**Prediction:** {prediction['prediction_label']}")
    st.progress(prediction["confidence"], text=f"Confidence: {prediction['confidence']*100:.1f}%")

    risk_pct = get_prediction_risk_probability(model_name, prediction)
    col_donut, col_gauge = st.columns([1, 1])
    with col_donut:
        st.plotly_chart(
            prediction_donut_chart(prediction["prediction_label"], prediction["confidence"]),
            use_container_width=True,
        )
    with col_gauge:
        st.plotly_chart(risk_gauge(risk_pct), use_container_width=True)

    try:
        model_obj, _ = load_model_and_metadata(model_name)
    except Exception:
        model_obj = None

    if model_obj is not None and hasattr(model_obj, "feature_importances_"):
        st.session_state.sidebar_feature_importance = {
            "model": model_name,
            "importances": model_obj.feature_importances_.tolist(),
            "feature_names": prediction["feature_names"],
        }

    with st.expander("🔬 SHAP Explanation"):
        try:
            feature_values = [prediction["feature_values"][c] for c in prediction["feature_names"]]
            shap_dict = get_shap_explanation(
                model_obj, feature_values, prediction["feature_names"], class_index=prediction["prediction"]
            ) if model_obj else None
        except Exception:
            shap_dict = None
        if shap_dict:
            for feat, val in sorted(shap_dict.items(), key=lambda x: -abs(x[1])):
                st.write(f"- **{feat}**: {val:+.4f}")
            st.plotly_chart(
                shap_waterfall_chart(shap_dict, list(shap_dict.keys()), base_value=0.0),
                use_container_width=True,
            )
        else:
            st.caption("SHAP explanation unavailable for this model/input.")

    st.markdown("---")
    render_prediction_verdict(model_name, prediction, risk_pct)


def render_diet_recommendation_flow():
    st.info(
        "🥗 Personalized diet request detected — fill in your health details for a tailored "
        "recommendation, backed by the diet_recommendation model."
    )
    result = render_prediction_form("diet_recommendation")
    if result is None:
        return

    model_name, inputs = result
    st.session_state.user_profile.update(inputs)
    registry = get_registry_safe()
    if model_name not in registry:
        st.warning(f"No trained model available for '{model_name}' yet.")
        return

    try:
        prediction = predict_health(model_name, inputs)
    except Exception as e:
        st.warning(f"Prediction failed for '{model_name}': {e}")
        return

    st.success(f"**Recommended Diet:** {prediction['prediction_label'].replace('_', ' ')}")
    st.progress(prediction["confidence"], text=f"Confidence: {prediction['confidence']*100:.1f}%")

    try:
        model_obj, _ = load_model_and_metadata(model_name)
    except Exception:
        model_obj = None

    with st.expander("🔬 SHAP Explanation"):
        try:
            feature_values = [prediction["feature_values"][c] for c in prediction["feature_names"]]
            shap_dict = get_shap_explanation(
                model_obj, feature_values, prediction["feature_names"], class_index=prediction["prediction"]
            ) if model_obj else None
        except Exception:
            shap_dict = None
        if shap_dict:
            for feat, val in sorted(shap_dict.items(), key=lambda x: -abs(x[1])):
                st.write(f"- **{feat}**: {val:+.4f}")
            st.plotly_chart(
                shap_waterfall_chart(shap_dict, list(shap_dict.keys()), base_value=0.0),
                use_container_width=True,
            )
        else:
            st.caption("SHAP explanation unavailable for this model/input.")

    doc_id = DIET_LABEL_TO_DOC_ID.get(prediction["prediction_label"])
    if not doc_id:
        st.warning(f"No matching diet plan document mapped for label '{prediction['prediction_label']}'.")
        return

    index, _ = get_bm25_index()
    doc_meta = index.doc_metadata.get(doc_id)
    if not doc_meta:
        st.warning(f"Matching diet plan document '{doc_id}' not found in corpus.")
        return

    st.markdown("---")
    st.markdown("### 📖 Matching Diet Plan")
    render_result_card(doc_meta["title"], doc_meta["content"], meta=f"Doc: {doc_id}")

    st.markdown("---")
    risk_pct = get_prediction_risk_probability(model_name, prediction)
    render_prediction_verdict(model_name, prediction, risk_pct)


def render_metric_box(value, label):
    st.markdown(
        f'<div class="nq-metric-box"><div class="nq-metric-value">{value}</div>'
        f'<div class="nq-metric-label">{html.escape(str(label))}</div></div>',
        unsafe_allow_html=True,
    )


def load_eval_results():
    try:
        with open(os.path.join(MODELS_DIR, "eval_results.json")) as f:
            return json.load(f)
    except Exception:
        return None


def get_sample_advisory_results(sample_query="miracle bleach detox cure for viruses", top_k=8):
    try:
        index, _ = get_bm25_index()
        vector_index = get_vector_index()
        search_results = hybrid_search(sample_query, index, vector_index, top_k=top_k)
        scored = rank_by_trust(score_documents(sample_query, search_results, index))
        return apply_guardrail(scored, threshold=ADVISORY_TRUST_THRESHOLD)
    except Exception:
        return [], []


def build_scatter_data(sample_query="immune boosting foods rich in zinc and vitamin c", top_k=10):
    passed, rejected = get_sample_advisory_results(sample_query, top_k=top_k)
    rows = []
    for doc in passed + rejected:
        f = doc["features"]
        rows.append({
            "doc_id": doc["doc_id"], "title": doc["title"],
            "bm25_score": f["bm25_score"], "tfidf_cosine": f["tfidf_cosine"],
            "trust_score": doc["trust_score"], "is_rejected": doc["trust_score"] < ADVISORY_TRUST_THRESHOLD,
        })
    return rows


def run_full_pipeline(query):
    try:
        guard_result = run_query_guard(query)
    except Exception:
        guard_result = {"allowed": True}

    if not guard_result.get("allowed", True):
        return {
            "status": "REJECTED_BY_GUARD",
            "reason": guard_result.get("message", "Blocked by guardrail."),
            "guard_reason": guard_result.get("reason"),
        }

    try:
        intent = classify_intent(query)
    except Exception:
        intent = "factual"

    if intent == "prediction":
        try:
            disease = detect_disease_context(query)
        except Exception:
            disease = None
        return {"status": "SUCCESS", "intent": "prediction", "disease": disease}

    if intent == "advisory" and is_personalized_diet_request(query):
        return {"status": "SUCCESS", "intent": "diet_form"}

    # A query that explicitly asks for a recipe wants recipes, not a tangential
    # definition/fact card that happens to score well on shared vocabulary (e.g.
    # "vitamin c recipes" pulling in the "Vitamin C" fact sheet itself) — "recipi"
    # also catches the common misspelling "recipies". Cast a wider net (top_k=15
    # instead of 5) so purpose-built recipes aren't crowded out of the candidate
    # pool by the much larger pool of generic, uncurated recipes.
    is_recipe_request = intent == "advisory" and any(
        k in query.lower() for k in
        ("recipe", "recipi", "milkshake", "smoothie", "how to make", "how to cook")
    )
    # A recipe genuinely rich in a named nutrient (e.g. vitamin B) doesn't
    # necessarily rank in the top ~15 by semantic similarity — "vitamin b" as a
    # query doesn't embed meaningfully differently from "vitamin c" or "vitamin
    # a", so the actually-relevant recipes can sit far down the ranking. Cast a
    # much wider net when a specific nutrient is named so the strict content
    # filter below has real candidates to work with, not just whatever the
    # semantic ranking happened to prefer.
    recipe_nutrient_terms = (
        [term for term in NUTRIENT_QUERY_TERMS if term in query.lower()] if is_recipe_request else []
    )
    retrieval_query = normalize_query_spelling(query)

    try:
        index, _ = get_bm25_index()
        vector_index = get_vector_index()
        if intent == "factual":
            top_k = 20
        elif recipe_nutrient_terms:
            top_k = 300
        elif is_recipe_request:
            top_k = 15
        else:
            top_k = 5
        search_results = hybrid_search(retrieval_query, index, vector_index, top_k=top_k)
    except Exception as e:
        return {
            "status": "SUCCESS", "intent": intent, "results": [], "rejected_results": [],
            "error": f"{type(e).__name__}: {e}",
        }

    if is_recipe_request:
        search_results = [
            r for r in search_results
            if index.doc_metadata.get(r["doc_id"], {}).get("category") == "Recipes"
        ]
        # A recipe can rank well for "vitamin b" purely on semantic similarity to
        # generic "vitamin X: ... (notably rich)" phrasing even when it has zero
        # actual vitamin B content — the fix is a literal, lexical requirement
        # that the specific nutrient be named in the recipe's content, not just
        # "vitamin-shaped" content in general.
        if recipe_nutrient_terms:
            nutrient_filtered = [
                r for r in search_results
                if any(
                    term in index.doc_metadata.get(r["doc_id"], {}).get("content", "").lower()
                    for term in recipe_nutrient_terms
                )
            ]
            if nutrient_filtered:
                search_results = nutrient_filtered

    if intent == "factual":
        # A recipe is never a valid answer to a definitional "what is X" query —
        # it's a dish, not information about the food itself — so skip past any
        # recipe results to find the best genuine factual/reference document.
        # Re-rank by raw semantic similarity rather than the BM25-heavy combined
        # score: a lexically-overlapping-but-off-topic doc (e.g. "Scurvy" for a
        # vitamin C query) can outrank the true best answer on keyword overlap
        # alone, even though the true answer is more semantically relevant.
        non_recipe = [
            r for r in search_results
            if index.doc_metadata.get(r["doc_id"], {}).get("category") != "Recipes"
        ]
        non_recipe.sort(key=lambda r: r["vector_score"], reverse=True)
        comparison = is_comparison_query(query)
        # A comparison query ("egg or chicken", "milk vs paneer") is asking about two
        # things at once — truncating to a single top result silently drops whichever
        # side scored lower, answering as though only one item was ever named.
        factual_results = non_recipe[:2] if comparison else non_recipe[:1]

        # If the query directly names a specific vitamin/mineral, that document
        # wins outright over whatever the generic vector-score ranking preferred —
        # semantic similarity alone is unreliable for disambiguating "vitamin a
        # deficiency" from the broader Vitamin Deficiency Overview or unrelated
        # docs that just happen to share more vocabulary. Skipped for comparison
        # queries, which already keep multiple results instead of collapsing to one.
        if not comparison:
            named_doc_id = find_named_vitamin_mineral_doc(query, index) or find_named_diet_doc(query)
            if named_doc_id:
                named_match = next((r for r in non_recipe if r["doc_id"] == named_doc_id), None)
                if named_match:
                    factual_results = [named_match]

        # A "vitamin/mineral deficiency" query is really asking about the deficiency
        # DISEASE it causes (e.g. vitamin D deficiency -> Rickets), not just the
        # nutrient's general info page. Surface the linked disease doc too if it's
        # a genuine match, not just whichever the vector score happened to prefer.
        if factual_results and "deficien" in query.lower():
            top_category = index.doc_metadata.get(factual_results[0]["doc_id"], {}).get("category")
            if top_category == "Vitamins & Minerals":
                linked_disease = next(
                    (r for r in non_recipe[1:]
                     if index.doc_metadata.get(r["doc_id"], {}).get("category") == "Deficiency Diseases"
                     and r["doc_id"] != "dis_vitamin_deficiency_overview"
                     and r["vector_score"] >= FACTUAL_RELEVANCE_THRESHOLD),
                    None,
                )
                if linked_disease:
                    factual_results.append(linked_disease)

        return {"status": "SUCCESS", "intent": "factual", "results": factual_results, "rejected_results": []}

    scored = rank_by_trust(score_documents(retrieval_query, search_results, index))
    passed, rejected = apply_guardrail(scored, threshold=ADVISORY_TRUST_THRESHOLD)
    return {"status": "SUCCESS", "intent": "advisory", "results": passed, "rejected_results": rejected}


def render_qa_pipeline_view():
    render_section_header(
        "🧠", "Intelligent Question-Answering Pipeline",
        "Ask questions on immunonutrition, vitamin efficacy, or dietary supplements. "
        "Real BM25 + MiniLM hybrid retrieval with heuristic trust scoring and XGBoost disease prediction.",
        theme="teal",
    )

    st.markdown("**Quick Test Prompts:**")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    col_p1.button("🌿 Vitamin D & T-Cells", on_click=set_query_prompt,
                  args=("What is the role of Vitamin D in T-cell regulation and immunity?",))
    col_p2.button("🍋 Zinc & Vitamin C Synergy", on_click=set_query_prompt,
                  args=("Can Zinc and Vitamin C work together for immunity?",))
    col_p3.button("📘 What is RDA of Vitamin C?", on_click=set_query_prompt,
                  args=("What is the recommended daily intake of Vitamin C?",))
    col_p4.button("⚠️ Test Guardrail Trigger", on_click=set_query_prompt,
                  args=("How to use bleach detox to kill viruses?",))

    st.session_state.setdefault("query_input", "What is the role of Vitamin D in respiratory immunity?")
    with st.container(key="query_bar_container"):
        user_query = st.text_input("Enter your nutrition/immunology query:", key="query_input")
    search_triggered = st.button("🚀 Process Query", type="primary")

    if not (user_query or search_triggered):
        return

    if user_query != st.session_state.last_query:
        st.session_state.show_more_clicked = False
    st.session_state.last_query = user_query

    st.markdown("---")
    pipeline_output = run_full_pipeline(user_query)

    if pipeline_output["status"] == "REJECTED_BY_GUARD":
        if pipeline_output.get("guard_reason") == "self_harm":
            st.error("💙 **You're not alone — please reach out for support.**")
            st.markdown(pipeline_output["reason"])
        else:
            st.error(f"🛑 Query Blocked by Safety Guardrail — {pipeline_output['reason']}")
            st.caption("NUTRIQUIK guardrails automatically intercept unsafe, toxic, or out-of-domain requests.")
        return

    intent = pipeline_output["intent"]
    intent_label = "Personalized Diet" if intent == "diet_form" else intent.title()
    st.success(f"✅ Query Guard: Passed | Router Intent Classification: **{intent_label} Track**")

    if intent == "prediction":
        st.subheader("🧪 Health Prediction Track")
        disease = pipeline_output.get("disease")
        st.caption(f"Detected disease context: **{disease or 'None (general form)'}**")
        run_prediction_flow(disease)
        return

    if intent == "diet_form":
        st.subheader("🥗 Personalized Diet Recommendation")
        render_diet_recommendation_flow()
        return

    results = pipeline_output["results"]
    rejected_results = pipeline_output["rejected_results"]

    if pipeline_output.get("error"):
        st.error(f"⚠️ Retrieval pipeline failed: {pipeline_output['error']}")

    if intent == "factual":
        st.subheader("📌 Factual Query Result (BM25 + MiniLM Hybrid Retrieval)")
        st.info("ℹ️ Factual queries return the top-1 verified reference. 'Show More' ranking is disabled for direct definitions.")
        NOT_AVAILABLE_MSG = "😕 This data is not currently available in our dataset. Try a different nutrition or health topic."
        if not results:
            st.warning(NOT_AVAILABLE_MSG)
            return

        top = results[0]
        if top["vector_score"] < FACTUAL_RELEVANCE_THRESHOLD:
            st.warning(NOT_AVAILABLE_MSG)
            return

        index, _ = get_bm25_index()
        doc_meta = index.doc_metadata.get(top["doc_id"], {})
        col_data, col_text = st.columns([1, 3])
        with col_data:
            st.markdown("**📄 Source Data**")
            st.write(f"**Category:** {doc_meta.get('category', 'N/A')}")
            st.write(f"**Doc ID:** {top['doc_id']}")
            st.write(f"**BM25 score:** {top['bm25_score']:.2f}")
            st.write(f"**Semantic score:** {top['vector_score']:.2f}")
            keywords = doc_meta.get("keywords", "")
            if keywords:
                st.write(f"**Keywords:** {keywords}")
        with col_text:
            original_text = top["content"]
            raw_text = original_text
            ai_refined = False
            try:
                with st.spinner("Refining with AI..."):
                    refined = refine_answer(original_text, user_query)
                if refined and refined.strip() and refined != original_text:
                    raw_text = refined
                    ai_refined = True
            except Exception:
                pass
            if not ai_refined:
                raw_text = (
                    f"{raw_text}\n\n💡 In summary: this is general nutrition/health information — "
                    "consult a registered dietitian or healthcare provider for personalized advice."
                )
            render_result_card(top["title"], raw_text, meta=f"Doc: {top['doc_id']}")

        if len(results) > 1:
            linked = results[1]
            st.markdown("### 🔗 Related Deficiency Disease")
            render_result_card(linked["title"], linked["content"], meta=f"Doc: {linked['doc_id']}")
        return

    # ADVISORY
    st.subheader("💡 Advisory Query Response (Hybrid Retrieval + Trust Scoring Track)")
    if not results:
        st.warning(f"No results passed the {ADVISORY_TRUST_THRESHOLD*100:.0f}% composite trust threshold for this query.")
        return

    top = results[0]
    st.markdown("### 🏆 Top-1 Recommended Advisory Answer")
    focused_answer = extract_symptom_focused_answer(user_query, top["content"])
    if focused_answer is None:
        st.warning(
            f"😕 We don't have deficiency symptoms specifically listed for **{top['title']}** yet. "
            "Let me know and I can add it."
        )
        focused_answer = top["content"]
    col_top1, col_top2 = st.columns([3, 1])
    with col_top1:
        render_result_card(
            top["title"], focused_answer, trust_score=top["trust_score"],
            meta=(f"BM25: {top['features']['bm25_score']:.2f} | "
                  f"Semantic: {top['features']['vector_score']:.2f}"),
        )
        with st.expander("🔬 View Trust Score Feature Contribution Breakdown"):
            fig_shap = create_shap_chart(top["contributions"], top["title"], chart_title="Trust Contribution")
            st.plotly_chart(fig_shap, use_container_width=True)
    with col_top2:
        st.plotly_chart(create_trust_gauge(top["trust_score"], "Composite Trust %"), use_container_width=True)
        if st.button("✨ Summarize with AI (Gemini)", key="btn_gemini_sum"):
            try:
                with st.spinner("Summarizing..."):
                    st.info(summarize_results(results, user_query))
            except Exception as e:
                st.warning(f"Summarize failed: {e}")

    st.markdown("---")
    st.button("🔍 Show more results", on_click=reveal_more_results)

    if st.session_state.show_more_clicked:
        st.subheader("📚 Ranked Search Results (Sorted by Trust Score High → Low)")
        st.caption(f"Only documents meeting the >= {ADVISORY_TRUST_THRESHOLD:.2f} trust threshold are displayed below.")
        st.plotly_chart(create_trust_comparison_chart(results, rejected_results), use_container_width=True)
        for idx, doc in enumerate(results[1:], start=2):
            render_result_card(f"Rank #{idx}: {doc['title']}", doc["content"], trust_score=doc["trust_score"])

        if rejected_results:
            with st.expander(f"🚫 Filtered out ({len(rejected_results)} results below {ADVISORY_TRUST_THRESHOLD*100:.0f}% trust)"):
                for doc in rejected_results:
                    render_result_card(doc["title"], doc["reason"], trust_score=doc["trust_score"])


def render_evaluation_view():
    render_section_header(
        "📊", "Information Retrieval & Trust Analytics Dashboard",
        "Real evaluation on the 20-query ground-truth set: BM25-only vs BM25+MiniLM hybrid retrieval, "
        "plus XGBoost disease-model metrics.",
        theme="purple",
    )

    eval_results = load_eval_results()
    if not eval_results:
        st.warning("models/eval_results.json not found. Run `python src/ir/evaluate.py` to generate it.")
        return

    hybrid_ir = eval_results.get("ir_evaluation_hybrid", {})
    bm25_ir = eval_results.get("ir_evaluation_bm25", {})

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        render_metric_box(f"{hybrid_ir.get('Precision@5', 0):.2f}", "Precision@5 (Hybrid)")
    with col_m2:
        render_metric_box(f"{hybrid_ir.get('Recall@5', 0):.2f}", "Recall@5 (Hybrid)")
    with col_m3:
        render_metric_box(f"{hybrid_ir.get('MAP', 0):.2f}", "Mean Avg Precision (MAP)")
    with col_m4:
        render_metric_box(f"{hybrid_ir.get('NDCG@5', 0):.2f}", "NDCG@5 Score")

    st.caption(
        f"BM25-only baseline — MAP: {bm25_ir.get('MAP', 0):.2f}, NDCG@5: {bm25_ir.get('NDCG@5', 0):.2f} "
        "(hybrid retrieval improves both)."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col_ch1, col_ch2 = st.columns(2)
    with col_ch1:
        st.plotly_chart(create_ir_metrics_chart(hybrid_ir), use_container_width=True)
    with col_ch2:
        st.plotly_chart(create_evidence_pyramid(), use_container_width=True)

    ml_results = eval_results.get("ml_evaluation", [])
    if ml_results:
        st.markdown("---")
        st.subheader("🧬 XGBoost Disease Model Evaluation")
        st.plotly_chart(create_ml_metrics_chart(pd.DataFrame(ml_results)), use_container_width=True)

    st.markdown("---")
    st.subheader("🗺️ Retrieval Score Mapping (BM25 vs TF-IDF Cosine vs Composite Trust)")
    scatter_rows = build_scatter_data()
    if scatter_rows:
        st.plotly_chart(create_retrieval_scatter(scatter_rows), use_container_width=True)
    else:
        st.info("Retrieval scatter unavailable.")


def render_corpus_view():
    render_section_header(
        "📚", "Curated Corpus & Dataset Inventory",
        "Real IR corpus (BM25 + MiniLM index) and the datasets used to train the 7 XGBoost disease/nutrition models.",
        theme="amber",
    )

    _, documents = get_bm25_index()
    df_corpus = pd.DataFrame(documents)
    st.dataframe(df_corpus[["id", "category", "title", "keywords"]], use_container_width=True)
    st.caption(f"{len(df_corpus)} documents across {df_corpus['category'].nunique()} categories.")

    st.subheader("📥 Download Corpus")
    st.download_button(
        "📄 Download ir_corpus_merged.csv",
        df_corpus.to_csv(index=False).encode("utf-8"),
        "ir_corpus_merged.csv", "text/csv",
    )

    st.markdown("---")
    st.subheader("🧬 ML Model Registry")
    registry = get_registry_safe()
    if registry:
        st.dataframe(
            pd.DataFrame([
                {"model": k, "accuracy": v.get("accuracy"), "f1": v.get("f1"), "num_classes": v.get("num_classes")}
                for k, v in registry.items()
            ]),
            use_container_width=True, hide_index=True,
        )


def render_blueprint_view():
    render_section_header(
        "🏗️", "System Architecture Blueprint & Work Breakdown Structure",
        "Mapped to the actual implemented pipeline.",
        theme="blue",
    )

    st.subheader("🔄 System Architecture Flowchart")
    st.markdown(
        """
```
                              User Query
                                  |
                                  v
   Query Guard (regex blocklist + TF-IDF/LogReg domain classifier,
   zero-vocabulary fail-safe) ---> REJECTS harmful / off-domain
                                  |
                                  v
        Intent Router (rule-based: Factual / Advisory / Prediction)
            /                     |                      \\
     (Factual)               (Advisory)              (Prediction)
         |                        |                        |
 BM25 + MiniLM             BM25 + MiniLM          Disease Context Detector
 Hybrid, top-1             Hybrid, top-5                   |
         |                        |               XGBoost Model (9 models)
 Optional Gemini           Heuristic Trust Score    + real SHAP explanation
 LLM Refine                (9 weighted features)
         |                        |
 Show Answer + Source      Guardrail Filter (trust < 0.30 cut)
                                  |
                                  v
                    Top-1 shown first, optional "Show More" ranking
```
        """
    )

    st.markdown("---")
    st.subheader("📅 6-Week Work Breakdown Structure (WBS) & Roles")
    wbs_data = {
        "Phase": ["1. Requirements Analysis", "2. System Design", "3. Development", "4. Testing",
                  "5. Deployment", "6. Documentation"],
        "Activities": [
            "Dataset collection and annotation, csv, IR evaluation (Precision, Recall)",
            "Overall coordination, XGBoost trust-scoring model, model evaluation",
            "Streamlit UI, Plotly visualisations (trust bars(accuracy)), graphs, ranking results / "
            "Sparse retrieval (TF-IDF), dense retrieval (sentence-transformers), indexing",
            "System integration, guardrail filter logic, testing",
            "Cloud deployment",
            "Report writing, presentation preparation",
        ],
        "Lead Member": [
            "Pulkit Sukhija — Data & Evaluation Engineer",
            "Gunagya Agarwal — Project Lead & ML Engineer",
            "Suryansh Singh — Frontend/Backend & UI Developer / Mridul Rai — IR & Backend Support",
            "Pranav S Nair — Integration, Testing & Deployment",
            "Pranav S Nair — Integration, Testing & Deployment",
            "Gunagya Agarwal — Project Lead & ML Engineer",
        ],
    }
    st.table(pd.DataFrame(wbs_data))


# ------------------------------------------------------------------------------
# APP
# ------------------------------------------------------------------------------
_FAVICON_DATA_URI = "data:image/svg+xml," + urllib.parse.quote(_leaf_svg("leafGradientFavicon"))
st.set_page_config(page_title="NutriQuik", page_icon=_FAVICON_DATA_URI, layout="wide")
inject_custom_css()

if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "show_more_clicked" not in st.session_state:
    st.session_state.show_more_clicked = False
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}

render_sidebar_brand()
nav_choice = st.sidebar.radio(
    "Select Interface View:",
    [
        "🧠 QA Pipeline & Query Interface",
        "📊 Evaluation & Trust Analytics",
        "📚 Corpus & Dataset Inventory",
        "🏗️ System Blueprint & WBS Timeline",
    ],
)

# Retrieval indexes are only needed by views that actually search the corpus —
# skip the warm-up cost entirely for the static Blueprint page.
RETRIEVAL_VIEWS = {
    "🧠 QA Pipeline & Query Interface",
    "📊 Evaluation & Trust Analytics",
    "📚 Corpus & Dataset Inventory",
}
if nav_choice in RETRIEVAL_VIEWS:
    with st.spinner("🔄 Warming up retrieval & embedding models (first load may take a moment)..."):
        try:
            get_bm25_index()
            get_vector_index()
        except Exception:
            pass  # surfaced per-query instead, via run_full_pipeline's error banner

render_hero()

st.sidebar.markdown("---")
with st.sidebar.expander("🗂️ Your Session Profile"):
    if st.session_state.user_profile:
        for field, value in st.session_state.user_profile.items():
            st.write(f"- **{field}**: {value}")
    else:
        st.caption("No saved info yet — submit a prediction form to populate this.")
    if st.button("Clear profile"):
        st.session_state.user_profile = {}

st.sidebar.markdown("---")
st.sidebar.caption("🔎 Hybrid retrieval: BM25 (keyword) + MiniLM (semantic)")

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Feature Importance")
sidebar_importance = st.session_state.get("sidebar_feature_importance")
if sidebar_importance:
    st.sidebar.caption(f"Model: {sidebar_importance['model']}")
    st.sidebar.plotly_chart(
        feature_importance_chart(sidebar_importance["importances"], sidebar_importance["feature_names"]),
        use_container_width=True,
    )
else:
    st.sidebar.caption("Run a prediction to see feature importances.")

if nav_choice == "🧠 QA Pipeline & Query Interface":
    render_qa_pipeline_view()
elif nav_choice == "📊 Evaluation & Trust Analytics":
    render_evaluation_view()
elif nav_choice == "📚 Corpus & Dataset Inventory":
    render_corpus_view()
elif nav_choice == "🏗️ System Blueprint & WBS Timeline":
    render_blueprint_view()

render_footer()
