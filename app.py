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

# diet_recommendation's predicted Diet_Label -> the matching curated plan doc in the IR corpus.
DIET_LABEL_TO_DOC_ID = {
    "High_Protein": "diet_high_protein",
    "Low_Fat": "diet_low_fat",
    "Low_Carb": "diet_low_carb",
    "Low_Sodium": "diet_low_sodium",
    "High_Fiber": "diet_high_fiber",
    "Balanced": "diet_balanced",
}

PERSONALIZED_DIET_PHRASES = ["diet plan", "custom", "for me", "rich in", "high in", "low in"]
NUTRIENT_GOAL_WORDS = ["protein", "fat", "fibre", "fiber", "sodium", "carb", "vitamin", "weight loss"]
_MORE_LESS_PATTERN = re.compile(r"\b(more|less)\s+\w+")


def is_personalized_diet_request(query):
    query_lower = query.lower()
    if any(phrase in query_lower for phrase in PERSONALIZED_DIET_PHRASES):
        return True
    if any(word in query_lower for word in NUTRIENT_GOAL_WORDS):
        return True
    return bool(_MORE_LESS_PATTERN.search(query_lower))


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
        .nq-section-title {
            color: #FFFFFF;
            font-size: 2.05rem;
            font-weight: 800;
            margin: 0.7rem 0 0.55rem;
        }
        .nq-muted {
            color: #9CA3AF;
            font-size: 0.94rem;
            margin-bottom: 1.35rem;
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
    skip_cols = grouped_cols | derived_cols | bmi_related_cols

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
                    col, value=float(saved) if saved is not None else None, key=f"{model_name}_{col}",
                )

        if prefilled_fields:
            st.info(f"📋 Using your saved info: {', '.join(prefilled_fields)}")
        submitted = st.form_submit_button("🔮 Predict")

    if not submitted:
        return None

    inputs = {}
    missing_fields = []
    for col in feature_cols:
        if col in grouped_cols or col in derived_cols or col in bmi_related_cols:
            continue
        val = field_values.get(col)
        if val is None:
            missing_fields.append(col)
            continue
        inputs[col] = val

    for group_name, group_cols in onehot_groups.items():
        chosen_col = f"{group_name}_{group_choice[group_name]}"
        for c in group_cols:
            inputs[c] = 1 if c == chosen_col else 0

    for lab_col, measured_col in measured_pairs.items():
        inputs[measured_col] = 1 if lab_col in inputs else 0
        inputs.setdefault(lab_col, 0.0)

    if bmi_config:
        if bmi_value is not None:
            inputs[bmi_config["bmi_col"]] = bmi_value
        else:
            missing_fields.append(bmi_config["bmi_col"])
            inputs[bmi_config["bmi_col"]] = 0.0
        if "weight_col" in bmi_config:
            inputs[bmi_config["weight_col"]] = weight_kg if weight_kg else 0.0
        if "height_col" in bmi_config:
            inputs[bmi_config["height_col"]] = height_cm if height_cm else 0.0
        st.session_state.user_profile["Height_cm"] = height_cm
        st.session_state.user_profile["Weight_kg"] = weight_kg

    if missing_fields:
        st.warning(
            f"⚠️ Left blank: {', '.join(missing_fields)} — treated as 0, prediction confidence may be reduced."
        )

    return model_name, inputs


def render_generic_form():
    registry = get_registry_safe()
    if not registry:
        st.warning("No trained models available yet.")
        return None

    st.subheader("🧾 General Health Assessment")
    st.caption("Select a condition below to open its dedicated assessment form.")
    selected_model = st.selectbox(
        "Select a condition to assess:", sorted(registry.keys()),
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

    st.caption("📈 Chart placeholders (probability distribution, feature trends) — added in a later pass.")


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

    try:
        index, _ = get_bm25_index()
        vector_index = get_vector_index()
        top_k = 8 if intent == "factual" else 5
        search_results = hybrid_search(query, index, vector_index, top_k=top_k)
    except Exception as e:
        return {
            "status": "SUCCESS", "intent": intent, "results": [], "rejected_results": [],
            "error": f"{type(e).__name__}: {e}",
        }

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
        factual_results = non_recipe[:1]
        return {"status": "SUCCESS", "intent": "factual", "results": factual_results, "rejected_results": []}

    scored = rank_by_trust(score_documents(query, search_results, index))
    passed, rejected = apply_guardrail(scored, threshold=ADVISORY_TRUST_THRESHOLD)
    return {"status": "SUCCESS", "intent": "advisory", "results": passed, "rejected_results": rejected}


def render_qa_pipeline_view(ai_refine_enabled):
    st.header("🔍 Intelligent Question-Answering Pipeline")
    st.caption(
        "Ask questions on immunonutrition, vitamin efficacy, or dietary supplements. "
        "Real BM25 + MiniLM hybrid retrieval with heuristic trust scoring and XGBoost disease prediction."
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
            raw_text = top["content_snippet"]
            if ai_refine_enabled:
                try:
                    with st.spinner("Refining with AI..."):
                        raw_text = refine_answer(raw_text, user_query)
                except Exception:
                    pass
            render_result_card(top["title"], raw_text, meta=f"Doc: {top['doc_id']}")
        return

    # ADVISORY
    st.subheader("💡 Advisory Query Response (Hybrid Retrieval + Trust Scoring Track)")
    if not results:
        st.warning(f"No results passed the {ADVISORY_TRUST_THRESHOLD*100:.0f}% composite trust threshold for this query.")
        return

    top = results[0]
    st.markdown("### 🏆 Top-1 Recommended Advisory Answer")
    col_top1, col_top2 = st.columns([3, 1])
    with col_top1:
        render_result_card(
            top["title"], top["content"][:500], trust_score=top["trust_score"],
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
            render_result_card(f"Rank #{idx}: {doc['title']}", doc["content"][:300], trust_score=doc["trust_score"])

        if rejected_results:
            with st.expander(f"🚫 Filtered out ({len(rejected_results)} results below {ADVISORY_TRUST_THRESHOLD*100:.0f}% trust)"):
                for doc in rejected_results:
                    render_result_card(doc["title"], doc["reason"], trust_score=doc["trust_score"])


def render_evaluation_view():
    st.header("📊 Information Retrieval & Trust Analytics Dashboard")
    st.caption(
        "Real evaluation on the 20-query ground-truth set: BM25-only vs BM25+MiniLM hybrid retrieval, "
        "plus XGBoost disease-model metrics."
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
    st.header("📚 Curated Corpus & Dataset Inventory")
    st.caption("Real IR corpus (BM25 + MiniLM index) and the datasets used to train the 9 XGBoost disease/nutrition models.")

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
    st.header("🏗️ System Architecture Blueprint & Work Breakdown Structure")
    st.caption("Mapped to the actual implemented pipeline.")

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
        "🔍 QA Pipeline & Query Interface",
        "📊 Evaluation & Trust Analytics",
        "📚 Corpus & Dataset Inventory",
        "🏗️ System Blueprint & WBS Timeline",
    ],
)

# Retrieval indexes are only needed by views that actually search the corpus —
# skip the warm-up cost entirely for the static Blueprint page.
RETRIEVAL_VIEWS = {
    "🔍 QA Pipeline & Query Interface",
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
ai_refine_enabled = st.sidebar.checkbox("Enable AI Refinement", value=False)
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

if nav_choice == "🔍 QA Pipeline & Query Interface":
    render_qa_pipeline_view(ai_refine_enabled)
elif nav_choice == "📊 Evaluation & Trust Analytics":
    render_evaluation_view()
elif nav_choice == "📚 Corpus & Dataset Inventory":
    render_corpus_view()
elif nav_choice == "🏗️ System Blueprint & WBS Timeline":
    render_blueprint_view()

render_footer()
