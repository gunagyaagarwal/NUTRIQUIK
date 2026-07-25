"""Prediction-form domain knowledge, ported verbatim from the former app.py
(Streamlit widget-building code) into a UI-framework-agnostic spec builder +
input encoder. No thresholds, encodings, or model behavior changed here — only
how the same field metadata is expressed (JSON-serializable dicts instead of
st.* widget calls) and how submitted values are turned into the same `inputs`
dict previously built by render_prediction_form's post-submit logic.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ml.predict import load_model_and_metadata, load_registry  # noqa: E402

DISEASE_TO_MODEL = {
    "diabetes": "diabetes", "anemia": "anemia", "heart": "heart",
    "kidney": "kidney", "vitamin_deficiency": "vitamin_deficiency",
    "supplement": "supplement", "weight": "weight", "diet_recommendation": "diet_recommendation",
}

# Not disease-risk assessments — they're recommendation models with their own
# dedicated entry points (personalized diet request flow / advisory track), so they
# don't belong in a "select a condition to assess" disease picker.
GENERIC_FORM_EXCLUDED_MODELS = {"supplement", "diet_recommendation"}

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

# "Sex" is a demographic category, not a yes/no clinical flag. Heart uses the
# standard UCI Heart Disease dataset convention (1=Male, 0=Female); anemia's
# encoding was inferred from its training data, where Sex=0 has the higher average
# hemoglobin, consistent with Sex=0 being Male.
SEX_FIELD_OPTIONS = {
    "anemia": ["Male", "Female"],
    "heart": ["Female", "Male"],
}
GENDER_OPTIONS = ["Female", "Male", "Other"]

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

# Above this risk/confidence level, the closing statement urges a doctor visit
# instead of the standard "be careful" advisory.
HIGH_RISK_THRESHOLD = 0.95

# Numeric fields a lay user can answer from self-knowledge, not a lab test/clinical
# measurement — everything else numeric defaults to "lab" tier (shown in the
# optional "lab values" expander instead of the main form).
_EASY_NUMERIC_FIELDS = {"Age", "age", "Weekly_Exercise_Hours", "symptoms_count"}

# Binary (Yes/No) fields that require a lab test rather than self-report, so they
# default-override out of the normal "easy" tier binary fields get.
_LAB_BINARY_OVERRIDES = {
    "heart": {"FBS over 120"},
}

# Notes surfaced once per model, verbatim from the equivalent Streamlit captions —
# not new claims, just carried over into the API response.
MODEL_NOTES = {
    "anemia": (
        "The anemia model's original training data was small (264 rows) and its labels were noisy — "
        "some clinically normal blood counts were marked anemic. It has since been retrained on the "
        "original data plus 550 clinically-realistic synthetic rows (added to correct the class balance "
        "and label consistency); treat predictions as illustrative rather than clinically validated."
    ),
}

# Per-field extra guidance beyond the label's own unit/range text — only added where
# there's genuine extra information (e.g. a documented near-zero feature importance
# found via this project's own SHAP/feature_importances_ analysis), not fabricated.
FIELD_HELP = {
    "Su": "This field has very low measured importance in the trained kidney model — "
          "unlikely to shift the prediction much either way.",
    "Rbc": "This field has very low measured importance in the trained kidney model — "
           "unlikely to shift the prediction much either way.",
    "Wbcc": "This field has very low measured importance in the trained kidney model — "
            "unlikely to shift the prediction much either way.",
}


def _onehot_option_label(prefix, col):
    return col[len(prefix) + 1:]


def get_registry_safe():
    try:
        return load_registry()
    except Exception:
        return {}


def build_field_spec(model_name):
    """Return {"disease": ..., "title": ..., "note": str|None, "fields": [...]}
    describing the form for this model, derived from its real trained feature_cols
    — the same source of truth render_prediction_form used, just serialized instead
    of turned into st.* widgets directly."""
    _, metadata = load_model_and_metadata(model_name)

    feature_cols = metadata["feature_cols"]
    encoders = metadata["encoders"]
    onehot_groups = ONE_HOT_GROUPS.get(model_name, {})
    binary_fields = set(BINARY_FIELDS.get(model_name, []))
    lab_binary_overrides = _LAB_BINARY_OVERRIDES.get(model_name, set())
    categorical_ranges = CATEGORICAL_RANGES.get(model_name, {})
    measured_pairs = MEASURED_FLAG_PAIRS.get(model_name, {})
    derived_cols = set(measured_pairs.values())
    grouped_cols = {c for cols in onehot_groups.values() for c in cols}
    bmi_config = BMI_FEATURE_CONFIG.get(model_name)
    bmi_related_cols = set(bmi_config.values()) if bmi_config else set()
    hidden_default_fields = HIDDEN_DEFAULT_FIELDS.get(model_name, {})
    skip_cols = grouped_cols | derived_cols | bmi_related_cols | set(hidden_default_fields)

    fields = []

    if bmi_config:
        if "height_col" in bmi_config or model_name in BMI_FEATURE_CONFIG:
            fields.append({
                "id": "Height_cm", "label": "Height (cm)", "type": "number",
                "tier": "easy", "help": "Used to compute your BMI live.",
                "min": 0, "unit": "cm",
            })
            fields.append({
                "id": "Weight_kg", "label": "Weight (kg)", "type": "number",
                "tier": "easy", "help": "Used to compute your BMI live.",
                "min": 0, "unit": "kg",
            })
        fields.append({
            "id": bmi_config["bmi_col"], "label": "BMI", "type": "derived",
            "tier": "derived", "help": "Computed automatically from height and weight.",
        })

    for group_name, group_cols in onehot_groups.items():
        options = [_onehot_option_label(group_name, c) for c in group_cols]
        fields.append({
            "id": f"__group__{group_name}", "label": group_name.replace("_", " ").title(),
            "type": "select", "options": options, "tier": "easy",
            "help": FIELD_HELP.get(group_name, ""),
        })

    for col in feature_cols:
        if col in skip_cols:
            continue

        if col == "Sex" and model_name in SEX_FIELD_OPTIONS:
            fields.append({
                "id": col, "label": "Gender", "type": "select",
                "options": SEX_FIELD_OPTIONS[model_name], "tier": "easy",
                "help": FIELD_HELP.get(col, ""),
            })
        elif col in encoders:
            fields.append({
                "id": col, "label": col.replace("_", " "), "type": "select",
                "options": list(encoders[col].classes_), "tier": "easy",
                "help": FIELD_HELP.get(col, ""),
            })
        elif col in binary_fields:
            tier = "lab" if col in lab_binary_overrides else "easy"
            fields.append({
                "id": col, "label": col.replace("_", " ").title(), "type": "select",
                "options": ["No", "Yes"], "tier": tier,
                "help": FIELD_HELP.get(col, ""),
            })
        elif col in categorical_ranges:
            fields.append({
                "id": col, "label": FIELD_LABELS.get(col, col), "type": "segment",
                "options": [str(v) for v in categorical_ranges[col]], "tier": "lab",
                "help": FIELD_HELP.get(col, ""),
            })
        else:
            tier = "easy" if col in _EASY_NUMERIC_FIELDS else "lab"
            fields.append({
                "id": col, "label": FIELD_LABELS.get(col, col.replace("_", " ")),
                "type": "number", "tier": tier, "help": FIELD_HELP.get(col, ""),
            })

    return {
        "disease": model_name,
        "title": model_name.replace("_", " ").title(),
        "note": MODEL_NOTES.get(model_name),
        "fields": fields,
    }


def encode_prediction_inputs(model_name, values):
    """Turn the submitted {field_id: value} dict into the exact `inputs` dict
    predict_health expects, replicating render_prediction_form's post-submit
    logic (one-hot construction, Sex/Gender resolution, hidden defaults, BMI
    injection, missing-field detection). Returns (inputs, missing_fields) — when
    missing_fields is non-empty, inputs is None and the caller should refuse to
    predict rather than silently defaulting blank lab values to 0 (0 is often the
    single most extreme possible reading for these models, e.g. 0% RDA intake,
    0 g/dL hemoglobin — not a neutral "unknown")."""
    _, metadata = load_model_and_metadata(model_name)
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

    field_values = {}
    missing_fields = []

    for col in feature_cols:
        if col in skip_cols:
            continue

        if col == "Sex" and model_name in SEX_FIELD_OPTIONS:
            options = SEX_FIELD_OPTIONS[model_name]
            gender = values.get(col)
            if gender not in options:
                missing_fields.append(col)
                continue
            field_values[col] = options.index(gender)
        elif col in encoders:
            classes = list(encoders[col].classes_)
            val = values.get(col)
            if val not in classes:
                missing_fields.append(col)
                continue
            field_values[col] = val
        elif col in binary_fields:
            choice = values.get(col)
            if choice not in ("No", "Yes"):
                missing_fields.append(col)
                continue
            field_values[col] = 1 if choice == "Yes" else 0
        elif col in categorical_ranges:
            options = categorical_ranges[col]
            raw = values.get(col)
            try:
                val = int(raw)
            except (TypeError, ValueError):
                val = None
            if val not in options:
                missing_fields.append(col)
                continue
            field_values[col] = val
        else:
            raw = values.get(col)
            if raw is None or raw == "":
                missing_fields.append(col)
                continue
            try:
                field_values[col] = float(raw)
            except (TypeError, ValueError):
                missing_fields.append(col)

    height_cm = weight_kg = bmi_value = None
    if bmi_config:
        try:
            height_cm = float(values.get("Height_cm"))
            weight_kg = float(values.get("Weight_kg"))
        except (TypeError, ValueError):
            height_cm = weight_kg = None
        if height_cm and height_cm > 0 and weight_kg and weight_kg > 0:
            bmi_value = weight_kg / ((height_cm / 100) ** 2)
        else:
            missing_fields.append(bmi_config["bmi_col"])

    for group_name, group_cols in onehot_groups.items():
        options = [_onehot_option_label(group_name, c) for c in group_cols]
        chosen = values.get(f"__group__{group_name}")
        if chosen not in options:
            missing_fields.append(group_name)
            continue
        chosen_col = f"{group_name}_{chosen}"
        for c in group_cols:
            field_values[c] = 1 if c == chosen_col else 0

    if missing_fields:
        return None, missing_fields

    inputs = dict(field_values)

    for lab_col, measured_col in measured_pairs.items():
        inputs[measured_col] = 1 if lab_col in inputs else 0
        inputs.setdefault(lab_col, 0.0)

    if bmi_config:
        inputs[bmi_config["bmi_col"]] = bmi_value
        if "weight_col" in bmi_config:
            inputs[bmi_config["weight_col"]] = weight_kg
        if "height_col" in bmi_config:
            inputs[bmi_config["height_col"]] = height_cm

    for col, default_value in hidden_default_fields.items():
        inputs[col] = default_value

    return inputs, []


def get_prediction_risk_probability(model_name, prediction):
    probabilities = prediction.get("all_probabilities", {})
    for positive_label in ("1", "Presence", "Positive", "Disease", "Yes"):
        if positive_label in probabilities:
            return probabilities[positive_label]
    if model_name in {"heart"} and "Absence" in probabilities:
        return 1.0 - probabilities["Absence"]
    return prediction["confidence"]
