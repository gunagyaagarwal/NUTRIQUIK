import json
import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")

RANDOM_STATE = 42


MODEL_CONFIGS = [
    {
        "name": "anemia",
        "path": "data/ml_training/Anemia_train_augmented.csv",
        "target_col": "target",
    },
    {
        "name": "diabetes",
        "path": "data/ml_training/diabetes_train.csv",
        "target_col": "target",
        "subsample": {"neg_n": 5000},
    },
    {
        "name": "vitamin_deficiency",
        "path": "data/ml_training/vitamin_deficiency_train.csv",
        "target_col": "target",
    },
    {
        "name": "kidney",
        "path": "data/ml_training/chronic_kidney_disease.csv",
        "target_col": "Class",
    },
    {
        "name": "heart",
        "path": "data/ml_training/Heart_Disease_Prediction.csv",
        "target_col": "Heart Disease",
        "target_map": {"Absence": 0, "Presence": 1},
    },
    {
        "name": "supplement",
        "path": "data/ml_training/NutriGuard_training_data.csv",
        "target_col": "Primary_Benefit",
        "id_cols": ["ID"],
        "drop_cols": ["Weight_Change", "Weight_Change_Pct"],
    },
    {
        "name": "diet_recommendation",
        "path": "data/ml_training/diet_recommendations_final.csv",
        "target_col": "Diet_Label",
        "id_cols": ["Patient_ID"],
        "drop_cols": ["Diet_Recommendation", "Diet_Recommendation_v2"],
        "fillna_none_cols": ["Disease_Type", "Dietary_Restrictions", "Allergies"],
    },
]


def clean_missing_values(X):
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == object:
            mode = X[col].mode(dropna=True)
            fill_val = mode.iloc[0] if not mode.empty else ""
            X[col] = X[col].fillna(fill_val)
        else:
            X[col] = X[col].fillna(X[col].median())
    return X


def encode_categorical_features(X):
    X = X.copy()
    encoders = {}
    for col in X.columns:
        if X[col].dtype == object:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
    return X, encoders


def encode_target(y, explicit_map=None):
    if explicit_map is not None:
        y_encoded = y.map(explicit_map).astype(int)
        label_mapping = {str(k): int(v) for k, v in explicit_map.items()}
        return y_encoded.values, label_mapping

    if y.dtype == object:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))
        label_mapping = {str(cls): int(idx) for idx, cls in enumerate(le.classes_)}
        return y_encoded, label_mapping

    uniques = sorted(y.astype(int).unique())
    remap = {v: i for i, v in enumerate(uniques)}
    y_encoded = y.astype(int).map(remap).values
    label_mapping = {str(v): remap[v] for v in uniques}
    return y_encoded, label_mapping


def train_one_model(config):
    name = config["name"]
    rel_path = config["path"]
    full_path = os.path.join(BASE_DIR, rel_path)

    if not os.path.exists(full_path):
        print(f"[SKIP] {name}: missing file '{rel_path}'")
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
        missing = set(config["feature_cols"]) - set(feature_cols)
        if missing:
            print(f"[WARN] {name}: expected feature columns not found: {missing}")
        X = df[feature_cols].copy()
    else:
        cols_to_drop = set(drop_cols) | {target_col}
        X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()

    y_raw = df[target_col]

    X = clean_missing_values(X)
    X, encoders = encode_categorical_features(X)
    y_encoded, label_mapping = encode_target(y_raw, explicit_map=config.get("target_map"))
    num_classes = len(label_mapping)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=RANDOM_STATE
    )

    params = dict(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    if num_classes == 2:
        neg_count = int((y_train == 0).sum())
        pos_count = int((y_train == 1).sum())
        params["scale_pos_weight"] = neg_count / pos_count if pos_count > 0 else 1.0
    else:
        params["objective"] = "multi:softprob"
        params["num_class"] = num_classes

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="binary" if num_classes == 2 else "weighted"))
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\n=== {name.upper()} ===")
    print(f"Rows used: {len(df)} | Classes: {num_classes}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1: {f1:.4f}")
    print(report)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_rel_path = f"models/{name}_model.pkl"
    metadata_rel_path = f"models/{name}_metadata.pkl"
    joblib.dump(model, os.path.join(BASE_DIR, model_rel_path))
    joblib.dump(
        {
            "feature_cols": list(X.columns),
            "label_mapping": label_mapping,
            "encoders": encoders,
            "accuracy": accuracy,
            "f1": f1,
        },
        os.path.join(BASE_DIR, metadata_rel_path),
    )

    return {
        "accuracy": accuracy,
        "f1": f1,
        "num_classes": num_classes,
        "feature_columns": list(X.columns),
        "label_mapping": label_mapping,
        "file_path": model_rel_path,
        "rows": len(df),
    }


def train_all_models():
    registry = {}
    rows_used = {}

    for config in MODEL_CONFIGS:
        result = train_one_model(config)
        if result is None:
            continue
        name = config["name"]
        rows_used[name] = result.pop("rows")
        registry[name] = result

    registry_path = os.path.join(MODELS_DIR, "model_registry.json")
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print("\n=== Summary ===")
    print(f"{'Model':<20}{'Accuracy':<12}{'F1':<12}{'Classes':<10}{'Rows':<10}")
    for name, entry in registry.items():
        print(
            f"{name:<20}{entry['accuracy']:<12.4f}{entry['f1']:<12.4f}"
            f"{entry['num_classes']:<10}{rows_used[name]:<10}"
        )

    print(f"\nSaved registry to: {registry_path}")
    return registry


if __name__ == "__main__":
    train_all_models()
