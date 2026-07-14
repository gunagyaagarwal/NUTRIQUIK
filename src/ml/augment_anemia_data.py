import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORIGINAL_PATH = os.path.join(BASE_DIR, "data", "ml_training", "Anemia_train.csv")
AUGMENTED_PATH = os.path.join(BASE_DIR, "data", "ml_training", "Anemia_train_augmented.csv")

RANDOM_STATE = 42
N_HEALTHY = 300
N_ANEMIC = 250

COLUMNS = ["Age", "Sex", "RBC", "PCV", "MCV", "MCH", "MCHC", "RDW", "TLC", "PLT/mm3", "HGB", "target"]


def _clip(values, lo, hi):
    return np.clip(values, lo, hi)


def generate_healthy(rng, n):
    hgb = rng.uniform(12.0, 16.5, n)
    rbc = rng.uniform(4.2, 5.8, n)
    pcv = _clip(3.0 * hgb + rng.normal(0, 1.5, n), 33, 52)
    mcv = rng.uniform(80, 96, n)
    mch = rng.uniform(27, 33, n)
    mchc = rng.uniform(32, 36, n)
    rdw = rng.uniform(11.5, 14.5, n)
    tlc = rng.uniform(4.0, 11.0, n)
    plt = rng.uniform(150, 450, n)
    age = rng.uniform(18, 75, n)
    sex = rng.integers(0, 2, n)
    return pd.DataFrame({
        "Age": age.round(0), "Sex": sex, "RBC": rbc.round(2), "PCV": pcv.round(1),
        "MCV": mcv.round(1), "MCH": mch.round(1), "MCHC": mchc.round(1), "RDW": rdw.round(1),
        "TLC": tlc.round(2), "PLT/mm3": plt.round(0), "HGB": hgb.round(1), "target": 0,
    })


def generate_anemic(rng, n):
    hgb = rng.uniform(4.0, 11.9, n)
    is_microcytic = rng.random(n) < 0.4
    mcv = np.where(is_microcytic, rng.uniform(60, 79, n), rng.uniform(80, 96, n))
    mch = np.where(is_microcytic, rng.uniform(18, 26, n), rng.uniform(27, 33, n))
    rbc = rng.uniform(2.5, 4.5, n)
    pcv = _clip(3.0 * hgb + rng.normal(0, 1.5, n), 12, 40)
    mchc = rng.uniform(28, 34, n)
    rdw = rng.uniform(14.0, 21.0, n)
    tlc = rng.uniform(3.0, 12.0, n)
    plt = rng.uniform(100, 450, n)
    age = rng.uniform(18, 80, n)
    sex = rng.integers(0, 2, n)
    return pd.DataFrame({
        "Age": age.round(0), "Sex": sex, "RBC": rbc.round(2), "PCV": pcv.round(1),
        "MCV": mcv.round(1), "MCH": mch.round(1), "MCHC": mchc.round(1), "RDW": rdw.round(1),
        "TLC": tlc.round(2), "PLT/mm3": plt.round(0), "HGB": hgb.round(1), "target": 1,
    })


def build_augmented_dataset():
    original = pd.read_csv(ORIGINAL_PATH)
    rng = np.random.default_rng(RANDOM_STATE)

    healthy = generate_healthy(rng, N_HEALTHY)
    anemic = generate_anemic(rng, N_ANEMIC)
    synthetic = pd.concat([healthy, anemic], ignore_index=True)[COLUMNS]

    combined = pd.concat([original[COLUMNS], synthetic], ignore_index=True)
    combined = combined.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    combined.to_csv(AUGMENTED_PATH, index=False)
    return combined


if __name__ == "__main__":
    combined = build_augmented_dataset()
    print(f"Original rows: {len(pd.read_csv(ORIGINAL_PATH))}")
    print(f"Synthetic added: {N_HEALTHY + N_ANEMIC} ({N_HEALTHY} healthy, {N_ANEMIC} anemic)")
    print(f"Combined rows: {len(combined)}")
    print(combined["target"].value_counts())
    print(f"Saved to: {AUGMENTED_PATH}")
