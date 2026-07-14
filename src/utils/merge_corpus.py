import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IR_CORPUS_DIR = os.path.join(BASE_DIR, "data", "ir_corpus")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "ir_corpus_merged.csv")

STANDARD_COLUMNS = ["id", "category", "title", "content", "keywords"]

GROUP_A_FILES = [
    "nutriguard_ir_corpus_master.csv",
    "immune_system_basics.csv",
    "nutritional_deficiency_diseases.csv",
    "superfoods_functional_foods.csv",
]

RECIPE_SAMPLE_SIZE = 300
FOOD_SAMPLE_SIZE = 300
RANDOM_STATE = 42

RECIPE_VITAMIN_MINERAL_COLUMNS = [
    "vitamin_a", "vitamin_b12", "vitamin_b6", "vitamin_c", "vitamin_e", "vitamin_k",
    "calcium", "iron", "magnesium", "zinc",
]


def _first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_group_a():
    frames = []
    for filename in GROUP_A_FILES:
        path = os.path.join(IR_CORPUS_DIR, filename)
        if not os.path.exists(path):
            print(f"[group A] SKIP (missing): {filename}")
            continue
        df = pd.read_csv(path)
        df = df[STANDARD_COLUMNS]
        frames.append(df)
        print(f"[group A] {filename}: {len(df)} rows")
    if not frames:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def load_group_b_diet_plans():
    path = os.path.join(IR_CORPUS_DIR, "diet_plans.csv")
    if not os.path.exists(path):
        print("[group B] SKIP (missing): diet_plans.csv")
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        title = f"{row.get('diet_type', '')} diet for {row.get('health_condition', '')} - {row.get('age_group', '')} {row.get('gender', '')}"
        content = (
            f"{row.get('recommended_plan', '')} "
            f"Daily calories: {row.get('daily_calories', '')}. "
            f"Protein: {row.get('protein_g', '')}g. "
            f"Fiber: {row.get('fiber_g', '')}g. "
            f"Key foods: {row.get('key_foods', '')}."
        )
        keywords = (
            f"{row.get('diet_type', '')}, {row.get('health_condition', '')}, "
            f"{row.get('goal', '')}, {row.get('age_group', '')}, {row.get('gender', '')}"
        )
        rows.append({
            "id": row.get("plan_id", ""),
            "category": "Diet Plans",
            "title": title,
            "content": content,
            "keywords": keywords,
        })
    result = pd.DataFrame(rows, columns=STANDARD_COLUMNS)
    print(f"[group B] diet_plans.csv: {len(result)} rows")
    return result


def load_group_c_sports_supplements():
    path = os.path.join(IR_CORPUS_DIR, "Sports_Supplements.csv")
    if not os.path.exists(path):
        print("[group C] SKIP (missing): Sports_Supplements.csv")
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    df = pd.read_csv(path)
    name_col = _first_existing_column(df, ["supplement", "supplement_name", "name"])
    aspect_col = _first_existing_column(df, ["claimed_fitness_aspect", "fitness_aspect", "claim"])
    content_col = _first_existing_column(df, ["notes", "description", "notes_description"])
    category_col = _first_existing_column(df, ["fitness_category", "category"])

    rows = []
    for idx, row in df.iterrows():
        content = str(row.get(content_col, "")).strip() if content_col else ""
        if not content or content.lower() == "nan":
            continue
        name = row.get(name_col, "") if name_col else ""
        aspect = row.get(aspect_col, "") if aspect_col else ""
        category = row.get(category_col, "") if category_col else ""
        rows.append({
            "id": f"supp_{idx}",
            "category": "Sports Supplements",
            "title": f"{name} - {aspect}",
            "content": content,
            "keywords": f"{name}, {category}",
        })
    result = pd.DataFrame(rows, columns=STANDARD_COLUMNS)
    print(f"[group C] Sports_Supplements.csv: {len(result)} rows")
    return result


def load_group_d_recipes():
    path = os.path.join(IR_CORPUS_DIR, "recipes_with_nutrients.csv")
    if not os.path.exists(path):
        print("[group D] SKIP (missing): recipes_with_nutrients.csv")
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    df = pd.read_csv(path)
    df = df[df["Name"].notna() & (df["Name"].astype(str).str.len() > 5)]
    sample_size = min(RECIPE_SAMPLE_SIZE, len(df))
    df = df.sample(n=sample_size, random_state=RANDOM_STATE)

    available_vm_cols = [c for c in RECIPE_VITAMIN_MINERAL_COLUMNS if c in df.columns]

    rows = []
    for idx, row in df.iterrows():
        name = row["Name"]
        content = (
            f"Recipe: {name}. "
            f"Calories: {row.get('Calories', '')}. "
            f"Protein: {row.get('ProteinContent', '')}g. "
            f"Fiber: {row.get('FiberContent', '')}g."
        )
        for col in available_vm_cols:
            label = col.replace("_", " ").title()
            content += f" {label}: {row[col]}."

        recipe_id = row["RecipeId"] if "RecipeId" in df.columns and pd.notna(row.get("RecipeId")) else idx
        rows.append({
            "id": f"recipe_{recipe_id}",
            "category": "Recipes",
            "title": name,
            "content": content,
            "keywords": ", ".join(str(name).split()),
        })
    result = pd.DataFrame(rows, columns=STANDARD_COLUMNS)
    print(f"[group D] recipes_with_nutrients.csv: sampled {len(result)} rows")
    return result


def load_group_e_food_data():
    path = os.path.join(IR_CORPUS_DIR, "food_data.csv")
    if not os.path.exists(path):
        print("[group E] SKIP (missing): food_data.csv")
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    df = pd.read_csv(path)
    desc_col = _first_existing_column(df, ["description", "food_description"])
    category_col = _first_existing_column(df, ["description_category", "category", "food_category"])
    id_col = _first_existing_column(df, ["fdc_id", "id"])

    df = df.dropna(subset=[desc_col])
    df = df.drop_duplicates(subset=[desc_col], keep="first")
    sample_size = min(FOOD_SAMPLE_SIZE, len(df))
    df = df.sample(n=sample_size, random_state=RANDOM_STATE)

    rows = []
    for idx, row in df.iterrows():
        description = row[desc_col]
        category = row.get(category_col, "") if category_col else ""
        food_id = row[id_col] if id_col and pd.notna(row.get(id_col)) else idx
        rows.append({
            "id": f"food_{food_id}",
            "category": "Food & Nutrients",
            "title": description,
            "content": f"{description}. Category: {category}.",
            "keywords": f"{description}, {category}",
        })
    result = pd.DataFrame(rows, columns=STANDARD_COLUMNS)
    print(f"[group E] food_data.csv: sampled {len(result)} rows")
    return result


def build_merged_corpus():
    groups = {
        "A": load_group_a(),
        "B": load_group_b_diet_plans(),
        "C": load_group_c_sports_supplements(),
        "D": load_group_d_recipes(),
        "E": load_group_e_food_data(),
    }

    merged = pd.concat(groups.values(), ignore_index=True)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    print("\n--- Merge summary ---")
    for name, df in groups.items():
        print(f"Group {name}: {len(df)} rows")
    print(f"Grand total: {len(merged)} rows")
    print(f"Saved to: {OUTPUT_PATH}")

    return merged


if __name__ == "__main__":
    build_merged_corpus()
