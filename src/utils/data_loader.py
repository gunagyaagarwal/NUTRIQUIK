import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IR_CORPUS_DIR = os.path.join(BASE_DIR, "data", "ir_corpus")
ML_TRAINING_DIR = os.path.join(BASE_DIR, "data", "ml_training")


def load_ir_files(ir_corpus_dir=IR_CORPUS_DIR):
    ir_files = {}
    for filename in sorted(os.listdir(ir_corpus_dir)):
        if not filename.endswith(".csv"):
            continue
        path = os.path.join(ir_corpus_dir, filename)
        df = pd.read_csv(path)
        print(f"[ir_corpus] {filename}: {len(df)} rows")
        ir_files[filename] = df
    return ir_files


def load_ml_files(ml_training_dir=ML_TRAINING_DIR):
    ml_files = {}
    for filename in sorted(os.listdir(ml_training_dir)):
        if not filename.endswith(".csv"):
            continue
        path = os.path.join(ml_training_dir, filename)
        df = pd.read_csv(path)
        print(f"[ml_training] {filename}: {len(df)} rows, columns: {list(df.columns)}")
        ml_files[filename] = df
    return ml_files


def load_all_data(ir_corpus_dir=IR_CORPUS_DIR, ml_training_dir=ML_TRAINING_DIR):
    return {
        "ir_files": load_ir_files(ir_corpus_dir),
        "ml_files": load_ml_files(ml_training_dir),
    }


if __name__ == "__main__":
    data = load_all_data()
    print(f"\nSummary: {len(data['ir_files'])} IR corpus files, {len(data['ml_files'])} ML training files")
