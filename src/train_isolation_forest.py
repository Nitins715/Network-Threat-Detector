import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.prediction import ALL_42_FEATURES


def train_isolation_forest(
    data_path: str = "data/raw/UNSW_NB15_training-set.csv",
    output_dir: str = "models",
    n_samples: int = 10000,
    contamination: float = 0.05,
    random_state: int = 42
):
    """
    Trains an Isolation Forest anomaly detector exclusively on benign (label == 0) network traffic.
    This enables the NIDS to detect zero-day or unseen attack anomalies without requiring attack labels.
    """
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Training dataset not found at: {data_file}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading dataset from {data_file}...")
    df = pd.read_csv(data_file)
    print(f"[+] Loaded {len(df):,} total flows.")

    # 1. Filter benign traffic for baseline profile
    normal_df = df[df["label"] == 0]
    attack_df = df[df["label"] == 1]
    print(f"[+] Found {len(normal_df):,} normal flows and {len(attack_df):,} attack flows.")

    # Subsample normal flows for fast, high-quality density estimation
    train_normal = normal_df.head(n_samples)
    print(f"[*] Training Isolation Forest on {len(train_normal):,} benign baseline flows...")

    # 2. Extract numerical columns from the 42 model feature list
    # Isolation Forest isolates points in continuous numerical space
    num_cols = [
        col for col in ALL_42_FEATURES
        if col in train_normal.select_dtypes(include=[np.number]).columns
    ]
    print(f"[+] Selected {len(num_cols)} numerical features for anomaly detection:")
    print(f"    {num_cols}")

    X_train = train_normal[num_cols].fillna(0)

    # 3. Fit Isolation Forest
    iso_model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    iso_model.fit(X_train)

    # 4. Quick validation & sanity check
    # Check decision score separation between held-out normal flows and attack flows
    val_normal = normal_df.iloc[n_samples:n_samples + 1000][num_cols].fillna(0)
    val_attack = attack_df.head(1000)[num_cols].fillna(0)

    score_normal = iso_model.decision_function(val_normal)
    score_attack = iso_model.decision_function(val_attack)

    print("\n--- Model Validation Check ---")
    print(f"Normal Flows   -> Mean decision score: {score_normal.mean():.4f} (Higher = Benign)")
    print(f"Attack Flows   -> Mean decision score: {score_attack.mean():.4f} (Lower = Anomaly)")
    separation = score_normal.mean() - score_attack.mean()
    print(f"Separation Gap -> {separation:+.4f} (Positive gap indicates effective anomaly isolation)")

    # 5. Save model and feature columns
    model_file = out_path / "isolation_forest.joblib"
    cols_file = out_path / "iso_num_cols.joblib"

    joblib.dump(iso_model, model_file)
    joblib.dump(num_cols, cols_file)

    print(f"\n[OK] Saved trained model to: {model_file}")
    print(f"[OK] Saved feature columns to: {cols_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Isolation Forest on normal baseline traffic.")
    parser.add_argument("--data-path", type=str, default="data/raw/UNSW_NB15_training-set.csv", help="Path to training CSV")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save model artifacts")
    parser.add_argument("--samples", type=int, default=10000, help="Number of normal flows to train on")
    args = parser.parse_args()

    train_isolation_forest(
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_samples=args.samples
    )
