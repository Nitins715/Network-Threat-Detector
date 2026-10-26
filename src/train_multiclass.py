import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.prediction import ALL_42_FEATURES


ATTACK_CATEGORIES = [
    'Normal', 'Generic', 'Exploits', 'Fuzzers', 'DoS',
    'Reconnaissance', 'Analysis', 'Backdoor', 'Shellcode', 'Worms'
]

CATEGORY_NORM_MAP = {
    'normal': 'Normal',
    'generic': 'Generic',
    'exploits': 'Exploits',
    'fuzzers': 'Fuzzers',
    'dos': 'DoS',
    'reconnaissance': 'Reconnaissance',
    'analysis': 'Analysis',
    'backdoor': 'Backdoor',
    'backdoors': 'Backdoor',
    'shellcode': 'Shellcode',
    'worms': 'Worms'
}


def train_multiclass_model(
    data_path: str = "data/raw/UNSW_NB15_training-set.csv",
    output_dir: str = "models",
    sample_size: int = 40000,
    random_state: int = 42
):
    """
    Trains a multi-class XGBoost classifier to categorize threats into 9 specific attack categories:
    Reconnaissance, DoS, Exploits, Fuzzers, Backdoors, Shellcode, Analysis, Generic, Worms, plus Normal.
    """
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found at {data_file}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading training data from {data_file}...")
    df = pd.read_csv(data_file)

    # Clean and normalize attack_cat column
    df['attack_cat'] = df['attack_cat'].astype(str).str.strip().replace({'Backdoors': 'Backdoor'})

    print(f"[+] Loaded {len(df):,} rows across {df['attack_cat'].nunique()} unique categories:")
    for cat, count in df['attack_cat'].value_counts().items():
        print(f"    * {cat:<15}: {count:>6,}")

    # Stratified subsampling for fast, balanced training across all 10 classes
    from sklearn.model_selection import train_test_split
    if sample_size and sample_size < len(df):
        print(f"\n[*] Subsampling {sample_size:,} rows with exact class stratification...")
        train_df, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df['attack_cat'],
            random_state=random_state
        )
        train_df = train_df.reset_index(drop=True)
    else:
        train_df = df

    print(f"[+] Training dataset prepared: {len(train_df):,} samples.")

    # Target variable encoding (fits directly on all 10 defined categories)
    label_encoder = LabelEncoder()
    label_encoder.fit(ATTACK_CATEGORIES)
    y_train = label_encoder.transform(train_df['attack_cat'])

    # Feature columns
    feature_cols = [c for c in ALL_42_FEATURES if c in train_df.columns]
    X_train = train_df[feature_cols].copy()

    # Preprocessing pipeline
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in ['proto', 'service', 'state'] if c in X_train.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_cols)
        ]
    )

    # Multi-class XGBoost classifier
    classifier = XGBClassifier(
        objective='multi:softprob',
        num_class=len(ATTACK_CATEGORIES),
        n_estimators=120,
        max_depth=6,
        learning_rate=0.15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        eval_metric='mlogloss'
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    print("\n[*] Training multi-class threat classifier...")
    pipeline.fit(X_train, y_train)

    # Evaluate training accuracy
    y_pred = pipeline.predict(X_train)
    acc = np.mean(y_pred == y_train) * 100
    print(f"[+] Model Training Accuracy: {acc:.2f}%")

    # Save artifacts
    model_path = out_path / "multiclass_model.joblib"
    labels_path = out_path / "multiclass_labels.joblib"

    # Compress pipeline for GitHub (<20MB)
    joblib.dump(pipeline, model_path, compress=3)
    joblib.dump(label_encoder, labels_path)

    print(f"\n[OK] Saved multi-class model to: {model_path}")
    print(f"[OK] Saved label encoder to: {labels_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-class attack category classifier.")
    parser.add_argument("--data-path", type=str, default="data/raw/UNSW_NB15_training-set.csv")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--sample-size", type=int, default=35000)
    args = parser.parse_args()

    train_multiclass_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        sample_size=args.sample_size
    )
