
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

RF_PATH = MODELS_DIR / "random_forest.joblib"
XGB_PATH = MODELS_DIR / "xgboost.joblib"
CONFIG_PATH = MODELS_DIR / "ensemble_config.json"

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
MAX_ROWS = 50_000

app = FastAPI(
    title="Network Intrusion Detection API",
    description="Predict normal or attack traffic from network-flow CSV data.",
    version="1.0.0",
)

# Load artifacts when the API starts
try:
    rf_model = joblib.load(RF_PATH)
    xgb_model = joblib.load(XGB_PATH)

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    FEATURE_COLUMNS = config["feature_columns"]
    RF_WEIGHT = float(config["rf_weight"])
    XGB_WEIGHT = float(config["xgb_weight"])
    THRESHOLD = float(config["threshold"])

    if not np.isclose(RF_WEIGHT + XGB_WEIGHT, 1.0):
        raise ValueError("Ensemble weights must sum to 1.")

    if not 0 <= THRESHOLD <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

except Exception as exc:
    raise RuntimeError(
        f"Could not load model artifacts from {MODELS_DIR}. "
        "Run Part 6 and verify the saved files."
    ) from exc


# These columns are metadata, not model input features
METADATA_COLUMNS = ["id", "label", "attack_cat"]


@app.get("/")
def home():
    return {
        "message": "Network Intrusion Detection API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": True,
        "model": "weighted_ensemble",
        "threshold": THRESHOLD,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate filename
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    # Read uploaded bytes and enforce file-size limit
    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 20 MB limit."
        )

    # Parse CSV
    try:
        df = pd.read_csv(pd.io.common.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not read the CSV. Check its format."
        ) from exc

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="The CSV contains no data rows."
        )

    if len(df) > MAX_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"Upload at most {MAX_ROWS:,} rows per request."
        )

    # Remove optional metadata if the uploaded file contains it
    df = df.drop(
        columns=METADATA_COLUMNS,
        errors="ignore"
    )

    # Check required model features
    missing_columns = [
        col for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "CSV is missing required UNSW-NB15 features.",
                "missing_columns": missing_columns,
            },
        )

    # Ignore extra columns and enforce training feature order
    X = df[FEATURE_COLUMNS].copy()

    # Convert numerical features; reject invalid values
    categorical_columns = {"proto", "service", "state"}
    numeric_columns = [
        col for col in FEATURE_COLUMNS
        if col not in categorical_columns
    ]

    for col in numeric_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    invalid_numeric = X[numeric_columns].isna().any(axis=1)

    if invalid_numeric.any():
        raise HTTPException(
            status_code=422,
            detail=(
                "Some rows contain missing or non-numeric values "
                "in numerical features. Correct these rows and retry."
            ),
        )

    if not np.isfinite(X[numeric_columns].to_numpy(dtype=float)).all():
        raise HTTPException(
            status_code=422,
            detail="Numerical features must contain finite values."
        )

    try:
        rf_probabilities = rf_model.predict_proba(X)[:, 1]
        xgb_probabilities = xgb_model.predict_proba(X)[:, 1]

        attack_probabilities = (
            RF_WEIGHT * rf_probabilities
            + XGB_WEIGHT * xgb_probabilities
        )

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="Prediction failed. Check that the CSV matches the trained features."
        ) from exc

    predictions = (attack_probabilities >= THRESHOLD).astype(int)

    results = pd.DataFrame({
        "prediction": [
            "Attack" if value == 1 else "Normal"
            for value in predictions
        ],
        "attack_probability": np.round(
            attack_probabilities, 4
        ),
    })

    attack_count = int((predictions == 1).sum())
    normal_count = int((predictions == 0).sum())

    return {
        "filename": file.filename,
        "total_records": len(results),
        "normal_count": normal_count,
        "attack_count": attack_count,
        "attack_percentage": round(
            attack_count / len(results) * 100, 2
        ),
        "threshold": THRESHOLD,
        "model": "weighted_ensemble",
        "predictions": results.to_dict(orient="records"),
    }
