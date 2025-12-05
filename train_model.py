# train_model_fixed.py
"""
Training script (fixed & ready-to-run)

- Persyaratan:
    pip install scikit-learn pandas joblib

- Cara pakai:
    python train_model_fixed.py

Hasil:
- model disimpan ke model_kelulusan.joblib
- menampilkan metrik evaluasi ke console
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from joblib import dump

RANDOM_STATE = 42
DATA_PATH = "dataset_kelulusan_mahasiswa.csv"
OUTPUT_MODEL = "model_kelulusan.joblib"
SUMMARY_JSON = "train_summary_fixed.json"

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    return df

def prepare_features(df):
    # Use the same 4 features as the Flask app for compatibility
    X = df[["ipk", "sks_lulus", "presensi", "mengulang"]].copy()
    y = df["lulus_tepat_waktu"].astype(int).copy()
    return X, y

def train_and_tune(X_train, y_train):
    # Pipeline: scaling + RandomForest
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
    ])

    # Hyperparameter distribution for randomized search
    param_dist = {
        "clf__n_estimators": [100, 200, 400, 600],
        "clf__max_depth": [5, 8, 12, None],
        "clf__min_samples_split": [2, 4, 6, 10],
        "clf__min_samples_leaf": [1, 2, 3, 4],
        "clf__class_weight": [None, "balanced"]
    }

    rs = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=24,
        scoring="roc_auc",
        cv=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    rs.fit(X_train, y_train)
    best_pipe = rs.best_estimator_
    return best_pipe, rs.best_params_, rs

def calibrate_model(pipe, X_train, y_train):
    # Calibrate probabilities to improve reliability
    calibrator = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
    calibrator.fit(X_train, y_train)
    return calibrator

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # Some calibrated wrappers expose predict_proba
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = None

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    return {
        "accuracy": acc,
        "roc_auc": auc,
        "report": report,
        "confusion_matrix": cm.tolist()
    }

def main():
    print("Loading dataset:", DATA_PATH)
    df = load_data(DATA_PATH)
    print("Rows:", len(df))
    print("Target distribution:\n", df["lulus_tepat_waktu"].value_counts(normalize=True))

    X, y = prepare_features(df)

    # Train/test split (stratify to keep class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("\nStarting hyperparameter search and training (RandomizedSearchCV)...")
    best_pipe, best_params, rs_obj = train_and_tune(X_train, y_train)
    print("Best params found:", best_params)

    print("\nCalibrating model probabilities (isotonic)...")
    calibrated = calibrate_model(best_pipe, X_train, y_train)

    print("\nEvaluating on test set...")
    metrics = evaluate_model(calibrated, X_test, y_test)
    print(f"Accuracy (test): {metrics['accuracy']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"ROC AUC (test): {metrics['roc_auc']:.4f}")
    print("\nClassification report:\n", metrics["report"])
    print("Confusion matrix:\n", np.array(metrics["confusion_matrix"]))

    # Save calibrated model
    dump(calibrated, OUTPUT_MODEL)
    print(f"\nSaved calibrated model to: {OUTPUT_MODEL}")

    # Save summary
    summary = {
        "dataset": DATA_PATH,
        "n_rows": len(df),
        "target_balance": df["lulus_tepat_waktu"].value_counts(normalize=True).to_dict(),
        "best_params": best_params,
        "metrics": {
            "accuracy": metrics["accuracy"],
            "roc_auc": metrics["roc_auc"]
        },
        "model_path": OUTPUT_MODEL
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote training summary to: {SUMMARY_JSON}")

if __name__ == "__main__":
    main()
