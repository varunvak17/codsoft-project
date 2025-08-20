# fraud_detection.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

CSV_PATH = "C:/Users/nanth/Downloads/archive (2)/creditcard.csv"  # change path if needed
MODEL_PATH = "credit_fraud_best_model.pkl"

def make_preprocessor(feature_df: pd.DataFrame):
    # Scale Time & Amount if present; keep V1..V28 as-is
    num_cols_to_scale = [c for c in ["Time", "Amount"] if c in feature_df.columns]
    remainder_cols = [c for c in feature_df.columns if c not in num_cols_to_scale]
    return ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), num_cols_to_scale),
            ("pass", "passthrough", remainder_cols),
        ]
    )

def build_pipeline_with_resampling(preprocessor, clf):
    """
    Try SMOTE + undersampling (imblearn). If unavailable, return a simple sklearn Pipeline.
    """
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.pipeline import Pipeline as ImbPipeline

        steps = [
            ("prep", preprocessor),
            ("smote", SMOTE(random_state=42, sampling_strategy=0.1)),
            ("under", RandomUnderSampler(random_state=42, sampling_strategy=0.5)),
            ("clf", clf),
        ]
        return ImbPipeline(steps)
    except Exception:
        return Pipeline([("prep", preprocessor), ("clf", clf)])

def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_score = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        s = model.decision_function(X_test)
        # normalize to [0,1] to compute ROC-AUC safely
        y_score = (s - s.min()) / (s.max() - s.min() + 1e-9)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    roc = roc_auc_score(y_test, y_score) if y_score is not None else float("nan")
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{name}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc:.4f}")
    print(f"  Confusion Matrix (rows=true 0/1, cols=pred 0/1):\n{cm}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Simple confusion matrix plot (matplotlib, no custom colors)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center")
    plt.tight_layout()
    plt.show()

    return {"precision": precision, "recall": recall, "f1": f1, "roc_auc": roc}

def main():
    # 1) Load
    df = pd.read_csv(CSV_PATH)
    assert "Class" in df.columns, "Expecting a 'Class' column (0 = genuine, 1 = fraud)."

    y = df["Class"].astype(int)
    X = df.drop(columns=["Class"])

    # 2) Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Preprocessor
    preprocessor = make_preprocessor(X)

    # 4) Models (class_weight handles imbalance even if imblearn isn't available)
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    rf = RandomForestClassifier(
        n_estimators=200,  # keep moderate for speed; increase if you want
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    )

    # 5) Pipelines w/ resampling (if available)
    pipe_lr = build_pipeline_with_resampling(preprocessor, logreg)
    pipe_rf = build_pipeline_with_resampling(preprocessor, rf)

    # 6) Fit
    print("Training Logistic Regression...")
    pipe_lr.fit(X_train, y_train)
    print("Training Random Forest...")
    pipe_rf.fit(X_train, y_train)

    # 7) Evaluate
    res_lr = evaluate("LogisticRegression", pipe_lr, X_test, y_test)
    res_rf = evaluate("RandomForest", pipe_rf, X_test, y_test)

    # 8) Choose best (by F1)
    best_name, best_model, best_res = (
        ("RandomForest", pipe_rf, res_rf)
        if res_rf["f1"] >= res_lr["f1"]
        else ("LogisticRegression", pipe_lr, res_lr)
    )

    # 9) Save best model
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nSaved best model ({best_name}) to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
