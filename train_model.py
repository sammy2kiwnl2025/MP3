"""
Machine Problem 3 - Student pass/fail model

Trains a Random Forest on the LMS activity dataset from Machine Problem 2
and saves the model + evaluation numbers for the Streamlit app.

Run:  python train_model.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "augmented_balanced_dataset_m.csv"
MODEL_PATH = BASE / "model.joblib"
METRICS_PATH = BASE / "metrics.json"

FEATURES = [
    "Submission.Actions",
    "Quiz.Actions",
    "Forum.Actions",
    "Create.Actions",
    "Read.Actions",
    "Update.Actions",
    "Delete.Actions",
    "Days.Logged.In",
]
TARGET = "rating"

# The dataset's labels are High / Low. For this problem they are read as:
#   High -> Pass,  Low -> Fail
LABEL_TO_OUTCOME = {"High": "Pass", "Low": "Fail"}


def make_model():
    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",  # after removing duplicates, High (1720) outnumbers Low (1000)
        random_state=42,
        n_jobs=-1,
    )


def load_clean_data(path=DATA_PATH):
    df = pd.read_csv(path)
    n_raw = len(df)
    # Duplicates must be removed BEFORE splitting. Otherwise identical rows land in
    # both train and test and the accuracy is inflated.
    df = df.drop_duplicates().reset_index(drop=True)
    return df, n_raw - len(df)


def train(path=DATA_PATH, save=True):
    df, n_dupes = load_clean_data(path)
    X, y = df[FEATURES], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = make_model().fit(X_train, y_train)
    pred = model.predict(X_test)

    labels = ["High", "Low"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(make_model(), X, y, cv=cv, scoring="accuracy")

    report = classification_report(y_test, pred, labels=labels, output_dict=True)
    metrics = {
        "sklearn_version": sklearn.__version__,
        "rows_raw": int(len(df) + n_dupes),
        "duplicates_removed": int(n_dupes),
        "rows_used": int(len(df)),
        "class_counts": {k: int(v) for k, v in y.value_counts().items()},
        "test_size": int(len(y_test)),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "report": {
            k: {m: float(v) for m, v in report[k].items()} for k in labels
        },
        "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
        "feature_importance": dict(
            sorted(
                zip(FEATURES, map(float, model.feature_importances_)),
                key=lambda kv: kv[1],
                reverse=True,
            )
        ),
        "feature_median": {f: float(X[f].median()) for f in FEATURES},
    }

    if save:
        joblib.dump(model, MODEL_PATH, compress=3)
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)

    return model, metrics


if __name__ == "__main__":
    _, m = train()
    print(f"Rows used: {m['rows_used']} (removed {m['duplicates_removed']} duplicates)")
    print(f"Hold-out test accuracy: {m['test_accuracy']:.2%}")
    print(f"5-fold CV accuracy:     {m['cv_accuracy_mean']:.2%} +/- {m['cv_accuracy_std']:.2%}")
    print("Confusion matrix [High, Low]:", m["confusion_matrix"])
    print("Saved:", MODEL_PATH.name, "and", METRICS_PATH.name)
