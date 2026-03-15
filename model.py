from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)


def time_series_split(x: pd.DataFrame, y: pd.Series, train_fraction: float = 0.9):
    split_idx = int(len(x) * train_fraction)
    return x.iloc[:split_idx], x.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def train_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    min_samples_leaf: int = 20,
    max_features: str = "sqrt",
    random_state: int = 42,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    return model


def evaluate_classifier(model: RandomForestClassifier, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }
    return metrics


def feature_importance_table(model: RandomForestClassifier, x: pd.DataFrame) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": x.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_run_artifacts(
    output_dir: str | Path,
    metrics: dict,
    feature_importance: pd.DataFrame,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    feature_importance.to_csv(output_path / "feature_importance.csv", index=False)
