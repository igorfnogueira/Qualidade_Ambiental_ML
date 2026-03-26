"""Avaliação do modelo: métricas, classification report e comparação."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


def predict(model: Pipeline, X_test: pd.DataFrame) -> pd.Series:
    """Gera predições no conjunto de teste."""
    return pd.Series(model.predict(X_test), index=X_test.index, name="y_pred")


def compute_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    average: str = "weighted",
) -> dict[str, float]:
    """accuracy, precision, recall e f1-score."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "f1_score": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def build_classification_report(y_true: pd.Series, y_pred: pd.Series) -> str:
    """Retorna o classification report como string."""
    return classification_report(y_true, y_pred)


def build_comparison_table(rows: list[dict[str, str | float]]) -> pd.DataFrame:
    """Monta um DataFrame com uma linha por modelo e colunas de métricas."""
    df = pd.DataFrame(rows)
    preferred = ["modelo", "accuracy", "precision", "recall", "f1_score"]
    cols = [c for c in preferred if c in df.columns]
    extra = [c for c in df.columns if c not in cols]
    return df[cols + extra]


def print_comparison_table(df: pd.DataFrame, float_format: str = "{:.4f}") -> None:
    """Imprime a tabela comparativa formatada."""
    print("\n=== COMPARAÇÃO DE MODELOS (média ponderada por classe) ===")
    display = df.copy()
    for col in ["accuracy", "precision", "recall", "f1_score"]:
        if col in display.columns:
            display[col] = display[col].map(lambda x: float_format.format(float(x)))
    print(display.to_string(index=False))


def avaliar_modelo(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    label_encoder: Any | None = None,
    average: str = "weighted",
) -> dict[str, Any]:
    """
    Avalia um modelo e retorna métricas e report.

    Se `label_encoder` for fornecido, decodifica y_true/y_pred para mostrar
    o classification report com os nomes originais das classes.
    """
    y_pred_enc = predict(model, X_test)

    if label_encoder is not None:
        y_true_dec = pd.Series(
            label_encoder.inverse_transform(y_test),
            index=y_test.index,
            name=y_test.name,
        )
        y_pred_dec = pd.Series(
            label_encoder.inverse_transform(y_pred_enc),
            index=X_test.index,
            name=y_pred_enc.name,
        )
        metrics = compute_classification_metrics(
            y_true_dec, y_pred_dec, average=average
        )
        report = build_classification_report(y_true_dec, y_pred_dec)
        return {"metrics": metrics, "report": report, "y_pred": y_pred_dec}

    metrics = compute_classification_metrics(y_test, y_pred_enc, average=average)
    report = build_classification_report(y_test, y_pred_enc)
    return {"metrics": metrics, "report": report, "y_pred": y_pred_enc}

