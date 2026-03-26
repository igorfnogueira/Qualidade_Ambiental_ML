"""Inferência e validação para Qualidade_Ambiental (uso em API)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"

FEATURE_COLUMNS = [
    "Temperatura",
    "Umidade",
    "CO2",
    "CO",
    "Pressao_Atm",
    "NO2",
    "SO2",
    "O3",
]

_model = None
_label_encoder = None

VALIDATION_RULES: dict[str, dict[str, object]] = {
    "Temperatura": {
        "unit": "°C",
        "hard_min": -60.0,
        "hard_max": 60.0,
        "soft_min": -10.0,
        "soft_max": 45.0,
        "hint": "faixa típica em ambiente: ~[-10, 45] °C",
    },
    "Umidade": {
        "unit": "%",
        "hard_min": 0.0,
        "hard_max": 100.0,
        "soft_min": 5.0,
        "soft_max": 95.0,
        "hint": "umidade relativa deve estar entre 0 e 100%",
    },
    "CO2": {
        "unit": "ppm",
        "hard_min": 0.0,
        "hard_max": None,
        "soft_min": None,
        "soft_max": 5000.0,
        "hint": "valores acima de 5000 ppm são incomuns em ambientes comuns",
    },
    "CO": {
        "unit": "unid",
        "hard_min": 0.0,
        "hard_max": None,
        "soft_min": None,
        "soft_max": 200.0,
        "hint": "valores muito altos são improváveis em ambiente comum",
    },
    "Pressao_Atm": {
        "unit": "hPa",
        "hard_min": 850.0,
        "hard_max": 1085.0,
        "soft_min": 930.0,
        "soft_max": 1050.0,
        "hint": "pressão ao nível do mar costuma ficar ~[930, 1050] hPa",
    },
    "NO2": {
        "unit": "unid",
        "hard_min": 0.0,
        "hard_max": None,
        "soft_min": None,
        "soft_max": 400.0,
        "hint": "valores muito altos são improváveis em ambiente comum",
    },
    "SO2": {
        "unit": "unid",
        "hard_min": 0.0,
        "hard_max": None,
        "soft_min": None,
        "soft_max": 400.0,
        "hint": "valores muito altos são improváveis em ambiente comum",
    },
    "O3": {
        "unit": "unid",
        "hard_min": 0.0,
        "hard_max": None,
        "soft_min": None,
        "soft_max": 400.0,
        "hint": "valores muito altos são improváveis em ambiente comum",
    },
}


def _load_artifacts() -> None:
    global _model, _label_encoder
    if _model is not None:
        return
    model_path = ARTIFACTS_DIR / "model.pkl"
    le_path = ARTIFACTS_DIR / "label_encoder.pkl"
    if not model_path.is_file() or not le_path.is_file():
        raise FileNotFoundError(f"Artefatos não encontrados em {ARTIFACTS_DIR.resolve()}")
    _model = joblib.load(model_path)
    _label_encoder = joblib.load(le_path)


def validate_inputs(values: tuple[float | None, ...]) -> tuple[list[str], list[str]]:
    """Retorna (errors, warnings) com mensagens amigáveis."""
    errors: list[str] = []
    warnings: list[str] = []

    if len(values) != len(FEATURE_COLUMNS):
        return [f"Esperado {len(FEATURE_COLUMNS)} valores."], []

    def fmt_interval(lo: float | None, hi: float | None) -> str:
        if lo is None and hi is None:
            return ""
        if lo is None:
            return f"<= {hi:g}"
        if hi is None:
            return f">= {lo:g}"
        return f"[{lo:g}, {hi:g}]"

    for name, value in zip(FEATURE_COLUMNS, values, strict=True):
        if value is None:
            errors.append(f"- {name}: valor ausente (preencha o campo).")
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            errors.append(f"- {name}: valor inválido ({value!r}); informe um número.")
            continue

        rule = VALIDATION_RULES.get(name, {})
        unit = str(rule.get("unit", ""))
        hard_min = rule.get("hard_min")
        hard_max = rule.get("hard_max")
        soft_min = rule.get("soft_min")
        soft_max = rule.get("soft_max")
        hint = str(rule.get("hint", "")).strip()

        hard_interval = fmt_interval(
            None if hard_min is None else float(hard_min),
            None if hard_max is None else float(hard_max),
        )
        soft_interval = fmt_interval(
            None if soft_min is None else float(soft_min),
            None if soft_max is None else float(soft_max),
        )

        hard_violation = (hard_min is not None and v < float(hard_min)) or (
            hard_max is not None and v > float(hard_max)
        )
        if hard_violation and hard_interval:
            msg = f"- {name}: {v:g} {unit} é incompatível; esperado {hard_interval} {unit}."
            if hint:
                msg += f" ({hint})"
            errors.append(msg)
            continue

        soft_violation = (soft_min is not None and v < float(soft_min)) or (
            soft_max is not None and v > float(soft_max)
        )
        if soft_violation and soft_interval:
            msg = f"- {name}: {v:g} {unit} é incomum; faixa típica {soft_interval} {unit}."
            if hint:
                msg += f" ({hint})"
            warnings.append(msg)

    return errors, warnings


def predict_qualidade(row: dict[str, float]) -> tuple[str, str]:
    """Retorna (classe prevista, texto auxiliar com probabilidades em %)."""
    _load_artifacts()
    missing = [c for c in FEATURE_COLUMNS if c not in row or row[c] is None]
    if missing:
        raise ValueError(f"Valores ausentes para: {missing}")

    X = pd.DataFrame([{c: float(row[c]) for c in FEATURE_COLUMNS}], columns=FEATURE_COLUMNS)
    y_hat = _model.predict(X)
    if hasattr(y_hat, "ravel"):
        y_hat = y_hat.ravel()
    classe = _label_encoder.inverse_transform(np.asarray(y_hat).astype(int))[0]

    extra = ""
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(X)[0]
        classes = _label_encoder.classes_
        lines: list[str] = []
        for cls, p in zip(classes, proba, strict=True):
            pct = f"{(float(p) * 100):.2f}%".replace(".", ",")
            lines.append(f"- {cls}: {pct}")
        extra = "Probabilidades (por classe):\n" + "\n".join(lines)

    return str(classe), extra

