"""Pré-processamento: carregamento, limpeza, separação X/y e encoding do alvo."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder


FEATURE_COLUMNS: list[str] = [
    "Temperatura",
    "Umidade",
    "CO2",
    "CO",
    "Pressao_Atm",
    "NO2",
    "SO2",
    "O3",
]


def carregar_dados(csv_path: str | Path) -> pd.DataFrame:
    """Carrega o CSV e retorna um DataFrame."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path.resolve()}")
    return pd.read_csv(csv_path)


def coerce_feature_columns_to_numeric(
    df: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    target: str | None = None,
) -> pd.DataFrame:
    """
    Converte colunas de features para numérico (float) usando `errors="coerce"`.

    Isso transforma valores inválidos (ex.: "erro_sensor") em NaN para poderem ser
    removidos por `dropna()` ou tratados por imputação.
    """
    cols = list(feature_columns or FEATURE_COLUMNS)
    if target and target in cols:
        cols = [c for c in cols if c != target]

    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def drop_rows_with_any_null(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas que contenham qualquer valor nulo."""
    cleaned = df.dropna().copy()
    print(f"\nLinhas após remoção de nulos: {len(cleaned)}")
    return cleaned


def preprocessar_dados(
    df: pd.DataFrame,
    *,
    target: str,
    drop_na: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Limpa e separa X (features) e y (alvo).
    """
    df = coerce_feature_columns_to_numeric(df, target=target)
    if drop_na:
        df = drop_rows_with_any_null(df)

    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def split_features_and_target(
    df: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Separação simples (mantida para compatibilidade)."""
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def encode_target(
    y_train: pd.Series, y_test: pd.Series
) -> tuple[pd.Series, pd.Series, LabelEncoder]:
    """
    Codifica o alvo em inteiros (0..n_classes-1) para ser compatível com XGBoost/estimators.
    """
    le = LabelEncoder()
    y_train_enc = pd.Series(
        le.fit_transform(y_train), index=y_train.index, name=y_train.name
    )
    y_test_enc = pd.Series(
        le.transform(y_test), index=y_test.index, name=y_test.name
    )
    return y_train_enc, y_test_enc, le


def encode_target_train_test(
    y_train: pd.Series, y_test: pd.Series
) -> tuple[pd.Series, pd.Series, LabelEncoder]:
    """Alias em português do `encode_target`."""
    return encode_target(y_train, y_test)

