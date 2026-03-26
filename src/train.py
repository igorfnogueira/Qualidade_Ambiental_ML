"""Treino e definição de pipelines para os modelos."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def dividir_treino_teste(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide dados em treino e teste com estratificação pela classe."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_random_forest_pipeline(
    *,
    n_estimators: int = 200,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    """Pipeline base com RandomForestClassifier."""
    return Pipeline(
        steps=[
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )


def build_logistic_regression_pipeline(
    *,
    max_iter: int = 5000,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    """Pipeline base com LogisticRegression."""
    return Pipeline(
        steps=[
            (
                "classifier",
                LogisticRegression(
                    max_iter=max_iter,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def build_xgboost_pipeline(
    *,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    """Pipeline base com XGBClassifier (import lazy de xgboost)."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "Instale o xgboost: pip install xgboost"
        ) from exc
    return Pipeline(
        steps=[
            (
                "classifier",
                XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    verbosity=0,
                ),
            ),
        ]
    )


ModelBuilder = Callable[[], Pipeline]
MODEL_BUILDERS: dict[str, ModelBuilder] = {
    "Random Forest": build_random_forest_pipeline,
    "Logistic Regression": build_logistic_regression_pipeline,
    "XGBoost": build_xgboost_pipeline,
}


def build_feature_preprocessor() -> ColumnTransformer:
    """Cria pré-processador para colunas numéricas e categóricas."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, make_column_selector(dtype_include=["number"])),
            (
                "cat",
                categorical_pipeline,
                make_column_selector(dtype_exclude=["number"]),
            ),
        ]
    )


def treinar_modelo(
    modelo: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Treina um modelo selecionado por nome e retorna o pipeline ajustado."""
    if modelo not in MODEL_BUILDERS:
        raise ValueError(
            f"Modelo '{modelo}' não encontrado. Opções: {list(MODEL_BUILDERS.keys())}"
        )
    base_model = MODEL_BUILDERS[modelo]()
    model = Pipeline(
        steps=[
            ("preprocessor", build_feature_preprocessor()),
            ("model", base_model),
        ]
    )
    model.fit(X_train, y_train)
    return model


def treinar_logistic_regression_tunado(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    cv: int = 5,
    random_state: int = 42,
    scoring: str = "f1_weighted",
    n_jobs: int = -1,
    verbose: int = 0,
) -> GridSearchCV:
    """
    Mesmo pré-processamento que `treinar_modelo`, com busca em grade na
    LogisticRegression (C e class_weight), usando validação cruzada estratificada.
    """
    inner = Pipeline(
        steps=[
            (
                "classifier",
                LogisticRegression(
                    max_iter=10_000,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipe = Pipeline(
        steps=[
            ("preprocessor", build_feature_preprocessor()),
            ("model", inner),
        ]
    )
    param_grid = {
        "model__classifier__C": np.logspace(-3, 3, num=13).tolist(),
        "model__classifier__class_weight": [None, "balanced"],
    }
    skf = StratifiedKFold(
        n_splits=cv, shuffle=True, random_state=random_state
    )
    gs = GridSearchCV(
        pipe,
        param_grid,
        cv=skf,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
        verbose=verbose,
    )
    gs.fit(X_train, y_train)
    return gs

