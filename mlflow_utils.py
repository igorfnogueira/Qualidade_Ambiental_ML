"""Integração MLflow: experimentos por modelo, métricas e artefatos."""

from __future__ import annotations

import re
from typing import Any

import mlflow
from sklearn.pipeline import Pipeline


def _resolved_fitted_estimator(estimator: Any) -> Any:
    """GridSearchCV / outros meta-estimadores expõem o pipeline final em best_estimator_."""
    if hasattr(estimator, "best_estimator_"):
        return estimator.best_estimator_
    return estimator


def _classifier_for_param_logging(pipe: Any) -> Any:
    """
    Localiza o classificador final no pipeline (suporta preprocessor + model + classifier).
    """
    if not hasattr(pipe, "named_steps"):
        return pipe
    steps = pipe.named_steps
    if "classifier" in steps:
        return steps["classifier"]
    if "model" in steps:
        return _classifier_for_param_logging(steps["model"])
    return pipe


def _slugify(text: str) -> str:
    """Nome seguro para experimento MLflow (sem espaços especiais)."""
    s = text.lower().strip()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s or "modelo"


def configure_tracking(tracking_uri: str | None) -> None:
    """
    Define onde o MLflow grava runs (ex.: file:./mlruns ou http://127.0.0.1:5000).

    Se `tracking_uri` for None, usa variável de ambiente MLFLOW_TRACKING_URI ou padrão local.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def experiment_name_for_model(
    model_display_name: str, *, study: str = "qualidade_ambiental"
) -> str:
    """
    Um experimento MLflow por tipo de modelo (pedido do projeto).

    Ex.: qualidade_ambiental / Random Forest -> qualidade_ambiental_random_forest
    """
    return f"{_slugify(study)}_{_slugify(model_display_name)}"


def registered_model_name(study: str, model_display_name: str) -> str:
    """Nome sugerido para o Model Registry (estudo + modelo)."""
    return f"{_slugify(study)}_{_slugify(model_display_name)}"


def _flatten_params_for_mlflow(params: dict[str, Any]) -> dict[str, str]:
    """MLflow aceita valores string; normaliza bool/números."""
    out: dict[str, str] = {}
    for k, v in params.items():
        if v is None:
            out[k] = "None"
        elif isinstance(v, bool):
            out[k] = str(v).lower()
        elif isinstance(v, (int, float)):
            out[k] = str(v)
        else:
            out[k] = str(v)[:250]
    return out


def extract_pipeline_hyperparams(pipeline: Pipeline | Any) -> dict[str, str]:
    """Extrai hiperparâmetros relevantes do classificador do pipeline (inclui aninhado)."""
    resolved = _resolved_fitted_estimator(pipeline)
    clf = _classifier_for_param_logging(resolved)
    shallow = clf.get_params(deep=False)
    keys = (
        "n_estimators",
        "max_depth",
        "learning_rate",
        "max_iter",
        "solver",
        "random_state",
        "n_jobs",
        "verbosity",
        "objective",
        "eval_metric",
    )
    picked: dict[str, Any] = {"classifier": clf.__class__.__name__}
    for key in keys:
        if key in shallow:
            picked[key] = shallow[key]
    return _flatten_params_for_mlflow(picked)


def log_sklearn_run(
    *,
    experiment_name: str,
    run_name: str,
    model: Pipeline | Any,
    metrics: dict[str, float],
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    artifact_path: str = "model",
    register_model_name: str | None = None,
) -> str:
    """
    Cria/uso o experimento, abre um run, registra params, métricas, tags e salva o modelo.

    Args:
        experiment_name: Nome único do experimento MLflow (por modelo).
        run_name: Nome legível do run dentro do experimento.
        model: Pipeline sklearn já treinado.
        metrics: Dicionário com floats (ex.: accuracy, precision, ...).
        params: Parâmetros adicionais de treino/dados (opcional).
        tags: Tags descritivas (ex.: caminho do CSV, coluna alvo).
        artifact_path: Pasta do artefato do modelo dentro do run.
        register_model_name: Se definido, tenta registrar no Model Registry após log_model.

    Returns:
        run_id do MLflow.
    """
    mlflow.set_experiment(experiment_name)

    model_to_log = _resolved_fitted_estimator(model)

    param_str: dict[str, str] = {}
    param_str.update(extract_pipeline_hyperparams(model))
    if params:
        param_str.update(_flatten_params_for_mlflow(params))

    metric_floats = {k: float(v) for k, v in metrics.items()}

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(param_str)
        mlflow.log_metrics(metric_floats)
        if tags:
            mlflow.set_tags(tags)

        mlflow.sklearn.log_model(model_to_log, artifact_path=artifact_path)

        if register_model_name:
            try:
                model_uri = f"runs:/{run.info.run_id}/{artifact_path}"
                mlflow.register_model(model_uri=model_uri, name=register_model_name)
            except Exception as exc:  # noqa: BLE001 — registry pode não existir no backend
                mlflow.set_tag("registry_error", str(exc)[:500])

        return run.info.run_id


def log_classification_training_run(
    *,
    model: Any,
    X_train: Any,
    X_test: Any,
    y_test_enc: Any,
    model_display_name: str,
    scenario: str,
    study: str = "qualidade_ambiental",
    metrics: dict[str, float] | None = None,
    params_extra: dict[str, Any] | None = None,
    tags_extra: dict[str, str] | None = None,
    csv_path: str | None = None,
    target_column: str | None = None,
    register_model_name: str | None = None,
) -> tuple[str, dict[str, float]]:
    """
    Opcionalmente calcula métricas no conjunto de teste e registra um run no MLflow.

    Compartilhado entre `main.py` e o notebook: use `scenario` e `tags_extra` para
    distinguir baseline, CO2 mediana, tuning, CLI etc.
    """
    from src.evaluate import compute_classification_metrics, predict

    fitted = _resolved_fitted_estimator(model)
    if metrics is None:
        y_pred = predict(fitted, X_test)
        out_metrics = compute_classification_metrics(y_test_enc, y_pred)
    else:
        out_metrics = {k: float(v) for k, v in metrics.items()}

    exp_name = experiment_name_for_model(model_display_name, study=study)
    run_name_ml = f"{_slugify(scenario)}__{_slugify(model_display_name)}"

    params: dict[str, Any] = {
        "test_size": 0.2,
        "random_state": 42,
        "n_features": int(X_train.shape[1]),
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
    }
    if hasattr(model, "best_params_"):
        params["gridsearch_best_params"] = str(model.best_params_)
    if hasattr(model, "best_score_"):
        params["gridsearch_best_cv_score"] = float(model.best_score_)
    if params_extra:
        params.update(params_extra)

    tags: dict[str, str] = {"scenario": scenario}
    if csv_path:
        tags["dataset_csv"] = csv_path[:500]
    if target_column:
        tags["target_column"] = target_column
    if tags_extra:
        tags.update(tags_extra)

    run_id = log_sklearn_run(
        experiment_name=exp_name,
        run_name=run_name_ml,
        model=model,
        metrics=out_metrics,
        params=params,
        tags=tags,
        register_model_name=register_model_name,
    )
    return run_id, out_metrics
