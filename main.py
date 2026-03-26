"""
Orquestra o pipeline: EDA → pré-processamento → treino → avaliação → MLflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eda import load_dataset, run_eda, validate_target_column
from src.evaluate import (
    build_comparison_table,
    compute_classification_metrics,
    predict,
    print_comparison_table,
)
from mlflow_utils import (
    configure_tracking,
    experiment_name_for_model,
    log_classification_training_run,
    registered_model_name,
)
from src.preprocess import (
    coerce_feature_columns_to_numeric,
    drop_rows_with_any_null,
    encode_target,
    split_features_and_target,
)
from src.train import MODEL_BUILDERS, dividir_treino_teste, treinar_modelo


def parse_args() -> argparse.Namespace:
    default_csv = Path(__file__).resolve().parent / "dataset_ambiental.csv"
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline ML: Qualidade_Ambiental — Random Forest, "
            "Regressão Logística e XGBoost (+ MLflow)."
        )
    )
    parser.add_argument(
        "--csv",
        default=str(default_csv),
        help=f'Caminho do CSV (padrão: "{default_csv}").',
    )
    parser.add_argument(
        "--target",
        default="Qualidade_Ambiental",
        help='Nome da coluna alvo (padrão: "Qualidade_Ambiental").',
    )
    parser.add_argument(
        "--mlflow-uri",
        default=None,
        help=(
            "URI do tracking MLflow (ex.: file:./mlruns ou http://127.0.0.1:5000). "
            "Se omitido, usa MLFLOW_TRACKING_URI ou padrão do MLflow."
        ),
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Não registra runs nem modelos no MLflow.",
    )
    parser.add_argument(
        "--mlflow-study",
        default="qualidade_ambiental",
        help="Prefixo dos nomes de experimento (padrão: qualidade_ambiental).",
    )
    parser.add_argument(
        "--register-models",
        action="store_true",
        help=(
            "Tenta registrar cada modelo no MLflow Model Registry após salvar o artefato. "
            "Requer backend com suporte a registry (pode falhar com apenas file store)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"\nCarregando arquivo: {args.csv}")
    df = load_dataset(args.csv)

    validate_target_column(df, args.target)
    df = coerce_feature_columns_to_numeric(df, target=args.target)
    run_eda(df, target=args.target)

    df = drop_rows_with_any_null(df)
    X, y = split_features_and_target(df, args.target)

    X_train, X_test, y_train, y_test = dividir_treino_teste(X, y)
    y_train_enc, y_test_enc, _ = encode_target(y_train, y_test)

    if not args.no_mlflow:
        configure_tracking(args.mlflow_uri)
        print("\n=== MLflow ===")
        print(f"Tracking URI: {args.mlflow_uri or '(padrão / env MLFLOW_TRACKING_URI)'}")

    rows: list[dict[str, str | float]] = []
    for name in MODEL_BUILDERS.keys():
        print(f"\n--- Treinando: {name} ---")
        model = treinar_modelo(name, X_train, y_train_enc)
        y_pred = predict(model, X_test)
        metrics = compute_classification_metrics(y_test_enc, y_pred)
        rows.append({"modelo": name, **metrics})

        if not args.no_mlflow:
            reg_name = (
                registered_model_name(args.mlflow_study, name)
                if args.register_models
                else None
            )

            run_id, _ = log_classification_training_run(
                model=model,
                X_train=X_train,
                X_test=X_test,
                y_test_enc=y_test_enc,
                model_display_name=name,
                scenario="cli_default",
                study=args.mlflow_study,
                metrics=metrics,
                csv_path=str(Path(args.csv).resolve()),
                target_column=args.target,
                tags_extra={"source": "cli"},
                register_model_name=reg_name,
            )
            exp_name = experiment_name_for_model(name, study=args.mlflow_study)
            print(f"MLflow — experimento: {exp_name} | run_id: {run_id}")

    table = build_comparison_table(rows)
    print_comparison_table(table)

    best = table.loc[table["f1_score"].idxmax(), "modelo"]
    print(f"\nMelhor F1-score (ponderado): {best}")

    if not args.no_mlflow:
        print(
            "\nPara ver a UI: mlflow ui --backend-store-uri file:./mlruns "
            "(ajuste o URI conforme --mlflow-uri)"
        )


if __name__ == "__main__":
    main()
