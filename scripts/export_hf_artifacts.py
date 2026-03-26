"""
Treina (mesmo fluxo do main.py) e exporta modelo + LabelEncoder para hf_space/artifacts.

Uso (na raiz do projeto):
  python scripts/export_hf_artifacts.py
  python scripts/export_hf_artifacts.py --model "Random Forest"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eda import load_dataset, validate_target_column
from src.preprocess import drop_rows_with_any_null, encode_target, split_features_and_target
from src.train import MODEL_BUILDERS, dividir_treino_teste, treinar_modelo

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta artefatos para Hugging Face Space.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "dataset_ambiental.csv",
        help="CSV com Qualidade_Ambiental e features.",
    )
    parser.add_argument("--target", default="Qualidade_Ambiental")
    parser.add_argument(
        "--model",
        default="XGBoost",
        choices=list(MODEL_BUILDERS.keys()),
        help="Modelo a treinar e exportar (padrão: XGBoost).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "hf_space" / "artifacts",
        help="Pasta de saída (model.pkl, label_encoder.pkl, metadata.json).",
    )
    args = parser.parse_args()

    df = load_dataset(args.csv)
    validate_target_column(df, args.target)
    df = drop_rows_with_any_null(df)
    X, y = split_features_and_target(df, args.target)

    missing = [c for c in FEATURE_COLUMNS if c not in X.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")
    X = X[FEATURE_COLUMNS]

    X_train, X_test, y_train, y_test = dividir_treino_teste(X, y)
    y_train_enc, _y_test_enc, le = encode_target(y_train, y_test)

    print(f"Treinando {args.model} …")
    model = treinar_modelo(args.model, X_train, y_train_enc)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out_dir / "model.pkl")
    joblib.dump(le, args.out_dir / "label_encoder.pkl")

    meta = {
        "model_name": args.model,
        "target": args.target,
        "features": FEATURE_COLUMNS,
        "csv": str(args.csv.resolve()),
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Artefatos gravados em {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
