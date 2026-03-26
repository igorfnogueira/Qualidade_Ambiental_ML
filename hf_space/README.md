---
title: Qualidade Ambiental
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.10.0
app_file: app.py
pinned: false
license: mit
---

# Classificação de Qualidade Ambiental

Interface **Gradio** que recebe as mesmas variáveis do dataset de treino e retorna a classe **`Qualidade_Ambiental`** (e probabilidades por classe, quando disponível).

## Variáveis de entrada (ordem do modelo)

| Campo | Descrição |
|-------|-----------|
| Temperatura | °C |
| Umidade | % |
| CO2 | ppm |
| CO | µg/m³ |
| Pressao_Atm | hPa |
| NO2 | µg/m³ |
| SO2 | µg/m³ |
| O3 | µg/m³ |

## Artefatos

- `artifacts/model.pkl` — pipeline sklearn (pré-processamento + classificador).
- `artifacts/label_encoder.pkl` — `LabelEncoder` do alvo.
- `artifacts/metadata.json` — metadados da exportação.

Para **regenerar** os artefatos a partir do repositório principal do projeto:

```bash
python scripts/export_hf_artifacts.py
python scripts/export_hf_artifacts.py --model "Random Forest"
```

## Publicar no Hugging Face Spaces

1. Crie um Space (Gradio ou Docker).
2. Envie o **conteúdo desta pasta** (`hf_space/`) como raiz do repositório do Space (ou copie `app.py`, `inference.py`, `requirements.txt`, `artifacts/` e este `README.md`).
3. Se usar template **Gradio**, o `README.md` acima já inclui o front matter YAML esperado.

## Docker (local ou Space tipo Docker)

Na pasta `hf_space/`:

```bash
docker build -t qualidade-ambiental .
docker run --rm -p 7860:7860 qualidade-ambiental
```

Abra `http://127.0.0.1:7860`.

## Notas

- O modelo padrão exportado é **XGBoost** (mesmo fluxo de `main.py`: remoção de linhas com nulos, split estratificado, encoding do alvo).
- Entradas devem ser numéricas e **sem nulos** (comportamento alinhado ao treino).
