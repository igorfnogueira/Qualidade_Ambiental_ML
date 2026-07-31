# Environmental Quality Classification (ML)

---

**Language / Idioma:** [English](README.md) | [Português](README.pt-BR.md)

Supervised **classification** pipeline that predicts environmental quality (`Qualidade_Ambiental`) from sensor variables — temperature, humidity, gases (CO₂, CO, NO₂, SO₂, O₃), and atmospheric pressure. Includes EDA, multi-model training, **MLflow** tracking (on by default), and a **FastAPI** web app for inference with a serialized model.

**Repository:** [igorfnogueira/ml-python-environmental-quality](https://github.com/igorfnogueira/ml-python-environmental-quality)  
**Live demo (Hugging Face Space):** [igorfn20/Qualidade_Ambiental_ML](https://huggingface.co/spaces/igorfn20/Qualidade_Ambiental_ML)

> **Educational use.** The data and outputs are illustrative and may not reflect real environmental conditions.

---

## Problem

Given environmental sensor readings (including noisy / invalid values such as `"erro_sensor"`), the goal is to assign a quality class among five labels (e.g. Boa, Moderada, Ruim, Muito Ruim, Excelente) and expose a reproducible inference API for interactive use.

The dataset ships with **10,000** rows and **9** columns (**8** features + target). After numeric coercion and dropping nulls, **9,604** rows remain.

---

## Approach

- **EDA:** descriptive stats, null checks, histograms, correlation heatmap, boxplots (`src/eda.py`, `notebooks/eda.ipynb`).
- **Cleaning:** convert features with `to_numeric(errors="coerce")`, then `dropna()`.
- **Models compared:** Random Forest, Logistic Regression, XGBoost (same stratified 80/20 split, `random_state=42`).
- **Metrics:** accuracy, precision, recall, and weighted F1 on the test set; `main.py` reports the best weighted F1 among configured models.
- **Optional experiments (notebook):** CO₂ outlier handling via IQR → median; Logistic Regression hyperparameter search (`GridSearchCV`).
- **Serving:** versioned artifacts under `artifacts/`; FastAPI `POST /predict` with hard/soft input validation and class probabilities.

---

## Repository layout

| Path | Description |
|------|-------------|
| `main.py` | Orchestrates EDA → preprocess → train → evaluate → MLflow (default on) |
| `src/` | EDA, preprocessing, training, evaluation |
| `notebooks/` | Exploratory notebook |
| `dataset_ambiental.csv` | Default training CSV |
| `mlflow_utils.py` | MLflow logging helpers |
| `qa_api/` | FastAPI app (`GET /`, `POST /predict`) |
| `web/` | HTML / CSS / JS UI served by the API |
| `artifacts/` | Source-of-truth inference artifacts (`model.pkl`, `label_encoder.pkl`, `metadata.json`) |
| `requirements.txt` | Training / EDA dependencies |
| `requirements.api.txt` | API / inference dependencies |
| `Dockerfile` | Uvicorn image on port **7860** |
| `hf_docker_space/README.md` | Hugging Face Docker Space YAML only; other files are **generated** by sync |
| `hf_space/` | Alternative Gradio package (code + assets); `artifacts/` there is **generated** by export |
| `scripts/sync_hf_docker_space.ps1` | Builds the full Docker Space folder from the repo root |
| `scripts/export_hf_artifacts.py` | Retrains and writes artifacts (e.g. into `hf_space/artifacts/`) |
| `scripts/push_hf_space.ps1` | Helps push a local Space clone (`HF_TOKEN`) |
| `LICENSE` | MIT |

---

## Input features

| Field | Unit |
|-------|------|
| Temperatura | °C |
| Umidade | % |
| CO2 | ppm |
| CO | µg/m³ |
| Pressao_Atm | hPa |
| NO2 | µg/m³ |
| SO2 | µg/m³ |
| O3 | µg/m³ |

---

## Training (local)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

Useful flags:

- `--csv PATH` — alternate CSV
- `--no-mlflow` — disable MLflow logging (enabled by default; tracking URI `file:./mlruns`)
- `--mlflow-uri URI` — override tracking URI (default `file:./mlruns`)

### Viewing MLflow runs

After `python main.py`, open the tracking UI (runs are stored under `./mlruns`, which is gitignored):

```bash
mlflow ui --backend-store-uri file:./mlruns --host 127.0.0.1 --port 5000
```

Then open **http://127.0.0.1:5000**. Experiments are named per model (e.g. `qualidade_ambiental_xgboost`).

Export artifacts (default model name: XGBoost, same cleaning flow as `main.py`):

```bash
python scripts/export_hf_artifacts.py
python scripts/export_hf_artifacts.py --model "Random Forest"
```

---

## API and UI (local)

```bash
pip install -r requirements.api.txt
uvicorn qa_api.main:app --host 127.0.0.1 --port 7860
```

Open **http://127.0.0.1:7860**. Endpoints: `GET /` (UI), `POST /predict`.

**Docker:**

```bash
docker build -t qa-ml .
docker run --rm -p 7860:7860 qa-ml
```

Then open **http://127.0.0.1:7860** (do not use `http://0.0.0.0:7860` in the browser).

### Example `POST /predict`

```bash
curl -X POST "http://127.0.0.1:7860/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Temperatura": 27.5,
    "Umidade": 62.0,
    "CO2": 780.0,
    "CO": 3.2,
    "Pressao_Atm": 1013.0,
    "NO2": 42.0,
    "SO2": 18.0,
    "O3": 25.0
  }'
```

Success (example):

```json
{
  "ok": true,
  "qualidade_ambiental": "Boa",
  "warnings": [],
  "details": "Probabilidades (por classe):\n- Boa: 82,10%\n- Moderada: 14,75%\n- Ruim: 3,15%"
}
```

Validation error (example):

```json
{
  "ok": false,
  "errors": [
    "- Umidade: 130 % é incompatível; esperado [0, 100] %."
  ],
  "warnings": []
}
```

Hard limits reject the request; soft limits only add warnings and still run prediction.

---

## Hugging Face

Public Space: [huggingface.co/spaces/igorfn20/Qualidade_Ambiental_ML](https://huggingface.co/spaces/igorfn20/Qualidade_Ambiental_ML).

Primary deploy path is **Docker + FastAPI** (`qa_api`). The Gradio tree under `hf_space/` is an optional alternative.

To refresh the Docker Space from this repo:

1. Run `scripts/sync_hf_docker_space.ps1` (copies `qa_api`, `web`, `artifacts`, `Dockerfile`, `requirements.api.txt`, and regenerates the Space `README.md` with Docker YAML).
2. Copy the generated `hf_docker_space/` contents into your local Space clone.
3. Commit and push (HF write token), or use `push_hf_space.ps1` with `$env:HF_TOKEN`.

For the Gradio package, run `python scripts/export_hf_artifacts.py` first so `hf_space/artifacts/` exists.

Do **not** put Hugging Face YAML front matter in this GitHub root README.

---

## Tech stack

- Python 3.11, pandas, NumPy  
- scikit-learn, XGBoost  
- matplotlib, seaborn  
- MLflow (enabled by default; disable with `--no-mlflow`)  
- FastAPI, Uvicorn, joblib  
- Docker  
- Gradio (optional package under `hf_space/`)

---

## License

MIT — see [LICENSE](LICENSE).
