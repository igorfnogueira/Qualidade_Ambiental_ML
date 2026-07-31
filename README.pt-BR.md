# Classificação de Qualidade Ambiental (ML)

---

**Language / Idioma:** [English](README.md) | [Português](README.pt-BR.md)

Pipeline de **classificação** supervisionada que prevê a qualidade ambiental (`Qualidade_Ambiental`) a partir de variáveis de sensores — temperatura, umidade, gases (CO₂, CO, NO₂, SO₂, O₃) e pressão atmosférica. Inclui EDA, treino com vários modelos, rastreamento no **MLflow** (ativo por padrão) e uma aplicação web com **FastAPI** para inferência com modelo serializado.

**Repositório:** [igorfnogueira/ml-python-environmental-quality](https://github.com/igorfnogueira/ml-python-environmental-quality)  
**Demo (Hugging Face Space):** [igorfn20/Qualidade_Ambiental_ML](https://huggingface.co/spaces/igorfn20/Qualidade_Ambiental_ML)

> **Uso educacional.** Os dados e resultados são ilustrativos e podem não corresponder a situações ambientais reais.

---

## Problema

A partir de leituras ambientais (incluindo valores inválidos como `"erro_sensor"`), o objetivo é atribuir uma classe de qualidade entre cinco rótulos (ex.: Boa, Moderada, Ruim, Muito Ruim, Excelente) e expor uma API de inferência reproduzível para uso interativo.

O dataset tem **10.000** linhas e **9** colunas (**8** features + alvo). Após coerção numérica e remoção de nulos, restam **9.604** linhas.

---

## Abordagem

- **EDA:** estatísticas descritivas, nulos, histogramas, correlação e boxplots (`src/eda.py`, `notebooks/eda.ipynb`).
- **Limpeza:** `to_numeric(errors="coerce")` nas features e `dropna()`.
- **Modelos comparados:** Random Forest, Regressão Logística, XGBoost (mesmo split estratificado 80/20, `random_state=42`).
- **Métricas:** accuracy, precision, recall e F1 ponderado no teste; `main.py` indica o melhor F1 ponderado entre os modelos configurados.
- **Experimentos opcionais (notebook):** tratamento de outliers de CO₂ (IQR → mediana); busca de hiperparâmetros na Regressão Logística (`GridSearchCV`).
- **Serviço:** artefatos em `artifacts/`; FastAPI `POST /predict` com validação hard/soft e probabilidades por classe.

---

## Estrutura do repositório

| Caminho | Descrição |
|---------|-----------|
| `main.py` | Orquestra EDA → pré-processamento → treino → avaliação → MLflow (ligado por padrão) |
| `src/` | EDA, pré-processamento, treino, avaliação |
| `notebooks/` | Notebook exploratório |
| `dataset_ambiental.csv` | CSV padrão de treino |
| `mlflow_utils.py` | Helpers de logging no MLflow |
| `qa_api/` | App FastAPI (`GET /`, `POST /predict`) |
| `web/` | Interface HTML / CSS / JS |
| `artifacts/` | Fonte da verdade para inferência (`model.pkl`, `label_encoder.pkl`, `metadata.json`) |
| `requirements.txt` | Dependências de treino / EDA |
| `requirements.api.txt` | Dependências da API / inferência |
| `Dockerfile` | Imagem Uvicorn na porta **7860** |
| `hf_docker_space/README.md` | YAML do Space Docker no HF; demais arquivos são **gerados** pelo sync |
| `hf_space/` | Pacote Gradio alternativo (código + assets); `artifacts/` ali é **gerado** pelo export |
| `scripts/sync_hf_docker_space.ps1` | Monta a pasta completa do Space Docker a partir da raiz |
| `scripts/export_hf_artifacts.py` | Retreina e grava artefatos (ex.: em `hf_space/artifacts/`) |
| `scripts/push_hf_space.ps1` | Ajuda no push do clone do Space (`HF_TOKEN`) |
| `LICENSE` | MIT |

---

## Variáveis de entrada

| Campo | Unidade |
|-------|---------|
| Temperatura | °C |
| Umidade | % |
| CO2 | ppm |
| CO | µg/m³ |
| Pressao_Atm | hPa |
| NO2 | µg/m³ |
| SO2 | µg/m³ |
| O3 | µg/m³ |

---

## Treino (local)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

Flags úteis:

- `--csv CAMINHO` — outro CSV
- `--no-mlflow` — desliga o MLflow (ativo por padrão; URI `file:./mlruns`)
- `--mlflow-uri URI` — sobrescreve a URI de tracking (padrão `file:./mlruns`)

### Ver runs no MLflow

Depois de `python main.py`, abra a UI de tracking (os runs ficam em `./mlruns`, ignorado pelo Git):

```bash
mlflow ui --backend-store-uri file:./mlruns --host 127.0.0.1 --port 5000
```

Em seguida abra **http://127.0.0.1:5000**. Os experimentos são nomeados por modelo (ex.: `qualidade_ambiental_xgboost`).

Exportar artefatos (modelo padrão: XGBoost, mesmo fluxo de limpeza do `main.py`):

```bash
python scripts/export_hf_artifacts.py
python scripts/export_hf_artifacts.py --model "Random Forest"
```

---

## API e interface (local)

```bash
pip install -r requirements.api.txt
uvicorn qa_api.main:app --host 127.0.0.1 --port 7860
```

Abra **http://127.0.0.1:7860**. Endpoints: `GET /` (UI), `POST /predict`.

**Docker:**

```bash
docker build -t qa-ml .
docker run --rm -p 7860:7860 qa-ml
```

Depois acesse **http://127.0.0.1:7860** (não use `http://0.0.0.0:7860` no navegador).

### Exemplo `POST /predict`

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

Sucesso (exemplo):

```json
{
  "ok": true,
  "qualidade_ambiental": "Boa",
  "warnings": [],
  "details": "Probabilidades (por classe):\n- Boa: 82,10%\n- Moderada: 14,75%\n- Ruim: 3,15%"
}
```

Erro de validação (exemplo):

```json
{
  "ok": false,
  "errors": [
    "- Umidade: 130 % é incompatível; esperado [0, 100] %."
  ],
  "warnings": []
}
```

Limites hard rejeitam a requisição; limites soft só geram avisos e a predição segue.

---

## Hugging Face

Space público: [huggingface.co/spaces/igorfn20/Qualidade_Ambiental_ML](https://huggingface.co/spaces/igorfn20/Qualidade_Ambiental_ML).

O deploy principal é **Docker + FastAPI** (`qa_api`). A árvore Gradio em `hf_space/` é uma alternativa opcional.

Para atualizar o Space Docker a partir deste repositório:

1. Execute `scripts/sync_hf_docker_space.ps1` (copia `qa_api`, `web`, `artifacts`, `Dockerfile`, `requirements.api.txt` e regenera o `README.md` do Space com YAML Docker).
2. Copie o conteúdo gerado de `hf_docker_space/` para o clone local do Space.
3. Faça commit e push (token HF com escrita), ou use `push_hf_space.ps1` com `$env:HF_TOKEN`.

Para o pacote Gradio, rode antes `python scripts/export_hf_artifacts.py` para criar `hf_space/artifacts/`.

**Não** coloque o front matter YAML do Hugging Face neste README da raiz do GitHub.

---

## Tecnologias

- Python 3.11, pandas, NumPy  
- scikit-learn, XGBoost  
- matplotlib, seaborn  
- MLflow (ativo por padrão; desligar com `--no-mlflow`)  
- FastAPI, Uvicorn, joblib  
- Docker  
- Gradio (pacote opcional em `hf_space/`)

---

## Licença

MIT — ver [LICENSE](LICENSE).
