# Qualidade ambiental (ML)

Projeto de **classificação da qualidade ambiental** a partir de variáveis como temperatura, umidade, gases (CO₂, CO, NO₂, SO₂, O₃) e pressão atmosférica. Inclui pipeline de EDA e treino com vários modelos, rastreamento opcional no **MLflow**, e uma **aplicação web** com **FastAPI** para inferência usando modelo serializado.

**Space (Hugging Face):** [igorfn20/Qualidade_Ambiental_ML](https://huggingface.co/spaces/igorfn20/Qualidade_Ambiental_ML)

---

## Objetivos (resumo)

- Explorar e validar o conjunto de dados ambiental.
- Treinar e comparar modelos (**Random Forest**, **Regressão logística**, **XGBoost**), com métricas no conjunto de teste (ex.: F1-score ponderado).
- Registrar experimentos no **MLflow** (opcional).
- Expor o modelo escolhido para produção via **API REST** + interface HTML, com validação de entradas e resposta em JSON (classe prevista e detalhes/probabilidades).

O script `main.py` treina **todos** os modelos configurados e indica o de melhor F1 no final; a API em produção usa o artefato versionado em `artifacts/` (por exemplo `model.pkl`), alinhado ao deploy no Hugging Face.

---

## Estrutura do repositório

| Caminho | Descrição |
|--------|-----------|
| `main.py` | Orquestra EDA → pré-processamento → treino → avaliação → MLflow (opcional). |
| `src/` | EDA, pré-processamento, treino, avaliação. |
| `notebooks/` | Análises exploratórias em notebook. |
| `dataset_ambiental.csv` | Dados de entrada do pipeline (padrão do CLI). |
| `mlflow_utils.py` | Helpers de logging/registro no MLflow. |
| `qa_api/` | FastAPI: rotas estáticas, `POST /predict`. |
| `web/` | Frontend (HTML, CSS, JS) servido pela API. |
| `artifacts/` | `model.pkl`, `label_encoder.pkl`, `metadata.json` usados na inferência. |
| `requirements.txt` | Dependências do pipeline de treino/EDA. |
| `requirements.api.txt` | Dependências da API + inferência. |
| `Dockerfile` | Imagem Docker (Uvicorn na porta **7860**). |
| `hf_docker_space/` | Pacote espelho para publicar no **Space** (README com `sdk: docker`). |
| `scripts/sync_hf_docker_space.ps1` | Copia fontes para `hf_docker_space/`. |
| `scripts/push_hf_space.ps1` | Ajuda no `git push` do clone local do Space (token via `HF_TOKEN`). |
| `hf_space/` | Versão alternativa com Gradio (referência; deploy principal é Docker + `qa_api`). |

Pastas como `Qualidade_Ambiental_ML/` e `api/` na raiz podem estar no `.gitignore` (clone aninhado / API legada); o fluxo oficial de deploy usa `qa_api/`.

---

## Pipeline de treino (local)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

### Tratamento de dados (valores inválidos)

O dataset pode conter **valores inválidos em colunas numéricas** (por exemplo, `Pressao_Atm` com texto como `"erro_sensor"`). O pipeline aplica uma etapa de limpeza que:

- converte as colunas de features para numérico com `errors="coerce"` (valores inválidos viram `NaN`);
- remove as linhas com `NaN` em qualquer coluna via `dropna()`.

Opções úteis:

- `--csv CAMINHO` — outro arquivo CSV.
- `--no-mlflow` — não registra runs no MLflow.
- `--mlflow-uri URI` — URI do servidor ou `file:./mlruns`.

---

## API e interface (local)

**Com Python:**

```bash
pip install -r requirements.api.txt
uvicorn qa_api.main:app --host 127.0.0.1 --port 7860
```

Abra **http://127.0.0.1:7860** no navegador. Endpoints relevantes: `GET /` (UI), `POST /predict` (JSON com as features esperadas).

**Com Docker** (na raiz do repositório):

```bash
docker build -t qa-ml .
docker run --rm -p 7860:7860 qa-ml
```

Depois acesse **http://127.0.0.1:7860** (não use `http://0.0.0.0:7860` no navegador).

### Exemplo de uso do modelo (`POST /predict`)

Exemplo com `curl`:

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

Resposta de sucesso (exemplo):

```json
{
  "ok": true,
  "qualidade_ambiental": "Boa",
  "warnings": [],
  "details": "Probabilidades (por classe):\n- Boa: 82,10%\n- Moderada: 14,75%\n- Ruim: 3,15%"
}
```

Resposta de erro de validação (exemplo):

```json
{
  "ok": false,
  "errors": [
    "- Umidade: 130 % é incompatível; esperado [0, 100] %."
  ],
  "warnings": []
}
```

---

## Hugging Face

O Space público está em [huggingface.co/spaces/igorfn20/Qualidade_Ambiental_ML](https://huggingface.co/spaces/igorfn20/Qualidade_Ambiental_ML). Para atualizar o Space a partir deste repositório:

1. Rode `scripts/sync_hf_docker_space.ps1` (PowerShell) para atualizar `hf_docker_space/`.
2. Copie o conteúdo de `hf_docker_space/` para o clone do repositório do Space (ou use o fluxo que você já utiliza).
3. `git add`, `git commit`, `git push` no repositório do Space (usuário Hugging Face + **access token** com escrita, ou `push_hf_space.ps1` com `$env:HF_TOKEN`).

O arquivo `hf_docker_space/README.md` contém o front matter YAML (`sdk: docker`, `app_port: 7860`) exigido pela plataforma; **não** duplique esse bloco no README da raiz do GitHub.

---

## Tecnologias

- Python, pandas, NumPy  
- scikit-learn, XGBoost  
- MLflow (tracking opcional)  
- FastAPI, Uvicorn  
- Docker  

---

## Licença e autor

Ajuste esta seção conforme a política da sua instituição ou escolha de licença (MIT, etc.).
