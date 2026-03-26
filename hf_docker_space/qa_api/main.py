from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .inference import FEATURE_COLUMNS, predict_qualidade, validate_inputs

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"

app = FastAPI(title="Qualidade Ambiental (IA)")

app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/styles.css")
def styles() -> FileResponse:
    return FileResponse(str(WEB_DIR / "styles.css"))


@app.get("/app.js")
def js() -> FileResponse:
    return FileResponse(str(WEB_DIR / "app.js"))


@app.post("/predict")
async def predict(payload: dict[str, Any]) -> JSONResponse:
    values = tuple(payload.get(k) for k in FEATURE_COLUMNS)
    errors, warnings = validate_inputs(values)
    if errors:
        return JSONResponse(
            {"ok": False, "errors": errors, "warnings": warnings},
            status_code=400,
        )

    row = {k: float(payload[k]) for k in FEATURE_COLUMNS}
    classe, detalhes = predict_qualidade(row)
    return JSONResponse(
        {
            "ok": True,
            "qualidade_ambiental": classe,
            "warnings": warnings,
            "details": detalhes,
        }
    )

