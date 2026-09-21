import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from triage_system.model_loader import ModelLoadError, load_local_model
from triage_system.predictor import PredictionError, predict_description

app = FastAPI(title="Medical Specialty Triage System")
WEB_ROOT = Path(__file__).with_name("web")


@app.get("/", include_in_schema=False)
def web_app() -> FileResponse:
    """Serve the small browser client from the same origin as the API."""
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    """Report API responsiveness, not model readiness."""
    return {"status": "ok"}


class PredictRequest(BaseModel):
    description: str


@lru_cache(maxsize=1)
def _loaded_model():
    model_dir = os.environ.get("TRIAGE_MODEL_DIR")
    if not model_dir:
        raise ModelLoadError("TRIAGE_MODEL_DIR must point to a local model export.")
    return load_local_model(Path(model_dir))


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, str | float]:
    """Classify one medical description with the local model."""
    try:
        if not request.description.strip():
            raise PredictionError("description must contain non-blank text.")
        result = predict_description(_loaded_model(), request.description)
    except (ModelLoadError, PredictionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return {"label": result.label, "score": result.score}
