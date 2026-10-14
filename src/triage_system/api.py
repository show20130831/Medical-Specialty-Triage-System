from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from triage_system.model_loader import ModelLoadError, load_local_model
from triage_system.predictor import PredictionError, predict_description
from triage_system.config import load_settings
from triage_system.logging_config import log_request

app = FastAPI(title="Medical Specialty Triage System")
MAX_DESCRIPTION_CHARS = 10_000
app.middleware("http")(log_request)
WEB_ROOT = Path(__file__).with_name("web")


@app.get("/", include_in_schema=False)
def web_app() -> FileResponse:
    """Serve the small browser client from the same origin as the API."""
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    """Report API responsiveness, not model readiness."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    """Report whether the configured local model can serve predictions."""
    try:
        _loaded_model()
    except ModelLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    return {"status": "ready"}


class PredictRequest(BaseModel):
    description: str = Field(max_length=MAX_DESCRIPTION_CHARS)

    @field_validator("description")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("description must contain non-blank text.")
        return value


class SpecialtyScore(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    label: str = Field(description="Highest scoring specialty.")
    score: float = Field(description="Temperature-calibrated confidence of the top label.")
    uncertain: bool = Field(
        description="True when the confidence is below the validated abstain threshold.")
    top: list[SpecialtyScore] = Field(description="Three highest scoring specialties.")


@lru_cache(maxsize=1)
def _loaded_model():
    model_dir = load_settings().model_dir
    if not model_dir:
        raise ModelLoadError("TRIAGE_MODEL_DIR must point to a local model export.")
    return load_local_model(model_dir)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Classify one medical description with the local model."""
    try:
        result = predict_description(_loaded_model(), request.description)
    except ModelLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except PredictionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None
    return PredictResponse(
        label=result.label, score=result.score, uncertain=result.uncertain,
        top=[SpecialtyScore(label=label, score=score) for label, score in result.top],
    )
