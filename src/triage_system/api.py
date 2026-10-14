import json
import time
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from triage_system.config import load_settings
from triage_system.logging_config import log_request
from triage_system.model_assets import SPECIALTIES
from triage_system.model_loader import LoadedModel, ModelLoadError, load_local_model
from triage_system.monitoring import PredictionMetrics
from triage_system.predictor import PredictionError, predict_description

app = FastAPI(title="Medical Specialty Triage System")
# Training descriptions are at most 433 characters; 1,000 leaves headroom while keeping
# every accepted input well below the 512-token model limit, so nothing is truncated.
MAX_DESCRIPTION_CHARS = 1_000
# Rejects oversized bodies before JSON parsing; a maximal valid request is far smaller.
MAX_BODY_BYTES = 16_384
WEB_ROOT = Path(__file__).with_name("web")
METRICS = PredictionMetrics(tuple(sorted(SPECIALTIES)))


@app.middleware("http")
async def _limit_body_and_count_status(request: Request, call_next):
    is_predict = request.url.path == "/predict"
    if is_predict and request.method == "POST":
        length = request.headers.get("content-length")
        if length is None:
            response = JSONResponse({"detail": "Content-Length is required."}, status_code=411)
        elif not length.isdigit():
            response = JSONResponse({"detail": "Invalid Content-Length."}, status_code=400)
        elif int(length) > MAX_BODY_BYTES:
            response = JSONResponse({"detail": "Request body is too large."}, status_code=413)
        else:
            response = await call_next(request)
    else:
        response = await call_next(request)
    if is_predict:
        METRICS.record_status(response.status_code)
    return response


app.middleware("http")(log_request)


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
def _loaded_model() -> LoadedModel:
    model_dir = load_settings().model_dir
    if not model_dir:
        raise ModelLoadError("TRIAGE_MODEL_DIR must point to a local model export.")
    return load_local_model(model_dir)


def _service_version() -> str:
    try:
        return version("medical-specialty-triage-system")
    except PackageNotFoundError:
        return "unknown"


def _model_version(loaded: LoadedModel) -> dict:
    """Identify the served bundle so metrics and incidents can be tied to it."""
    manifest_path = loaded.assets.directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) \
        if manifest_path.is_file() else {}
    return {
        "weights_sha256": loaded.weights_sha256,
        "training_commit": manifest.get("git", {}).get("commit"),
        "trained_utc": manifest.get("created_utc"),
        "base_model": manifest.get("base_model", {}).get("name"),
        "temperature": loaded.assets.temperature,
        "uncertain_threshold": loaded.assets.uncertain_threshold,
    }


@app.get("/version")
def service_version() -> dict:
    """Report the service and model versions; loads the model like /ready."""
    try:
        loaded = _loaded_model()
    except ModelLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    return {"service_version": _service_version(), "model": _model_version(loaded)}


@app.get("/metrics")
def metrics() -> dict:
    """Aggregate counters since process start; no request text or per-request records."""
    try:
        model = _model_version(_loaded_model())["weights_sha256"]
    except ModelLoadError:
        model = None
    return {"model_weights_sha256": model, **METRICS.snapshot()}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Classify one medical description with the local model."""
    try:
        loaded = _loaded_model()
        started = time.perf_counter()
        result = predict_description(loaded, request.description)
        latency_ms = (time.perf_counter() - started) * 1000
    except ModelLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except PredictionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None
    METRICS.record_prediction(result.label, result.score, result.uncertain,
                              len(request.description), latency_ms)
    return PredictResponse(
        label=result.label, score=result.score, uncertain=result.uncertain,
        top=[SpecialtyScore(label=label, score=score) for label, score in result.top],
    )
