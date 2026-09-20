from fastapi import FastAPI

app = FastAPI(title="Medical Specialty Triage System")


@app.get("/health")
def health() -> dict[str, str]:
    """Report API responsiveness, not model readiness."""
    return {"status": "ok"}
