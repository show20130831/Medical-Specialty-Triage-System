"""Start the service with validated settings: python -m triage_system."""

import sys

from triage_system.config import ConfigurationError, load_settings


def main() -> None:
    try:
        settings = load_settings()
    except ConfigurationError as exc:
        sys.exit(f"Configuration error: {exc}")
    import uvicorn

    uvicorn.run("triage_system.api:app", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
