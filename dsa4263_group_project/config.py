"""Project configuration utilities."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"


def ensure_directories() -> None:
    """Ensure core directories exist at runtime."""

    for path in (RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
