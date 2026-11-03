import os
from pathlib import Path
from typing import Union, Optional

# Paths
APP_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

DATA_DIR = Path(os.getenv("SIGNALML_DATA_DIR", PROJECT_ROOT / "data"))
STORAGE_DIR = Path(os.getenv("SIGNALML_STORAGE_DIR", PROJECT_ROOT / "storage"))

DATASETS_DIR = STORAGE_DIR / "datasets"
MODELS_DIR = STORAGE_DIR / "models"
CHECKPOINTS_DIR = STORAGE_DIR / "checkpoints"
EXPORTS_DIR = STORAGE_DIR / "exports"
FEATURES_DIR = STORAGE_DIR / "features"
CLUSTERS_DIR = STORAGE_DIR / "clusters"
WORKFLOWS_DIR = STORAGE_DIR / "workflows"
TEMP_DIR = STORAGE_DIR / "temp"

SQLITE_DB_PATH = STORAGE_DIR / "signalml.db"

# Ensure all critical storage directories exist
for directory in [
    DATA_DIR,
    STORAGE_DIR,
    DATASETS_DIR,
    MODELS_DIR,
    CHECKPOINTS_DIR,
    EXPORTS_DIR,
    FEATURES_DIR,
    CLUSTERS_DIR,
    WORKFLOWS_DIR,
    TEMP_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Limits & Security
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", 150 * 1024 * 1024))  # 150 MB
ALLOWED_DATA_EXTENSIONS = {".csv", ".txt", ".tsv", ".json", ".jsonl", ".npy", ".npz", ".bin"}
ALLOWED_ARCHIVE_EXTENSIONS = {".zip"}

# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8080))
DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")


def resolve_storage_path(saved_path: Union[str, Path, None]) -> Optional[Path]:
    """
    Safely resolves storage paths across different run environments
    (e.g., /app/storage inside Docker vs /workspace/SignalML/storage on host).
    """
    if not saved_path:
        return None
    p = Path(saved_path)
    if p.exists():
        return p
    p_str = str(saved_path).replace("\\", "/")
    if "/storage/" in p_str:
        sub = p_str.split("/storage/", 1)[1]
        candidate = STORAGE_DIR / sub
        if candidate.exists() or candidate.parent.exists():
            return candidate
    candidate = STORAGE_DIR / p.name
    if candidate.exists():
        return candidate
    return p

