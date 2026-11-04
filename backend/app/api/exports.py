import os
from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.core.config import EXPORTS_DIR, MODELS_DIR, DATASETS_DIR, resolve_storage_path
from app.core.database import get_db_connection

router = APIRouter(prefix="/api/exports", tags=["Exports & Downloads"])


@router.get("/list/{project_id}")
def list_exports(project_id: str):
    """Lists all available export artifacts for the project."""
    files = []
    # Scan project exports
    if EXPORTS_DIR.exists():
        for entry in os.scandir(EXPORTS_DIR):
            if entry.is_file():
                stat = entry.stat()
                files.append({
                    "name": entry.name,
                    "size_bytes": stat.st_size,
                    "extension": Path(entry.name).suffix.lower(),
                    "download_url": f"/api/exports/download/{entry.name}"
                })
    return files


@router.get("/download/{filename}")
def download_export_file(filename: str):
    """Safely downloads an exported file."""
    # Prevent traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    target = EXPORTS_DIR / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail="Export file not found")

    return FileResponse(
        path=str(target),
        filename=filename,
        media_type="application/octet-stream"
    )


@router.get("/model/{model_id}")
def download_model_weights(model_id: str):
    """Downloads saved Keras model file (.keras)."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT name, weights_path FROM models WHERE id = ?", (model_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Model not found")

        weights_path = resolve_storage_path(row["weights_path"])
        if not weights_path or not weights_path.exists():
            raise HTTPException(status_code=404, detail="Model weights file missing on disk")

        clean_name = f"{row['name'].replace(' ', '_')}_{model_id[:8]}.keras"
        return FileResponse(
            path=str(weights_path),
            filename=clean_name,
            media_type="application/octet-stream"
        )


@router.delete("/file/{filename}")
def delete_export_file(filename: str):
    """Deletes an exported file from EXPORTS_DIR."""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    target = EXPORTS_DIR / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail="Export file not found")

    try:
        target.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not delete file: {str(e)}")

    return {"message": f"Export file '{filename}' deleted successfully", "filename": filename}

