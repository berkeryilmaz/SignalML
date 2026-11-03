from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query
from app.core.database import get_db_connection
from app.services.training_service import TrainingService, TrainingJob
from app.schemas.training import TrainingConfig, TrainingJobStatus

router = APIRouter(prefix="/api/training", tags=["Model Training"])


@router.post("/start", response_model=Dict[str, Any])
def start_training(payload: TrainingConfig):
    """
    Starts asynchronous training in a background worker thread.
    Returns immediately with job_id for non-blocking UI polling.
    """
    # Verify model and dataset exist
    with get_db_connection() as conn:
        model_row = conn.execute("SELECT project_id FROM models WHERE id = ?", (payload.model_id,)).fetchone()
        if not model_row:
            raise HTTPException(status_code=404, detail="Model not found")
        project_id = model_row["project_id"]

        dataset_row = conn.execute("SELECT id FROM datasets WHERE id = ?", (payload.dataset_id,)).fetchone()
        if not dataset_row:
            raise HTTPException(status_code=404, detail="Dataset not found")

    job = TrainingService.start_training(
        project_id=project_id,
        model_id=payload.model_id,
        dataset_id=payload.dataset_id,
        config=payload.model_dump()
    )

    return {
        "job_id": job.job_id,
        "status": job.status,
        "total_epochs": job.total_epochs,
        "message": "Training job dispatched successfully"
    }


@router.get("/status/{job_id}", response_model=Dict[str, Any])
def get_training_status(job_id: str):
    """
    Returns live training progress: current epoch, loss/accuracy curves, elapsed time, and status.
    """
    job = TrainingService.get_job(job_id)
    if job:
        return job.to_dict()

    # If not active in memory, check SQLite database
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM training_runs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Training job not found")

        r = dict(row)
        import json
        import numpy as np

        def _clean_obj(val):
            if isinstance(val, float):
                return val if np.isfinite(val) else None
            elif isinstance(val, dict):
                return {k: _clean_obj(v) for k, v in val.items()}
            elif isinstance(val, (list, tuple)):
                return [_clean_obj(v) for v in val]
            return val

        try:
            history = _clean_obj(json.loads(r.get("history_json", "{}")))
        except Exception:
            history = {}

        try:
            evaluation = _clean_obj(json.loads(r.get("evaluation_json", "{}")))
        except Exception:
            evaluation = {}

        try:
            config = json.loads(r.get("config_json", "{}"))
        except Exception:
            config = {}

        return {
            "job_id": r["id"],
            "project_id": r["project_id"],
            "model_id": r["model_id"],
            "dataset_id": r["dataset_id"],
            "status": r["status"],
            "current_epoch": len(history.get("loss", [])) if "loss" in history else 0,
            "total_epochs": config.get("epochs", 0),
            "history": history,
            "evaluation": evaluation,
            "error": r.get("error_message"),
            "elapsed_seconds": 0.0
        }


@router.post("/cancel/{job_id}")
def cancel_training(job_id: str):
    """Signals training worker to halt gracefully at the next epoch boundary."""
    success = TrainingService.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot cancel job: job not running or already terminated.")
    return {"message": "Cancellation request submitted"}


@router.get("/history/{model_id}")
def get_model_training_history(model_id: str):
    """Retrieves all past training runs for a specific model."""
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM training_runs WHERE model_id = ? ORDER BY created_at DESC", (model_id,)).fetchall()
        import json
        runs = []
        for r in rows:
            d = dict(r)
            d["config"] = json.loads(d.get("config_json", "{}"))
            d["history"] = json.loads(d.get("history_json", "{}"))
            d["evaluation"] = json.loads(d.get("evaluation_json", "{}"))
            runs.append(d)
        return runs
