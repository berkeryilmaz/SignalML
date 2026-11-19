import csv
import io
import json
import numpy as np
import keras
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, Response
from app.core.config import resolve_storage_path
from app.core.database import get_db_connection
from app.services.training_service import TrainingService, TrainingJob
from app.services.data_manager import DataManager
from app.services.keras_model_builder import ClusteringLayer
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


@router.delete("/runs/{run_id}")
def delete_training_run(run_id: str):
    """Deletes a training run record from the database."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT id FROM training_runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Training run not found")
        conn.execute("DELETE FROM training_runs WHERE id = ?", (run_id,))
        conn.commit()
    return {"message": "Training run deleted successfully", "id": run_id}


@router.get("/predict_csv/{model_id}/{dataset_id}")
@router.get("/predict_batch_csv/{model_id}/{dataset_id}")
def export_predictions_csv(
    model_id: str,
    dataset_id: str,
    extra_cols: Optional[str] = Query(default=None, description="Comma-separated list of extra columns from main table to include in CSV")
):
    """
    Runs batch inference on the target dataset using the specified model,
    and returns a downloadable CSV containing input_id (from sample_ids),
    optional extra columns from the main table, predicted labels/values,
    class probabilities (for classification), and ground truth labels (if available in dataset).
    """
    with get_db_connection() as conn:
        model_row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        if not model_row:
            raise HTTPException(status_code=404, detail="Model not found")
        model_record = dict(model_row)

    weights_path = resolve_storage_path(model_record.get("weights_path"))
    if not weights_path or not weights_path.exists():
        raise HTTPException(status_code=404, detail=f"Model weights file not found on disk: {model_record.get('weights_path')}")

    try:
        model = keras.models.load_model(str(weights_path), custom_objects={"ClusteringLayer": ClusteringLayer})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    try:
        X, y, dataset_record = TrainingService.prepare_model_inputs(model, dataset_id, model_record=model_record)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to prepare dataset inputs: {str(e)}")

    n_samples = len(X[0]) if isinstance(X, list) else len(X)
    sample_ids = dataset_record.get("sample_ids")
    if not sample_ids or len(sample_ids) != n_samples or DataManager._is_default_zero_indexed(list(sample_ids)):
        try:
            loaded_ids = DataManager.load_sample_ids(dataset_id, expected_count=n_samples)
            if loaded_ids and len(loaded_ids) == n_samples:
                sample_ids = loaded_ids
        except Exception:
            pass

    if not sample_ids or len(sample_ids) != n_samples:
        sample_ids = [str(i) for i in range(n_samples)]

    selected_extra_cols = [c.strip() for c in extra_cols.split(",") if c.strip()] if extra_cols else None
    extra_dict = {}
    extra_cols_to_use = []
    if selected_extra_cols:
        try:
            extra_dict = DataManager.load_extra_columns(dataset_id, selected_columns=selected_extra_cols, expected_count=n_samples)
            extra_cols_to_use = [c for c in selected_extra_cols if c in extra_dict]
        except Exception:
            extra_dict = {}
            extra_cols_to_use = []

    try:
        preds = model.predict(X, verbose=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

    task_type = model_record.get("task_type", "classification")

    output = io.StringIO()
    writer = csv.writer(output)

    # Check if preds is from a multi-output model or IDEC model
    q_preds = None
    if isinstance(preds, list):
        outputs = model.outputs if isinstance(model.outputs, list) else [model.output]
        for idx, out_tensor in enumerate(outputs):
            hist = getattr(out_tensor, "_keras_history", None)
            l = hist[0] if isinstance(hist, tuple) else getattr(hist, "operation", None)
            if isinstance(l, ClusteringLayer) or (l and "clustering" in getattr(l, "name", "").lower()):
                if idx < len(preds):
                    q_preds = preds[idx]
                    break
        if q_preds is None:
            # Fallback heuristic: find 2D output with shape (n_samples, n_clusters)
            for p in preds:
                if hasattr(p, "ndim") and p.ndim == 2 and p.shape[1] >= 2 and p.shape[0] == n_samples:
                    if np.all(p >= 0.0) and np.all(p <= 1.001):
                        q_preds = p
                        break

    if q_preds is not None or task_type == "idec_clustering":
        if q_preds is None:
            q_preds = preds[0] if isinstance(preds, list) else preds
        if q_preds.ndim != 2:
            q_preds = q_preds.reshape(n_samples, -1)

        n_clusters = q_preds.shape[1]
        pred_clusters = np.argmax(q_preds, axis=1)
        confidences = np.max(q_preds, axis=1)

        header = ["input_id"] + extra_cols_to_use + ["predicted_cluster", "confidence"]
        if y is not None:
            header.append("ground_truth")
        for c in range(n_clusters):
            header.append(f"cluster_{c}_prob")
        writer.writerow(header)

        y_flat = y.flatten() if y is not None else None
        for i in range(n_samples):
            extra_vals = [str(extra_dict[c][i]) if i < len(extra_dict[c]) else "" for c in extra_cols_to_use]
            row = [sample_ids[i]] + extra_vals + [int(pred_clusters[i]), float(round(confidences[i], 4))]
            if y is not None:
                row.append(int(y_flat[i]) if (y_flat is not None and i < len(y_flat)) else "")
            for c in range(n_clusters):
                row.append(float(round(q_preds[i, c], 4)))
            writer.writerow(row)

    elif task_type == "classification":
        p_arr = preds[0] if isinstance(preds, list) else preds
        if p_arr.ndim > 1 and p_arr.shape[1] > 1:
            n_classes = p_arr.shape[1]
            header = ["input_id"] + extra_cols_to_use + ["predicted_class"]
            if y is not None:
                header.append("ground_truth")
            for c in range(n_classes):
                header.append(f"prob_class_{c}")
            writer.writerow(header)

            pred_classes = np.argmax(p_arr, axis=1)
            y_flat = y.flatten() if (y is not None and y.ndim == 1) else (np.argmax(y, axis=1) if (y is not None and y.ndim > 1 and y.shape[1] > 1) else y)

            for i in range(n_samples):
                extra_vals = [str(extra_dict[c][i]) if i < len(extra_dict[c]) else "" for c in extra_cols_to_use]
                row = [sample_ids[i]] + extra_vals + [int(pred_classes[i])]
                if y is not None:
                    row.append(int(y_flat[i]) if (y_flat is not None and i < len(y_flat)) else "")
                for c in range(n_classes):
                    row.append(float(round(p_arr[i, c], 6)))
                writer.writerow(row)
        else:
            preds_flat = p_arr.flatten()
            pred_classes = (preds_flat >= 0.5).astype(int)
            y_flat = y.flatten() if y is not None else None

            header = ["input_id"] + extra_cols_to_use + ["predicted_class"]
            if y is not None:
                header.append("ground_truth")
            header.append("probability")
            writer.writerow(header)

            for i in range(n_samples):
                extra_vals = [str(extra_dict[c][i]) if i < len(extra_dict[c]) else "" for c in extra_cols_to_use]
                row = [sample_ids[i]] + extra_vals + [int(pred_classes[i])]
                if y is not None:
                    row.append(int(y_flat[i]) if (y_flat is not None and i < len(y_flat)) else "")
                row.append(float(round(preds_flat[i], 6)))
                writer.writerow(row)
    else:
        p_arr = preds[0] if isinstance(preds, list) else preds
        preds_flat = p_arr.reshape(n_samples, -1)
        y_flat = y.reshape(n_samples, -1) if y is not None else None
        n_out = preds_flat.shape[1]

        header = ["input_id"] + extra_cols_to_use
        for c in range(n_out):
            header.append(f"pred_{c}" if n_out > 1 else "prediction")
        if y is not None:
            for c in range(n_out):
                header.append(f"ground_truth_{c}" if n_out > 1 else "ground_truth")
        writer.writerow(header)

        for i in range(n_samples):
            extra_vals = [str(extra_dict[c][i]) if i < len(extra_dict[c]) else "" for c in extra_cols_to_use]
            row = [sample_ids[i]] + extra_vals
            for c in range(n_out):
                row.append(float(round(preds_flat[i, c], 6)))
            if y is not None and y_flat is not None:
                for c in range(min(n_out, y_flat.shape[1])):
                    row.append(float(round(y_flat[i, c], 6)))
            writer.writerow(row)

    csv_content = output.getvalue()
    filename = f"predictions_{model_id[:8]}_{dataset_id[:8]}.csv"

    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


# Alias for batch inference CSV export
predict_batch_csv = export_predictions_csv

