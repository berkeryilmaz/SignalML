import io
import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Response, Query
import numpy as np

from app.core.config import EXPORTS_DIR, DATASETS_DIR
from app.core.database import get_db_connection
from app.services.data_manager import DataManager
from app.services.signal_processing import (
    run_pipeline,
    export_signals,
    to_fixed_length
)
from app.schemas.signal import (
    SignalWaveformRequest,
    SignalProcessRequest,
    SignalExportRequest
)

router = APIRouter(prefix="/api/signals", tags=["Signal Processing"])


@router.post("/waveform")
def get_waveforms(payload: SignalWaveformRequest):
    """
    Returns time series waveform points for the specified sample row indices.
    Supports downsampling for responsive browser rendering.
    """
    try:
        data, labels, record = DataManager.load_dataset_arrays(payload.dataset_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim == 0 or data.size == 0:
        return {
            "dataset_id": payload.dataset_id,
            "total_samples": 0,
            "sequence_length": 0,
            "waveforms": []
        }

    num_samples, seq_len = data.shape
    waveforms = []

    for idx in payload.sample_indices:
        if 0 <= idx < num_samples:
            sig = np.nan_to_num(data[idx], copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            finite_sig = sig[np.isfinite(sig)]
            # Downsample if length exceeds max_points
            if seq_len > payload.max_points and payload.max_points > 0:
                step = int(np.ceil(seq_len / payload.max_points))
                raw_pts = sig[::step]
                x_axis = list(range(0, seq_len, step))
            else:
                raw_pts = sig
                x_axis = list(range(seq_len))

            points = [float(v) if np.isfinite(v) else 0.0 for v in raw_pts]

            label_val = None
            if labels is not None and idx < len(labels):
                raw_l = labels[idx]
                label_val = int(raw_l) if np.issubdtype(type(raw_l), np.integer) else str(raw_l)

            min_v = float(np.min(finite_sig)) if len(finite_sig) > 0 and np.isfinite(np.min(finite_sig)) else 0.0
            max_v = float(np.max(finite_sig)) if len(finite_sig) > 0 and np.isfinite(np.max(finite_sig)) else 0.0
            mean_v = float(np.mean(finite_sig)) if len(finite_sig) > 0 and np.isfinite(np.mean(finite_sig)) else 0.0
            std_v = float(np.std(finite_sig)) if len(finite_sig) > 0 and np.isfinite(np.std(finite_sig)) else 0.0

            waveforms.append({
                "sample_index": idx,
                "label": label_val,
                "length": seq_len,
                "x": x_axis,
                "y": points,
                "min": min_v,
                "max": max_v,
                "mean": mean_v,
                "std": std_v
            })

    return {
        "dataset_id": payload.dataset_id,
        "total_samples": num_samples,
        "sequence_length": seq_len,
        "waveforms": waveforms
    }


@router.post("/process_preview")
def process_preview(payload: SignalProcessRequest):
    """
    Applies preprocessing pipeline to sample signals and returns before/after waveforms.
    """
    try:
        data, labels, record = DataManager.load_dataset_arrays(payload.dataset_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if len(data) == 0:
        return {
            "dataset_id": payload.dataset_id,
            "preview_count": 0,
            "pipeline_steps_applied": 0,
            "results": []
        }

    # Take first 3 signals for preview
    preview_indices = [0, 1, 2] if len(data) >= 3 else list(range(len(data)))
    orig_signals = np.nan_to_num(data[preview_indices], copy=True, nan=0.0, posinf=0.0, neginf=0.0)

    steps = [s.model_dump() for s in payload.pipeline]
    try:
        proc_signals = run_pipeline(orig_signals, steps)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Pipeline error: {str(e)}")

    if proc_signals.ndim == 1:
        proc_signals = proc_signals.reshape(1, -1)

    proc_signals = np.nan_to_num(proc_signals, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    results = []
    for i, idx in enumerate(preview_indices):
        orig_row = orig_signals[i]
        proc_row = proc_signals[i]
        orig_y = [float(v) if np.isfinite(v) else 0.0 for v in orig_row.tolist()]
        proc_y = [float(v) if np.isfinite(v) else 0.0 for v in proc_row.tolist()]
        orig_finite = [v for v in orig_y if np.isfinite(v)]
        proc_finite = [v for v in proc_y if np.isfinite(v)]

        results.append({
            "sample_index": idx,
            "original": {
                "length": orig_signals.shape[1],
                "y": orig_y,
                "min": float(np.min(orig_finite)) if orig_finite else 0.0,
                "max": float(np.max(orig_finite)) if orig_finite else 0.0
            },
            "processed": {
                "length": proc_signals.shape[1],
                "y": proc_y,
                "min": float(np.min(proc_finite)) if proc_finite else 0.0,
                "max": float(np.max(proc_finite)) if proc_finite else 0.0
            }
        })

    return {
        "dataset_id": payload.dataset_id,
        "preview_count": len(results),
        "pipeline_steps_applied": len(steps),
        "results": results
    }


@router.post("/apply_and_save")
def apply_and_save_pipeline(payload: SignalProcessRequest):
    """
    Applies the preprocessing pipeline to the entire dataset and saves as a new dataset.
    """
    try:
        data, labels, record = DataManager.load_dataset_arrays(payload.dataset_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    steps = [s.model_dump() for s in payload.pipeline]
    try:
        processed_data = run_pipeline(data, steps)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Pipeline processing failed: {str(e)}")

    project_id = record["project_id"]
    new_name = payload.new_dataset_name or f"{record['name']} (Processed)"

    new_id = str(uuid.uuid4())
    dest_path = DATASETS_DIR / project_id / f"{new_id}.npz"
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {"data": processed_data}
    if labels is not None:
        save_dict["labels"] = labels
    np.savez_compressed(str(dest_path), **save_dict)

    now = datetime.now(timezone.utc).isoformat()
    meta = {
        "original_dataset_id": payload.dataset_id,
        "pipeline": steps,
        "shape": list(processed_data.shape)
    }

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO datasets (
                id, project_id, name, file_path, file_format,
                row_count, col_count, has_header, label_col, created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id, project_id, new_name, str(dest_path), "npz",
                processed_data.shape[0], processed_data.shape[1], 0, record.get("label_col"),
                now, json.dumps(meta)
            )
        )
        conn.execute(
            "UPDATE projects SET active_dataset_id = ?, updated_at = ? WHERE id = ?",
            (new_id, now, project_id)
        )
        conn.commit()

    return {
        "new_dataset_id": new_id,
        "name": new_name,
        "row_count": processed_data.shape[0],
        "col_count": processed_data.shape[1]
    }


@router.post("/export")
def export_signal_dataset(payload: SignalExportRequest):
    """
    Exports signals with exact N sample points per row.
    Guarantees no extraneous index columns and verifies exact row/column dimensions.
    """
    try:
        data, labels, record = DataManager.load_dataset_arrays(payload.dataset_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Apply pipeline if provided
    if payload.pipeline:
        steps = [s.model_dump() for s in payload.pipeline]
        data = run_pipeline(data, steps)

    if payload.target_length is not None and payload.target_length > 0:
        data = to_fixed_length(data, payload.target_length)

    content_bytes, mime, verification = export_signals(
        signals=data,
        labels=labels,
        output_format=payload.output_format,
        target_length=payload.target_length,
        separate_labels=payload.separate_labels
    )

    filename = f"signals_export_{payload.target_length or data.shape[1]}_{uuid.uuid4().hex[:6]}.{payload.output_format}"
    return Response(
        content=content_bytes,
        media_type=mime,
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "X-Signal-Rows": str(verification["num_rows"]),
            "X-Signal-Cols": str(verification["num_cols"])
        }
    )


@router.post("/group_waveform_analysis")
def group_waveform_analysis(payload: Dict[str, Any]):
    """
    Computes faded waveforms and prominent average for a set of sample indices or label class.
    """
    dataset_id = payload.get("dataset_id")
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id is required")

    try:
        data, labels, record = DataManager.load_dataset_arrays(dataset_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    if data.ndim == 1:
        data = data.reshape(1, -1)

    num_samples, seq_len = data.shape
    sample_indices = payload.get("sample_indices")
    label_value = payload.get("label_value")

    if sample_indices is not None:
        indices = [int(i) for i in sample_indices if 0 <= int(i) < num_samples]
    elif label_value is not None and labels is not None:
        indices = [i for i, l in enumerate(labels) if str(l) == str(label_value)]
    else:
        indices = list(range(min(num_samples, 50)))

    if len(indices) == 0:
        raise HTTPException(status_code=400, detail="No matching samples found")

    group_signals = data[indices]
    clean_signals = np.nan_to_num(group_signals, nan=0.0, posinf=0.0, neginf=0.0)
    mean_wf = np.mean(clean_signals, axis=0)
    std_wf = np.std(clean_signals, axis=0)
    min_wf = np.min(clean_signals, axis=0)
    max_wf = np.max(clean_signals, axis=0)

    max_points = int(payload.get("max_points", 1000))
    if seq_len > max_points and max_points > 0:
        step = int(np.ceil(seq_len / max_points))
        x_axis = list(range(0, seq_len, step))
        mean_y = [float(v) if np.isfinite(v) else 0.0 for v in mean_wf[::step]]
        std_y = [float(v) if np.isfinite(v) else 0.0 for v in std_wf[::step]]
        min_y = [float(v) if np.isfinite(v) else 0.0 for v in min_wf[::step]]
        max_y = [float(v) if np.isfinite(v) else 0.0 for v in max_wf[::step]]
    else:
        step = 1
        x_axis = list(range(seq_len))
        mean_y = [float(v) if np.isfinite(v) else 0.0 for v in mean_wf]
        std_y = [float(v) if np.isfinite(v) else 0.0 for v in std_wf]
        min_y = [float(v) if np.isfinite(v) else 0.0 for v in min_wf]
        max_y = [float(v) if np.isfinite(v) else 0.0 for v in max_wf]

    max_wfs = int(payload.get("max_waveforms", 100))
    if len(indices) > max_wfs:
        sub_indices = np.linspace(0, len(indices) - 1, max_wfs, dtype=int)
        sampled_signals = group_signals[sub_indices]
        sampled_indices = [indices[i] for i in sub_indices]
    else:
        sampled_signals = group_signals
        sampled_indices = indices

    rendered = []
    for s_idx, sig in zip(sampled_indices, sampled_signals):
        sig_down = sig[::step] if step > 1 else sig
        rendered.append({
            "sample_index": int(s_idx),
            "y": [float(v) if np.isfinite(v) else 0.0 for v in sig_down.tolist()]
        })

    return {
        "dataset_id": dataset_id,
        "sample_count": len(indices),
        "sequence_length": seq_len,
        "x": x_axis,
        "mean": [float(v) if np.isfinite(v) else 0.0 for v in mean_y],
        "std": [float(v) if np.isfinite(v) else 0.0 for v in std_y],
        "upper_std": [float(m + s) if np.isfinite(m + s) else 0.0 for m, s in zip(mean_y, std_y)],
        "lower_std": [float(m - s) if np.isfinite(m - s) else 0.0 for m, s in zip(mean_y, std_y)],
        "min": [float(v) if np.isfinite(v) else 0.0 for v in min_y],
        "max": [float(v) if np.isfinite(v) else 0.0 for v in max_y],
        "waveforms": rendered
    }
