from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Body
from app.services.feature_extraction import FeatureExtractor, FeatureExtractionError
from app.schemas.clustering import FeatureExtractionRequest
from app.core.database import get_db_connection
from app.core.config import resolve_storage_path

router = APIRouter(prefix="/api/features", tags=["Feature Extraction"])


@router.get("/layers/{model_id}")
def get_model_layers(model_id: str):
    """
    Inspects trained Keras model and returns all selectable intermediate layers with shapes.
    """
    try:
        layers_info = FeatureExtractor.get_model_layers(model_id)
        return {
            "model_id": model_id,
            "layers": layers_info
        }
    except FeatureExtractionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Layer inspection failed: {str(e)}")


@router.post("/extract")
def extract_features(payload: FeatureExtractionRequest):
    """
    Computes intermediate layer tensor outputs for the dataset using a Keras submodel.
    Saves features persistently and returns metadata and sample preview.
    """
    with get_db_connection() as conn:
        row = conn.execute("SELECT project_id FROM models WHERE id = ?", (payload.model_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Model not found")
        project_id = row["project_id"]

    try:
        res = FeatureExtractor.extract_features(
            project_id=project_id,
            model_id=payload.model_id,
            dataset_id=payload.dataset_id,
            layer_name=payload.layer_name,
            sample_limit=payload.sample_limit,
            batch_size=payload.batch_size
        )
        return res
    except FeatureExtractionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")


@router.get("/list/{project_id}")
def list_extracted_features(project_id: str):
    """Lists all saved feature sets in the project."""
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM features WHERE project_id = ? ORDER BY created_at DESC", (project_id,)).fetchall()
        import json
        res = []
        for r in rows:
            d = dict(r)
            d["shape"] = json.loads(d.get("shape_json", "[]"))
            res.append(d)
        return res


@router.get("/{feature_id}/export_csv")
def export_features_csv(feature_id: str):
    """Exports extracted feature matrix as a CSV file download (with Input ID column and custom layer name)."""
    try:
        features, labels, record = FeatureExtractor.load_feature_matrix(feature_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Feature not found: {str(e)}")

    import io
    import csv
    import numpy as np
    from fastapi.responses import StreamingResponse

    num_samples, dim = features.shape
    sample_ids = [str(i) for i in range(num_samples)]

    # Try loading sample_ids from feature .npz or parent dataset .npz
    try:
        fp = resolve_storage_path(record.get("file_path"))
        if fp and fp.exists():
            with np.load(str(fp), allow_pickle=False) as fnpz:
                if "sample_ids" in fnpz:
                    s_ids = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in fnpz["sample_ids"]]
                    if len(s_ids) == num_samples:
                        sample_ids = s_ids
        if sample_ids == [str(i) for i in range(num_samples)] and record.get("dataset_id"):
            with get_db_connection() as conn:
                ds_row = conn.execute("SELECT file_path FROM datasets WHERE id = ?", (record["dataset_id"],)).fetchone()
                if ds_row and ds_row["file_path"]:
                    dsp = resolve_storage_path(ds_row["file_path"])
                    if dsp and dsp.exists():
                        with np.load(str(dsp), allow_pickle=False) as dnpz:
                            if "sample_ids" in dnpz:
                                s_ids = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in dnpz["sample_ids"]]
                                if len(s_ids) >= num_samples:
                                    sample_ids = s_ids[:num_samples]
    except Exception:
        pass

    layer_prefix = (record.get("layer_id") or "feat").replace(" ", "_")

    output = io.StringIO()
    writer = csv.writer(output)

    header = ["input_id"] + [f"{layer_prefix}_{i}" for i in range(dim)]
    if labels is not None:
        header.append("label")
    writer.writerow(header)

    for i in range(num_samples):
        row = [sample_ids[i]] + [float(val) for val in features[i]]
        if labels is not None:
            try:
                row.append(float(labels[i]))
            except Exception:
                row.append(str(labels[i]))
        writer.writerow(row)

    output.seek(0)
    filename = f"features_{feature_id[:8]}_{layer_prefix.replace('/', '_')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@router.put("/{feature_id}")
def update_feature_name(feature_id: str, payload: Dict[str, Any] = Body(...)):
    """Updates custom name or metadata for a saved feature extraction."""
    new_name = payload.get("name") or payload.get("custom_name")
    if not new_name:
        raise HTTPException(status_code=400, detail="Name is required")

    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM features WHERE id = ?", (feature_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Feature extraction not found")
        conn.execute("UPDATE features SET layer_id = ? WHERE id = ?", (new_name, feature_id))
        conn.commit()

    return {"status": "success", "feature_id": feature_id, "custom_name": new_name}


@router.delete("/{feature_id}")
def delete_feature(feature_id: str):
    """Deletes saved feature set from database and disk."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM features WHERE id = ?", (feature_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Feature extraction not found")
        file_path = row["file_path"]

    if file_path:
        p = resolve_storage_path(file_path)
        if p and p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    with get_db_connection() as conn:
        conn.execute("DELETE FROM features WHERE id = ?", (feature_id,))
        conn.commit()

    return {"status": "success", "deleted_feature_id": feature_id}
