from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
from app.services.feature_extraction import FeatureExtractor, FeatureExtractionError
from app.schemas.clustering import FeatureExtractionRequest
from app.core.database import get_db_connection

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
    """Exports extracted feature matrix as a CSV file download (with Input ID column)."""
    try:
        features, labels, record = FeatureExtractor.load_feature_matrix(feature_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Feature not found: {str(e)}")

    import io
    import csv
    from fastapi.responses import StreamingResponse

    output = io.StringIO()
    writer = csv.writer(output)

    num_samples, dim = features.shape
    header = ["input_id"] + [f"feat_{i}" for i in range(dim)]
    if labels is not None:
        header.append("label")
    writer.writerow(header)

    for i in range(num_samples):
        row = [i] + [float(val) for val in features[i]]
        if labels is not None:
            try:
                row.append(float(labels[i]))
            except Exception:
                row.append(str(labels[i]))
        writer.writerow(row)

    output.seek(0)
    filename = f"features_{feature_id[:8]}_{record.get('layer_id', 'layer').replace('/', '_')}.csv"
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
