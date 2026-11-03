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
