from typing import Dict, Any, List, Optional
import json
from fastapi import APIRouter, HTTPException, Query
from app.core.responses import SafeJSONResponse, sanitize_for_json
from app.services.clustering_service import ClusteringService
from app.schemas.clustering import ClusteringRequest
from app.core.database import get_db_connection

router = APIRouter(prefix="/api/clustering", tags=["Clustering & PCA"], default_response_class=SafeJSONResponse)


@router.post("/run", response_class=SafeJSONResponse)
def run_clustering(payload: ClusteringRequest):
    """
    Executes K-Means or DBSCAN clustering along with 2D/3D PCA.
    Returns 3D scatter plot coordinates, cluster distribution, and evaluation metrics.
    """
    # Look up project_id from source
    with get_db_connection() as conn:
        if payload.source_type in ("dataset", "processed"):
            row = conn.execute("SELECT project_id FROM datasets WHERE id = ?", (payload.source_id,)).fetchone()
        else:
            row = conn.execute("SELECT project_id FROM features WHERE id = ?", (payload.source_id,)).fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Source dataset or features not found")
        project_id = row["project_id"]

    try:
        results = ClusteringService.run_clustering(
            project_id=project_id,
            source_type=payload.source_type,
            source_id=payload.source_id,
            algorithm=payload.algorithm,
            n_clusters=payload.n_clusters,
            init=payload.init,
            n_init=payload.n_init,
            eps=payload.eps,
            min_samples=payload.min_samples,
            metric=payload.metric,
            standardize=payload.standardize,
            random_state=payload.random_state,
            compute_pca=payload.compute_pca
        )
        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Clustering execution failed: {str(e)}")


@router.get("/list/{project_id}", response_class=SafeJSONResponse)
def list_clusterings(project_id: str):
    """Lists saved clustering runs for a project."""
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM clusterings WHERE project_id = ? ORDER BY created_at DESC", (project_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["params"] = sanitize_for_json(json.loads(d.get("params_json", "{}")))
            except Exception:
                d["params"] = {}
            try:
                d["metrics"] = sanitize_for_json(json.loads(d.get("metrics_json", "{}")))
            except Exception:
                d["metrics"] = {}
            out.append(d)
        return out


@router.get("/group_waveforms/{clustering_id}/{cluster_label}", response_model=Dict[str, Any], response_class=SafeJSONResponse)
def get_cluster_group_waveforms(
    clustering_id: str,
    cluster_label: int,
    max_waveforms: int = Query(default=100, ge=1, le=500),
    max_points: int = Query(default=1000, ge=50, le=5000)
):
    """
    Returns individual waveforms (for faded rendering) and mean/std summary (for prominent rendering)
    for a specific cluster group.
    """
    try:
        return ClusteringService.get_cluster_group_waveforms(
            clustering_id=clustering_id,
            cluster_label=cluster_label,
            max_waveforms=max_waveforms,
            max_points=max_points
        )
    except HTTPException:
        raise
    except (ValueError, FileNotFoundError) as e:
        status = 404 if "not found" in str(e).lower() else 400
        raise HTTPException(status_code=status, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
